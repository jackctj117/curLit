"""Durable FX emergency-order attempts and their evidence model (CL-pksi).

The kill-switch manager (``src/risk/kill_switches.py``) closes or halves FX
positions in an emergency. CL-o9sq made it fence a symbol while such an
order's outcome is unknown, but the fence lived only in memory: nothing ever
resolved it, and a restart forgot it. This module supplies the missing parts:

* :class:`EmergencyAttempt` — one emergency order attempt, persisted BEFORE the
  broker call (``SUBMITTING``) and updated on every status change
  (migration 025, ``fx_emergency_attempts``).
* :class:`DeriskEvidence` — the ONLY input that can move an attempt out of an
  unresolved state: a streamed fill for the attempt's client id, a broker
  order lookup, or the OMS's own placement outcome. A position snapshot is not
  representable here on purpose — a flat book proves nothing about whether an
  order is still working.
* :func:`apply_evidence` — the pure state machine. Cumulative fills are kept
  per venue transaction id (so a redelivered fill is counted once, across
  restarts too) and are modelled independently of terminal status: a partial
  fill followed by a cancel is ``PARTIAL_TERMINAL`` — neither done nor
  untouched — and stays fenced until an operator reconciles it.
* :class:`SqlEmergencyAttemptStore` / :class:`InMemoryEmergencyAttemptStore` —
  persistence. SQL transactions contain only SQL; callers never hold one
  across a broker call.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy import text

from src.execution.broker import OrderStatus, canonical_symbol
from src.execution.oms import SubmissionResult, SubmissionStatus

logger = logging.getLogger(__name__)


class AttemptStatus(StrEnum):
    """Lifecycle of one emergency order attempt (CL-pksi)."""

    #: Persisted, broker call not yet answered. A crash here is ambiguous.
    SUBMITTING = "SUBMITTING"
    #: Broker acknowledged a live order (or a fill below the requested size).
    WORKING = "WORKING"
    #: Outcome unknown (lost response, unparseable acceptance, generic cancel).
    UNKNOWN = "UNKNOWN"
    #: Terminal at the broker with a PARTIAL fill. Fenced until an operator
    #: reconciles: resubmitting the full close could over-close, and treating
    #: it as complete would hide residual exposure.
    PARTIAL_TERMINAL = "PARTIAL_TERMINAL"
    #: Broker evidence shows the requested quantity filled.
    FILLED = "FILLED"
    #: Broker rejected it; verified zero fill.
    REJECTED = "REJECTED"
    #: Broker cancelled it; verified zero fill.
    CANCELLED = "CANCELLED"
    #: Broker confirms no order with this client id exists (after a grace).
    NOT_FOUND = "NOT_FOUND"
    #: The OMS decided not to send anything (already at target, dust,
    #: blocked as non-reducing). No broker order exists.
    NOT_SUBMITTED = "NOT_SUBMITTED"
    #: An operator reconciled the attempt and released its fence explicitly.
    OPERATOR_RELEASED = "OPERATOR_RELEASED"


#: States that keep the symbol fenced: the broker may still change the
#: position (or already did, by an amount not yet reconciled).
UNRESOLVED_STATUSES: frozenset[AttemptStatus] = frozenset(
    {
        AttemptStatus.SUBMITTING,
        AttemptStatus.WORKING,
        AttemptStatus.UNKNOWN,
        AttemptStatus.PARTIAL_TERMINAL,
    }
)

#: Verified terminal outcomes with ZERO fill: the position is untouched, so
#: one bounded retry against the ORIGINAL target is safe on a later tick.
RETRYABLE_STATUSES: frozenset[AttemptStatus] = frozenset(
    {
        AttemptStatus.REJECTED,
        AttemptStatus.CANCELLED,
        AttemptStatus.NOT_FOUND,
        AttemptStatus.NOT_SUBMITTED,
    }
)

#: Fill completeness tolerance in units. OANDA accepts whole units only (the
#: adapter sends ``int(quantity)``) and the OMS treats < 1 unit as dust
#: (``OrderManager._min_trade_size``), so a fill within one unit of the
#: requested size is the whole order.
FILL_TOLERANCE_UNITS: float = 1.0

#: How long after the attempt was created a broker "no such order" answer
#: is trusted as proof that nothing was created. OANDA creates the order
#: synchronously inside the POST, which the adapter bounds at 10 s per hop and
#: 3 redirect hops (30 s); 120 s is 4x that, so an order still being created
#: cannot be mistaken for one that never existed.
NOT_FOUND_GRACE_SEC: float = 120.0

EVIDENCE_FILL_EVENT = "fill_event"
EVIDENCE_ORDER_LOOKUP = "order_lookup"
EVIDENCE_SUBMISSION = "submission_outcome"
EVIDENCE_OPERATOR = "operator_release"


def _now() -> datetime:
    return datetime.now(UTC)


def _as_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True)
class EmergencyAttempt:
    """One kill-switch emergency order attempt (one row of migration 025)."""

    intent_id: str
    client_order_id: str
    action: str
    symbol: str  # canonical (fence key)
    route_symbol: str  # broker routing form
    original_qty: float  # net position when the risk decision was made
    target: float  # the FIXED target position for this action/symbol
    requested_qty: float  # absolute units requested (OMS-reported when known)
    status: AttemptStatus
    created_at: datetime
    updated_at: datetime
    broker_order_id: str | None = None
    #: venue fill transaction id -> absolute units (idempotency key)
    fills: dict[str, float] = field(default_factory=dict)
    #: Largest cumulative filled quantity a broker order lookup reported. The
    #: lookup and the fill transactions describe the SAME order, so the
    #: cumulative fill is their max, never their sum.
    reported_fill_qty: float = 0.0
    last_evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.intent_id and self.client_order_id, "attempt identity required"
        assert self.symbol == canonical_symbol(self.symbol), "symbol must be canonical"
        assert self.requested_qty >= 0, "requested_qty must be non-negative"

    @property
    def cumulative_fill_qty(self) -> float:
        return max(float(sum(self.fills.values())), float(self.reported_fill_qty))

    @property
    def unresolved(self) -> bool:
        return self.status in UNRESOLVED_STATUSES

    def evidence_json(self) -> str:
        return json.dumps(
            {
                "fills": self.fills,
                "reported_fill_qty": self.reported_fill_qty,
                "last": self.last_evidence,
            },
            sort_keys=True,
            default=str,
        )


@dataclass(frozen=True)
class DeriskEvidence:
    """Verified broker evidence about one emergency attempt (CL-pksi).

    ``source`` is one of the ``EVIDENCE_*`` constants. A position snapshot
    can never be expressed as evidence: it cannot distinguish "the order
    filled" from "the order is still working" or "the book is stale".
    """

    source: str
    client_order_id: str | None = None
    order_id: str | None = None
    #: Broker order state from a lookup (``order_lookup`` only).
    order_state: OrderStatus | None = None
    #: The broker confirmed no such order exists (``order_lookup`` only).
    not_found: bool = False
    #: A venue fill transaction and its absolute units.
    fill_transaction_id: str | None = None
    fill_qty: float | None = None
    #: Venue-reported cumulative filled quantity from a lookup; None = unknown.
    filled_quantity: float | None = None
    #: OMS placement outcome (``submission_outcome`` only).
    submission: SubmissionResult | None = None
    detail: str = ""


def _record_fill(fills: dict[str, float], ev: DeriskEvidence) -> dict[str, float]:
    if ev.fill_transaction_id and ev.fill_qty is not None:
        qty = abs(float(ev.fill_qty))
        if ev.fill_transaction_id in fills:
            logger.info(
                "emergency attempt: fill %s already recorded — duplicate delivery ignored",
                ev.fill_transaction_id,
            )
            return fills
        merged = dict(fills)
        merged[ev.fill_transaction_id] = qty
        return merged
    return fills


def _complete(cumulative: float, requested: float) -> bool:
    return requested > 0 and cumulative >= requested - FILL_TOLERANCE_UNITS + 1e-9


def _from_submission(
    attempt: EmergencyAttempt, sub: SubmissionResult, cumulative: float
) -> AttemptStatus:
    status = sub.status
    if status is SubmissionStatus.FILLED:
        return AttemptStatus.FILLED
    if status in (SubmissionStatus.AT_TARGET, SubmissionStatus.SKIPPED, SubmissionStatus.BLOCKED):
        return AttemptStatus.NOT_SUBMITTED
    if status is SubmissionStatus.REJECTED:
        # A synchronous broker reject is terminal with zero fill — unless a
        # fill for this client id has already been seen (then it is partial).
        return AttemptStatus.REJECTED if cumulative == 0 else AttemptStatus.PARTIAL_TERMINAL
    if status is SubmissionStatus.WORKING:
        return AttemptStatus.WORKING
    return AttemptStatus.UNKNOWN


def apply_evidence(
    attempt: EmergencyAttempt,
    ev: DeriskEvidence,
    *,
    now: datetime | None = None,
    not_found_grace_sec: float = NOT_FOUND_GRACE_SEC,
) -> EmergencyAttempt:
    """Pure transition: the attempt after ``ev`` (CL-pksi).

    Rules (the oracle the tests encode):

    * Fills accumulate per transaction id; a repeated id changes nothing.
    * A resolved attempt never changes status again (late fills are still
      recorded for audit). Operator release is a separate, explicit path.
    * Complete = cumulative fill within :data:`FILL_TOLERANCE_UNITS` of the
      requested size. A fill short of that keeps the attempt WORKING.
    * A terminal broker state (cancelled / rejected / filled-short) with a
      partial cumulative fill is PARTIAL_TERMINAL (still fenced).
    * Cancelled/rejected is a retryable zero-fill outcome ONLY when the
      broker reported zero filled quantity and no fill was ever seen.
    * "No such order" is trusted only after ``not_found_grace_sec`` since the
      attempt was created, and never when a fill was seen.
    """
    when = now or _now()
    fills = _record_fill(attempt.fills, ev)
    reported = attempt.reported_fill_qty
    if ev.filled_quantity is not None:
        # Monotonic: a lagging lookup can never undo a fill already seen.
        reported = max(reported, abs(float(ev.filled_quantity)))
    cumulative = max(float(sum(fills.values())), reported)
    requested = attempt.requested_qty
    if ev.submission is not None and ev.submission.requested_qty is not None:
        requested = float(ev.submission.requested_qty)
    order_id = ev.order_id or attempt.broker_order_id
    if ev.submission is not None and ev.submission.order_id and not attempt.broker_order_id:
        order_id = ev.submission.order_id

    status = attempt.status
    if attempt.unresolved and attempt.status is not AttemptStatus.PARTIAL_TERMINAL:
        if ev.source == EVIDENCE_SUBMISSION and ev.submission is not None:
            status = _from_submission(attempt, ev.submission, cumulative)
            if status is AttemptStatus.FILLED and ev.submission.order_id:
                # Synchronous fill: the placement response IS the fill txn.
                fills = _record_fill(
                    fills,
                    DeriskEvidence(
                        source=ev.source,
                        fill_transaction_id=ev.submission.order_id,
                        fill_qty=requested,
                    ),
                )
                cumulative = max(float(sum(fills.values())), reported)
            if status is AttemptStatus.FILLED and not _complete(cumulative, requested):
                # A sync FILLED with a short fill amount is not "done".
                status = AttemptStatus.PARTIAL_TERMINAL if cumulative > 0 else AttemptStatus.FILLED
            if status in (AttemptStatus.WORKING, AttemptStatus.UNKNOWN) and _complete(
                cumulative, requested
            ):
                status = AttemptStatus.FILLED  # a fill raced ahead of the response
        elif ev.source == EVIDENCE_FILL_EVENT:
            status = (
                AttemptStatus.FILLED if _complete(cumulative, requested) else AttemptStatus.WORKING
            )
        elif ev.source == EVIDENCE_ORDER_LOOKUP:
            status = _from_lookup(attempt, ev, cumulative, requested, when, not_found_grace_sec)
    elif attempt.status is AttemptStatus.PARTIAL_TERMINAL and _complete(cumulative, requested):
        # A delayed fill completed what looked partial: now verifiably done.
        status = AttemptStatus.FILLED

    last = {
        "source": ev.source,
        "at": when.isoformat(),
        "order_id": order_id,
        "order_state": ev.order_state.value if ev.order_state is not None else None,
        "not_found": ev.not_found,
        "fill_transaction_id": ev.fill_transaction_id,
        "filled_quantity": ev.filled_quantity,
        "submission": ev.submission.status.value if ev.submission is not None else None,
        "detail": ev.detail[:300],
        "previous_status": attempt.status.value,
    }
    return replace(
        attempt,
        status=status,
        requested_qty=requested,
        broker_order_id=order_id,
        fills=fills,
        reported_fill_qty=reported,
        last_evidence=last,
        updated_at=when,
    )


def _from_lookup(
    attempt: EmergencyAttempt,
    ev: DeriskEvidence,
    cumulative: float,
    requested: float,
    when: datetime,
    grace: float,
) -> AttemptStatus:
    if ev.not_found:
        if cumulative > 0:
            logger.critical(
                "emergency attempt %s: broker says no order but %.4f filled — UNKNOWN",
                attempt.client_order_id,
                cumulative,
            )
            return AttemptStatus.UNKNOWN
        # Anchored on creation: the row is written immediately before the one
        # POST, and later lookups refresh updated_at, which must not reset it.
        age = (when - attempt.created_at).total_seconds()
        if age >= grace:
            return AttemptStatus.NOT_FOUND
        logger.info(
            "emergency attempt %s: not found yet but only %.0fs old (< %.0fs grace) — kept",
            attempt.client_order_id,
            age,
            grace,
        )
        return attempt.status
    state = ev.order_state
    if state in (OrderStatus.PENDING, OrderStatus.PARTIAL):
        return AttemptStatus.WORKING
    if state is OrderStatus.FILLED:
        if _complete(cumulative, requested):
            return AttemptStatus.FILLED
        if ev.filled_quantity is None and cumulative == 0:
            # Filled, but the amount could not be read: not yet verified.
            return attempt.status
        return AttemptStatus.PARTIAL_TERMINAL
    if state in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
        if cumulative > 0:
            return (
                AttemptStatus.FILLED
                if _complete(cumulative, requested)
                else AttemptStatus.PARTIAL_TERMINAL
            )
        if ev.filled_quantity is None:
            # Terminal, but zero fill is not verified: keep fenced.
            return attempt.status
        return AttemptStatus.CANCELLED if state is OrderStatus.CANCELLED else AttemptStatus.REJECTED
    return attempt.status


class EmergencyAttemptStore(Protocol):
    """Persistence for :class:`EmergencyAttempt` (CL-pksi)."""

    def insert(self, attempt: EmergencyAttempt) -> None: ...

    def update(self, attempt: EmergencyAttempt) -> None: ...

    def load_unresolved(self) -> list[EmergencyAttempt]: ...


class InMemoryEmergencyAttemptStore:
    """Process-local store: tests and engines constructed without a DB.

    Survives nothing — production wires :class:`SqlEmergencyAttemptStore`.
    """

    durable = False

    def __init__(self) -> None:
        self._rows: dict[str, EmergencyAttempt] = {}
        self._lock = threading.Lock()

    def insert(self, attempt: EmergencyAttempt) -> None:
        with self._lock:
            if attempt.intent_id in self._rows:
                msg = f"attempt {attempt.intent_id} already exists"
                raise ValueError(msg)
            self._rows[attempt.intent_id] = attempt

    def update(self, attempt: EmergencyAttempt) -> None:
        with self._lock:
            if attempt.intent_id not in self._rows:
                msg = f"attempt {attempt.intent_id} does not exist"
                raise KeyError(msg)
            self._rows[attempt.intent_id] = attempt

    def load_unresolved(self) -> list[EmergencyAttempt]:
        with self._lock:
            return sorted(
                (a for a in self._rows.values() if a.unresolved), key=lambda a: a.created_at
            )


_COLUMNS = (
    "intent_id, client_order_id, action, symbol, route_symbol, original_qty, target, "
    "requested_qty, status, broker_order_id, cumulative_fill_qty, last_evidence, "
    "created_at, updated_at"
)


class SqlEmergencyAttemptStore:
    """``fx_emergency_attempts`` (migration 025) over SQLAlchemy (CL-pksi).

    Portable across Postgres (production) and sqlite (unit tests). Every
    transaction here is SQL only — no broker/network call ever runs inside it.
    """

    durable = True

    def __init__(self, engine: Any) -> None:
        assert engine is not None, "SqlEmergencyAttemptStore requires an engine"
        self.engine = engine

    @staticmethod
    def _params(attempt: EmergencyAttempt) -> dict[str, Any]:
        return {
            "intent_id": attempt.intent_id,
            "client_order_id": attempt.client_order_id,
            "action": attempt.action,
            "symbol": attempt.symbol,
            "route_symbol": attempt.route_symbol,
            "original_qty": float(attempt.original_qty),
            "target": float(attempt.target),
            "requested_qty": float(attempt.requested_qty),
            "status": attempt.status.value,
            "broker_order_id": attempt.broker_order_id,
            "cumulative_fill_qty": attempt.cumulative_fill_qty,
            "last_evidence": attempt.evidence_json(),
            "created_at": attempt.created_at,
            "updated_at": attempt.updated_at,
        }

    def insert(self, attempt: EmergencyAttempt) -> None:
        logger.info(
            "emergency attempts: persisting %s %s %s (%s) before submission",
            attempt.intent_id,
            attempt.action,
            attempt.symbol,
            attempt.status,
        )
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO fx_emergency_attempts ({_COLUMNS}) VALUES "
                    "(:intent_id, :client_order_id, :action, :symbol, :route_symbol, "
                    ":original_qty, :target, :requested_qty, :status, :broker_order_id, "
                    ":cumulative_fill_qty, :last_evidence, :created_at, :updated_at)"
                ),
                self._params(attempt),
            )

    def update(self, attempt: EmergencyAttempt) -> None:
        logger.info(
            "emergency attempts: %s -> %s (filled %.4f / %.4f)",
            attempt.intent_id,
            attempt.status,
            attempt.cumulative_fill_qty,
            attempt.requested_qty,
        )
        with self.engine.begin() as conn:
            updated = conn.execute(
                text(
                    "UPDATE fx_emergency_attempts SET requested_qty = :requested_qty, "
                    "status = :status, broker_order_id = :broker_order_id, "
                    "cumulative_fill_qty = :cumulative_fill_qty, "
                    "last_evidence = :last_evidence, updated_at = :updated_at "
                    "WHERE intent_id = :intent_id"
                ),
                self._params(attempt),
            ).rowcount
        if updated != 1:
            msg = f"emergency attempt {attempt.intent_id} not found for update"
            raise KeyError(msg)

    def load_unresolved(self) -> list[EmergencyAttempt]:
        wanted = sorted(s.value for s in UNRESOLVED_STATUSES)
        placeholders = ", ".join(f":s{i}" for i in range(len(wanted)))
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {_COLUMNS} FROM fx_emergency_attempts "
                    f"WHERE status IN ({placeholders}) ORDER BY created_at"
                ),
                {f"s{i}": v for i, v in enumerate(wanted)},
            ).all()
        return [self._row(r) for r in rows]

    @staticmethod
    def _row(r: Any) -> EmergencyAttempt:
        """Parse one row; a corrupt row RAISES (fail loud: an unreadable
        unresolved attempt must not silently drop its fence)."""
        doc = json.loads(r[11] or "{}")
        if not isinstance(doc, dict):
            msg = f"emergency attempt {r[0]}: last_evidence is not an object"
            raise ValueError(msg)
        fills_raw = doc.get("fills") or {}
        if not isinstance(fills_raw, dict):
            msg = f"emergency attempt {r[0]}: fills is not an object"
            raise ValueError(msg)
        fills = {str(k): float(v) for k, v in fills_raw.items()}
        attempt = EmergencyAttempt(
            intent_id=str(r[0]),
            client_order_id=str(r[1]),
            action=str(r[2]),
            symbol=str(r[3]),
            route_symbol=str(r[4]),
            original_qty=float(r[5]),
            target=float(r[6]),
            requested_qty=float(r[7]),
            status=AttemptStatus(str(r[8])),
            broker_order_id=str(r[9]) if r[9] else None,
            fills=fills,
            reported_fill_qty=float(doc.get("reported_fill_qty") or 0.0),
            last_evidence=dict(doc.get("last") or {}),
            created_at=_as_dt(r[12]),
            updated_at=_as_dt(r[13]),
        )
        if abs(attempt.cumulative_fill_qty - float(r[10])) > 1e-6:
            msg = (
                f"emergency attempt {r[0]}: cumulative_fill_qty {r[10]} disagrees with "
                f"its fill evidence {attempt.cumulative_fill_qty}"
            )
            raise ValueError(msg)
        return attempt
