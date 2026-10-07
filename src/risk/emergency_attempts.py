"""Durable FX emergency-order attempts and their evidence model (CL-pksi).

The kill-switch manager (``src/risk/kill_switches.py``) closes or halves FX
positions in an emergency. CL-o9sq made it fence a symbol while such an
order's outcome is unknown, but the fence lived only in memory: nothing ever
resolved it, and a restart forgot it. This module supplies the missing parts:

* :class:`EmergencyAttempt` — one emergency order attempt, persisted BEFORE the
  broker call (``SUBMITTING``) and updated on every status change
  (migration 027, ``fx_emergency_attempts``).
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
        AttemptStatus.NOT_SUBMITTED,
    }
)

#: Fill completeness tolerance in units. OANDA accepts whole units only (the
#: adapter sends ``int(quantity)``) and the OMS treats < 1 unit as dust
#: (``OrderManager._min_trade_size``), so a fill within one unit of the
#: requested size is the whole order.
FILL_TOLERANCE_UNITS: float = 1.0

# A broker "no such order" answer is recorded but NEVER resolves an attempt:
# OANDA also returns NO_SUCH_ORDER for an order that executed and aged out of
# its recent-order list, so a 404 cannot prove zero execution. Such an attempt
# stays fenced until a fill is seen or an operator reconciles it.

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
    """One kill-switch emergency order attempt (one row of migration 027)."""

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
    #: The kill-switch action episode whose FIXED targets this attempt serves.
    episode_id: str | None = None

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
    #: The broker answered "no such order" (``order_lookup`` only). Recorded
    #: as evidence but never a resolution: it cannot prove zero execution.
    not_found: bool = False
    #: A venue fill transaction and its absolute units.
    fill_transaction_id: str | None = None
    fill_qty: float | None = None
    #: Venue-reported cumulative filled quantity from a lookup; None = unknown.
    filled_quantity: float | None = None
    #: OMS placement outcome (``submission_outcome`` only).
    submission: SubmissionResult | None = None
    #: Absolute units of the order as the BROKER holds it (``order_lookup``):
    #: authoritative over the pre-submission estimate.
    order_qty: float | None = None
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
    # Inclusive boundary: a fill exactly FILL_TOLERANCE_UNITS short is complete
    # (the tolerance exists for broker unit rounding).  The epsilon guards the
    # float comparison; it must widen the band, never narrow it (CL-pksi).
    return requested > 0 and cumulative >= requested - FILL_TOLERANCE_UNITS - 1e-9


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
    * "No such order" never resolves anything (it cannot prove zero
      execution); a synchronous fill counts only its REPORTED quantity.
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
    elif ev.order_qty is not None and ev.order_qty > 0:
        requested = abs(float(ev.order_qty))
    order_id = ev.order_id or attempt.broker_order_id
    if ev.submission is not None and ev.submission.order_id and not attempt.broker_order_id:
        order_id = ev.submission.order_id

    status = attempt.status
    if attempt.unresolved and attempt.status is not AttemptStatus.PARTIAL_TERMINAL:
        if ev.source == EVIDENCE_SUBMISSION and ev.submission is not None:
            status = _from_submission(attempt, ev.submission, cumulative)
            if status is AttemptStatus.FILLED:
                # Synchronous fill: count ONLY the quantity the venue reported,
                # never the requested size. Unknown amount -> UNKNOWN, left for
                # an order lookup or a streamed fill to verify.
                filled = ev.submission.filled_qty
                if filled is None and ev.submission.target_reached:
                    # OMS contract: target_reached is only True once the venue
                    # REPORTED the full executed quantity (OrderManager checks
                    # placed.filled_quantity), so it is evidence of the amount.
                    filled = requested
                if filled is not None and ev.submission.order_id:
                    fills = _record_fill(
                        fills,
                        DeriskEvidence(
                            source=ev.source,
                            fill_transaction_id=ev.submission.order_id,
                            fill_qty=filled,
                        ),
                    )
                elif filled is not None:
                    reported = max(reported, abs(float(filled)))
                cumulative = max(float(sum(fills.values())), reported)
                if _complete(cumulative, requested):
                    status = AttemptStatus.FILLED
                elif filled is None and cumulative == 0:
                    status = AttemptStatus.UNKNOWN
                else:
                    status = AttemptStatus.PARTIAL_TERMINAL
            if status in (AttemptStatus.WORKING, AttemptStatus.UNKNOWN) and _complete(
                cumulative, requested
            ):
                status = AttemptStatus.FILLED  # a fill raced ahead of the response
        elif ev.source == EVIDENCE_FILL_EVENT:
            if attempt.status is AttemptStatus.SUBMITTING:
                # The size actually sent is not known yet (the OMS re-reads
                # the book and may send more than the decision estimate):
                # record the fill, decide once the submission outcome lands.
                status = AttemptStatus.SUBMITTING
            else:
                status = (
                    AttemptStatus.FILLED
                    if _complete(cumulative, requested)
                    else AttemptStatus.WORKING
                )
        elif ev.source == EVIDENCE_ORDER_LOOKUP:
            status = _from_lookup(attempt, ev, cumulative, requested)
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
) -> AttemptStatus:
    if ev.not_found:
        if cumulative > 0:
            logger.critical(
                "emergency attempt %s: broker says no order but %.4f filled — UNKNOWN",
                attempt.client_order_id,
                cumulative,
            )
            return AttemptStatus.UNKNOWN
        logger.critical(
            "emergency attempt %s: broker reports no such order — NOT proof of zero "
            "execution (may have aged out); fence kept for fill evidence or operator",
            attempt.client_order_id,
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

    def load_for_episodes(self, episode_ids: list[str]) -> list[EmergencyAttempt]: ...

    def save_episode(
        self, episode_id: str, action: str, targets: EpisodeTargets, now: datetime
    ) -> None: ...

    def close_episodes(self, episode_ids: list[str], now: datetime) -> None: ...

    def load_open_episodes(self) -> list[tuple[str, str, EpisodeTargets]]: ...


#: canonical symbol -> (route symbol, FIXED target position) for one action.
EpisodeTargets = dict[str, tuple[str, float]]


def _targets_json(targets: EpisodeTargets) -> str:
    return json.dumps({k: [r, float(t)] for k, (r, t) in targets.items()}, sort_keys=True)


def _targets_from_json(episode_id: str, raw: Any) -> EpisodeTargets:
    doc = json.loads(raw or "{}")
    if not isinstance(doc, dict):
        msg = f"emergency episode {episode_id}: targets is not an object"
        raise ValueError(msg)
    out: EpisodeTargets = {}
    for key, value in doc.items():
        if not isinstance(value, list) or len(value) != 2:
            msg = f"emergency episode {episode_id}: bad target for {key!r}"
            raise ValueError(msg)
        out[str(key)] = (str(value[0]), float(value[1]))
    return out


class InMemoryEmergencyAttemptStore:
    """Process-local store: tests and engines constructed without a DB.

    Survives nothing — production wires :class:`SqlEmergencyAttemptStore`.
    """

    durable = False

    def __init__(self) -> None:
        self._rows: dict[str, EmergencyAttempt] = {}
        self._episodes: dict[str, tuple[str, EpisodeTargets, bool]] = {}
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

    def load_for_episodes(self, episode_ids: list[str]) -> list[EmergencyAttempt]:
        wanted = set(episode_ids)
        with self._lock:
            return sorted(
                (a for a in self._rows.values() if a.episode_id in wanted),
                key=lambda a: a.created_at,
            )

    def save_episode(
        self, episode_id: str, action: str, targets: EpisodeTargets, now: datetime
    ) -> None:
        with self._lock:
            self._episodes[episode_id] = (action, dict(targets), True)

    def close_episodes(self, episode_ids: list[str], now: datetime) -> None:
        with self._lock:
            for eid in episode_ids:
                if eid in self._episodes:
                    action, targets, _ = self._episodes[eid]
                    self._episodes[eid] = (action, targets, False)

    def load_open_episodes(self) -> list[tuple[str, str, EpisodeTargets]]:
        with self._lock:
            return [
                (eid, action, dict(targets))
                for eid, (action, targets, is_open) in self._episodes.items()
                if is_open
            ]


_COLUMNS = (
    "intent_id, client_order_id, action, symbol, route_symbol, original_qty, target, "
    "requested_qty, status, broker_order_id, cumulative_fill_qty, last_evidence, "
    "created_at, updated_at, episode_id"
)


class SqlEmergencyAttemptStore:
    """``fx_emergency_attempts`` (migration 027) over SQLAlchemy (CL-pksi).

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
            "episode_id": attempt.episode_id,
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
                    f"INSERT INTO fx_emergency_attempts ({_COLUMNS}) VALUES "  # nosec B608 — constant columns, bound values
                    "(:intent_id, :client_order_id, :action, :symbol, :route_symbol, "
                    ":original_qty, :target, :requested_qty, :status, :broker_order_id, "
                    ":cumulative_fill_qty, :last_evidence, :created_at, :updated_at, "
                    ":episode_id)"
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
                    f"SELECT {_COLUMNS} FROM fx_emergency_attempts "  # nosec B608 — constant columns, bound placeholders
                    f"WHERE status IN ({placeholders}) ORDER BY created_at"
                ),
                {f"s{i}": v for i, v in enumerate(wanted)},
            ).all()
        return [self._row(r) for r in rows]

    def load_for_episodes(self, episode_ids: list[str]) -> list[EmergencyAttempt]:
        if not episode_ids:
            return []
        placeholders = ", ".join(f":e{i}" for i in range(len(episode_ids)))
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT {_COLUMNS} FROM fx_emergency_attempts "  # nosec B608 — constant columns, bound placeholders
                    f"WHERE episode_id IN ({placeholders}) ORDER BY created_at"
                ),
                {f"e{i}": v for i, v in enumerate(episode_ids)},
            ).all()
        return [self._row(r) for r in rows]

    def save_episode(
        self, episode_id: str, action: str, targets: EpisodeTargets, now: datetime
    ) -> None:
        """Upsert one OPEN episode's fixed targets (SQL only, no network)."""
        logger.info("emergency episodes: persisting %s %s targets %s", episode_id, action, targets)
        params = {
            "id": episode_id,
            "action": action,
            "targets": _targets_json(targets),
            "now": now,
        }
        with self.engine.begin() as conn:
            updated = conn.execute(
                text(
                    "UPDATE fx_emergency_episodes SET targets = :targets, status = 'OPEN', "
                    "updated_at = :now WHERE episode_id = :id"
                ),
                params,
            ).rowcount
            if updated == 0:
                conn.execute(
                    text(
                        "INSERT INTO fx_emergency_episodes "
                        "(episode_id, action, targets, status, created_at, updated_at) "
                        "VALUES (:id, :action, :targets, 'OPEN', :now, :now)"
                    ),
                    params,
                )

    def close_episodes(self, episode_ids: list[str], now: datetime) -> None:
        if not episode_ids:
            return
        logger.info("emergency episodes: closing %s", episode_ids)
        with self.engine.begin() as conn:
            for eid in episode_ids:
                conn.execute(
                    text(
                        "UPDATE fx_emergency_episodes SET status = 'CLOSED', updated_at = :now "
                        "WHERE episode_id = :id"
                    ),
                    {"id": eid, "now": now},
                )

    def load_open_episodes(self) -> list[tuple[str, str, EpisodeTargets]]:
        with self.engine.connect() as conn:
            found = conn.execute(
                text(
                    "SELECT episode_id, action, targets FROM fx_emergency_episodes "
                    "WHERE status = 'OPEN' ORDER BY created_at"
                )
            ).all()
        return [(str(r[0]), str(r[1]), _targets_from_json(str(r[0]), r[2])) for r in found]

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
            episode_id=str(r[14]) if r[14] else None,
        )
        if abs(attempt.cumulative_fill_qty - float(r[10])) > 1e-6:
            msg = (
                f"emergency attempt {r[0]}: cumulative_fill_qty {r[10]} disagrees with "
                f"its fill evidence {attempt.cumulative_fill_qty}"
            )
            raise ValueError(msg)
        return attempt
