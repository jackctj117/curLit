"""Durable account-wide entry halt shared by every execution path (CL-0deu.2).

Before this module "halt" meant three unrelated, process-local things: the FX
OMS's in-memory ``_halted`` flag (cleared by any restart and invisible to the
Alpaca daemons), ``/api/system/halt`` flipping that same flag, and a boot-time
``ALPACA_LEDGER_CLOSE_ONLY`` env var. None covered the whole account, none was
a recorded decision, and nothing could say whether every writer honored it.

The contract here:

* ONE durable record (``trading_halt_state``, migration 024) holds the
  requested :class:`HaltMode`. Every change bumps ``version`` and appends an
  audit row (who / why / when / from where).
* Every writer asks :meth:`TradingHaltStore.entry_decision` IMMEDIATELY before
  submitting new exposure. Anything other than a readable ``ACTIVE`` state
  blocks: an unreadable table, a missing row, or an unknown mode all fail
  CLOSED. Risk-reducing exits are not gated here; each exit manager keeps its
  own reduce-only semantics.
* Resume is explicit: :meth:`resume` requires a reason and a named actor and
  is logged. Restarting a process re-reads the record, so a restart can never
  clear a halt.
* Each path records an acknowledgement (:meth:`observe` / :meth:`acknowledge`)
  of the version it saw at a QUIESCENT point (none of its own entry
  submissions in flight). :meth:`application_status` only reports a halt as
  applied once every registered path has acknowledged the current version or
  reported itself visibly unavailable.

Deliberately out of scope for this slice (tracked as follow-up beads):
cancelling/reconciling opening orders still working at the broker when a halt
lands and keeping them reserved until terminal (needs the CL-0deu.1
reservations and CL-0deu.3 order reconciliation); verified CLOSE_ONLY
reductions (oversize/flip rejection when ownership is unknown); and an
EMERGENCY_FLATTEN action with its own price policy. :meth:`request` refuses
``EMERGENCY_FLATTEN`` until a path can actually honor it, rather than record
a mode no writer implements.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


class HaltMode(StrEnum):
    #: Entries allowed (subject to every other gate).
    ACTIVE = "ACTIVE"
    #: No new exposure; exits continue.
    PAUSE_ENTRIES = "PAUSE_ENTRIES"
    #: No new exposure; only reductions. Verified-reduction checks are a
    #: follow-up — today both non-ACTIVE modes block entries identically.
    CLOSE_ONLY = "CLOSE_ONLY"
    #: Reserved for an explicitly authorized flatten; refused by request().
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"


#: Modes :meth:`TradingHaltStore.request` accepts today. EMERGENCY_FLATTEN is
#: excluded because no execution path implements it yet; recording a mode
#: nothing honors would make "applied" a lie.
REQUESTABLE_HALT_MODES: frozenset[HaltMode] = frozenset(
    {HaltMode.PAUSE_ENTRIES, HaltMode.CLOSE_ONLY}
)

#: Every writer that can open exposure. A halt is applied only when all of
#: these have acknowledged it (or reported visibly unavailable).
PATH_FX_OMS = "fx_oms"
PATH_ALPACA_OPTIONS = "alpaca_options"
PATH_ALPACA_EQUITIES = "alpaca_equities"
EXECUTION_PATHS: tuple[str, ...] = (PATH_FX_OMS, PATH_ALPACA_OPTIONS, PATH_ALPACA_EQUITIES)

#: Machine-readable reason codes carried on every :class:`EntryDecision`.
REASON_ACTIVE = "active"
REASON_HALTED = "halted"
REASON_UNAVAILABLE = "halt_state_unavailable"

#: Optimistic-concurrency retries for a version bump. Two simultaneous
#: requests are rare (operator + kill switch); three attempts is ample and a
#: persistent conflict is surfaced loudly instead of spinning.
_MAX_WRITE_ATTEMPTS = 3


class HaltStateUnavailableError(RuntimeError):
    """The durable halt record could not be read or is invalid."""


@dataclass(frozen=True)
class HaltState:
    mode: HaltMode
    version: int
    reason: str
    source: str
    changed_by: str
    changed_at: datetime | None

    @property
    def entries_allowed(self) -> bool:
        return self.mode is HaltMode.ACTIVE


@dataclass(frozen=True)
class EntryDecision:
    """Result of the pre-submission gate. ``allowed`` is False unless the
    record was read successfully AND says ACTIVE."""

    allowed: bool
    reason_code: str
    mode: HaltMode | None
    version: int | None
    detail: str = ""


@dataclass(frozen=True)
class HaltApplication:
    """Whether the current request has reached every execution path.

    ``paths`` maps each registered path to one of: ``applied`` (acked the
    current version), ``unavailable`` (reported it cannot read the record —
    its entries are blocked fail-closed), ``lagging`` (acked an older
    version), or ``missing`` (never acknowledged).
    """

    state: HaltState
    applied: bool
    paths: dict[str, str] = field(default_factory=dict)


def _as_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _require_text(name: str, value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        msg = f"trading halt: {name} is required and must be non-empty"
        raise ValueError(msg)
    return cleaned


class TradingHaltStore:
    """Reads and writes the durable halt record (migration 024).

    Portable across Postgres (production) and sqlite (unit tests). The only
    network-free work done inside a transaction is the halt record itself;
    callers never hold one of these transactions across a broker call.
    """

    def __init__(self, engine: Any) -> None:
        assert engine is not None, "TradingHaltStore requires a SQLAlchemy engine"
        self.engine = engine

    # -- reads -----------------------------------------------------------

    def read(self) -> HaltState:
        """The current record. Raises :class:`HaltStateUnavailableError` on ANY
        problem — callers deciding on exposure must treat that as blocked."""
        try:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT mode, version, reason, source, changed_by, changed_at "
                        "FROM trading_halt_state WHERE id = 1"
                    )
                ).one_or_none()
        except Exception as exc:
            msg = f"halt record unreadable: {type(exc).__name__}: {exc}"
            raise HaltStateUnavailableError(msg) from exc
        if row is None:
            msg = "halt record missing (migration 024 not applied or row deleted)"
            raise HaltStateUnavailableError(msg)
        try:
            mode = HaltMode(str(row[0]))
            version = int(row[1])
        except (ValueError, TypeError) as exc:
            msg = f"halt record invalid: mode={row[0]!r} version={row[1]!r}"
            raise HaltStateUnavailableError(msg) from exc
        if version < 1:
            msg = f"halt record invalid: version={version}"
            raise HaltStateUnavailableError(msg)
        return HaltState(
            mode=mode,
            version=version,
            reason=str(row[2] or ""),
            source=str(row[3] or ""),
            changed_by=str(row[4] or ""),
            changed_at=_as_dt(row[5]),
        )

    def entry_decision(self) -> EntryDecision:
        """The pre-submission gate. Never raises; fails CLOSED."""
        try:
            state = self.read()
        except HaltStateUnavailableError as exc:
            logger.error("trading halt: entry BLOCKED — %s", exc)
            return EntryDecision(
                allowed=False,
                reason_code=REASON_UNAVAILABLE,
                mode=None,
                version=None,
                detail=str(exc),
            )
        if state.entries_allowed:
            return EntryDecision(
                allowed=True, reason_code=REASON_ACTIVE, mode=state.mode, version=state.version
            )
        return EntryDecision(
            allowed=False,
            reason_code=REASON_HALTED,
            mode=state.mode,
            version=state.version,
            detail=f"{state.mode} v{state.version} by {state.changed_by}: {state.reason}",
        )

    # -- writes ----------------------------------------------------------

    def request(
        self,
        mode: HaltMode,
        *,
        reason: str,
        source: str,
        changed_by: str,
        now: datetime | None = None,
    ) -> HaltState:
        """Record a halt request. Same-mode requests are no-ops (no version
        churn, so repeated kill-switch firings do not reset acknowledgements).
        """
        mode = HaltMode(mode)
        if mode not in REQUESTABLE_HALT_MODES:
            msg = (
                f"trading halt: {mode} cannot be requested — no execution path "
                "implements it yet (see the CL-0deu.2 follow-up beads)"
            )
            raise ValueError(msg)
        return self._write(mode, reason=reason, source=source, changed_by=changed_by, now=now)

    def resume(
        self,
        *,
        reason: str,
        source: str,
        changed_by: str,
        now: datetime | None = None,
    ) -> HaltState:
        """Explicit, attributed return to ACTIVE. No-op when already ACTIVE."""
        return self._write(
            HaltMode.ACTIVE, reason=reason, source=source, changed_by=changed_by, now=now
        )

    def _write(
        self,
        mode: HaltMode,
        *,
        reason: str,
        source: str,
        changed_by: str,
        now: datetime | None,
    ) -> HaltState:
        reason = _require_text("reason", reason)
        source = _require_text("source", source)
        changed_by = _require_text("changed_by", changed_by)
        when = now or datetime.now(UTC)
        for attempt in range(1, _MAX_WRITE_ATTEMPTS + 1):
            current = self.read()  # raises HaltStateUnavailableError: fail loud
            if current.mode is mode:
                logger.info(
                    "trading halt: %s requested by %s but already in effect (v%d) — no change",
                    mode,
                    changed_by,
                    current.version,
                )
                return current
            new_version = current.version + 1
            logger.warning(
                "trading halt: CHANGING %s -> %s (v%d -> v%d) source=%s by=%s reason=%s",
                current.mode,
                mode,
                current.version,
                new_version,
                source,
                changed_by,
                reason,
            )
            with self.engine.begin() as conn:
                # Optimistic concurrency: only the writer that still sees the
                # version it read may bump it; a racing writer retries.
                updated = conn.execute(
                    text(
                        "UPDATE trading_halt_state SET mode = :mode, version = :nv, "
                        "reason = :reason, source = :source, changed_by = :by, "
                        "changed_at = :at WHERE id = 1 AND version = :ov"
                    ),
                    {
                        "mode": mode.value,
                        "nv": new_version,
                        "reason": reason,
                        "source": source,
                        "by": changed_by,
                        "at": when,
                        "ov": current.version,
                    },
                ).rowcount
                if updated == 1:
                    conn.execute(
                        text(
                            "INSERT INTO trading_halt_events "
                            "(version, mode, reason, source, changed_by, changed_at) "
                            "VALUES (:v, :mode, :reason, :source, :by, :at)"
                        ),
                        {
                            "v": new_version,
                            "mode": mode.value,
                            "reason": reason,
                            "source": source,
                            "by": changed_by,
                            "at": when,
                        },
                    )
            if updated == 1:
                state = self.read()
                assert state.version == new_version and state.mode is mode, (
                    "halt write committed but re-read disagrees",
                    state,
                )
                logger.warning("trading halt: now %s (v%d)", state.mode, state.version)
                return state
            logger.warning(
                "trading halt: concurrent change detected (attempt %d/%d) — re-reading",
                attempt,
                _MAX_WRITE_ATTEMPTS,
            )
        msg = f"trading halt: could not record {mode} after {_MAX_WRITE_ATTEMPTS} attempts"
        raise RuntimeError(msg)

    # -- acknowledgements ------------------------------------------------

    def acknowledge(
        self,
        path: str,
        *,
        status: str,
        state: HaltState | None,
        detail: str = "",
        now: datetime | None = None,
    ) -> None:
        """Record what ``path`` observed. Call ONLY at a quiescent point for
        that path (no entry submission of its own in flight). Never raises:
        a failed ack leaves the path visibly ``lagging``/``missing``."""
        assert status in ("applied", "unavailable"), status
        _require_text("path", path)
        version = state.version if state is not None else 0
        mode = state.mode.value if state is not None else "UNKNOWN"
        try:
            logger.debug("trading halt: %s acknowledging %s v%d (%s)", path, mode, version, status)
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO trading_halt_acks "
                        "(path, version, mode, status, detail, acked_at) "
                        "VALUES (:p, :v, :m, :s, :d, :at) "
                        "ON CONFLICT (path) DO UPDATE SET version = excluded.version, "
                        "mode = excluded.mode, status = excluded.status, "
                        "detail = excluded.detail, acked_at = excluded.acked_at"
                    ),
                    {
                        "p": path,
                        "v": version,
                        "m": mode,
                        "s": status,
                        "d": detail or None,
                        "at": now or datetime.now(UTC),
                    },
                )
        except Exception:
            logger.warning(
                "trading halt: acknowledgement write failed for %s — path will show lagging",
                path,
                exc_info=True,
            )

    def observe(self, path: str, *, now: datetime | None = None) -> EntryDecision:
        """Gate + acknowledgement in one call, for single-threaded writers at
        the top of a cycle (their quiescent point)."""
        decision = self.entry_decision()
        if decision.reason_code == REASON_UNAVAILABLE:
            self.acknowledge(
                path, status="unavailable", state=None, detail=decision.detail, now=now
            )
            return decision
        try:
            state = self.read()
        except HaltStateUnavailableError as exc:
            # Raced with a breakage between the two reads: report, stay blocked.
            self.acknowledge(path, status="unavailable", state=None, detail=str(exc), now=now)
            return EntryDecision(
                allowed=False,
                reason_code=REASON_UNAVAILABLE,
                mode=None,
                version=None,
                detail=str(exc),
            )
        self.acknowledge(path, status="applied", state=state, now=now)
        return decision

    def application_status(self, paths: Iterable[str] = EXECUTION_PATHS) -> HaltApplication:
        """Has the CURRENT request reached every path? Raises
        :class:`HaltStateUnavailableError` when the record itself is unreadable."""
        state = self.read()
        wanted = tuple(paths)
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT path, version, status FROM trading_halt_acks")).all()
        acks = {str(r[0]): (int(r[1]), str(r[2])) for r in rows}
        per_path: dict[str, str] = {}
        for path in wanted:
            ack = acks.get(path)
            if ack is None:
                per_path[path] = "missing"
            elif ack[1] == "unavailable":
                per_path[path] = "unavailable"
            elif ack[0] >= state.version:
                per_path[path] = "applied"
            else:
                per_path[path] = "lagging"
        applied = all(v in ("applied", "unavailable") for v in per_path.values())
        return HaltApplication(state=state, applied=applied, paths=per_path)
