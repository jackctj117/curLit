"""Per-event position-book state for the event-driven strategy (CL-mnhw).

Extracted from ``src/strategies/event_driven.py`` (structural residual,
CL-e6lx — the same split pattern as the CL-ikz2 EventNotifier
extraction): everything that coheres as BOOK STATE lives here —

  * the ``open_positions`` legs (:class:`EventPosition`),
  * cumulative realized P&L + closed-trade count, persisted atomically
    (tmp+rename) in ``data/event_book_state.json``,
  * the loss-cap freeze (``event_book_max_loss_pct``) that blocks NEW
    entries while exits always still flow,
  * exit bookkeeping for the hard stop + hard TIME STOP — TWO-PHASE
    since CL-8cw1 (exit half of CL-hqyj): triggering MOVES the leg to
    ``pending_exits`` and the exit intent re-emits every tick until
    :meth:`EventBook.confirm_exits` sees the broker flat; only then is
    P&L realized (from the trigger price captured at trigger time),
  * phantom-position reconciliation (CL-v9g4),
  * the concentration caps (CL-wbmw generalizing CL-5mkf).

:class:`~src.strategies.event_driven.EventDrivenStrategy` owns signal
evaluation, sizing, intent emission, and alerting, and delegates all of
the above to :class:`EventBook`. The knobs arrive as scalars (not the
strategy config object) so this module never imports strategy
internals. Serialized state format was byte-identical to the
pre-extraction code until CL-8cw1 added the ADDITIVE ``pending_exits``
key — a state file WITHOUT it (the pre-CL-8cw1 live format) still loads
cleanly with no pending exits.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_VERSION = 1

#: Pure safe-haven OANDA instruments (CL-5mkf). Combined open notional
#: across these is capped at ``haven_max_pct`` of equity — a tighter
#: CLUSTER cap on top of the per-instrument cap (CL-wbmw), because a
#: per-name limit can't see correlated metals as one basket.
HAVEN_INSTRUMENTS = frozenset({"XAU_USD", "XAG_USD"})


@dataclass
class EventPosition:
    symbol: str
    event_id: Any
    entry_ts: datetime
    entry_price: float
    quantity: float  # signed units
    direction: int   # +1 long / -1 short
    stop_price: float
    headline: str = ""


@dataclass
class ExitRecord:
    """One exit-intent emission from :meth:`EventBook.check_exits`
    (trigger or pending re-emit) or a confirmed finalization from
    :meth:`EventBook.confirm_exits` (CL-8cw1) — carries everything the
    strategy needs to emit the exit intent and feature snapshot. ``pnl``
    and ``held_hours`` are computed from the trigger price/timestamp
    captured at TRIGGER time (idempotent across retries).
    ``book_realized_pnl`` is the book's cumulative realized P&L
    immediately AFTER this exit is booked — or, for a not-yet-confirmed
    emission, the value it WILL be once the broker confirms flat
    (matching the pre-CL-8cw1 snapshot semantics)."""

    symbol: str
    position: EventPosition
    reason: str          # "hard_stop" | "time_stop"
    pnl: float
    held_hours: float
    current_price: float | None
    book_realized_pnl: float


@dataclass
class PendingExit:
    """A leg whose stop/time-stop TRIGGERED but whose broker-side exit
    is not yet CONFIRMED flat (CL-8cw1, exit half of CL-hqyj).

    Before CL-8cw1 the book finalized on intent EMISSION, so an OMS or
    broker REJECT of the exit order left the book flat while the broker
    still held the risk — stops gone, nothing retrying. Now the leg
    parks here: :meth:`EventBook.check_exits` re-emits the exit intent
    every tick until :meth:`EventBook.confirm_exits` sees the broker
    flat, and only then is P&L realized (from ``trigger_price``,
    captured at trigger time)."""

    position: EventPosition
    reason: str              # "hard_stop" | "time_stop"
    triggered_ts: datetime
    trigger_price: float | None
    #: Exit-intent emissions so far (1 = the trigger tick). Emissions
    #: 2+ log at WARNING — an unconfirmed exit means a rejected order or
    #: a slow fill, and the operator should see it.
    emit_count: int = 1


class EventBook:
    """Owns the event strategy's per-event position state.

    Constructed by :class:`EventDrivenStrategy` from its config scalars;
    loads any persisted state immediately (so open legs survive an
    engine restart and the hard time stop still fires after a bounce).
    """

    def __init__(
        self,
        *,
        state_path: str,
        max_loss_pct: float,
        per_instrument_max_pct: float,
        haven_max_pct: float,
        max_holding_hours: float,
        reconcile_grace_sec: int,
    ) -> None:
        self._state_path_str = state_path
        self._max_loss_pct = max_loss_pct
        self._per_instrument_max_pct = per_instrument_max_pct
        self._haven_max_pct = haven_max_pct
        self._max_holding_hours = max_holding_hours
        self._reconcile_grace_sec = reconcile_grace_sec
        self.open_positions: dict[str, EventPosition] = {}
        # Triggered-but-not-broker-confirmed exits (CL-8cw1) — see
        # PendingExit. Keys never overlap open_positions (legs MOVE here).
        self.pending_exits: dict[str, PendingExit] = {}
        self.realized_pnl: float = 0.0
        self.closed_trades: int = 0
        # Loss-cap breach is CRITICAL once per activation, WARNING after.
        self._breach_logged = False
        self.load()

    # ------------------------------------------------------------------
    # State file (atomic tmp+rename, like equity trailing stop)
    # ------------------------------------------------------------------

    def _path(self) -> Path:
        return Path(self._state_path_str)

    def load(self) -> None:
        path = self._path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            # A silently reset book under-counts losses, but refusing to
            # boot over a torn file is worse. Back the file up loudly and
            # start fresh — the operator can restore realized_pnl by hand.
            corrupt = path.with_name(path.name + ".corrupt")
            logger.error(
                "Event book state %s is corrupt — backing up to %s and starting "
                "a FRESH book (realized P&L reset to 0; loss cap restarts).",
                path, corrupt,
            )
            try:
                os.replace(path, corrupt)
            except OSError:
                logger.exception("Could not back up corrupt event book state")
            return
        self.realized_pnl = float(payload.get("realized_pnl", 0.0))
        self.closed_trades = int(payload.get("closed_trades", 0))
        for sym, pos in (payload.get("open_positions") or {}).items():
            try:
                self.open_positions[sym] = self._position_from_payload(sym, pos)
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Skipping unparseable persisted event position %r", sym,
                )
        # pending_exits is ABSENT from pre-CL-8cw1 state files (the live
        # format at rollout) — a missing key MUST load as an empty dict.
        for sym, entry in (payload.get("pending_exits") or {}).items():
            try:
                triggered_ts = datetime.fromisoformat(entry["triggered_ts"])
                if triggered_ts.tzinfo is None:
                    triggered_ts = triggered_ts.replace(tzinfo=UTC)
                price_raw = entry.get("trigger_price")
                self.pending_exits[sym] = PendingExit(
                    position=self._position_from_payload(sym, entry["position"]),
                    reason=str(entry["reason"]),
                    triggered_ts=triggered_ts,
                    trigger_price=float(price_raw) if price_raw is not None else None,
                    emit_count=int(entry.get("emit_count", 1)),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Skipping unparseable persisted pending exit %r", sym,
                )

    @staticmethod
    def _position_from_payload(sym: str, pos: dict[str, Any]) -> EventPosition:
        """Parse one persisted position payload (raises on bad shape —
        callers decide the skip-with-warning policy)."""
        entry_ts = datetime.fromisoformat(pos["entry_ts"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=UTC)
        return EventPosition(
            symbol=sym,
            event_id=pos.get("event_id"),
            entry_ts=entry_ts,
            entry_price=float(pos["entry_price"]),
            quantity=float(pos["quantity"]),
            direction=int(pos["direction"]),
            stop_price=float(pos["stop_price"]),
            headline=str(pos.get("headline", "")),
        )

    @staticmethod
    def _position_payload(pos: EventPosition) -> dict[str, Any]:
        return {
            "event_id": pos.event_id,
            "entry_ts": pos.entry_ts.isoformat(),
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "direction": pos.direction,
            "stop_price": pos.stop_price,
            "headline": pos.headline,
        }

    def save(self) -> None:
        path = self._path()
        payload = {
            "version": _STATE_VERSION,
            "realized_pnl": self.realized_pnl,
            "closed_trades": self.closed_trades,
            "open_positions": {
                sym: self._position_payload(pos)
                for sym, pos in self.open_positions.items()
            },
            "pending_exits": {
                sym: {
                    "position": self._position_payload(entry.position),
                    "reason": entry.reason,
                    "triggered_ts": entry.triggered_ts.isoformat(),
                    "trigger_price": entry.trigger_price,
                    "emit_count": entry.emit_count,
                }
                for sym, entry in self.pending_exits.items()
            },
            "updated_at": datetime.now(UTC).isoformat(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True))
            os.replace(tmp, path)
        except OSError:
            logger.exception("Failed to persist event book state to %s", path)

    # ------------------------------------------------------------------
    # Leg lifecycle
    # ------------------------------------------------------------------

    def record_entry(self, position: EventPosition) -> None:
        """Track a newly-entered leg and persist immediately (the intent
        is already emitted — the book must survive a crash right after)."""
        self.open_positions[position.symbol] = position
        self.save()

    def held_positions(self) -> dict[str, EventPosition]:
        """Every leg the broker may still hold: OPEN legs plus
        PENDING-EXIT legs (triggered, exit intent emitted, not yet
        confirmed flat — CL-8cw1). This is the view the strategy exposes
        as ``open_positions`` to the portfolio reconciler and uses for
        entry-slot accounting: until the broker confirms flat, a
        pending-exit leg is REAL broker risk that must not be classified
        ``orphaned_broker`` (and double-flattened) and must keep its
        concurrency slot occupied. Returns a fresh merged dict — mutate
        ``open_positions`` / ``pending_exits`` directly, not this."""
        merged = dict(self.open_positions)
        for sym, entry in self.pending_exits.items():
            merged.setdefault(sym, entry.position)
        return merged

    def _pending_exit_record(self, symbol: str, entry: PendingExit) -> ExitRecord:
        """ExitRecord for a pending exit — every value derives from the
        trigger-time capture, so re-emissions are idempotent.
        ``book_realized_pnl`` is the projected cumulative P&L after this
        exit CONFIRMS (pre-CL-8cw1 snapshot semantics preserved)."""
        pos = entry.position
        pnl = (
            (entry.trigger_price - pos.entry_price) * pos.quantity
            if entry.trigger_price is not None else 0.0
        )
        held_hours = (entry.triggered_ts - pos.entry_ts).total_seconds() / 3600.0
        return ExitRecord(
            symbol=symbol, position=pos, reason=entry.reason, pnl=pnl,
            held_hours=held_hours, current_price=entry.trigger_price,
            book_realized_pnl=self.realized_pnl + pnl,
        )

    def check_exits(
        self,
        current_price: Callable[[str], float | None],
        now: datetime,
    ) -> list[ExitRecord]:
        """Two-phase exit emission (CL-8cw1).

        Phase 1 — every leg already in ``pending_exits`` is RE-returned
        for intent re-emission (an OMS/broker-rejected exit self-heals
        next tick), logging at WARNING from the 2nd emission on. Pending
        legs get no further stop/time-stop evaluation — they already
        triggered.

        Phase 2 — open legs that hit the hard stop or the hard TIME STOP
        (``max_holding_hours``, fires regardless of P&L or even a
        current price) MOVE to ``pending_exits`` (persisted) and are
        returned for the initial intent emission. NOTHING is finalized
        here: realized P&L and ``closed_trades`` book exclusively in
        :meth:`confirm_exits`, once the broker shows flat."""
        records: list[ExitRecord] = []
        # Phase 1 first, so a leg triggered below isn't emitted twice in
        # the same call.
        for symbol, entry in self.pending_exits.items():
            entry.emit_count += 1
            logger.warning(
                "exit for %s not confirmed, re-emitting (emission %d, "
                "reason=%s, event_id=%s)",
                symbol, entry.emit_count, entry.reason, entry.position.event_id,
            )
            records.append(self._pending_exit_record(symbol, entry))
        # Phase 2: trigger detection on open legs (conditions unchanged).
        for symbol, pos in list(self.open_positions.items()):
            current = current_price(symbol)
            exit_reason = None
            if current is not None and (
                (pos.direction > 0 and current <= pos.stop_price)
                or (pos.direction < 0 and current >= pos.stop_price)
            ):
                exit_reason = "hard_stop"
            held_hours = (now - pos.entry_ts).total_seconds() / 3600.0
            if held_hours >= self._max_holding_hours:
                # TIME STOP fires regardless of P&L or even a current price.
                exit_reason = "time_stop"
            if exit_reason is None:
                continue
            entry = PendingExit(
                position=pos, reason=exit_reason, triggered_ts=now,
                trigger_price=current,
            )
            del self.open_positions[symbol]
            self.pending_exits[symbol] = entry
            self.save()
            logger.info(
                "Event exit triggered %s: %s trigger_price=%s held=%.1fh "
                "event_id=%s — awaiting broker flat confirmation",
                symbol, exit_reason,
                "n/a" if current is None else f"{current:.5f}",
                held_hours, pos.event_id,
            )
            records.append(self._pending_exit_record(symbol, entry))
        return records

    def confirm_exits(self, broker_positions: Iterable[Any]) -> list[ExitRecord]:
        """Finalize pending exits the broker CONFIRMS are flat (CL-8cw1).

        ``broker_positions`` is the SAME snapshot :meth:`reconcile`
        fetched this tick — one broker call per tick, never a second.
        Symbols match via the shared canonical_symbol. A pending symbol
        the broker still holds stays pending (:meth:`check_exits` keeps
        re-emitting). A flat one finalizes exactly as the pre-CL-8cw1
        trigger path did — realized P&L from the trigger price captured
        at trigger time, ``closed_trades`` bump, persisted, ExitRecord
        returned — and exactly ONCE: finalized legs leave
        ``pending_exits``, so a repeat call with the same snapshot is a
        no-op."""
        if not self.pending_exits:
            return []
        held = {
            self._norm_symbol(str(getattr(p, "symbol", "")))
            for p in broker_positions
        }
        records: list[ExitRecord] = []
        for symbol, entry in list(self.pending_exits.items()):
            if self._norm_symbol(symbol) in held:
                continue  # broker still holds it — keep retrying the exit
            record = self._pending_exit_record(symbol, entry)
            self.realized_pnl += record.pnl
            self.closed_trades += 1
            del self.pending_exits[symbol]
            self.save()
            logger.info(
                "Event exit %s: %s pnl=%.2f held=%.1fh event_id=%s (broker "
                "confirmed flat after %d emission(s))",
                symbol, entry.reason, record.pnl, record.held_hours,
                entry.position.event_id, entry.emit_count,
            )
            records.append(record)
        return records

    # ------------------------------------------------------------------
    # Phantom-position reconciliation (CL-v9g4)
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_symbol(sym: str) -> str:
        """Compare-form for position matching — delegates to the shared
        canonical_symbol (CL-qqra) so there is ONE normalizer repo-wide."""
        from src.execution.broker import canonical_symbol  # noqa: PLC0415
        return canonical_symbol(sym)

    def reconcile(self, broker: Any, now: datetime) -> list[Any] | None:
        """Prune phantom open_positions (CL-v9g4) and return the broker
        position snapshot for reuse (CL-8cw1).

        A leg is recorded in ``open_positions`` when its OrderIntent is emitted,
        before the fill is known, so a REJECTED order leaves a phantom that eats
        the concurrency cap. Each cycle, drop entries OLDER than the grace window
        (which protects a just-recorded position not yet visible broker-side)
        that the broker does not actually hold. Fail-safe: if the broker's
        positions can't be read, prune NOTHING — a transient broker error can
        never drop a real position.

        Returns the raw ``broker.get_positions()`` list so the caller can
        feed :meth:`confirm_exits` from the SAME snapshot (one broker call
        per tick), or None when there was nothing to fetch or the broker
        was unreadable (fail-safe: prune nothing, confirm nothing).
        Pending-exit legs are never phantom-pruned — a pending symbol the
        broker no longer holds is a CONFIRMED exit whose P&L
        :meth:`confirm_exits` must book, not a phantom to drop."""
        if not self.open_positions and not self.pending_exits:
            return None
        try:
            positions = list(broker.get_positions())
        except Exception:
            logger.debug(
                "event_driven: broker positions unavailable — skipping phantom "
                "reconciliation", exc_info=True,
            )
            return None
        held = {self._norm_symbol(p.symbol) for p in positions}
        grace = timedelta(seconds=self._reconcile_grace_sec)
        pruned = 0
        for symbol in list(self.open_positions.keys()):
            pos = self.open_positions[symbol]
            if now - pos.entry_ts <= grace:
                continue  # too fresh — a real fill may not show broker-side yet
            if self._norm_symbol(symbol) not in held:
                logger.warning(
                    "event_driven: pruning phantom position %s (event id=%s) — "
                    "broker does not hold it (order likely rejected)",
                    symbol, pos.event_id,
                )
                del self.open_positions[symbol]
                pruned += 1
        if pruned:
            self.save()
        return positions

    # ------------------------------------------------------------------
    # Event-book protection (loss-cap freeze on NEW entries)
    # ------------------------------------------------------------------

    def breached(self, equity: float | None) -> bool:
        if equity is None or equity <= 0:
            return False
        cap = self._max_loss_pct * equity
        breached = -self.realized_pnl >= cap
        if breached:
            if not self._breach_logged:
                logger.critical(
                    "EVENT BOOK LOSS CAP BREACHED: cumulative realized P&L "
                    "%.2f <= -%.2f (%.1f%% of equity %.0f). NO new event "
                    "positions will be opened; exits still flow. Reset "
                    "requires operator action on %s. (Kill-switch "
                    "integration pending — this is the loud log.)",
                    self.realized_pnl, cap,
                    self._max_loss_pct * 100, equity,
                    self._state_path_str,
                )
                self._breach_logged = True
            else:
                logger.warning(
                    "Event book loss cap still breached (realized P&L %.2f) — "
                    "new entries blocked", self.realized_pnl,
                )
        elif self._breach_logged:
            logger.warning(
                "Event book back under the loss cap — new entries re-enabled",
            )
            self._breach_logged = False
        return breached

    # ------------------------------------------------------------------
    # Concentration caps (CL-wbmw generalizes CL-5mkf) — semantics in
    # concentration_capped_size's docstring. Additive to event_risk_pct
    # sizing + the loss cap; not a substitute for the correlation switch.
    # ------------------------------------------------------------------

    def _open_instrument_notional(self, symbol: str) -> float:
        """Open notional (|quantity| * entry_price, the sizing units) in a
        SINGLE instrument, from tracked positions (open + pending-exit —
        a pending leg is still broker exposure until confirmed flat,
        CL-8cw1) — nothing new persisted."""
        total = 0.0
        for sym, pos in self.held_positions().items():
            if sym == symbol:
                total += abs(pos.quantity) * pos.entry_price
        return total

    def _open_haven_notional(self) -> float:
        """Combined open notional (|quantity| * entry_price) across
        HAVEN_INSTRUMENTS (gold/silver), from tracked positions (open +
        pending-exit, as above)."""
        total = 0.0
        for sym, pos in self.held_positions().items():
            if sym in HAVEN_INSTRUMENTS:
                total += abs(pos.quantity) * pos.entry_price
        return total

    def concentration_capped_size(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        equity: float,
    ) -> float:
        """Reduce a NEW event leg's signed ``size`` so it satisfies the
        concentration caps (CL-wbmw). Two layers: (1) per-instrument cap
        (``per_instrument_max_pct``) — ALWAYS, for every symbol; (2)
        haven-cluster cap (``haven_max_pct``) — additionally for havens,
        on combined gold+silver notional. The SMALLER headroom binds; the
        cap is a ceiling the base sizing grows toward, so under-cap legs
        pass through UNCHANGED. Returns the (possibly reduced) signed
        size — 0.0 when EITHER cap is already at/over (skip). Logs at
        WARNING naming the binding cap. Preserves sign."""
        if equity <= 0:
            return size

        # Layer 1: per-instrument headroom (every symbol).
        per_cap = self._per_instrument_max_pct * equity
        per_open = self._open_instrument_notional(symbol)
        headroom = per_cap - per_open
        binding = "per-instrument"
        cap_pct = self._per_instrument_max_pct
        open_exposure = per_open
        cap_notional = per_cap

        # Layer 2: haven-cluster headroom (havens only) — take the tighter.
        if symbol in HAVEN_INSTRUMENTS:
            haven_cap = self._haven_max_pct * equity
            haven_open = self._open_haven_notional()
            haven_headroom = haven_cap - haven_open
            if haven_headroom < headroom:
                headroom = haven_headroom
                cluster = "/".join(sorted(HAVEN_INSTRUMENTS))
                binding = f"haven-cluster ({cluster})"
                cap_pct = self._haven_max_pct
                open_exposure = haven_open
                cap_notional = haven_cap

        proposed_notional = abs(size) * entry_price
        if headroom <= 0:
            logger.warning(
                "Concentration cap [%s]: already at/over %.0f%% of equity "
                "(open notional %.0f >= cap %.0f) — SKIPPING new %s entry",
                binding, cap_pct * 100, open_exposure, cap_notional, symbol,
            )
            return 0.0
        if proposed_notional <= headroom:
            return size  # fits under the (more binding) cap — unchanged
        # Trim the position to exactly fill the remaining headroom.
        max_units = headroom / entry_price
        capped = max_units if size > 0 else -max_units
        logger.warning(
            "Concentration cap [%s]: %s entry reduced from %.0f to %.0f "
            "units (open notional %.0f + proposed %.0f would exceed cap "
            "%.0f = %.0f%% of equity)",
            binding, symbol, size, capped, open_exposure, proposed_notional,
            cap_notional, cap_pct * 100,
        )
        return capped
