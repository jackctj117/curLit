"""Per-event position-book state for the event-driven strategy (CL-mnhw).

Extracted from ``src/strategies/event_driven.py`` (structural residual,
CL-e6lx — the same split pattern as the CL-ikz2 EventNotifier
extraction): everything that coheres as BOOK STATE lives here —

  * the ``open_positions`` legs (:class:`EventPosition`),
  * cumulative realized P&L + closed-trade count, persisted atomically
    (tmp+rename) in ``data/event_book_state.json``,
  * the loss-cap freeze (``event_book_max_loss_pct``) that blocks NEW
    entries while exits always still flow,
  * exit bookkeeping for the hard stop + hard TIME STOP,
  * phantom-position reconciliation (CL-v9g4),
  * the concentration caps (CL-wbmw generalizing CL-5mkf).

:class:`~src.strategies.event_driven.EventDrivenStrategy` owns signal
evaluation, sizing, intent emission, and alerting, and delegates all of
the above to :class:`EventBook`. The knobs arrive as scalars (not the
strategy config object) so this module never imports strategy
internals. Serialized state format, log message text, and cap semantics
are byte-identical to the pre-extraction code — this is a pure move.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
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
    """One leg closed by :meth:`EventBook.check_exits` — carries
    everything the strategy needs to emit the exit intent and feature
    snapshot. ``book_realized_pnl`` is the book's cumulative realized
    P&L immediately AFTER this exit was booked (not after the whole
    batch), matching the pre-extraction snapshot semantics."""

    symbol: str
    position: EventPosition
    reason: str          # "hard_stop" | "time_stop"
    pnl: float
    held_hours: float
    current_price: float | None
    book_realized_pnl: float


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
                entry_ts = datetime.fromisoformat(pos["entry_ts"])
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.replace(tzinfo=UTC)
                self.open_positions[sym] = EventPosition(
                    symbol=sym,
                    event_id=pos.get("event_id"),
                    entry_ts=entry_ts,
                    entry_price=float(pos["entry_price"]),
                    quantity=float(pos["quantity"]),
                    direction=int(pos["direction"]),
                    stop_price=float(pos["stop_price"]),
                    headline=str(pos.get("headline", "")),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Skipping unparseable persisted event position %r", sym,
                )

    def save(self) -> None:
        path = self._path()
        payload = {
            "version": _STATE_VERSION,
            "realized_pnl": self.realized_pnl,
            "closed_trades": self.closed_trades,
            "open_positions": {
                sym: {
                    "event_id": pos.event_id,
                    "entry_ts": pos.entry_ts.isoformat(),
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "direction": pos.direction,
                    "stop_price": pos.stop_price,
                    "headline": pos.headline,
                }
                for sym, pos in self.open_positions.items()
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

    def check_exits(
        self,
        current_price: Callable[[str], float | None],
        now: datetime,
    ) -> list[ExitRecord]:
        """Close legs that hit the hard stop or the hard TIME STOP
        (``max_holding_hours``, fires regardless of P&L or even a
        current price). Books realized P&L and persists per closed leg;
        returns the closed legs for intent/snapshot emission."""
        records: list[ExitRecord] = []
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
            pnl = (current - pos.entry_price) * pos.quantity if current is not None else 0.0
            self.realized_pnl += pnl
            self.closed_trades += 1
            del self.open_positions[symbol]
            self.save()
            logger.info(
                "Event exit %s: %s pnl=%.2f held=%.1fh event_id=%s",
                symbol, exit_reason, pnl, held_hours, pos.event_id,
            )
            records.append(ExitRecord(
                symbol=symbol, position=pos, reason=exit_reason, pnl=pnl,
                held_hours=held_hours, current_price=current,
                book_realized_pnl=self.realized_pnl,
            ))
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

    def reconcile(self, broker: Any, now: datetime) -> None:
        """Prune phantom open_positions (CL-v9g4).

        A leg is recorded in ``open_positions`` when its OrderIntent is emitted,
        before the fill is known, so a REJECTED order leaves a phantom that eats
        the concurrency cap. Each cycle, drop entries OLDER than the grace window
        (which protects a just-recorded position not yet visible broker-side)
        that the broker does not actually hold. Fail-safe: if the broker's
        positions can't be read, prune NOTHING — a transient broker error can
        never drop a real position."""
        if not self.open_positions:
            return
        try:
            held = {self._norm_symbol(p.symbol) for p in broker.get_positions()}
        except Exception:
            logger.debug(
                "event_driven: broker positions unavailable — skipping phantom "
                "reconciliation", exc_info=True,
            )
            return
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
        SINGLE instrument, from tracked positions — nothing new persisted."""
        total = 0.0
        for sym, pos in self.open_positions.items():
            if sym == symbol:
                total += abs(pos.quantity) * pos.entry_price
        return total

    def _open_haven_notional(self) -> float:
        """Combined open notional (|quantity| * entry_price) across
        HAVEN_INSTRUMENTS (gold/silver), from tracked positions."""
        total = 0.0
        for sym, pos in self.open_positions.items():
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
