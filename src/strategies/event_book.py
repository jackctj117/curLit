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
cleanly with no pending exits. CL-9dhg added the additive per-entry
``trigger_broker_qty`` (missing loads as None → flat-only confirmation)
and made a symbol appearing in BOTH ``open_positions`` and
``pending_exits`` a fail-loud load error (corrupt state refuses to
start rather than silently losing realized P&L).

ENTRY half of CL-hqyj (mirrors the CL-8cw1 exit lifecycle): before this,
:meth:`record_entry` put the INTENDED-size leg straight into
``open_positions`` and only the phantom pruner (CL-v9g4) ever corrected
it — and it checks PRESENCE, not QUANTITY, so a PARTIAL fill (book
intends -308, broker fills -214) was never corrected and permanently
tripped the portfolio reconciler's ``reconciliation_failure`` kill
switch. Now a new leg parks in ``pending_entries`` (additive state key —
a file WITHOUT it loads as empty) at intended size for SLOT/CAP
accounting only; :meth:`confirm_entries` polls the broker each tick,
computes the co-held-safe fill delta from a submit-time baseline, and
PROMOTES the leg into ``open_positions`` at the ACTUAL broker fill (or
REJECTS it, booking nothing, after grace with no fill). Only a promoted
leg is reconciler-facing exposure and only it gets stop/time-stop
evaluation — so the reconciler never sees the intended phantom.
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
    direction: int  # +1 long / -1 short
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
    reason: str  # "hard_stop" | "time_stop"
    pnl: float
    held_hours: float
    current_price: float | None
    book_realized_pnl: float
    #: Exit-intent emissions so far for this leg (1 = the trigger tick).
    #: The strategy records the exit FeatureSnapshot only on the first
    #: emission (CL-9dhg finding 10) — re-emissions would write one
    #: near-identical row per tick per pending exit.
    emit_count: int = 1


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
    reason: str  # "hard_stop" | "time_stop"
    triggered_ts: datetime
    trigger_price: float | None
    #: Exit-intent emissions so far (1 = the trigger tick). Emissions
    #: 2+ log at WARNING — an unconfirmed exit means a rejected order or
    #: a slow fill, and the operator should see it.
    emit_count: int = 1
    #: The broker's account-wide NET quantity for this symbol AT TRIGGER
    #: TIME (canonical-symbol matched, from the same snapshot
    #: :meth:`EventBook.reconcile` fetched that tick) — CL-9dhg
    #: findings 1 + 2. Broker positions are account-wide, so a
    #: co-holding sibling strategy means "flat" never happens; the
    #: trigger capture lets :meth:`EventBook.confirm_exits` recognize
    #: "our share is out, the residual is the co-holders'". A capture of
    #: 0 means the broker demonstrably never held the leg at trigger (a
    #: rejected entry whose stop crossed inside the reconcile grace) —
    #: it finalizes as PHANTOM with NO realized P&L. ``None`` = broker
    #: unreadable at trigger (and every pre-CL-9dhg persisted entry):
    #: confirm only on broker-flat and book P&L as before — never guess.
    trigger_broker_qty: float | None = None


@dataclass
class PendingEntry:
    """A leg whose entry OrderIntent was emitted but whose broker fill is
    not yet CONFIRMED (ENTRY half of CL-hqyj — mirror of :class:`PendingExit`).

    ``position`` carries the INTENDED signed quantity (for logging,
    analytics, and SLOT/CAP accounting) — it NEVER drives live exposure or
    the portfolio reconciler, which see ``confirmed_qty`` instead.
    :meth:`EventBook.confirm_entries` computes the co-held-safe observed
    fill as ``broker_net_now(canon) - entry_broker_qty`` and, once it
    crosses the min-fill threshold with the intended sign, PROMOTES the leg
    into ``open_positions`` at the OBSERVED fill (the partial-fill fix). A
    leg with no fill after ``grace`` is REJECTED, booking nothing (mirror
    of the phantom prune, for the pending path)."""

    position: EventPosition
    submitted_ts: datetime
    #: Broker ACCOUNT-NET quantity for this symbol's canonical form captured
    #: from the snapshot AT SUBMIT TIME (co-held siblings included). The fill
    #: is the DELTA from this baseline — never the raw account net (a
    #: co-holder would otherwise be counted as our fill). ``None`` = broker
    #: unreadable at submit (rare double-degradation): best-effort
    #: promote/reject after grace (see :meth:`EventBook.confirm_entries`).
    entry_broker_qty: float | None = None
    #: Observed signed fill from the latest :meth:`confirm_entries` pass
    #: (0.0 until the broker shows the fill). The reconciler-facing and
    #: concentration-cap-*notional* quantity for a not-yet-promoted leg.
    confirmed_qty: float = 0.0


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
        # Emitted-but-not-broker-confirmed entries (ENTRY half of CL-hqyj)
        # — see PendingEntry. A leg lives here from record_entry until
        # confirm_entries PROMOTES it (fill seen) or REJECTS it (no fill
        # after grace). Keys never overlap open_positions/pending_exits.
        self.pending_entries: dict[str, PendingEntry] = {}
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
                path,
                corrupt,
            )
            try:
                os.replace(path, corrupt)
            except OSError:
                logger.exception("Could not back up corrupt event book state")
            return
        # Fail LOUD on a symbol present in more than one book (CL-9dhg
        # finding 11; extended to pending_entries by the entry half of
        # CL-hqyj): a leg awaits its stop (open), awaits broker flat
        # confirmation (pending_exit), or awaits its fill (pending_entry)
        # — never more than one. Silently preferring one would double-track
        # broker risk, drop a triggered exit's realized P&L, or resurrect a
        # rejected entry. Repo rule: corrupt state refuses to start.
        open_syms = set(payload.get("open_positions") or {})
        pending_exit_syms = set(payload.get("pending_exits") or {})
        pending_entry_syms = set(payload.get("pending_entries") or {})
        overlap = sorted(
            (open_syms & pending_exit_syms) | (pending_entry_syms & (open_syms | pending_exit_syms))
        )
        if overlap:
            raise ValueError(
                f"Event book state {path} is corrupt: symbol(s) {overlap} "
                "appear in more than one of open_positions / pending_exits / "
                "pending_entries — refusing to start. Repair the state file "
                "by hand (a leg belongs in exactly one book)."
            )
        self.realized_pnl = float(payload.get("realized_pnl", 0.0))
        self.closed_trades = int(payload.get("closed_trades", 0))
        for sym, pos in (payload.get("open_positions") or {}).items():
            try:
                self.open_positions[sym] = self._position_from_payload(sym, pos)
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Skipping unparseable persisted event position %r",
                    sym,
                )
        # pending_exits is ABSENT from pre-CL-8cw1 state files (the live
        # format at rollout) — a missing key MUST load as an empty dict.
        for sym, entry in (payload.get("pending_exits") or {}).items():
            try:
                triggered_ts = datetime.fromisoformat(entry["triggered_ts"])
                if triggered_ts.tzinfo is None:
                    triggered_ts = triggered_ts.replace(tzinfo=UTC)
                price_raw = entry.get("trigger_price")
                # trigger_broker_qty is ABSENT from pre-CL-9dhg pending
                # entries — missing loads as None (confirm on broker-flat
                # only, P&L booked as before; never guess a capture).
                qty_raw = entry.get("trigger_broker_qty")
                self.pending_exits[sym] = PendingExit(
                    position=self._position_from_payload(sym, entry["position"]),
                    reason=str(entry["reason"]),
                    triggered_ts=triggered_ts,
                    trigger_price=float(price_raw) if price_raw is not None else None,
                    emit_count=int(entry.get("emit_count", 1)),
                    trigger_broker_qty=(float(qty_raw) if qty_raw is not None else None),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Skipping unparseable persisted pending exit %r",
                    sym,
                )
        # pending_entries is ABSENT from every pre-CL-hqyj-entry state file
        # (the live format at rollout) — a missing key MUST load as an empty
        # dict (backward compat with data/event_book_state.json).
        for sym, entry in (payload.get("pending_entries") or {}).items():
            try:
                submitted_ts = datetime.fromisoformat(entry["submitted_ts"])
                if submitted_ts.tzinfo is None:
                    submitted_ts = submitted_ts.replace(tzinfo=UTC)
                qty_raw = entry.get("entry_broker_qty")
                self.pending_entries[sym] = PendingEntry(
                    position=self._position_from_payload(sym, entry["position"]),
                    submitted_ts=submitted_ts,
                    entry_broker_qty=(float(qty_raw) if qty_raw is not None else None),
                    confirmed_qty=float(entry.get("confirmed_qty", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Skipping unparseable persisted pending entry %r",
                    sym,
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
                sym: self._position_payload(pos) for sym, pos in self.open_positions.items()
            },
            "pending_exits": {
                sym: {
                    "position": self._position_payload(entry.position),
                    "reason": entry.reason,
                    "triggered_ts": entry.triggered_ts.isoformat(),
                    "trigger_price": entry.trigger_price,
                    "emit_count": entry.emit_count,
                    "trigger_broker_qty": entry.trigger_broker_qty,
                }
                for sym, entry in self.pending_exits.items()
            },
            "pending_entries": {
                sym: {
                    "position": self._position_payload(entry.position),
                    "submitted_ts": entry.submitted_ts.isoformat(),
                    "entry_broker_qty": entry.entry_broker_qty,
                    "confirmed_qty": entry.confirmed_qty,
                }
                for sym, entry in self.pending_entries.items()
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

    def record_entry(
        self,
        position: EventPosition,
        broker_positions: Iterable[Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Park a newly-submitted entry leg in ``pending_entries`` at its
        INTENDED size and persist immediately (ENTRY half of CL-hqyj —
        mirror of the CL-8cw1 exit trigger capture).

        The intent is already emitted, so the book must survive a crash
        right after — but the fill is UNKNOWN. The leg does NOT enter
        ``open_positions`` (and thus is not reconciler-facing exposure and
        gets no stop/time-stop evaluation) until :meth:`confirm_entries`
        sees the actual broker fill and PROMOTES it at the observed size.

        ``broker_positions`` is the snapshot :meth:`reconcile` already
        fetched this tick (None = broker unreadable at submit): the
        account-wide NET quantity for this symbol's canonical form is
        captured as ``entry_broker_qty`` — the BASELINE from which
        :meth:`confirm_entries` measures OUR fill as a delta (a co-holding
        sibling strategy is thus never counted as our fill). A None baseline
        means the broker was unreadable at submit (rare double-degradation):
        confirm falls back to a best-effort promote/reject after grace."""
        now = now or datetime.now(UTC)
        entry_broker_qty: float | None = None
        if broker_positions is not None:
            net = self._net_broker_quantities(broker_positions)
            entry_broker_qty = net.get(self._norm_symbol(position.symbol), 0.0)
        self.pending_entries[position.symbol] = PendingEntry(
            position=position,
            submitted_ts=now,
            entry_broker_qty=entry_broker_qty,
            confirmed_qty=0.0,
        )
        self.save()

    def held_positions(self) -> dict[str, EventPosition]:
        """RECONCILER-facing view: every leg valued at the quantity the
        broker actually holds for us.

        OPEN legs (their signed quantity), plus PENDING-EXIT legs
        (triggered, exit intent emitted, not yet confirmed flat — CL-8cw1;
        still REAL broker risk, must not be classified ``orphaned_broker``
        and double-flattened), plus PENDING-ENTRY legs valued at
        ``confirmed_qty`` — the ACTUAL observed fill, NOT the intended size
        (ENTRY half of CL-hqyj). Valuing a pending entry at its intended
        size is exactly the bug that tripped the reconciliation_failure kill
        switch (book -308 vs broker -214); until the broker shows the fill,
        confirmed_qty is 0.0 and the leg contributes nothing to the
        reconciler's internal_qty. Returns a fresh merged dict — mutate
        ``open_positions`` / ``pending_exits`` / ``pending_entries``
        directly, not this. For SLOT/CAP accounting (which must count a
        pending entry at its intended magnitude so the strategy can't
        over-submit past max legs while a fill is in flight) use
        :meth:`accounting_positions` instead."""
        merged = dict(self.open_positions)
        for sym, entry in self.pending_exits.items():
            merged.setdefault(sym, entry.position)
        for sym, pending in self.pending_entries.items():
            if sym in merged:
                continue
            pos = pending.position
            # Reconciler-facing size is the CONFIRMED fill, never intended.
            merged[sym] = EventPosition(
                symbol=pos.symbol,
                event_id=pos.event_id,
                entry_ts=pos.entry_ts,
                entry_price=pos.entry_price,
                quantity=pending.confirmed_qty,
                direction=pos.direction,
                stop_price=pos.stop_price,
                headline=pos.headline,
            )
        return merged

    def accounting_positions(self) -> dict[str, EventPosition]:
        """SLOT/CAP-accounting view (ENTRY half of CL-hqyj): like
        :meth:`held_positions` but a not-yet-confirmed PENDING-ENTRY leg is
        valued at its INTENDED magnitude, not ``confirmed_qty``.

        The two views split because they answer different questions. The
        reconciler asks "what does the broker hold for us right now?" —
        confirmed only, or an in-flight order looks like a phantom. Slot and
        concentration-cap enforcement ask "how much are we COMMITTED to?" —
        a submitted-but-unfilled leg must occupy its concurrency slot and
        count toward the per-instrument/haven notional so the strategy
        cannot over-submit past ``max_concurrent_event_positions`` (or blow
        the concentration cap) while an entry is pending. Once
        ``confirm_entries`` has an observed fill, both views agree on it.
        Returns a fresh merged dict — mutate the underlying books, not
        this."""
        merged = dict(self.open_positions)
        for sym, entry in self.pending_exits.items():
            merged.setdefault(sym, entry.position)
        for sym, pending in self.pending_entries.items():
            if sym in merged:
                continue
            pos = pending.position
            # Count the LARGER of intended and observed fill: a pending leg
            # commits its intended magnitude; a partially-promoted-but-still-
            # pending leg (shouldn't happen — promotion removes it) would
            # still not under-count.
            qty = (
                pos.quantity
                if pending.confirmed_qty == 0.0
                else (
                    pos.quantity
                    if abs(pos.quantity) >= abs(pending.confirmed_qty)
                    else pending.confirmed_qty
                )
            )
            merged[sym] = EventPosition(
                symbol=pos.symbol,
                event_id=pos.event_id,
                entry_ts=pos.entry_ts,
                entry_price=pos.entry_price,
                quantity=qty,
                direction=pos.direction,
                stop_price=pos.stop_price,
                headline=pos.headline,
            )
        return merged

    def _pending_exit_record(self, symbol: str, entry: PendingExit) -> ExitRecord:
        """ExitRecord for a pending exit — every value derives from the
        trigger-time capture, so re-emissions are idempotent.
        ``book_realized_pnl`` is the projected cumulative P&L after this
        exit CONFIRMS (pre-CL-8cw1 snapshot semantics preserved)."""
        pos = entry.position
        pnl = (
            (entry.trigger_price - pos.entry_price) * pos.quantity
            if entry.trigger_price is not None
            else 0.0
        )
        held_hours = (entry.triggered_ts - pos.entry_ts).total_seconds() / 3600.0
        return ExitRecord(
            symbol=symbol,
            position=pos,
            reason=entry.reason,
            pnl=pnl,
            held_hours=held_hours,
            current_price=entry.trigger_price,
            book_realized_pnl=self.realized_pnl + pnl,
            emit_count=entry.emit_count,
        )

    def _net_broker_quantities(
        self,
        broker_positions: Iterable[Any],
    ) -> dict[str, float | None]:
        """Account-wide NET quantity per canonical symbol from a broker
        position snapshot (CL-9dhg finding 1). A symbol whose quantity
        cannot be read maps to None — "held, size unknown": callers must
        neither confirm against it nor capture it at trigger (never
        finalize or classify blind). A symbol ABSENT from the returned
        dict is one the broker does not hold at all (net 0)."""
        net: dict[str, float | None] = {}
        for p in broker_positions:
            sym = self._norm_symbol(str(getattr(p, "symbol", "")))
            prev = net.get(sym, 0.0)
            if prev is None:
                continue  # one unreadable row poisons the symbol's net
            try:
                qty = float(p.quantity)
            except (AttributeError, TypeError, ValueError):
                net[sym] = None
                continue
            net[sym] = prev + qty
        return net

    def check_exits(
        self,
        current_price: Callable[[str], float | None],
        now: datetime,
        broker_positions: Iterable[Any] | None = None,
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
        :meth:`confirm_exits`, once the broker confirms.

        ``broker_positions`` is the snapshot :meth:`reconcile` already
        fetched this tick (None = broker unreadable): a triggering leg
        captures the broker's net quantity for its symbol as
        ``trigger_broker_qty`` (CL-9dhg findings 1 + 2) so
        :meth:`confirm_exits` can confirm a co-held symbol from the
        residual and finalize a never-filled leg as phantom."""
        trigger_net = (
            self._net_broker_quantities(broker_positions) if broker_positions is not None else None
        )
        records: list[ExitRecord] = []
        # Phase 1 first, so a leg triggered below isn't emitted twice in
        # the same call.
        for symbol, entry in self.pending_exits.items():
            entry.emit_count += 1
            logger.warning(
                "exit for %s not confirmed, re-emitting (emission %d, reason=%s, event_id=%s)",
                symbol,
                entry.emit_count,
                entry.reason,
                entry.position.event_id,
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
            # Trigger-time broker capture (CL-9dhg): symbol absent from a
            # READABLE snapshot nets to 0.0 (broker demonstrably does not
            # hold it); an unreadable snapshot/quantity captures None.
            trigger_broker_qty: float | None = None
            if trigger_net is not None:
                trigger_broker_qty = trigger_net.get(
                    self._norm_symbol(symbol),
                    0.0,
                )
            entry = PendingExit(
                position=pos,
                reason=exit_reason,
                triggered_ts=now,
                trigger_price=current,
                trigger_broker_qty=trigger_broker_qty,
            )
            del self.open_positions[symbol]
            self.pending_exits[symbol] = entry
            self.save()
            logger.info(
                "Event exit triggered %s: %s trigger_price=%s held=%.1fh "
                "event_id=%s — awaiting broker flat confirmation",
                symbol,
                exit_reason,
                "n/a" if current is None else f"{current:.5f}",
                held_hours,
                pos.event_id,
            )
            records.append(self._pending_exit_record(symbol, entry))
        return records

    #: Broker net quantities within this of zero count as FLAT (units are
    #: broker position units — 1 unit of an FX pair is dust).
    _FLAT_QTY = 1.0

    #: Minimum |observed fill| (broker units) for a pending entry to PROMOTE
    #: (ENTRY half of CL-hqyj). Below this the fill is dust — a rejected or
    #: not-yet-visible order — and the leg stays pending until grace expires,
    #: then REJECTS. Same 1-unit dust floor the exit side uses for "flat".
    _MIN_ENTRY_FILL = 1.0

    def confirm_exits(self, broker_positions: Iterable[Any]) -> list[ExitRecord]:
        """Finalize pending exits the broker CONFIRMS are out (CL-8cw1;
        residual + phantom semantics CL-9dhg findings 1 + 2).

        ``broker_positions`` is the SAME snapshot :meth:`reconcile`
        fetched this tick — one broker call per tick, never a second.
        Symbols match via the shared canonical_symbol on account-wide
        NET quantities. A pending leg confirms when EITHER

          (a) the broker is flat in the symbol (|net qty| < 1 unit), OR
          (b) ``trigger_broker_qty`` was captured at trigger and the
              current net equals the expected co-holder residual
              ``trigger_broker_qty - position.quantity`` within
              ``max(1, 1% of |position.quantity|)`` — our share is out;
              the remainder belongs to a sibling strategy that co-holds
              the instrument, which would otherwise keep the account
              non-flat FOREVER (re-emit loop, occupied slot, unbooked
              P&L).

        Legacy pending entries (``trigger_broker_qty`` None — persisted
        pre-CL-9dhg, or broker unreadable at trigger) confirm only via
        (a). A confirming leg whose trigger capture shows the broker
        NEVER held it (``trigger_broker_qty == 0`` — a rejected entry
        whose stop crossed inside the reconcile grace) finalizes as
        PHANTOM: WARNING, no realized P&L, no ``closed_trades`` bump, no
        ExitRecord — booking a loss for a trade that never existed would
        poison reflective_review and the loss-cap freeze. Everything
        else finalizes exactly as the pre-CL-8cw1 trigger path did —
        realized P&L from the trigger price captured at trigger time,
        ``closed_trades`` bump, persisted, ExitRecord returned — and
        exactly ONCE: finalized legs leave ``pending_exits``, so a
        repeat call with the same snapshot is a no-op."""
        if not self.pending_exits:
            return []
        net = self._net_broker_quantities(broker_positions)
        records: list[ExitRecord] = []
        for symbol, entry in list(self.pending_exits.items()):
            current_qty = net.get(self._norm_symbol(symbol), 0.0)
            if current_qty is None:
                continue  # quantity unreadable — never finalize blind
            confirmed = abs(current_qty) < self._FLAT_QTY  # (a) broker flat
            if not confirmed and entry.trigger_broker_qty is not None:
                # (b) co-holder residual: our share left the account.
                expected_residual = entry.trigger_broker_qty - entry.position.quantity
                tolerance = max(1.0, 0.01 * abs(entry.position.quantity))
                confirmed = abs(current_qty - expected_residual) <= tolerance
            if not confirmed:
                continue  # broker still holds our share — keep retrying
            del self.pending_exits[symbol]
            if entry.trigger_broker_qty is not None and abs(entry.trigger_broker_qty) < 1e-9:
                # PHANTOM (CL-9dhg finding 2): the broker demonstrably
                # never held this leg at trigger — the entry order never
                # filled. Drop it WITHOUT booking P&L: a fabricated
                # realized loss would poison reflective_review and could
                # trip the loss-cap freeze on a trade that never existed.
                self.save()
                logger.warning(
                    "Event exit %s: PHANTOM — broker never held the leg at "
                    "trigger (trigger_broker_qty=0, reason=%s, event_id=%s, "
                    "%d emission(s)). Entry order never filled; dropping "
                    "WITHOUT booking realized P&L.",
                    symbol,
                    entry.reason,
                    entry.position.event_id,
                    entry.emit_count,
                )
                continue
            record = self._pending_exit_record(symbol, entry)
            self.realized_pnl += record.pnl
            self.closed_trades += 1
            self.save()
            logger.info(
                "Event exit %s: %s pnl=%.2f held=%.1fh event_id=%s (broker "
                "confirmed %s after %d emission(s))",
                symbol,
                entry.reason,
                record.pnl,
                record.held_hours,
                entry.position.event_id,
                "flat"
                if abs(current_qty) < self._FLAT_QTY
                else f"co-holder residual {current_qty:.0f}",
                entry.emit_count,
            )
            records.append(record)
        return records

    def confirm_entries(
        self,
        broker_positions: Iterable[Any],
        now: datetime,
    ) -> list[EventPosition]:
        """Confirm pending entries against the broker (ENTRY half of
        CL-hqyj — mirror of :meth:`confirm_exits`).

        ``broker_positions`` is the SAME snapshot :meth:`reconcile` fetched
        this tick (one broker call per tick). For each pending entry the
        OBSERVED fill is the co-held-safe delta from the submit-time
        baseline: ``broker_net_now(canon) - entry_broker_qty``. Then:

          * PROMOTE when ``|observed_fill| >= _MIN_ENTRY_FILL`` AND its sign
            matches the intended direction — move the leg into
            ``open_positions`` as an :class:`EventPosition` whose quantity
            is the OBSERVED FILL (the partial-fill fix: intended -308,
            broker delta -214 → books -214), preserving
            entry_price/stop/event_id/headline/entry_ts. Only now is it
            reconciler-facing exposure and eligible for stop/time-stop.
          * REJECT when ``now - submitted_ts > grace`` and no qualifying
            fill — drop from ``pending_entries``, book NOTHING, WARNING
            (mirror of the phantom prune for the pending path).
          * else stay pending (``confirmed_qty`` updated each call).

        ``entry_broker_qty is None`` (broker unreadable at submit — rare
        double-degradation): best-effort. After grace, promote at
        ``min(|intended|, |broker_net_now|)`` with the intended sign, or
        REJECT if the broker is flat in the symbol. We cannot isolate our
        share of a co-held net without the baseline, so we cap the promoted
        size at our intended magnitude (never over-book someone else's
        position) and never wait past grace.

        Idempotent: promoted/rejected legs leave ``pending_entries``, so a
        repeat call with the same snapshot is a no-op. Returns the list of
        newly-PROMOTED positions (for logging/analytics; the strategy does
        not re-emit on promotion)."""
        if not self.pending_entries:
            return []
        net = self._net_broker_quantities(broker_positions)
        grace = timedelta(seconds=self._reconcile_grace_sec)
        promoted: list[EventPosition] = []
        for symbol, entry in list(self.pending_entries.items()):
            pos = entry.position
            broker_now = net.get(self._norm_symbol(symbol), 0.0)
            if broker_now is None:
                continue  # quantity unreadable — never promote/reject blind
            aged_out = now - entry.submitted_ts > grace

            if entry.entry_broker_qty is not None:
                observed_fill = broker_now - entry.entry_broker_qty
                entry.confirmed_qty = observed_fill
                sign_ok = (observed_fill > 0) == (pos.direction > 0)
                if abs(observed_fill) >= self._MIN_ENTRY_FILL and sign_ok:
                    self._promote_entry(symbol, entry, observed_fill, promoted)
                    continue
                if aged_out:
                    self._reject_entry(symbol, entry, broker_now)
                    continue
                # A qualifying-magnitude fill in the WRONG direction before
                # grace is left pending (a co-holder moving against us);
                # grace will REJECT it if our fill never materializes.
                self.save()  # persist the updated confirmed_qty
                continue

            # entry_broker_qty is None — broker unreadable at submit. We
            # cannot measure a delta, so wait for grace, then best-effort.
            if not aged_out:
                continue
            if abs(broker_now) < self._FLAT_QTY:
                self._reject_entry(symbol, entry, broker_now)
                continue
            # Promote at min(|intended|, |broker net|) with the intended
            # sign — never over-book a co-holder's share we can't isolate.
            magnitude = min(abs(pos.quantity), abs(broker_now))
            best_effort = magnitude if pos.direction > 0 else -magnitude
            entry.confirmed_qty = best_effort
            logger.warning(
                "Event entry %s: broker was unreadable at submit — "
                "best-effort promoting at %.0f (min of intended %.0f and "
                "broker net %.0f, intended sign) after grace (event_id=%s)",
                symbol,
                best_effort,
                pos.quantity,
                broker_now,
                pos.event_id,
            )
            self._promote_entry(symbol, entry, best_effort, promoted)
        return promoted

    def _promote_entry(
        self,
        symbol: str,
        entry: PendingEntry,
        fill: float,
        promoted: list[EventPosition],
    ) -> None:
        """Move a confirmed pending entry into ``open_positions`` at the
        ACTUAL broker fill (the partial-fill fix), persist, and log."""
        pos = entry.position
        confirmed = EventPosition(
            symbol=pos.symbol,
            event_id=pos.event_id,
            entry_ts=pos.entry_ts,
            entry_price=pos.entry_price,
            quantity=fill,
            direction=pos.direction,
            stop_price=pos.stop_price,
            headline=pos.headline,
        )
        del self.pending_entries[symbol]
        self.open_positions[symbol] = confirmed
        promoted.append(confirmed)
        self.save()
        if abs(abs(fill) - abs(pos.quantity)) > self._MIN_ENTRY_FILL:
            logger.warning(
                "Event entry %s PARTIALLY filled: booked %.0f vs intended "
                "%.0f (event_id=%s) — book now tracks the ACTUAL broker size",
                symbol,
                fill,
                pos.quantity,
                pos.event_id,
            )
        else:
            logger.info(
                "Event entry %s confirmed: filled %.0f (intended %.0f) "
                "event_id=%s — promoted to open, stop/time-stop now active",
                symbol,
                fill,
                pos.quantity,
                pos.event_id,
            )

    def _reject_entry(
        self,
        symbol: str,
        entry: PendingEntry,
        broker_now: float,
    ) -> None:
        """Drop a pending entry that never filled within grace — book
        NOTHING (mirror of the phantom prune, for the pending path)."""
        pos = entry.position
        del self.pending_entries[symbol]
        self.save()
        logger.warning(
            "Event entry %s REJECTED — no qualifying fill within grace "
            "(broker net %.0f, intended %.0f, event_id=%s); dropping WITHOUT "
            "booking any position (order likely rejected)",
            symbol,
            broker_now,
            pos.quantity,
            pos.event_id,
        )

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
        feed :meth:`confirm_exits` and :meth:`confirm_entries` from the SAME
        snapshot (one broker call per tick), or None when there was nothing
        to fetch or the broker was unreadable (fail-safe: prune nothing,
        confirm nothing). Pending-exit legs are never phantom-pruned — a
        pending symbol the broker no longer holds is a CONFIRMED exit whose
        P&L :meth:`confirm_exits` must book, not a phantom to drop.
        Pending-ENTRY legs are likewise never pruned here — their own
        promote/reject lifecycle (:meth:`confirm_entries`) is the correct
        backstop; the phantom pruner now only backstops LEGACY
        ``open_positions`` legs (record_entry no longer creates prunable
        open legs — ENTRY half of CL-hqyj)."""
        if not self.open_positions and not self.pending_exits and not self.pending_entries:
            return None
        try:
            positions = list(broker.get_positions())
        except Exception:
            logger.debug(
                "event_driven: broker positions unavailable — skipping phantom reconciliation",
                exc_info=True,
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
                    symbol,
                    pos.event_id,
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
                    self.realized_pnl,
                    cap,
                    self._max_loss_pct * 100,
                    equity,
                    self._state_path_str,
                )
                self._breach_logged = True
            else:
                logger.warning(
                    "Event book loss cap still breached (realized P&L %.2f) — new entries blocked",
                    self.realized_pnl,
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
        SINGLE instrument, from tracked positions (open + pending-exit +
        pending-ENTRY at its intended magnitude — a pending exit is still
        broker exposure until confirmed flat (CL-8cw1) and a pending entry
        is a committed submission that must count against the cap before it
        confirms, ENTRY half of CL-hqyj) — uses the accounting view,
        nothing new persisted."""
        total = 0.0
        for sym, pos in self.accounting_positions().items():
            if sym == symbol:
                total += abs(pos.quantity) * pos.entry_price
        return total

    def _open_haven_notional(self) -> float:
        """Combined open notional (|quantity| * entry_price) across
        HAVEN_INSTRUMENTS (gold/silver), from tracked positions (open +
        pending-exit + pending-entry at intended magnitude, as above)."""
        total = 0.0
        for sym, pos in self.accounting_positions().items():
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
                binding,
                cap_pct * 100,
                open_exposure,
                cap_notional,
                symbol,
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
            binding,
            symbol,
            size,
            capped,
            open_exposure,
            proposed_notional,
            cap_notional,
            cap_pct * 100,
        )
        return capped
