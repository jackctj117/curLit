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

ACCOUNT CURRENCY (CL-vfw7): every amount compared against account equity
is converted from the leg's QUOTE currency first, via an injected
``quote -> account`` rate (:mod:`src.risk.currency`). Each leg carries
its ``quote_ccy`` and entry conversion; each exit captures its own exit
conversion (at trigger time, retried at confirmation). ``realized_pnl``
— the field the loss cap reads — is the ACCOUNT-currency sum from state
version 2 on. A version-1 file's aggregate was a MIXED-currency sum (e.g.
¥44.29 booked as "+44.29"); it is preserved verbatim, labelled
``legacy_mixed_currency_pnl``, and never added to ``realized_pnl``. Its
account-currency value is UNKNOWN (either sign can hide losses), so the
loss budget is unknown and NEW entries are blocked until the operator
records ``legacy_reconciled_account_pnl`` from broker records; then
loss consumed = ``-realized_pnl - legacy_reconciled_account_pnl``. A close
whose exit conversion is unavailable (or only stale) books ``pnl_quote``
into ``unconverted_closes``; while any of those is a LOSS new entries are
blocked until the operator records its ``reconciled_pnl_account`` — no
rate is ever guessed or back-filled from a later tick.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.risk.currency import MAX_RATE_AGE, Conversion, ConversionUnavailable, quote_currency

logger = logging.getLogger(__name__)

#: v2 (CL-vfw7): ``realized_pnl`` is ACCOUNT currency; adds
#: ``account_currency``, ``legacy_mixed_currency_pnl``,
#: ``currency_migrated_at``, ``unconverted_closes``, ``recent_closed`` and
#: per-leg ``quote_ccy`` / conversions. v1 files (no ``version`` key or 1)
#: still load; anything else refuses to start.
_STATE_VERSION = 2
_SUPPORTED_STATE_VERSIONS = frozenset({1, 2})

#: Closed-trade provenance rows retained in the state file. Event legs
#: close a few times a week, so 100 rows is months of audit trail while
#: keeping the atomically-rewritten file small.
_RECENT_CLOSED_MAX = 100

#: ``quote -> account`` rate resolver: returns a Conversion or raises
#: ConversionUnavailable (never a default rate).
ConvertFn = Callable[[str, datetime], Conversion]

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
    #: Quote currency of ``entry_price``/``stop_price`` (CL-vfw7). None on
    #: legs persisted before the currency migration — derived from the
    #: symbol when needed, never assumed to be the account currency.
    quote_ccy: str | None = None
    #: The quote→account rate used to SIZE this leg (provenance only).
    entry_conversion: Conversion | None = None


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
    #: P&L in the leg's QUOTE currency, from the trigger price (an
    #: ESTIMATE of the fill — the broker-flat confirmation is not fill
    #: evidence). CL-vfw7 split ``pnl`` into the two denominations.
    pnl_quote: float
    #: ``pnl_quote`` converted at the exit conversion; None when no fresh
    #: rate was available (never booked into ``realized_pnl`` then).
    pnl_account: float | None
    held_hours: float
    current_price: float | None
    #: None when this exit's account-currency amount is unknown (CL-vfw7).
    book_realized_pnl: float | None
    #: Exit-intent emissions so far for this leg (1 = the trigger tick).
    #: The strategy records the exit FeatureSnapshot only on the first
    #: emission (CL-9dhg finding 10) — re-emissions would write one
    #: near-identical row per tick per pending exit.
    emit_count: int = 1
    quote_ccy: str | None = None
    exit_conversion: Conversion | None = None


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
    #: quote→account rate captured at TRIGGER time (CL-vfw7); None when
    #: unavailable then (retried at confirmation; missing on legacy rows).
    exit_conversion: Conversion | None = None


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
        account_currency: str = "USD",
        convert: ConvertFn | None = None,
    ) -> None:
        self._state_path_str = state_path
        # CL-vfw7: the currency every equity-comparable amount is booked in,
        # and the quote→account rate resolver (None = identity only — any
        # other quote currency is unavailable, i.e. fail closed).
        self.account_currency = account_currency.strip().upper()
        assert len(self.account_currency) == 3, "account currency must be ISO-4217"
        self._convert: ConvertFn | None = convert
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
        #: ACCOUNT-currency realized P&L (CL-vfw7) — what the loss cap reads.
        self.realized_pnl: float = 0.0
        self.closed_trades: int = 0
        #: Pre-CL-vfw7 aggregate: a MIXED-currency sum of quote-currency
        #: P&L. Kept verbatim for the record, NEVER added to realized_pnl.
        #: None = no legacy history (fresh book or already-clean v2).
        self.legacy_mixed_currency_pnl: float | None = None
        #: Closed-trade count at migration (the legacy history's size).
        self.legacy_closed_trades: int = 0
        #: OPERATOR-supplied account-currency realized P&L of the legacy
        #: history (from broker transaction records). None = unreconciled →
        #: the loss budget is unknown and new entries are blocked.
        self.legacy_reconciled_account_pnl: float | None = None
        #: When the v1→v2 migration ran (None = book born on v2).
        self.currency_migrated_at: datetime | None = None
        #: Closed trades whose account P&L is UNKNOWN (no fresh exit rate):
        #: dicts with symbol/quote_ccy/pnl_quote/closed_at/... — see
        #: resolve_unconverted. Losses here block new entries.
        self.unconverted_closes: list[dict[str, Any]] = []
        #: Bounded closed-trade provenance (newest last).
        self.recent_closed: list[dict[str, Any]] = []
        # Loss-cap breach is CRITICAL once per activation, WARNING after.
        self._breach_logged = False
        self.load()

    # ------------------------------------------------------------------
    # Currency conversion (CL-vfw7)
    # ------------------------------------------------------------------

    def set_converter(self, convert: ConvertFn | None) -> None:
        self._convert = convert

    def conversion(self, quote_ccy: str | None, now: datetime) -> Conversion:
        """quote→account rate or :class:`ConversionUnavailable`. Identity
        needs no market data; anything else needs the injected resolver."""
        if quote_ccy is None:
            raise ConversionUnavailable("instrument quote currency unknown")
        if quote_ccy == self.account_currency:
            return Conversion(quote_ccy, self.account_currency, 1.0, "identity", now)
        if self._convert is None:
            raise ConversionUnavailable(
                f"no rate source for {quote_ccy}->{self.account_currency}",
            )
        conv = self._convert(quote_ccy, now)
        if conv.quote_ccy != quote_ccy or conv.account_ccy != self.account_currency:
            raise ConversionUnavailable(
                f"converter returned {conv.quote_ccy}->{conv.account_ccy} for "
                f"{quote_ccy}->{self.account_currency}"
            )
        return conv

    @staticmethod
    def leg_quote_ccy(pos: EventPosition) -> str | None:
        return pos.quote_ccy or quote_currency(pos.symbol)

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
        except (ValueError, OSError) as exc:
            # CL-74u9: preserve the evidence in place. A fresh book would
            # silently erase owned positions and grant a new loss budget.
            raise ValueError(
                f"Event book state {path} unreadable; reconciliation required"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Event book state {path} must be an object")
        for collection in ("open_positions", "pending_exits", "pending_entries"):
            if collection in payload and not isinstance(payload[collection], dict):
                raise ValueError(f"Event book {collection} must be an object")
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
        version = payload.get("version", 1)
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in _SUPPORTED_STATE_VERSIONS
        ):
            raise ValueError(f"Event book state {path} has unsupported version {version!r}")
        try:
            stored_realized = float(payload.get("realized_pnl", 0.0))
            self.closed_trades = int(payload.get("closed_trades", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Event book has invalid loss/trade history") from exc
        if not math.isfinite(stored_realized) or self.closed_trades < 0:
            raise ValueError("Event book has invalid loss/trade history; reconciliation required")
        self._load_currency_fields(path, version, stored_realized, payload)
        for sym, pos in (payload.get("open_positions") or {}).items():
            try:
                self.open_positions[sym] = self._position_from_payload(sym, pos)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unparseable event position {sym}; refusing partial book"
                ) from exc
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
                    exit_conversion=self._conversion_from_payload(entry.get("exit_conversion")),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Unparseable pending exit {sym}; refusing partial book") from exc
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
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Unparseable pending entry {sym}; refusing partial book") from exc
        if version == 1:
            # CL-vfw7: persist the v1→v2 migration as soon as the whole file
            # validated, so the operator can reconcile the legacy figure in
            # the v2 file (a flat book would otherwise never save, and the v1
            # branch never reads legacy_reconciled_account_pnl).
            logger.warning("Event book %s: persisting v1→v2 currency migration", path)
            self.save()

    def _load_currency_fields(
        self,
        path: Path,
        version: int,
        stored_realized: float,
        payload: dict[str, Any],
    ) -> None:
        """CL-vfw7 currency semantics of the persisted aggregate.

        v1: ``realized_pnl`` was a mixed-currency sum → preserved as
        ``legacy_mixed_currency_pnl``; account realized starts at 0 and the
        migration marker is stamped (persisted on the next save). v2: the
        file's ``account_currency`` must match ours (a USD sum read as EUR
        would be silently wrong) and every new field must be sane."""
        if version == 1:
            self.legacy_mixed_currency_pnl = stored_realized
            self.legacy_closed_trades = self.closed_trades
            self.realized_pnl = 0.0
            self.currency_migrated_at = datetime.now(UTC)
            logger.warning(
                "Event book %s: v1 state — realized_pnl %.2f over %d trade(s) is a "
                "MIXED-currency sum (quote-currency P&L of every pair added "
                "together). Preserved as legacy_mixed_currency_pnl, NOT counted as "
                "%s; account-currency realized P&L starts at 0.00 from %s (CL-vfw7). "
                "Its %s value is UNKNOWN, so new event entries stay BLOCKED until the "
                "operator sets legacy_reconciled_account_pnl (docs/CURRENT_OPERATIONS.md §1a).",
                path,
                stored_realized,
                self.legacy_closed_trades,
                self.account_currency,
                self.currency_migrated_at.isoformat(),
                self.account_currency,
            )
            return
        stored_ccy = payload.get("account_currency")
        if not isinstance(stored_ccy, str) or stored_ccy.upper() != self.account_currency:
            raise ValueError(
                f"Event book state {path} is denominated in {stored_ccy!r}, but the "
                f"account currency is {self.account_currency!r} — refusing to start "
                "(a realized sum in another currency cannot be compared to equity)"
            )
        self.realized_pnl = stored_realized
        legacy = payload.get("legacy_mixed_currency_pnl")
        if legacy is not None:
            try:
                legacy_f = float(legacy)
            except (TypeError, ValueError) as exc:
                raise ValueError("Event book legacy_mixed_currency_pnl unparseable") from exc
            if not math.isfinite(legacy_f):
                raise ValueError("Event book legacy_mixed_currency_pnl is not finite")
            self.legacy_mixed_currency_pnl = legacy_f
            # Absent count on a legacy figure = unknown size → treat as
            # non-empty (never silently clears the reconciliation gate).
            self.legacy_closed_trades = self._finite_int(
                payload.get("legacy_closed_trades", 1), "legacy_closed_trades"
            )
        self.legacy_reconciled_account_pnl = self._optional_finite(
            payload.get("legacy_reconciled_account_pnl"), "legacy_reconciled_account_pnl"
        )
        migrated = payload.get("currency_migrated_at")
        if migrated is not None:
            try:
                ts = datetime.fromisoformat(str(migrated))
            except ValueError as exc:
                raise ValueError("Event book currency_migrated_at unparseable") from exc
            self.currency_migrated_at = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        for key in ("unconverted_closes", "recent_closed"):
            raw = payload.get(key, [])
            if not isinstance(raw, list) or not all(isinstance(r, dict) for r in raw):
                raise ValueError(f"Event book {key} must be a list of objects")
        for row in payload.get("unconverted_closes", []):
            try:
                pnl_q = float(row["pnl_quote"])
                ccy = str(row["quote_ccy"])
                str(row["symbol"])
                datetime.fromisoformat(str(row["closed_at"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Event book unconverted close unparseable: {row!r}") from exc
            if not math.isfinite(pnl_q) or len(ccy) != 3 or not ccy.isalpha():
                raise ValueError(f"Event book unconverted close invalid: {row!r}")
            self._optional_finite(row.get("reconciled_pnl_account"), "reconciled_pnl_account")
        self.recent_closed = list(payload.get("recent_closed", []))[-_RECENT_CLOSED_MAX:]
        # Operator reconciliation of an unconverted close: book the
        # operator-supplied ACCOUNT amount (broker-statement evidence), never
        # a rate we chose. Unreconciled rows stay and keep blocking.
        remaining: list[dict[str, Any]] = []
        for row in payload.get("unconverted_closes", []):
            reconciled = self._optional_finite(
                row.get("reconciled_pnl_account"), "reconciled_pnl_account"
            )
            if reconciled is None:
                remaining.append(row)
                continue
            self.realized_pnl += reconciled
            self._append_recent_closed(
                {**row, "pnl_account": reconciled, "pnl_account_source": "operator_reconciled"}
            )
            logger.warning(
                "Event book: operator-reconciled close %s (%s %.2f) booked as %.2f %s",
                row["symbol"],
                row["quote_ccy"],
                float(row["pnl_quote"]),
                reconciled,
                self.account_currency,
            )
        self.unconverted_closes = remaining

    @staticmethod
    def _optional_finite(raw: Any, name: str) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, bool):
            raise ValueError(f"Event book {name} must be a number")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Event book {name} unparseable: {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Event book {name} is not finite")
        return value

    @staticmethod
    def _finite_int(raw: Any, name: str) -> int:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"Event book {name} must be a non-negative integer: {raw!r}")
        return int(raw)

    @staticmethod
    def _conversion_from_payload(raw: Any) -> Conversion | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"conversion record must be an object: {raw!r}")
        return Conversion.from_payload(raw)

    @classmethod
    def _position_from_payload(cls, sym: str, pos: dict[str, Any]) -> EventPosition:
        """Parse one persisted position; invalid state must prevent startup.
        ``quote_ccy``/``entry_conversion`` are ABSENT on pre-CL-vfw7 legs."""
        entry_ts = datetime.fromisoformat(pos["entry_ts"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=UTC)
        quote_raw = pos.get("quote_ccy")
        if quote_raw is not None and (
            not isinstance(quote_raw, str) or len(quote_raw) != 3 or not quote_raw.isalpha()
        ):
            raise ValueError(f"invalid quote_ccy {quote_raw!r}")
        return EventPosition(
            symbol=sym,
            event_id=pos.get("event_id"),
            entry_ts=entry_ts,
            entry_price=float(pos["entry_price"]),
            quantity=float(pos["quantity"]),
            direction=int(pos["direction"]),
            stop_price=float(pos["stop_price"]),
            headline=str(pos.get("headline", "")),
            quote_ccy=quote_raw.upper() if isinstance(quote_raw, str) else None,
            entry_conversion=cls._conversion_from_payload(pos.get("entry_conversion")),
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
            "quote_ccy": pos.quote_ccy,
            "entry_conversion": (
                pos.entry_conversion.to_payload() if pos.entry_conversion is not None else None
            ),
        }

    def save(self) -> None:
        path = self._path()
        payload = {
            "version": _STATE_VERSION,
            # ACCOUNT-currency realized P&L (CL-vfw7) — the loss-cap input.
            "realized_pnl": self.realized_pnl,
            "account_currency": self.account_currency,
            "legacy_mixed_currency_pnl": self.legacy_mixed_currency_pnl,
            "legacy_closed_trades": self.legacy_closed_trades,
            "legacy_reconciled_account_pnl": self.legacy_reconciled_account_pnl,
            "currency_migrated_at": (
                self.currency_migrated_at.isoformat()
                if self.currency_migrated_at is not None
                else None
            ),
            "unconverted_closes": self.unconverted_closes,
            "recent_closed": self.recent_closed[-_RECENT_CLOSED_MAX:],
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
                    "exit_conversion": (
                        entry.exit_conversion.to_payload()
                        if entry.exit_conversion is not None
                        else None
                    ),
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
                quote_ccy=pos.quote_ccy,
                entry_conversion=pos.entry_conversion,
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
                quote_ccy=pos.quote_ccy,
                entry_conversion=pos.entry_conversion,
            )
        return merged

    def _pending_exit_record(self, symbol: str, entry: PendingExit) -> ExitRecord:
        """ExitRecord for a pending exit — every value derives from the
        trigger-time capture, so re-emissions are idempotent.
        ``book_realized_pnl`` is the projected cumulative P&L after this
        exit CONFIRMS (pre-CL-8cw1 snapshot semantics preserved)."""
        pos = entry.position
        # Quote-currency P&L: price units of the QUOTE currency × units.
        pnl_quote = (
            (entry.trigger_price - pos.entry_price) * pos.quantity
            if entry.trigger_price is not None
            else 0.0
        )
        conv = entry.exit_conversion
        pnl_account = pnl_quote * conv.rate if conv is not None else None
        held_hours = (entry.triggered_ts - pos.entry_ts).total_seconds() / 3600.0
        return ExitRecord(
            symbol=symbol,
            position=pos,
            reason=entry.reason,
            pnl_quote=pnl_quote,
            pnl_account=pnl_account,
            held_hours=held_hours,
            current_price=entry.trigger_price,
            # Projected ACCOUNT-currency total; None when this exit's account
            # amount is unknown (never a number that silently omits it).
            # None also while ANY earlier close is unconverted: the
            # cumulative total is then unknown, not a partial subtotal.
            book_realized_pnl=(
                self.realized_pnl + pnl_account
                if pnl_account is not None and not self.unconverted_closes
                else None
            ),
            emit_count=entry.emit_count,
            quote_ccy=self.leg_quote_ccy(pos),
            exit_conversion=conv,
        )

    def _try_exit_conversion(self, entry: PendingExit, now: datetime) -> None:
        """Ensure the exit quote→account rate is FRESH as of ``now``
        (captured at trigger, re-checked at confirmation). A rate older than
        MAX_RATE_AGE — an exit that stayed pending, or survived a restart —
        is discarded and re-resolved; failure leaves it None (→ the close
        books as unconverted, never at the stale rate)."""
        conv = entry.exit_conversion
        if conv is not None and now - conv.observed_at <= MAX_RATE_AGE:
            return
        if conv is not None:
            logger.warning(
                "Event exit %s: exit conversion observed %s is stale at %s — re-resolving",
                entry.position.symbol,
                conv.observed_at.isoformat(),
                now.isoformat(),
            )
            entry.exit_conversion = None
        try:
            entry.exit_conversion = self.conversion(self.leg_quote_ccy(entry.position), now)
        except ConversionUnavailable as exc:
            logger.warning(
                "Event exit %s: quote->%s conversion unavailable (%s) — P&L "
                "stays quote-only until a fresh rate exists",
                entry.position.symbol,
                self.account_currency,
                exc,
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
                raw_qty = p.quantity
                qty = float(raw_qty)
            except (AttributeError, TypeError, ValueError):
                net[sym] = None
                continue
            if isinstance(raw_qty, bool) or not math.isfinite(qty):
                # CL-oqos: NaN/inf compares False against every threshold, so
                # confirm_entries would REJECT a possibly-filled leg after
                # grace as if the broker were flat. Unknown, not flat.
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
            # CL-vfw7: rate captured at the same moment as the trigger price.
            self._try_exit_conversion(entry, now)
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

    def confirm_exits(
        self,
        broker_positions: Iterable[Any],
        now: datetime | None = None,
    ) -> list[ExitRecord]:
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
        repeat call with the same snapshot is a no-op.

        CL-vfw7: ``realized_pnl`` grows by the ACCOUNT-currency P&L only.
        If no exit conversion was captured at trigger and none is fresh now,
        the close books into ``unconverted_closes`` (quote amount only) and
        ``realized_pnl`` is untouched — never a guessed rate."""
        if not self.pending_exits:
            return []
        now = now or datetime.now(UTC)
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
            late = now - entry.triggered_ts > MAX_RATE_AGE
            if late and self.leg_quote_ccy(entry.position) != self.account_currency:
                # (Identity — quote == account — is time-invariant and exempt.)
                # The fill happened somewhere between trigger and now (an
                # exit pending across re-emits or a restart); no rate we hold
                # or can fetch is known to be the fill-time rate → the close
                # books as UNCONVERTED (operator reconciliation), never at a
                # trigger-time or today's rate.
                logger.warning(
                    "Event exit %s confirmed %.0fs after trigger (> %s): fill-time "
                    "rate unknown — booking as unconverted (trigger-time rate %s "
                    "not used)",
                    symbol,
                    (now - entry.triggered_ts).total_seconds(),
                    MAX_RATE_AGE,
                    entry.exit_conversion.to_payload() if entry.exit_conversion else None,
                )
                entry.exit_conversion = None
            else:
                self._try_exit_conversion(entry, now)
            record = self._pending_exit_record(symbol, entry)
            self.closed_trades += 1
            closed_row = self._closed_row(record, now)
            if record.pnl_account is not None:
                self.realized_pnl += record.pnl_account
            else:
                self.unconverted_closes.append(closed_row)
            self._append_recent_closed(closed_row)
            self.save()
            logger.info(
                "Event exit %s: %s pnl_quote=%.2f %s pnl_account=%s %s held=%.1fh "
                "event_id=%s (broker confirmed %s after %d emission(s))",
                symbol,
                entry.reason,
                record.pnl_quote,
                record.quote_ccy,
                "UNCONVERTED" if record.pnl_account is None else f"{record.pnl_account:.2f}",
                self.account_currency,
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
            quote_ccy=pos.quote_ccy,
            entry_conversion=pos.entry_conversion,
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

    @staticmethod
    def validated_snapshot(raw: Any) -> list[Any] | None:
        """The broker snapshot as a list, or None when it is malformed (CL-oqos).

        Same whole-snapshot contract as the cold-start reconciler
        (:func:`src.portfolio.reconciler.validate_broker_snapshot`): a
        non-list, an empty/unreadable symbol, a nonfinite/bool quantity, or
        two rows for one canonical symbol (``USD_CAD`` + ``USDCAD``) make
        the WHOLE snapshot unknown. Partial use is unsafe — ``{}`` would read
        as flat (rejecting a filled pending entry) and duplicate rows would
        sum into an overstated fill.
        """
        from src.portfolio.reconciler import (  # noqa: PLC0415
            SnapshotUnavailableError,
            validate_broker_snapshot,
        )

        try:
            validate_broker_snapshot(raw)
        except SnapshotUnavailableError as exc:
            logger.warning(
                "event_driven: broker snapshot malformed (%s) — no prune/confirm this tick",
                exc.reason,
            )
            return None
        return list(raw)

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
            raw = broker.get_positions()
        except Exception:
            logger.debug(
                "event_driven: broker positions unavailable — skipping phantom reconciliation",
                exc_info=True,
            )
            return None
        positions = self.validated_snapshot(raw)
        if positions is None:
            return None  # malformed snapshot: prune nothing, confirm nothing
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

    # ------------------------------------------------------------------
    # Closed-trade provenance + loss-cap knowability (CL-vfw7)
    # ------------------------------------------------------------------

    def _closed_row(self, record: ExitRecord, now: datetime) -> dict[str, Any]:
        pos = record.position
        return {
            "symbol": record.symbol,
            "event_id": pos.event_id,
            "reason": record.reason,
            # "XXX" = ISO-4217 "no currency": an unparseable quote stays
            # unconvertible (fail closed) instead of being assumed USD.
            "quote_ccy": record.quote_ccy if record.quote_ccy is not None else "XXX",
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "exit_price": record.current_price,
            # The exit price is the TRIGGER-time mid, not broker fill
            # evidence — an estimate, labelled as such.
            "exit_price_basis": "trigger_price_estimate",
            "pnl_quote": record.pnl_quote,
            "pnl_account": record.pnl_account,
            "account_ccy": self.account_currency,
            "entry_conversion": (
                pos.entry_conversion.to_payload() if pos.entry_conversion is not None else None
            ),
            "exit_conversion": (
                record.exit_conversion.to_payload() if record.exit_conversion is not None else None
            ),
            "closed_at": now.isoformat(),
        }

    def _append_recent_closed(self, row: dict[str, Any]) -> None:
        self.recent_closed.append(row)
        del self.recent_closed[:-_RECENT_CLOSED_MAX]

    def has_unconverted_losses(self) -> bool:
        """True while any closed trade's ACCOUNT-currency loss is unknown —
        the loss cap cannot be evaluated, so new entries must be blocked."""
        return any(float(r["pnl_quote"]) < 0 for r in self.unconverted_closes)

    def legacy_unreconciled(self) -> bool:
        """True while a pre-CL-vfw7 mixed-currency history exists whose
        account-currency value the operator has not supplied. Its USD value
        is UNKNOWN (a +1000 sum can hide -$5,000 and +¥6,000), so neither
        sign of it can be used — the loss budget is unknown."""
        if self.legacy_mixed_currency_pnl is None:
            return False
        if self.legacy_reconciled_account_pnl is not None:
            return False
        # An empty v1 history (no trades, zero sum) carries no unknown.
        return not (self.legacy_mixed_currency_pnl == 0.0 and self.legacy_closed_trades == 0)

    def loss_cap_unknown_reason(self) -> str | None:
        """Why the loss cap cannot be evaluated (→ block new entries), or
        None. Exits are never affected. Each is cleared only by operator
        reconciliation in the state file (see docs/CURRENT_OPERATIONS.md
        §1a): ``legacy_reconciled_account_pnl`` for the legacy history,
        ``reconciled_pnl_account`` on an ``unconverted_closes`` row."""
        if self.legacy_unreconciled():
            return "legacy_pnl_unreconciled"
        if self.has_unconverted_losses():
            return "unconverted_realized_loss"
        return None

    def loss_cap_consumed(self) -> float:
        """KNOWN loss-cap budget consumed, in ACCOUNT currency (CL-vfw7):
        ``-realized_pnl`` (account-currency since the migration) minus the
        operator-reconciled account value of the legacy history. Only
        meaningful when :meth:`loss_cap_unknown_reason` is None."""
        return -self.realized_pnl - (self.legacy_reconciled_account_pnl or 0.0)

    def breached(self, equity: float | None) -> bool:
        if equity is None or equity <= 0:
            return False
        cap = self._max_loss_pct * equity
        consumed = self.loss_cap_consumed()
        breached = consumed >= cap
        if breached:
            if not self._breach_logged:
                logger.critical(
                    "EVENT BOOK LOSS CAP BREACHED: loss consumed %.2f %s >= cap "
                    "%.2f (%.1f%% of equity %.0f; account realized %.2f, legacy "
                    "reconciled %s). NO new event positions will be opened; "
                    "exits still flow. Reset requires operator action on %s. "
                    "(Kill-switch integration pending — this is the loud log.)",
                    consumed,
                    self.account_currency,
                    cap,
                    self._max_loss_pct * 100,
                    equity,
                    self.realized_pnl,
                    self.legacy_reconciled_account_pnl,
                    self._state_path_str,
                )
                self._breach_logged = True
            else:
                logger.warning(
                    "Event book loss cap still breached (consumed %.2f %s) — new entries blocked",
                    consumed,
                    self.account_currency,
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

    def _account_notional(self, pos: EventPosition, now: datetime) -> float:
        """|quantity| * entry_price (QUOTE currency) converted to ACCOUNT
        currency at the current rate (CL-vfw7). Raises
        :class:`ConversionUnavailable` — the cap cannot be evaluated."""
        conv = self.conversion(self.leg_quote_ccy(pos), now)
        return abs(pos.quantity) * pos.entry_price * conv.rate

    def _open_instrument_notional(self, symbol: str, now: datetime | None = None) -> float:
        """Open ACCOUNT-currency notional in a SINGLE instrument, from
        tracked positions (open + pending-exit + pending-ENTRY at its
        intended magnitude — a pending exit is still broker exposure until
        confirmed flat (CL-8cw1) and a pending entry is a committed
        submission that must count against the cap before it confirms,
        ENTRY half of CL-hqyj) — uses the accounting view, nothing new
        persisted. Raises ConversionUnavailable (CL-vfw7)."""
        now = now or datetime.now(UTC)
        total = 0.0
        for sym, pos in self.accounting_positions().items():
            if sym == symbol:
                total += self._account_notional(pos, now)
        return total

    def _open_haven_notional(self, now: datetime | None = None) -> float:
        """Combined open ACCOUNT-currency notional across HAVEN_INSTRUMENTS
        (gold/silver), from tracked positions (open + pending-exit +
        pending-entry at intended magnitude, as above)."""
        now = now or datetime.now(UTC)
        total = 0.0
        for sym, pos in self.accounting_positions().items():
            if sym in HAVEN_INSTRUMENTS:
                total += self._account_notional(pos, now)
        return total

    def concentration_capped_size(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        equity: float,
        now: datetime | None = None,
    ) -> float:
        """Reduce a NEW event leg's signed ``size`` so it satisfies the
        concentration caps (CL-wbmw). Two layers: (1) per-instrument cap
        (``per_instrument_max_pct``) — ALWAYS, for every symbol; (2)
        haven-cluster cap (``haven_max_pct``) — additionally for havens,
        on combined gold+silver notional. The SMALLER headroom binds; the
        cap is a ceiling the base sizing grows toward, so under-cap legs
        pass through UNCHANGED. Returns the (possibly reduced) signed
        size — 0.0 when EITHER cap is already at/over (skip). Logs at
        WARNING naming the binding cap. Preserves sign. All notionals are
        ACCOUNT currency (CL-vfw7): raises ConversionUnavailable when the new
        leg or any counted leg has no fresh quote->account rate."""
        now = now or datetime.now(UTC)
        if equity <= 0:
            return size
        # Quote→account rate of the NEW leg (raises when unavailable).
        rate = self.conversion(quote_currency(symbol), now).rate

        # Layer 1: per-instrument headroom (every symbol).
        per_cap = self._per_instrument_max_pct * equity
        per_open = self._open_instrument_notional(symbol, now)
        headroom = per_cap - per_open
        binding = "per-instrument"
        cap_pct = self._per_instrument_max_pct
        open_exposure = per_open
        cap_notional = per_cap

        # Layer 2: haven-cluster headroom (havens only) — take the tighter.
        if symbol in HAVEN_INSTRUMENTS:
            haven_cap = self._haven_max_pct * equity
            haven_open = self._open_haven_notional(now)
            haven_headroom = haven_cap - haven_open
            if haven_headroom < headroom:
                headroom = haven_headroom
                cluster = "/".join(sorted(HAVEN_INSTRUMENTS))
                binding = f"haven-cluster ({cluster})"
                cap_pct = self._haven_max_pct
                open_exposure = haven_open
                cap_notional = haven_cap

        proposed_notional = abs(size) * entry_price * rate
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
        max_units = headroom / (entry_price * rate)
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
