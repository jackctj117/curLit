"""Event-driven strategy (CL-mnhw) — paper-trades CONFIRMED current events.

Consumer half of the current-events pipeline. Every poll it:

  1. Emits exit intents for open event positions that hit the hard stop
     or the hard TIME STOP (``event_max_holding_hours``, fires
     regardless of P&L — event edges decay in hours, not days).
     Triggered legs park in the book's ``pending_exits`` and the exit
     intent RE-emits every tick until the broker confirms flat
     (CL-8cw1, exit half of CL-hqyj) — realized P&L books only on
     confirmation, so a rejected exit self-heals instead of leaving the
     broker holding unstopped risk against a flat book.
  2. Polls ``geo_events`` for ASSESSED rows and runs them through
     :class:`src.events.confluence.EventConfluence` (Gate A quality +
     Gate B market confirmation).
  3. For events it newly CONFIRMED: alerts the operator (via the
     injected :class:`src.events.event_notifier.EventNotifier`, which
     owns all alert formatting + ``notify_operator`` dispatch) and emits
     tightly-risked OrderIntents for the tradable affected instruments,
     then marks the row TRADED. Sizing:
     ``equity * event_risk_pct / stop_distance`` with the stop
     ``event_stop_pct`` from entry.
  4. EXPIRED events with urgency >= ``expired_alert_min_urgency`` get a
     brief "expired unconfirmed" info alert (capped at one per run).

Event-book protection: cumulative realized P&L persists in
``data/event_book_state.json`` (atomic tmp+rename). Breaching
``event_book_max_loss_pct`` of equity blocks NEW entries and logs
CRITICAL — exits always still flow. All per-event position-book state
(open legs, realized P&L, loss-cap freeze, phantom reconciliation,
concentration caps) lives in :class:`src.strategies.event_book.EventBook`
(CL-e6lx extraction); this module delegates and keeps signal
evaluation, sizing, and intent emission.

Boot safety: if the ``geo_events`` table doesn't exist yet (producer
migration not applied), the strategy is a NO-OP that logs ONCE — the
engine must never fail to boot because the sibling half hasn't landed.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from src.events.confluence import (
    TRADABLE_KINDS,
    TRADE_DIRECTIONS,
    ConfluenceConfig,
    EventConfluence,
    mid_price,
)
from src.events.event_notifier import EventNotifier
from src.events.playbooks import Playbook, load_playbooks
from src.execution.oms import OrderIntent, Urgency
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    attach_snapshot_payload,
)
from src.risk.liquidity_window import LiquidityProfile, spread_bps_from_tick
from src.risk.sizing import PositionSizer
from src.strategies.event_book import (
    HAVEN_INSTRUMENTS,
    EventBook,
    EventPosition,
)
from src.strategies.event_driven_config import (
    EventDrivenConfig,
    _default_instrument_map,  # noqa: F401 — legacy import path
)

__all__ = [
    "HAVEN_INSTRUMENTS",
    "EventBook",
    "EventDrivenConfig",
    "EventDrivenStrategy",
    "EventPosition",
]

logger = logging.getLogger(__name__)

_FEATURE_SET_NAME = "event_driven"
_FEATURE_SET_VERSION = "v1"


def _build_theme_primary(playbooks: dict[str, Playbook]) -> dict[str, frozenset[str]]:
    """theme key → the set of TRADABLE (oanda/fx) instrument names that
    theme's playbook lists (CL-9nvq). This is the whitelist a CONFIRMED
    event of that theme may open machine legs on — equity_watch/polymarket
    are excluded (already advisory-only), and instruments belonging only to
    OTHER themes are absent, so they get demoted to advisory at trade time."""
    return {
        key: frozenset(i.instrument for i in pb.tradable_instruments)
        for key, pb in playbooks.items()
    }


class EventDrivenStrategy:
    #: The strategy sizes its own legs by risk (50bps at the stop) — the
    #: coordinator passes its intents through UNSCALED (CL-8lv6): applying
    #: the allocation weight on top double-applied sizing (broker held 1/n
    #: of the book -> persistent boot-time size_mismatch).
    self_sized = True

    def __init__(
        self,
        config: EventDrivenConfig | None = None,
        data_provider: Any = None,
        state_store: Any = None,
        snapshot_store: FeatureSnapshotStore | None = None,
        db_engine: Any = None,
        notifier: EventNotifier | None = None,
        liquidity_profile: LiquidityProfile | None = None,
    ) -> None:
        self.config = config or EventDrivenConfig()
        self.data = data_provider
        self.state = state_store
        self.snapshot_store = snapshot_store
        self.db = db_engine
        # CL-y412: hour-of-week liquidity gate for NEW event entries. Event
        # news breaks at all hours — including dead-liquidity windows where a
        # blown-out spread eats the edge — so entries are trimmed (or blocked)
        # by the window multiplier. None = inert (no refresh has run yet):
        # every entry passes at full size. Exits are NEVER gated (below).
        self._liquidity_profile = liquidity_profile
        # Operator alerting lives in the events layer (CL-ikz2) — the
        # notifier owns alert formatting + notify_operator dispatch.
        self.notifier = notifier or EventNotifier(
            event_risk_pct=self.config.event_risk_pct,
            event_stop_pct=self.config.event_stop_pct,
            event_max_holding_hours=self.config.event_max_holding_hours,
            confirm_window_max_minutes=self.config.confirm_window_max_minutes,
            db_engine=db_engine,
            price_resolver=self._current_price,
        )
        # Cross-asset corroboration config (CL-6mzn) — best-effort load.
        cross_asset_config = self._load_cross_asset_config()
        # Event playbooks — loaded ONCE, fail-SOFT: an unloadable/absent
        # config leaves an EMPTY map, which makes both consumers INERT rather
        # than break engine boot — theme-primary scoping (CL-9nvq) below, and
        # the confluence stale-fact confidence ceiling (CL-ylak, which reads
        # each theme's `last_reviewed` date passed in here).
        self._playbooks: dict[str, Playbook] = {}
        try:
            self._playbooks = load_playbooks(self.config.event_playbooks_path)
        except Exception:
            logger.warning(
                "event playbooks unloadable at %s — theme-primary scoping and "
                "the stale-fact confidence ceiling are INERT",
                self.config.event_playbooks_path,
                exc_info=True,
            )
        self.confluence = EventConfluence(
            config=ConfluenceConfig(
                min_urgency=self.config.min_urgency,
                min_confidence=self.config.min_confidence,
                confirm_window_min_minutes=self.config.confirm_window_min_minutes,
                confirm_window_max_minutes=self.config.confirm_window_max_minutes,
                confirm_move_frac=self.config.confirm_move_frac,
                min_confirmed_instruments=self.config.min_confirmed_instruments,
                realized_vol_window=self.config.realized_vol_window,
                vol_spike_check_enabled=self.config.vol_spike_check_enabled,
                vol_spike_window=self.config.vol_spike_window,
                vol_spike_ratio=self.config.vol_spike_ratio,
                stale_review_days=self.config.stale_review_days,
                stale_confidence_ceiling=self.config.stale_confidence_ceiling,
            ),
            data_provider=data_provider,
            db_engine=db_engine,
            instrument_map=self.config.instrument_map,
            cross_asset_config=cross_asset_config,
            playbooks=self._playbooks,
        )
        # Per-event position-book state (CL-e6lx extraction) — loads any
        # persisted legs/P&L immediately, so open positions survive an
        # engine restart and the hard time stop still fires after a bounce.
        self.book = EventBook(
            state_path=self.config.event_book_state_path,
            max_loss_pct=self.config.event_book_max_loss_pct,
            per_instrument_max_pct=self.config.per_instrument_max_pct,
            haven_max_pct=self.config.haven_max_pct,
            max_holding_hours=self.config.event_max_holding_hours,
            reconcile_grace_sec=self.config.position_reconcile_grace_sec,
        )
        # Theme-primary scoping (CL-9nvq): matched-theme → tradable
        # instrument-name sets, derived from the playbooks loaded above. When
        # scoping is off (or playbooks failed to load) this is empty, so
        # _theme_primary_instruments returns None → scoping inert (fail-open).
        self._theme_primary: dict[str, frozenset[str]] = (
            _build_theme_primary(self._playbooks) if self.config.theme_primary_only else {}
        )
        # geo_events missing (producer migration not applied) is logged
        # ONCE, not every poll — engine boot must never break or spam.
        self._table_missing_logged = False
        # Lazily-built symbol universe for ticker→company-name enrichment in
        # operator alerts (CL-ikz2). Built once on first use from self.db;
        # None when there's no engine (the alerts just show bare tickers).
        self._symbol_universe: Any = None
        self._symbol_universe_tried = False

    def _load_cross_asset_config(self) -> Any:
        """Load the cross-asset checks config (CL-6mzn), or None on any
        problem — a missing/broken config must never break confirmation,
        it just means alerts carry no cross-asset line."""
        try:
            from src.events.cross_asset import load_cross_asset_config  # noqa: PLC0415

            return load_cross_asset_config(self.config.cross_asset_checks_path)
        except Exception:
            logger.warning(
                "cross-asset checks config unavailable at %s — confirmed-event "
                "alerts will omit the cross-asset corroboration line",
                self.config.cross_asset_checks_path,
                exc_info=True,
            )
            return None

    def _resolve_names(self, assessment: dict[str, Any]) -> dict[str, str]:
        """Ticker→company-name map for an assessment's advisory idea tickers
        (CL-ikz2), so the operator sees "VG (Venture Global, Inc.)" instead of
        a bare ticker. Reads the cached symbol universe (no per-alert SQL).

        Fail-soft BY DESIGN: no engine, no universe, or any lookup error
        returns ``{}`` — the alert then renders bare tickers exactly as
        before. Never allowed to break an alert."""
        if self.db is None:
            return {}
        if not self._symbol_universe_tried:
            self._symbol_universe_tried = True
            try:
                from src.data.symbols import SymbolUniverse  # noqa: PLC0415

                self._symbol_universe = SymbolUniverse(self.db)
            except Exception:
                logger.debug("symbol universe unavailable for name enrichment", exc_info=True)
                self._symbol_universe = None
        if self._symbol_universe is None:
            return {}
        tickers = {
            str(i.get("ticker"))
            for i in (assessment.get("trade_ideas") or [])
            if isinstance(i, dict) and i.get("ticker")
        }
        if not tickers:
            return {}
        try:
            names: dict[str, str] = self._symbol_universe.company_names(tickers)
        except Exception:
            logger.debug("company-name enrichment failed", exc_info=True)
            return {}
        return names

    # ------------------------------------------------------------------
    # Strategy protocol
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def symbols(self) -> list[str]:
        return sorted(set(self.config.instrument_map.values()))

    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds

    # ------------------------------------------------------------------
    # Book delegation (state lives in src/strategies/event_book.py)
    # ------------------------------------------------------------------

    @property
    def open_positions(self) -> dict[str, EventPosition]:
        """The live per-symbol leg book, owned by :class:`EventBook` —
        OPEN legs plus PENDING-EXIT legs, merged per access (CL-8cw1).

        Kept as a plain-dict attribute contract: the portfolio
        reconciler reads ``strategy.open_positions`` directly (CL-8s1e)
        and tests assign it wholesale — both predate the extraction.

        Pending-exit legs are INCLUDED deliberately: a triggered exit
        whose order the broker has not yet confirmed flat is still a
        REAL broker holding. Hiding it would make the reconciler
        classify the leg ``orphaned_broker`` and flatten it a second
        time. Entry logic reads this same merged view, so pending legs
        also keep their concurrency slot occupied (no re-entry until the
        exit confirms). The getter returns a FRESH merged dict — mutate
        ``self.book.open_positions`` / ``self.book.pending_exits``, not
        the returned value."""
        return self.book.held_positions()

    @open_positions.setter
    def open_positions(self, value: dict[str, EventPosition]) -> None:
        # Wholesale assignment replaces the OPEN book only; pending
        # exits keep their own lifecycle (confirm_exits).
        self.book.open_positions = value

    @staticmethod
    def _norm_symbol(sym: str) -> str:
        """Compare-form for position matching — delegates to the shared
        canonical_symbol (CL-qqra) via EventBook so there is ONE
        normalizer repo-wide."""
        return EventBook._norm_symbol(sym)

    def _reconcile_positions(self, broker: Any, now: datetime) -> list[Any] | None:
        """Prune phantom open_positions (CL-v9g4) — see
        :meth:`EventBook.reconcile` for the full rationale. Returns the
        broker position snapshot (or None) for confirm_exits reuse."""
        return self.book.reconcile(broker, now)

    def _concentration_capped_size(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        equity: float,
    ) -> float:
        """Concentration caps (CL-wbmw) — see
        :meth:`EventBook.concentration_capped_size` for the two-layer
        (per-instrument + haven-cluster) semantics."""
        return self.book.concentration_capped_size(symbol, size, entry_price, equity)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _emit_snapshot(self, values: dict[str, Any]) -> dict[str, Any]:
        """Build, persist, and return a snapshot-reference payload (or {})."""
        if self.snapshot_store is None:
            return {}
        snapshot = FeatureSnapshot.create(
            feature_set_name=_FEATURE_SET_NAME,
            feature_set_version=_FEATURE_SET_VERSION,
            data_snapshot_id="live",
            model_version=(
                f"move_frac={self.config.confirm_move_frac},min_urgency={self.config.min_urgency}"
            ),
            ts=datetime.now(UTC),
            values=values,
        )
        try:
            self.snapshot_store.store(snapshot)
        except Exception:
            logger.exception(
                "Failed to store feature snapshot for %s — intent will lack snapshot reference",
                self.id,
            )
            return {}
        return attach_snapshot_payload(snapshot)

    # ------------------------------------------------------------------
    # DB polling
    # ------------------------------------------------------------------

    # BOUNDED ASSESSED scan (CL-9ts9): the old query was an unbounded
    # `WHERE status='ASSESSED' ORDER BY seen_at` that grew with the backlog
    # and never shed a bad/never-confirming row. Now it is windowed
    # (seen_at >= :floor) and LIMITed, ordered FRESHEST-FIRST (seen_at DESC)
    # so the most-recent / most-tradable events are always processed; the
    # count query below detects when the LIMIT elided rows so the cap is
    # logged, never silent. seen_at is a bind param (portable across the
    # pg live path and the sqlite ISO-TEXT test fixtures).
    _POLL_SQL = text(
        "SELECT id, seen_at, source, external_id, headline, url, theme, "
        "assessment, status "
        "FROM geo_events WHERE status = 'ASSESSED' AND seen_at >= :floor "
        "ORDER BY seen_at DESC LIMIT :lim"
    )
    _POLL_COUNT_SQL = text(
        "SELECT count(*) FROM geo_events WHERE status = 'ASSESSED' AND seen_at >= :floor"
    )

    def _fetch_assessed(self) -> list[dict[str, Any]] | None:
        """ASSESSED rows within the poll window, freshest first, capped at
        ``assessed_poll_limit`` (CL-9ts9). None = table unreachable (NO-OP).
        When the cap elides rows it is LOGGED — never silently dropped."""
        if self.db is None:
            if not self._table_missing_logged:
                logger.warning(
                    "EventDrivenStrategy has no DB handle — running as a NO-OP (logged once)",
                )
                self._table_missing_logged = True
            return None
        floor = (
            datetime.now(UTC) - timedelta(hours=self.config.assessed_poll_window_hours)
        ).isoformat()
        limit = self.config.assessed_poll_limit
        params = {"floor": floor, "lim": limit}
        try:
            with self.db.connect() as conn:
                rows = [dict(r) for r in conn.execute(self._POLL_SQL, params).mappings().all()]
                # Only pay for the count when we actually hit the cap — a
                # full page means there MAY be elided rows worth logging.
                total = (
                    conn.execute(
                        self._POLL_COUNT_SQL,
                        {"floor": floor},
                    ).scalar()
                    if len(rows) >= limit
                    else len(rows)
                )
        except Exception as exc:
            if not self._table_missing_logged:
                logger.warning(
                    "geo_events unavailable (%s: %s) — event strategy idles as a "
                    "NO-OP until the producer migration (005_geo_events.sql) is "
                    "applied (logged once)",
                    type(exc).__name__,
                    exc,
                )
                self._table_missing_logged = True
            return None
        if self._table_missing_logged:
            logger.info("geo_events table now reachable — event polling active")
            self._table_missing_logged = False
        if total and total > len(rows):
            logger.warning(
                "ASSESSED poll capped: processing %d of %d in-window rows "
                "(limit=%d, window=%.1fh, freshest first) — %d older row(s) "
                "elided this cycle (CL-9ts9)",
                len(rows),
                total,
                limit,
                self.config.assessed_poll_window_hours,
                total - len(rows),
            )
        return rows

    # ------------------------------------------------------------------
    # Exits: hard stop + hard TIME STOP (bookkeeping in EventBook).
    # Two-phase since CL-8cw1: check_exits returns trigger emissions AND
    # pending re-emissions; finalization happens in confirm_exits.
    # ------------------------------------------------------------------

    def _current_price(self, symbol: str, prices: dict[str, Any], now: datetime) -> float | None:
        price = mid_price(prices.get(symbol))
        if price is not None:
            return price
        if self.data is None:
            return None
        try:
            value = self.data.get_latest_value(symbol, now)
        except Exception as exc:
            # Broad by design: the exit-check price fallback must not break
            # the tick — but a dead data provider is not "no price", so it
            # must not be swallowed silently (CL-gmr1). warning without
            # traceback per CL-2yta.
            logger.warning(
                "%s: price fallback get_latest_value failed for %s: %s: %s",
                self.id,
                symbol,
                type(exc).__name__,
                exc,
            )
            return None
        return float(value) if value is not None else None

    def _check_exits(
        self,
        prices: dict[str, Any],
        now: datetime,
        broker_positions: list[Any] | None = None,
    ) -> list[OrderIntent]:
        """Emit exit intents. ``broker_positions`` is the snapshot
        :meth:`EventBook.reconcile` fetched this tick (None = broker
        unreadable) — a triggering leg captures its symbol's net broker
        quantity for residual/phantom confirmation (CL-9dhg)."""
        exits: list[OrderIntent] = []
        closed = self.book.check_exits(
            lambda symbol: self._current_price(symbol, prices, now),
            now,
            broker_positions,
        )
        for rec in closed:
            pos = rec.position
            # Snapshot only on the FIRST emission (CL-9dhg finding 10):
            # every value in a re-emission derives from the trigger-time
            # capture, so per-tick re-recording would just write a
            # near-identical row per pending exit per tick.
            meta = (
                {}
                if rec.emit_count > 1
                else self._emit_snapshot(
                    {
                        "trigger": "exit",
                        "exit_reason": rec.reason,
                        "symbol": rec.symbol,
                        "event_id": pos.event_id,
                        "entry_price": float(pos.entry_price),
                        "current_price": (
                            float(rec.current_price) if rec.current_price is not None else None
                        ),
                        "direction": int(pos.direction),
                        "pnl": float(rec.pnl),
                        "held_hours": float(rec.held_hours),
                        "book_realized_pnl": float(rec.book_realized_pnl),
                    }
                )
            )
            # Canonical URGENT (CL-ikz2): the legacy "high" string was unknown
            # to the coordinator's rank map, so exits could never escalate.
            exits.append(
                OrderIntent(
                    strategy_id=self.id,
                    symbol=rec.symbol,
                    target_position=0,
                    urgency=Urgency.URGENT.value,
                    max_slippage_bps=self.config.max_slippage_bps,
                    metadata=meta,
                )
            )
        return exits

    def _theme_primary_instruments(self, theme_key: str) -> frozenset[str] | None:
        """The tradable instrument names the event's OWN theme may open
        machine legs on (CL-9nvq), or ``None`` to disable scoping — which is
        FAIL-OPEN (every affected leg passes, the pre-CL-9nvq behavior).

        ``None`` when scoping is off / playbooks failed to load, or the theme
        is empty/unknown (not a configured key). A KNOWN theme with no
        tradable instruments returns an empty frozenset — correctly scoping
        machine legs to nothing (that theme is watch-only)."""
        if not theme_key or not self._theme_primary:
            return None
        return self._theme_primary.get(theme_key)

    # ------------------------------------------------------------------
    # Entries for a newly-CONFIRMED event
    # ------------------------------------------------------------------

    def _enter_confirmed(
        self,
        row: dict[str, Any],
        assessment: dict[str, Any],
        prices: dict[str, Any],
        equity: float | None,
        now: datetime,
        cross_asset: Any = None,
        broker_positions: list[Any] | None = None,
        confirmed_instruments: set[str] | None = None,
    ) -> tuple[
        list[OrderIntent],
        list[tuple[str, str, str, float, str]],
        list[tuple[str, str]],
    ]:
        """Emit entry intents for the tradable affected instruments.

        ``confirmed_instruments`` (CL-tbl8, P0): the set of instrument ids that
        INDIVIDUALLY passed Gate B. Only those are machine-traded — with
        ``min_confirmed_instruments=1`` a single leg's move confirms the whole
        EVENT, but that must not open a full-size leg on every affected
        instrument, including co-legs that showed no post-headline move.
        ``None`` disables the per-leg filter (direct-call/back-compat).

        Returns (intents, entered, skipped): entered is [(symbol,
        direction_str, size_str, entry_price, reason)] with reason from
        ``affected[].reason``; skipped is [(instrument_or_symbol,
        reason)]. Both feed the operator alert."""
        intents: list[OrderIntent] = []
        entered: list[tuple[str, str, str, float, str]] = []
        skipped: list[tuple[str, str]] = []
        event_id = row.get("id")
        headline = str(row.get("headline") or "")

        all_tradables = [
            aff
            for aff in (assessment.get("affected") or [])
            if isinstance(aff, dict)
            and str(aff.get("kind") or "") in TRADABLE_KINDS
            and str(aff.get("direction") or "") in TRADE_DIRECTIONS
        ]
        # Theme-primary scoping (CL-9nvq): a CONFIRMED event may only open
        # machine legs on instruments its OWN theme playbook lists. The impact
        # agent's whitelist admits any instrument known to ANY theme
        # (all_tradable_instruments), so gold/USDJPY/indices leak across
        # themes; here the cross-theme ones are demoted to advisory (recorded
        # in `skipped` so the confirmed alert still lists them) and never
        # auto-trade. FAIL-OPEN: empty/unknown theme or unloaded playbook
        # returns None → no scoping (see _theme_primary_instruments).
        theme_primary = self._theme_primary_instruments(str(row.get("theme") or ""))
        if theme_primary is not None:
            in_theme = []
            for aff in all_tradables:
                if str(aff.get("instrument") or "") in theme_primary:
                    in_theme.append(aff)
                else:
                    skipped.append((str(aff.get("instrument") or ""), "cross_theme"))
            all_tradables = in_theme
        # CL-tbl8: machine-trade ONLY the legs that individually confirmed.
        # Unconfirmed co-legs stay advisory (recorded in `skipped` so the
        # operator alert still lists them).
        if confirmed_instruments is None:
            tradables = all_tradables
        else:
            tradables = []
            for aff in all_tradables:
                if str(aff.get("instrument") or "") in confirmed_instruments:
                    tradables.append(aff)
                else:
                    skipped.append(
                        (str(aff.get("instrument") or ""), "leg_unconfirmed"),
                    )
        if not tradables:
            return intents, entered, skipped

        # Cross-asset ENTRY GATE (CL-6mzn): a contradictory read vetoes the
        # machine legs (fade risk isn't machine-traded); an UNAVAILABLE read
        # only vetoes under block_on_missing — see the config comments.
        # Advisory ideas and the confirmed alert are unaffected either way.
        if self.config.cross_asset_gate_enabled:
            ca_confirmed = getattr(cross_asset, "confirmed", None)
            veto = ca_confirmed is False or (
                ca_confirmed is None and self.config.cross_asset_block_on_missing
            )
            if veto:
                reason = "cross_asset_veto" if ca_confirmed is False else "cross_asset_no_data"
                logger.warning(
                    "cross-asset gate BLOCKED event id=%s entries (%s): "
                    "related assets %s — skipping %d leg(s)",
                    event_id,
                    reason,
                    "contradict the theme" if ca_confirmed is False else "unreadable",
                    len(tradables),
                )
                return (
                    intents,
                    entered,
                    [(str(aff.get("instrument") or ""), reason) for aff in tradables],
                )

        if equity is None or equity <= 0:
            logger.warning(
                "Cannot size event entries (broker account unavailable) — "
                "skipping trades for event id=%s",
                event_id,
            )
            return (
                intents,
                entered,
                [(str(aff.get("instrument") or ""), "no_account") for aff in tradables],
            )

        if self.book.breached(equity):
            return (
                intents,
                entered,
                [(str(aff.get("instrument") or ""), "event_book_loss_cap") for aff in tradables],
            )

        for aff in tradables:
            instrument = str(aff.get("instrument") or "")
            dir_str = str(aff.get("direction"))
            symbol = self.config.instrument_map.get(instrument)
            if symbol is None:
                logger.warning(
                    "Event instrument %r not in instrument_map — skipping "
                    "(event id=%s). Add a mapping to trade it.",
                    instrument,
                    event_id,
                )
                skipped.append((instrument, "unknown_instrument"))
                continue
            if symbol in self.book.pending_exits:
                # CL-8cw1: exit triggered but not broker-confirmed — the
                # slot is still occupied by REAL broker risk, and a fresh
                # entry would race the in-flight/retrying close order.
                logger.info(
                    "Event entry %s skipped — exit pending broker confirmation (event id=%s)",
                    symbol,
                    event_id,
                )
                skipped.append((symbol, "pending_exit"))
                continue
            if symbol in self.book.pending_entries:
                # ENTRY half of CL-hqyj: an entry for this symbol is already
                # submitted and awaiting its fill — a second submit would
                # race it and double the position. The slot is committed.
                logger.info(
                    "Event entry %s skipped — entry pending broker fill confirmation (event id=%s)",
                    symbol,
                    event_id,
                )
                skipped.append((symbol, "pending_entry"))
                continue
            # SLOT/CAP accounting counts pending entries at their intended
            # magnitude (accounting_positions), so a submitted-but-unfilled
            # leg still occupies a slot and can't be over-submitted past the
            # cap while its fill is in flight (ENTRY half of CL-hqyj).
            accounting = self.book.accounting_positions()
            if len(accounting) >= self.config.max_concurrent_event_positions:
                logger.warning(
                    "max_concurrent_event_positions=%d reached — skipping %s (event id=%s)",
                    self.config.max_concurrent_event_positions,
                    symbol,
                    event_id,
                )
                skipped.append((symbol, "max_concurrent"))
                continue
            if symbol in accounting:
                skipped.append((symbol, "already_open"))
                continue

            direction = 1 if dir_str == "long" else -1
            tick = prices.get(symbol)
            if isinstance(tick, dict) and tick.get("bid") is not None:
                entry_price = float(tick["ask"] if direction > 0 else tick["bid"])
            else:
                fallback = self._current_price(symbol, prices, now)
                if fallback is None:
                    logger.warning(
                        "No price for %s — skipping event entry (event id=%s)",
                        symbol,
                        event_id,
                    )
                    skipped.append((symbol, "no_price"))
                    continue
                entry_price = fallback

            stop_price = entry_price * (1 - direction * self.config.event_stop_pct)
            stop_distance = abs(entry_price - stop_price)
            size = equity * self.config.event_risk_pct / max(stop_distance, 1e-9) * direction

            # Concentration caps (CL-wbmw): trim to fit the per-instrument
            # (and, for havens, cluster) headroom, or skip at 0 — see
            # EventBook.concentration_capped_size. Additive to the loss cap.
            capped = self.book.concentration_capped_size(
                symbol,
                size,
                entry_price,
                equity,
            )
            if capped == 0.0:
                skipped.append((symbol, "concentration_cap"))
                continue
            size = capped

            # CL-y412: liquidity-window gate (NEW entries only — the exit
            # manager above runs unconditionally). A wide spread in a dead
            # window trims the entry (0.5×) or blocks it (0.0×). Skipped when
            # the tick has no two-sided quote — we never widen an entry on a
            # guessed spread, only shrink it on a measured one.
            if self._liquidity_profile is not None:
                spread_bps = spread_bps_from_tick(tick)
                if spread_bps is not None:
                    liq_size = PositionSizer.adjust_for_liquidity(
                        size,
                        symbol,
                        now,
                        spread_bps,
                        self._liquidity_profile,
                    )
                    if liq_size == 0.0:
                        logger.info(
                            "Event entry %s skipped — dead liquidity window "
                            "(spread=%.1fbps, event id=%s)",
                            symbol,
                            spread_bps,
                            event_id,
                        )
                        skipped.append((symbol, "liquidity_window"))
                        continue
                    if liq_size != size:
                        logger.info(
                            "Event entry %s trimmed for thin liquidity "
                            "(spread=%.1fbps): %.0f -> %.0f",
                            symbol,
                            spread_bps,
                            size,
                            liq_size,
                        )
                    size = liq_size

            # Park the leg in pending_entries at its INTENDED size and
            # capture the submit-time broker baseline (ENTRY half of
            # CL-hqyj): confirm_entries promotes it into open_positions at
            # the ACTUAL fill next tick, so the book never records the
            # intended phantom the reconciler would flag.
            self.book.record_entry(
                EventPosition(
                    symbol=symbol,
                    event_id=event_id,
                    entry_ts=now,
                    entry_price=entry_price,
                    quantity=size,
                    direction=direction,
                    stop_price=stop_price,
                    headline=headline[:200],
                ),
                broker_positions,
                now,
            )
            logger.info(
                "Event entry %s %s: size=%.0f entry=%.5f stop=%.5f event_id=%s headline=%r",
                symbol,
                dir_str,
                size,
                entry_price,
                stop_price,
                event_id,
                headline[:80],
            )
            meta = self._emit_snapshot(
                {
                    "trigger": "entry",
                    "event_id": event_id,
                    "headline": headline[:200],
                    "instrument": instrument,
                    "symbol": symbol,
                    "direction": int(direction),
                    "urgency": int(assessment.get("urgency") or 0),
                    "confidence": float(assessment.get("confidence") or 0.0),
                    "entry_price": float(entry_price),
                    "stop_price": float(stop_price),
                    "size": float(size),
                    "reason": str(aff.get("reason") or ""),
                }
            )
            intents.append(
                OrderIntent(
                    strategy_id=self.id,
                    symbol=symbol,
                    target_position=size,
                    urgency=Urgency.URGENT.value,
                    max_slippage_bps=self.config.max_slippage_bps,
                    metadata=meta,
                )
            )
            entered.append(
                (
                    symbol,
                    dir_str,
                    f"{size:.0f}",
                    entry_price,
                    str(aff.get("reason") or ""),
                )
            )
        return intents, entered, skipped

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @staticmethod
    def _get_equity(broker: Any) -> float | None:
        try:
            return float(broker.get_account().equity)
        except Exception as exc:
            # Broad by design: broker hiccups must not break the tick, but
            # include the actual error for diagnosis (CL-gmr1).
            logger.warning(
                "broker.get_account() failed (%s: %s) — cannot size event entries",
                type(exc).__name__,
                exc,
            )
            return None

    @staticmethod
    def _fetch_entry_baseline(broker: Any) -> list[Any] | None:
        """One broker position snapshot for the entry submit-time baseline
        (ENTRY half of CL-hqyj), used only when reconcile() returned None
        (all books empty). Fail-safe: an unreadable broker yields None, and
        record_entry then falls back to its best-effort baseline-None
        promote/reject path — never crash the tick, never guess a baseline."""
        if not hasattr(broker, "get_positions"):
            return None
        try:
            return list(broker.get_positions())
        except Exception:
            logger.debug(
                "event_driven: broker positions unavailable for entry "
                "baseline — record_entry will fall back to baseline-None",
                exc_info=True,
            )
            return None

    async def generate_intents(
        self,
        prices: dict[str, Any],
        broker: Any,
    ) -> list[OrderIntent]:
        now = datetime.now(UTC)
        # Prune phantom positions from earlier rejected orders BEFORE the cap
        # check, so a rejected leg can't keep blocking real entries (CL-v9g4).
        # reconcile() hands back the broker snapshot it already fetched, and
        # confirm_exits reuses it (CL-8cw1) — ONE broker positions call per
        # tick. Confirmation runs BEFORE check_exits so a just-confirmed exit
        # is not re-emitted on the same tick; broker-unreadable (None) means
        # confirm nothing and keep retrying — never finalize blind.
        broker_positions = self.book.reconcile(broker, now)
        if broker_positions is not None:
            # confirm_entries runs FIRST, from the freshest snapshot, so a
            # just-filled entry promotes into open_positions THIS tick and
            # is immediately eligible for stop/time-stop evaluation below —
            # and the book's reconciler-facing size converges to the broker
            # size within one tick (ENTRY half of CL-hqyj). Same-tick
            # promotion is also what keeps the submit→confirm window from
            # accumulating an alignment-mismatch streak: the 300s alignment
            # timer would need TWO consecutive mismatched checks
            # (_ALIGNMENT_MISMATCH_STREAK_TO_FLAG=2) to flag, and promotion
            # closes the window inside one strategy tick.
            self.book.confirm_entries(broker_positions, now)
            self.book.confirm_exits(broker_positions)
        # The same snapshot feeds check_exits so a leg triggering THIS
        # tick captures trigger_broker_qty for residual/phantom
        # confirmation (CL-9dhg findings 1 + 2).
        intents = self._check_exits(prices, now, broker_positions)

        poll_started = time.monotonic()
        rows = self._fetch_assessed()
        if rows is None:
            return intents  # geo_events unreachable — NO-OP (logged once)

        # Batch the Gate-B current-mid / daily-vol reads for the WHOLE
        # instrument set of this tick ONCE (CL-9ts9 / CL-8s2a) — one
        # connection, one query per metric — instead of a fresh connection
        # per event × per instrument. Single-cycle only: a fresh cache each
        # poll, so no cross-tick staleness. Byte-identical verdicts: the
        # cache serves the same value the per-call read would have.
        poll_cache = self.confluence.build_poll_cache(rows, now)

        equity = self._get_equity(broker) if rows else None
        # ENTRY half of CL-hqyj: record_entry captures the submit-time
        # broker baseline (entry_broker_qty) from a snapshot so
        # confirm_entries can measure OUR fill as a delta and stay co-held-
        # safe. reconcile() returns None when all books were empty (the
        # common case on the very first entry), so fetch ONE snapshot here
        # if we have events to trade and don't already hold one — never a
        # baseline of None when the broker is actually readable.
        if rows and broker_positions is None:
            broker_positions = self._fetch_entry_baseline(broker)
        expired_alerts_sent = 0

        for row in rows:
            try:
                result = self.confluence.evaluate_and_transition(
                    row,
                    prices=prices,
                    now=now,
                    poll_cache=poll_cache,
                )
            except Exception:
                logger.exception(
                    "Confluence evaluation failed for geo_event id=%s — skipping",
                    row.get("id"),
                )
                continue

            if result.outcome == "expired":
                # Big events the operator should hear about even without
                # confirmation — capped at 1 alert per run to avoid a
                # backlog flood.
                if (
                    result.transitioned
                    and result.urgency >= self.config.expired_alert_min_urgency
                    and expired_alerts_sent < 1
                ):
                    exp_assessment = EventConfluence.parse_assessment(row.get("assessment")) or {}
                    self.notifier.alert_expired(
                        row,
                        result.urgency,
                        result.confidence,
                        names=self._resolve_names(exp_assessment),
                    )
                    expired_alerts_sent += 1
                continue

            if result.outcome != "confirmed" or not result.transitioned:
                continue  # pending, or another writer won the transition

            assessment = EventConfluence.parse_assessment(row.get("assessment")) or {}
            # CL-tbl8 (P0): only the instruments that INDIVIDUALLY passed
            # Gate B are machine-traded — not every affected leg just because
            # the event-level confirm gate (min_confirmed_instruments) tripped.
            confirmed_instruments = {str(c.instrument) for c in result.checks if c.confirmed}
            entry_intents, entered, skipped = self._enter_confirmed(
                row,
                assessment,
                prices,
                equity,
                now,
                cross_asset=result.cross_asset,
                broker_positions=broker_positions,
                confirmed_instruments=confirmed_instruments,
            )
            # Stamp the cross-asset read onto the persisted trade ideas'
            # notes so `idea <id>` surfaces it later (CL-6mzn). Additive,
            # best-effort — never blocks the alert or the trades.
            self.notifier.stamp_cross_asset_on_ideas(row.get("id"), result.cross_asset)
            # Alert on every CONFIRMED event — even when caps/mapping
            # meant nothing was tradable (operator can act manually).
            self.notifier.alert_confirmed(
                row,
                assessment,
                entered,
                skipped,
                prices=prices,
                now=now,
                cross_asset=result.cross_asset,
                names=self._resolve_names(assessment),
            )
            if entry_intents:
                intents.extend(entry_intents)
                self.confluence.transition(row.get("id"), "CONFIRMED", "TRADED")
            # else: row stays CONFIRMED — visible in the table as
            # "confirmed but not traded" (caps / unknown instruments).

        # One-line poll-cycle timing (CL-3lga): events scanned, instruments
        # batched, batch DB queries issued, wall-ms. Makes the batch win
        # (queries flat as the event set grows) measurable in the logs.
        logger.info(
            "event poll cycle: events=%d instruments=%d batch_queries=%d wall_ms=%.1f",
            len(rows),
            poll_cache.instruments_looked_up,
            poll_cache.batch_queries,
            (time.monotonic() - poll_started) * 1000.0,
        )

        return intents
