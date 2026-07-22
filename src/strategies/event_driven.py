"""Event-driven strategy (CL-mnhw) — paper-trades CONFIRMED current events.

Consumer half of the current-events pipeline. Every poll it:

  1. Emits exit intents for open event positions that hit the hard stop
     or the hard TIME STOP (``event_max_holding_hours``, fires
     regardless of P&L — event edges decay in hours, not days).
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
CRITICAL — exits always still flow.

Boot safety: if the ``geo_events`` table doesn't exist yet (producer
migration not applied), the strategy is a NO-OP that logs ONCE — the
engine must never fail to boot because the sibling half hasn't landed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
from src.execution.oms import OrderIntent, Urgency
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    attach_snapshot_payload,
)

logger = logging.getLogger(__name__)

_FEATURE_SET_NAME = "event_driven"
_FEATURE_SET_VERSION = "v1"
_STATE_VERSION = 1

#: Pure safe-haven OANDA instruments (CL-5mkf). Combined open notional
#: across these is capped at ``haven_max_pct`` of equity — a tighter
#: CLUSTER cap on top of the per-instrument cap (CL-wbmw), because a
#: per-name limit can't see correlated metals as one basket.
HAVEN_INSTRUMENTS = frozenset({"XAU_USD", "XAG_USD"})


def _default_instrument_map() -> dict[str, str]:
    """Assessment instrument id → OANDA broker instrument. Covers every
    tradable id in configs/event_playbooks.yaml plus compact aliases an
    LLM might emit; anything NOT in the map is skipped with a warning at
    trade time — never traded blind, never a crash."""
    return {
        # OANDA-native ids pass through unchanged.
        "EUR_USD": "EUR_USD", "GBP_USD": "GBP_USD", "USD_JPY": "USD_JPY",
        "USD_CAD": "USD_CAD", "USD_CHF": "USD_CHF", "AUD_USD": "AUD_USD",
        "NZD_USD": "NZD_USD", "USD_MXN": "USD_MXN", "USD_NOK": "USD_NOK",
        "USD_SEK": "USD_SEK", "USD_CNH": "USD_CNH", "EUR_GBP": "EUR_GBP",
        "EUR_JPY": "EUR_JPY", "XAU_USD": "XAU_USD", "XAG_USD": "XAG_USD",
        "BCO_USD": "BCO_USD", "WTICO_USD": "WTICO_USD",
        "NATGAS_USD": "NATGAS_USD", "SPX500_USD": "SPX500_USD",
        "NAS100_USD": "NAS100_USD",
        # Compact aliases → OANDA ids.
        "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY",
        "USDCAD": "USD_CAD", "USDCHF": "USD_CHF", "AUDUSD": "AUD_USD",
        "NZDUSD": "NZD_USD", "XAUUSD": "XAU_USD", "XAGUSD": "XAG_USD",
        "GOLD": "XAU_USD", "BRENT": "BCO_USD", "WTI": "WTICO_USD",
    }


@dataclass
class EventDrivenConfig:
    # ---- Confirmation (mirrors ConfluenceConfig; see src/events/confluence.py)
    min_urgency: int = 7
    min_confidence: float = 0.75
    confirm_window_min_minutes: int = 30
    confirm_window_max_minutes: int = 120
    confirm_move_frac: float = 0.25
    min_confirmed_instruments: int = 1
    realized_vol_window: int = 20
    vol_spike_check_enabled: bool = False
    vol_spike_window: int = 5
    vol_spike_ratio: float = 1.5
    # ---- Trading -----------------------------------------------------
    # Risk per event trade as a fraction of equity. 0.005 = 50bps —
    # deliberately half the CB-sentiment risk; event assessments are the
    # least-proven signal in the book.
    event_risk_pct: float = 0.005
    # Hard stop distance from entry (fraction of entry price).
    event_stop_pct: float = 0.01
    # Max simultaneous event positions across ALL events.
    max_concurrent_event_positions: int = 2
    # Phantom-position reconciliation grace window (CL-v9g4): open_positions
    # entries OLDER than this that the broker doesn't hold are pruned each
    # cycle — see _reconcile_positions for the full rationale.
    position_reconcile_grace_sec: int = 120
    # Hard TIME STOP: exit after this many hours regardless of P&L.
    event_max_holding_hours: float = 4.0
    # Per-instrument concentration cap (CL-wbmw): combined open notional in
    # ANY single event instrument as a fraction of equity. A CEILING that
    # catches ACCUMULATION, sized just ABOVE one base leg (base notional =
    # event_risk_pct / event_stop_pct = ~50% of equity, only 0.5% RISK at
    # the 1% stop): the first leg in a name passes intact, a SECOND leg in
    # the same instrument is trimmed/skipped. 0.55 = one base leg + buffer.
    per_instrument_max_pct: float = 0.55
    # Combined open notional across HAVEN_INSTRUMENTS (gold/silver) as a
    # fraction of equity (CL-5mkf) — the tighter CLUSTER cap correlated
    # metals need (per-name 0.55 alone would allow 1.10 of havens). Allows
    # one full haven leg (~0.50) plus a small second, never two full metal
    # legs of stacked gap risk. 0.60 = ~1.2 base legs of combined metals.
    haven_max_pct: float = 0.60
    # ---- Event-book protection ----------------------------------------
    # Cumulative realized loss (fraction of equity) that freezes NEW
    # event entries. Exits always still flow.
    event_book_max_loss_pct: float = 0.02
    event_book_state_path: str = "data/event_book_state.json"
    # ---- Cross-asset corroboration (CL-6mzn) ---------------------------
    # Path to the per-theme corroborating-instruments config. The read is
    # a DISPLAY annotation on confirmed events, and — when the gate below
    # is enabled — an ENTRY gate on the machine legs.
    cross_asset_checks_path: str = "configs/cross_asset_checks.yaml"
    # ENTRY GATE (CL-6mzn, gating half): when True, a confirmed event whose
    # cross-asset read POSITIVELY FAILS (confirmed is False — contradictory)
    # has its machine legs SKIPPED (reason 'cross_asset_veto'). The event
    # still confirms/alerts and advisory ideas still flow — only the
    # auto-traded legs are blocked; fade risk isn't machine-traded.
    cross_asset_gate_enabled: bool = False
    # Stricter mode: ALSO veto when the cross-asset read is UNAVAILABLE
    # (confirmed is None — instruments unmapped / no price data). Default
    # False on purpose: this repo's history shows silent data gaps already
    # zeroed the book once (CL-5lpp/CL-gr8o); fail-closed-on-missing would
    # recreate that failure mode. Enable once cross-asset data coverage is
    # proven complete.
    cross_asset_block_on_missing: bool = False
    # ---- Alerts --------------------------------------------------------
    # EXPIRED events at/above this urgency get a brief info alert.
    expired_alert_min_urgency: int = 8
    # ---- Plumbing ------------------------------------------------------
    # Event entries chase news moves — allow more slippage than the
    # 2bps default before refusing a fill.
    max_slippage_bps: float = 10.0
    signal_interval_seconds: int = 300
    instrument_map: dict[str, str] = field(default_factory=_default_instrument_map)
    id: str = "event_driven"


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


class EventDrivenStrategy:
    def __init__(
        self,
        config: EventDrivenConfig | None = None,
        data_provider: Any = None,
        state_store: Any = None,
        snapshot_store: FeatureSnapshotStore | None = None,
        db_engine: Any = None,
        notifier: EventNotifier | None = None,
    ) -> None:
        self.config = config or EventDrivenConfig()
        self.data = data_provider
        self.state = state_store
        self.snapshot_store = snapshot_store
        self.db = db_engine
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
            ),
            data_provider=data_provider,
            db_engine=db_engine,
            instrument_map=self.config.instrument_map,
            cross_asset_config=cross_asset_config,
        )
        self.open_positions: dict[str, EventPosition] = {}
        self._realized_pnl: float = 0.0
        self._closed_trades: int = 0
        # geo_events missing (producer migration not applied) is logged
        # ONCE, not every poll — engine boot must never break or spam.
        self._table_missing_logged = False
        # Loss-cap breach is CRITICAL once per activation, WARNING after.
        self._breach_logged = False
        self._load_state()

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

    def fit(self, train_data: Any) -> None:
        pass

    def generate_signals(self, data: Any) -> None:
        return None

    # ------------------------------------------------------------------
    # Event-book state file (atomic tmp+rename, like equity trailing stop)
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return Path(self.config.event_book_state_path)

    def _load_state(self) -> None:
        path = self._state_path()
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
        self._realized_pnl = float(payload.get("realized_pnl", 0.0))
        self._closed_trades = int(payload.get("closed_trades", 0))
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

    def _save_state(self) -> None:
        path = self._state_path()
        payload = {
            "version": _STATE_VERSION,
            "realized_pnl": self._realized_pnl,
            "closed_trades": self._closed_trades,
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
                f"move_frac={self.config.confirm_move_frac},"
                f"min_urgency={self.config.min_urgency}"
            ),
            ts=datetime.now(UTC),
            values=values,
        )
        try:
            self.snapshot_store.store(snapshot)
        except Exception:
            logger.exception(
                "Failed to store feature snapshot for %s — intent will lack "
                "snapshot reference",
                self.id,
            )
            return {}
        return attach_snapshot_payload(snapshot)

    # ------------------------------------------------------------------
    # DB polling
    # ------------------------------------------------------------------

    _POLL_SQL = text(
        "SELECT id, seen_at, source, external_id, headline, url, theme, "
        "assessment, status "
        "FROM geo_events WHERE status = 'ASSESSED' ORDER BY seen_at"
    )

    def _fetch_assessed(self) -> list[dict[str, Any]] | None:
        """ASSESSED rows, oldest first. None = table unreachable (NO-OP)."""
        if self.db is None:
            if not self._table_missing_logged:
                logger.warning(
                    "EventDrivenStrategy has no DB handle — running as a NO-OP "
                    "(logged once)",
                )
                self._table_missing_logged = True
            return None
        try:
            with self.db.connect() as conn:
                rows = [dict(r) for r in conn.execute(self._POLL_SQL).mappings().all()]
        except Exception as exc:
            if not self._table_missing_logged:
                logger.warning(
                    "geo_events unavailable (%s: %s) — event strategy idles as a "
                    "NO-OP until the producer migration (005_geo_events.sql) is "
                    "applied (logged once)",
                    type(exc).__name__, exc,
                )
                self._table_missing_logged = True
            return None
        if self._table_missing_logged:
            logger.info("geo_events table now reachable — event polling active")
            self._table_missing_logged = False
        return rows

    # ------------------------------------------------------------------
    # Exits: hard stop + hard TIME STOP
    # ------------------------------------------------------------------

    def _current_price(self, symbol: str, prices: dict[str, Any], now: datetime) -> float | None:
        price = mid_price(prices.get(symbol))
        if price is not None:
            return price
        if self.data is None:
            return None
        try:
            value = self.data.get_latest_value(symbol, now)
        except Exception:
            return None
        return float(value) if value is not None else None

    @staticmethod
    def _norm_symbol(sym: str) -> str:
        """Compare-form for position matching — delegates to the shared
        canonical_symbol (CL-qqra) so there is ONE normalizer repo-wide."""
        from src.execution.broker import canonical_symbol  # noqa: PLC0415
        return canonical_symbol(sym)

    def _reconcile_positions(self, broker: Any, now: datetime) -> None:
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
        grace = timedelta(seconds=self.config.position_reconcile_grace_sec)
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
            self._save_state()

    def _check_exits(self, prices: dict[str, Any], now: datetime) -> list[OrderIntent]:
        exits: list[OrderIntent] = []
        for symbol, pos in list(self.open_positions.items()):
            current = self._current_price(symbol, prices, now)
            exit_reason = None
            if current is not None and (
                (pos.direction > 0 and current <= pos.stop_price)
                or (pos.direction < 0 and current >= pos.stop_price)
            ):
                exit_reason = "hard_stop"
            held_hours = (now - pos.entry_ts).total_seconds() / 3600.0
            if held_hours >= self.config.event_max_holding_hours:
                # TIME STOP fires regardless of P&L or even a current price.
                exit_reason = "time_stop"
            if exit_reason is None:
                continue
            pnl = (current - pos.entry_price) * pos.quantity if current is not None else 0.0
            self._realized_pnl += pnl
            self._closed_trades += 1
            del self.open_positions[symbol]
            self._save_state()
            logger.info(
                "Event exit %s: %s pnl=%.2f held=%.1fh event_id=%s",
                symbol, exit_reason, pnl, held_hours, pos.event_id,
            )
            meta = self._emit_snapshot({
                "trigger": "exit",
                "exit_reason": exit_reason,
                "symbol": symbol,
                "event_id": pos.event_id,
                "entry_price": float(pos.entry_price),
                "current_price": float(current) if current is not None else None,
                "direction": int(pos.direction),
                "pnl": float(pnl),
                "held_hours": float(held_hours),
                "book_realized_pnl": float(self._realized_pnl),
            })
            # Canonical URGENT (CL-ikz2): the legacy "high" string was unknown
            # to the coordinator's rank map, so exits could never escalate.
            exits.append(OrderIntent(
                strategy_id=self.id, symbol=symbol, target_position=0,
                urgency=Urgency.URGENT.value,
                max_slippage_bps=self.config.max_slippage_bps,
                metadata=meta,
            ))
        return exits

    # ------------------------------------------------------------------
    # Event-book protection
    # ------------------------------------------------------------------

    def _book_breached(self, equity: float | None) -> bool:
        if equity is None or equity <= 0:
            return False
        cap = self.config.event_book_max_loss_pct * equity
        breached = -self._realized_pnl >= cap
        if breached:
            if not self._breach_logged:
                logger.critical(
                    "EVENT BOOK LOSS CAP BREACHED: cumulative realized P&L "
                    "%.2f <= -%.2f (%.1f%% of equity %.0f). NO new event "
                    "positions will be opened; exits still flow. Reset "
                    "requires operator action on %s. (Kill-switch "
                    "integration pending — this is the loud log.)",
                    self._realized_pnl, cap,
                    self.config.event_book_max_loss_pct * 100, equity,
                    self.config.event_book_state_path,
                )
                self._breach_logged = True
            else:
                logger.warning(
                    "Event book loss cap still breached (realized P&L %.2f) — "
                    "new entries blocked", self._realized_pnl,
                )
        elif self._breach_logged:
            logger.warning(
                "Event book back under the loss cap — new entries re-enabled",
            )
            self._breach_logged = False
        return breached

    # ------------------------------------------------------------------
    # Concentration caps (CL-wbmw generalizes CL-5mkf) — semantics in
    # _concentration_capped_size's docstring. Additive to event_risk_pct
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

    def _concentration_capped_size(
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
        per_cap = self.config.per_instrument_max_pct * equity
        per_open = self._open_instrument_notional(symbol)
        headroom = per_cap - per_open
        binding = "per-instrument"
        cap_pct = self.config.per_instrument_max_pct
        open_exposure = per_open
        cap_notional = per_cap

        # Layer 2: haven-cluster headroom (havens only) — take the tighter.
        if symbol in HAVEN_INSTRUMENTS:
            haven_cap = self.config.haven_max_pct * equity
            haven_open = self._open_haven_notional()
            haven_headroom = haven_cap - haven_open
            if haven_headroom < headroom:
                headroom = haven_headroom
                cluster = "/".join(sorted(HAVEN_INSTRUMENTS))
                binding = f"haven-cluster ({cluster})"
                cap_pct = self.config.haven_max_pct
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
    ) -> tuple[
        list[OrderIntent],
        list[tuple[str, str, str, float, str]],
        list[tuple[str, str]],
    ]:
        """Emit entry intents for the tradable affected instruments.

        Returns (intents, entered, skipped): entered is [(symbol,
        direction_str, size_str, entry_price, reason)] with reason from
        ``affected[].reason``; skipped is [(instrument_or_symbol,
        reason)]. Both feed the operator alert."""
        intents: list[OrderIntent] = []
        entered: list[tuple[str, str, str, float, str]] = []
        skipped: list[tuple[str, str]] = []
        event_id = row.get("id")
        headline = str(row.get("headline") or "")

        tradables = [
            aff for aff in (assessment.get("affected") or [])
            if isinstance(aff, dict)
            and str(aff.get("kind") or "") in TRADABLE_KINDS
            and str(aff.get("direction") or "") in TRADE_DIRECTIONS
        ]
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
                reason = (
                    "cross_asset_veto"
                    if ca_confirmed is False else "cross_asset_no_data"
                )
                logger.warning(
                    "cross-asset gate BLOCKED event id=%s entries (%s): "
                    "related assets %s — skipping %d leg(s)",
                    event_id, reason,
                    "contradict the theme" if ca_confirmed is False
                    else "unreadable",
                    len(tradables),
                )
                return intents, entered, [
                    (str(aff.get("instrument") or ""), reason)
                    for aff in tradables
                ]

        if equity is None or equity <= 0:
            logger.warning(
                "Cannot size event entries (broker account unavailable) — "
                "skipping trades for event id=%s", event_id,
            )
            return intents, entered, [
                (str(aff.get("instrument") or ""), "no_account") for aff in tradables
            ]

        if self._book_breached(equity):
            return intents, entered, [
                (str(aff.get("instrument") or ""), "event_book_loss_cap")
                for aff in tradables
            ]

        for aff in tradables:
            instrument = str(aff.get("instrument") or "")
            dir_str = str(aff.get("direction"))
            symbol = self.config.instrument_map.get(instrument)
            if symbol is None:
                logger.warning(
                    "Event instrument %r not in instrument_map — skipping "
                    "(event id=%s). Add a mapping to trade it.",
                    instrument, event_id,
                )
                skipped.append((instrument, "unknown_instrument"))
                continue
            if len(self.open_positions) >= self.config.max_concurrent_event_positions:
                logger.warning(
                    "max_concurrent_event_positions=%d reached — skipping %s "
                    "(event id=%s)",
                    self.config.max_concurrent_event_positions, symbol, event_id,
                )
                skipped.append((symbol, "max_concurrent"))
                continue
            if symbol in self.open_positions:
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
                        symbol, event_id,
                    )
                    skipped.append((symbol, "no_price"))
                    continue
                entry_price = fallback

            stop_price = entry_price * (1 - direction * self.config.event_stop_pct)
            stop_distance = abs(entry_price - stop_price)
            size = equity * self.config.event_risk_pct / max(stop_distance, 1e-9) * direction

            # Concentration caps (CL-wbmw): trim to fit the per-instrument
            # (and, for havens, cluster) headroom, or skip at 0 — see
            # _concentration_capped_size. Additive to the loss cap above.
            capped = self._concentration_capped_size(
                symbol, size, entry_price, equity,
            )
            if capped == 0.0:
                skipped.append((symbol, "concentration_cap"))
                continue
            size = capped

            self.open_positions[symbol] = EventPosition(
                symbol=symbol, event_id=event_id, entry_ts=now,
                entry_price=entry_price, quantity=size, direction=direction,
                stop_price=stop_price, headline=headline[:200],
            )
            self._save_state()
            logger.info(
                "Event entry %s %s: size=%.0f entry=%.5f stop=%.5f event_id=%s "
                "headline=%r",
                symbol, dir_str, size, entry_price, stop_price, event_id,
                headline[:80],
            )
            meta = self._emit_snapshot({
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
            })
            intents.append(OrderIntent(
                strategy_id=self.id, symbol=symbol, target_position=size,
                urgency=Urgency.URGENT.value,
                max_slippage_bps=self.config.max_slippage_bps,
                metadata=meta,
            ))
            entered.append((
                symbol, dir_str, f"{size:.0f}", entry_price,
                str(aff.get("reason") or ""),
            ))
        return intents, entered, skipped

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @staticmethod
    def _get_equity(broker: Any) -> float | None:
        try:
            return float(broker.get_account().equity)
        except Exception:
            logger.warning("broker.get_account() failed — cannot size event entries")
            return None

    async def generate_intents(
        self, prices: dict[str, Any], broker: Any,
    ) -> list[OrderIntent]:
        now = datetime.now(UTC)
        # Prune phantom positions from earlier rejected orders BEFORE the cap
        # check, so a rejected leg can't keep blocking real entries (CL-v9g4).
        self._reconcile_positions(broker, now)
        intents = self._check_exits(prices, now)

        rows = self._fetch_assessed()
        if rows is None:
            return intents  # geo_events unreachable — NO-OP (logged once)

        equity = self._get_equity(broker) if rows else None
        expired_alerts_sent = 0

        for row in rows:
            try:
                result = self.confluence.evaluate_and_transition(
                    row, prices=prices, now=now,
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
                    self.notifier.alert_expired(row, result.urgency, result.confidence)
                    expired_alerts_sent += 1
                continue

            if result.outcome != "confirmed" or not result.transitioned:
                continue  # pending, or another writer won the transition

            assessment = EventConfluence.parse_assessment(row.get("assessment")) or {}
            entry_intents, entered, skipped = self._enter_confirmed(
                row, assessment, prices, equity, now,
                cross_asset=result.cross_asset,
            )
            # Stamp the cross-asset read onto the persisted trade ideas'
            # notes so `idea <id>` surfaces it later (CL-6mzn). Additive,
            # best-effort — never blocks the alert or the trades.
            self.notifier.stamp_cross_asset_on_ideas(row.get("id"), result.cross_asset)
            # Alert on every CONFIRMED event — even when caps/mapping
            # meant nothing was tradable (operator can act manually).
            self.notifier.alert_confirmed(
                row, assessment, entered, skipped, prices=prices, now=now,
                cross_asset=result.cross_asset,
            )
            if entry_intents:
                intents.extend(entry_intents)
                self.confluence.transition(row.get("id"), "CONFIRMED", "TRADED")
            # else: row stays CONFIRMED — visible in the table as
            # "confirmed but not traded" (caps / unknown instruments).

        return intents
