"""Event-driven strategy (CL-mnhw) — paper-trades CONFIRMED current events.

Consumer half of the current-events pipeline. Every poll it:

  1. Emits exit intents for open event positions that hit the hard stop
     or the hard TIME STOP (``event_max_holding_hours``, fires
     regardless of P&L — event edges decay in hours, not days).
  2. Polls ``geo_events`` for ASSESSED rows and runs them through
     :class:`src.events.confluence.EventConfluence` (Gate A quality +
     Gate B market confirmation).
  3. For events it newly CONFIRMED: alerts the operator (Telegram /
     Telegram via ``notify_operator``) and emits tightly-risked
     OrderIntents for the tradable affected instruments, then marks the
     row TRADED. Sizing: ``equity * event_risk_pct / stop_distance``
     with the stop ``event_stop_pct`` from entry.
  4. EXPIRED events with urgency >= ``expired_alert_min_urgency`` get a
     brief "expired unconfirmed" info alert (capped at one per run).

Event-book protection: cumulative realized P&L of event trades persists
in ``data/event_book_state.json`` (atomic tmp+rename, same pattern as
the equity trailing stop). Breaching ``event_book_max_loss_pct`` of
equity blocks NEW entries and logs CRITICAL — exits always still flow.
Kill-switch integration can come later; for now this logs loudly.

Boot safety: if the ``geo_events`` table doesn't exist yet (producer
migration not applied), the strategy is a NO-OP that logs ONCE — the
engine must never fail to boot because the sibling half hasn't landed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from src.events.prices import format_age
from src.execution.oms import OrderIntent
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    attach_snapshot_payload,
)
from src.research.notifications import notify_operator

logger = logging.getLogger(__name__)

_FEATURE_SET_NAME = "event_driven"
_FEATURE_SET_VERSION = "v1"
_STATE_VERSION = 1


def _default_instrument_map() -> dict[str, str]:
    """Assessment instrument id → OANDA broker instrument.

    Covers every tradable id in configs/event_playbooks.yaml (OANDA ids
    pass through) plus compact aliases an LLM might emit. Anything NOT
    in this map is skipped with a warning at trade time — never traded
    blind, never a crash.
    """
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
    # Hard TIME STOP: exit after this many hours regardless of P&L.
    event_max_holding_hours: float = 4.0
    # ---- Event-book protection ----------------------------------------
    # Cumulative realized loss (fraction of equity) that freezes NEW
    # event entries. Exits always still flow.
    event_book_max_loss_pct: float = 0.02
    event_book_state_path: str = "data/event_book_state.json"
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
    ) -> None:
        self.config = config or EventDrivenConfig()
        self.data = data_provider
        self.state = state_store
        self.snapshot_store = snapshot_store
        self.db = db_engine
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
                # TIME STOP fires regardless of P&L — and regardless of
                # whether we even have a current price.
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
            exits.append(OrderIntent(
                strategy_id=self.id, symbol=symbol, target_position=0,
                urgency="high", max_slippage_bps=self.config.max_slippage_bps,
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
    # Entries for a newly-CONFIRMED event
    # ------------------------------------------------------------------

    def _enter_confirmed(
        self,
        row: dict[str, Any],
        assessment: dict[str, Any],
        prices: dict[str, Any],
        equity: float | None,
        now: datetime,
    ) -> tuple[
        list[OrderIntent],
        list[tuple[str, str, str, float, str]],
        list[tuple[str, str]],
    ]:
        """Emit entry intents for the tradable affected instruments.

        Returns (intents, entered, skipped) where entered is
        [(symbol, direction_str, size_str, entry_price, reason)] —
        reason is the assessment's per-instrument ``affected[].reason``
        — and skipped is [(instrument_or_symbol, reason)]; both feed
        the operator alert.
        """
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
                urgency="high", max_slippage_bps=self.config.max_slippage_bps,
                metadata=meta,
            ))
            entered.append((
                symbol, dir_str, f"{size:.0f}", entry_price,
                str(aff.get("reason") or ""),
            ))
        return intents, entered, skipped

    # ------------------------------------------------------------------
    # Operator alerts (plain text — short lines, one fact per line)
    # ------------------------------------------------------------------

    @staticmethod
    def _watch_list(assessment: dict[str, Any]) -> list[str]:
        watch: list[str] = []
        for aff in assessment.get("affected") or []:
            if not isinstance(aff, dict):
                continue
            kind = str(aff.get("kind") or "")
            direction = str(aff.get("direction") or "")
            if kind not in TRADABLE_KINDS or direction == "watch":
                name = str(aff.get("instrument") or "")
                if name:
                    watch.append(name)
        return watch

    @staticmethod
    def _ideas_block(
        assessment: dict[str, Any], limit: int = 5,
    ) -> list[str]:
        """Advisory trade-ideas lines (CL-mgcp) — clearly separated
        from the machine trades above them: the system never trades
        equities or options; these are for the operator's own hands.
        Empty list when the assessment carries no ideas."""
        ideas = assessment.get("trade_ideas")
        if not isinstance(ideas, list) or not ideas:
            return []
        lines = ["Operator ideas (not machine-traded):"]
        for idea in ideas[:limit]:
            if not isinstance(idea, dict):
                continue
            parts = [
                str(idea.get("ticker") or "?"),
                str(idea.get("action") or "?"),
            ]
            horizon = str(idea.get("time_horizon") or "").strip()
            if horizon:
                parts.append(horizon)
            time_stop = idea.get("time_stop_days")
            if time_stop is not None:
                parts.append(f"stop{time_stop}d")
            line = f"- {' '.join(parts)}"
            rationale = str(idea.get("rationale") or "").strip()
            if rationale:
                line += f" — {rationale[:60]}"
            lines.append(line)
        return lines if len(lines) > 1 else []

    @staticmethod
    def _event_age(row: dict[str, Any], now: datetime | None = None) -> str:
        """``2h`` / ``3d`` since the event was first seen, or ``""``."""
        return format_age(row.get("seen_at"), now=now)

    def _alert_confirmed(
        self,
        row: dict[str, Any],
        assessment: dict[str, Any],
        entered: list[tuple[str, str, str, float, str]],
        skipped: list[tuple[str, str]],
        prices: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(UTC)
        lines = [f"Headline: {str(row.get('headline') or '')[:140]}"]
        age = self._event_age(row, now)
        if age:
            lines.append(f"Age: {age} since first seen")
        for symbol, dir_str, size_str, entry_price, reason in entered:
            lines.append(
                f"Trade: {symbol} {dir_str} ({size_str} units) @ {entry_price:g}",
            )
            if reason:
                lines.append(f"  Why: {reason[:100]}")
        for name, reason in skipped:
            line = f"Skipped: {name} ({reason})"
            current = self._current_price(name, prices or {}, now)
            if current is not None:
                line += f" @ {current:g}"
            lines.append(line)
        lines.append(f"Risk: {self.config.event_risk_pct * 100:.2f}% of equity per trade")
        lines.append(f"Stop: {self.config.event_stop_pct * 100:.2f}% from entry")
        lines.append(f"Time stop: {self.config.event_max_holding_hours:g}h")
        lines.append(f"Urgency: {assessment.get('urgency')}/10")
        with contextlib.suppress(TypeError, ValueError):
            lines.append(f"Confidence: {float(assessment.get('confidence') or 0.0):.2f}")
        watch = self._watch_list(assessment)
        if watch:
            lines.append("Watch: " + ", ".join(watch))
        ideas = self._ideas_block(assessment)
        if ideas:
            lines.append("")
            lines.extend(ideas)
        try:
            notify_operator("Event confirmed", "\n".join(lines), priority=1)
        except Exception:
            logger.exception("Confirmed-event alert dispatch failed")

    def _alert_expired(self, row: dict[str, Any], urgency: int, confidence: float) -> None:
        lines = [f"Headline: {str(row.get('headline') or '')[:140]}"]
        age = self._event_age(row)
        if age:
            lines.append(f"Age: {age} since first seen")
        lines += [
            f"Urgency: {urgency}/10",
            f"Confidence: {confidence:.2f}",
            f"No market confirmation within {self.config.confirm_window_max_minutes}min",
            "No trade taken",
        ]
        # Top advisory idea (highest confidence) still surfaces — an
        # expired-unconfirmed event can be an operator opportunity even
        # when the machine passes (CL-mgcp).
        assessment = EventConfluence.parse_assessment(row.get("assessment")) or {}
        ideas = [
            i for i in (assessment.get("trade_ideas") or [])
            if isinstance(i, dict) and i.get("ticker")
        ]
        if ideas:
            top = max(
                ideas, key=lambda i: float(i.get("confidence") or 0.0),
            )
            line = (
                f"Top idea: {top.get('ticker')} {top.get('action') or '?'}"
            )
            if top.get("time_stop_days") is not None:
                line += f" (stop {top.get('time_stop_days')}d)"
            rationale = str(top.get("rationale") or "").strip()
            if rationale:
                line += f" — {rationale[:60]}"
            lines.append(line)
        try:
            notify_operator("Event expired unconfirmed", "\n".join(lines), priority=0)
        except Exception:
            logger.exception("Expired-event alert dispatch failed")

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
                    self._alert_expired(row, result.urgency, result.confidence)
                    expired_alerts_sent += 1
                continue

            if result.outcome != "confirmed" or not result.transitioned:
                continue  # pending, or another writer won the transition

            assessment = EventConfluence.parse_assessment(row.get("assessment")) or {}
            entry_intents, entered, skipped = self._enter_confirmed(
                row, assessment, prices, equity, now,
            )
            # Alert on every CONFIRMED event — even when caps/mapping
            # meant nothing was tradable (operator can act manually).
            self._alert_confirmed(
                row, assessment, entered, skipped, prices=prices, now=now,
            )
            if entry_intents:
                intents.extend(entry_intents)
                self.confluence.transition(row.get("id"), "CONFIRMED", "TRADED")
            # else: row stays CONFIRMED — visible in the table as
            # "confirmed but not traded" (caps / unknown instruments).

        return intents
