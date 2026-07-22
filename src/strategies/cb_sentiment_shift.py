"""Strategy 2: CB sentiment shift — event-driven entries on hawkish/dovish extreme diffs."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from src.execution.oms import OrderIntent
from src.models.feature_versioning import (
    FeatureSnapshot,
    FeatureSnapshotStore,
    attach_snapshot_payload,
)

logger = logging.getLogger(__name__)


_FEATURE_SET_NAME = "cb_sentiment_shift"
_FEATURE_SET_VERSION = "v1"


@dataclass
class CBSentimentConfig:
    cb_to_pair: dict[str, Any] = field(default_factory=lambda: {
        "fed": ("EURUSD", "short"), "ecb": ("EURUSD", "long"),
        "boe": ("GBPUSD", "long"), "boj": ("USDJPY", "short"), "boc": ("USDCAD", "short"),
    })
    strong_shift_percentile: float = 0.15
    min_diff_score_abs: float = 0.3
    holding_days: int = 10
    hard_stop_pct: float = 0.015
    trailing_trigger_pct: float = 0.02
    trailing_distance_pct: float = 0.015
    risk_per_trade_pct: float = 0.01
    max_concurrent_positions: int = 3
    signal_interval_seconds: int = 300
    id: str = "cb_sentiment_shift"


@dataclass
class OpenPosition:
    symbol: str
    entry_ts: datetime
    entry_price: float
    quantity: float
    direction: int
    stop_loss: float
    source_cb: str = ""
    trailing_stop: float | None = None
    peak_price: float | None = None


class CBSentimentShiftStrategy:
    def __init__(
        self,
        config: CBSentimentConfig | None = None,
        data_provider: Any = None,
        nlp_provider: Any = None,
        state_store: Any = None,
        snapshot_store: FeatureSnapshotStore | None = None,
    ) -> None:
        self.config = config or CBSentimentConfig()
        self.data = data_provider
        self.nlp = nlp_provider
        self.state = state_store
        self.snapshot_store = snapshot_store
        self.open_positions: dict[str, OpenPosition] = {}
        self._historical_diffs: dict[str, list[float]] = {}
        self._last_refresh: datetime | None = None

    def _emit_snapshot(self, values: dict[str, Any]) -> dict[str, Any]:
        """Build, persist, and return a snapshot-reference payload (or {})."""
        if self.snapshot_store is None:
            return {}
        snapshot = FeatureSnapshot.create(
            feature_set_name=_FEATURE_SET_NAME,
            feature_set_version=_FEATURE_SET_VERSION,
            data_snapshot_id="live",
            model_version=f"thresh_pct={self.config.strong_shift_percentile}",
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

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def symbols(self) -> list[str]:
        return list(set(p for p, _ in self.config.cb_to_pair.values()))

    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds

    # NOTE: no fit()/generate_signals() — those belong to the walk-forward
    # backtest protocol (src/backtest/walkforward.Strategy); live-only
    # strategies are driven exclusively via generate_intents() and the
    # dead no-op stubs were removed (CL-e6lx).

    def _refresh_thresholds(self) -> None:
        if self.nlp is None:
            return
        now = datetime.now(UTC)
        if self._last_refresh and (now - self._last_refresh).days < 7:
            return
        try:
            events = self.nlp.get_historical_diff_scores(
                cbs=list(self.config.cb_to_pair.keys()), lookback_years=5,
            )
            if events is not None and not events.empty:
                for cb in self.config.cb_to_pair:
                    cb_data = events[events["cb"] == cb]["net_shift"]
                    if len(cb_data) > 0:
                        self._historical_diffs[cb] = cb_data.tolist()
                self._last_refresh = now
        except Exception as exc:
            # Broad by design: a threshold refresh must never break the tick
            # (previous thresholds stay in force), but the failure has to be
            # visible (CL-gmr1). warning without traceback per CL-2yta.
            logger.warning(
                "%s: threshold refresh failed (%s: %s) — keeping previous "
                "thresholds", self.id, type(exc).__name__, exc,
            )

    def _get_thresholds(self, cb: str) -> tuple[float, float]:
        diffs = self._historical_diffs.get(cb, [])
        if len(diffs) < 10:
            return (-self.config.min_diff_score_abs, self.config.min_diff_score_abs)
        pct = self.config.strong_shift_percentile
        dovish = float(np.percentile(diffs, pct * 100))
        hawkish = float(np.percentile(diffs, (1 - pct) * 100))
        return (
            min(dovish, -self.config.min_diff_score_abs),
            max(hawkish, self.config.min_diff_score_abs),
        )

    def _check_new_events(self) -> list[dict[str, Any]]:
        if self.nlp is None:
            return []
        now = datetime.now(UTC)
        try:
            events = self.nlp.get_recent_diff_events(
                since=now - timedelta(hours=2),
                cbs=list(self.config.cb_to_pair.keys()),
            )
        except Exception as exc:
            # Broad by design: the strategy tick must survive a provider
            # failure — but a dead NLP feed is NOT "no events", so it must
            # never be swallowed silently (CL-gmr1 review call-out).
            logger.warning(
                "%s: get_recent_diff_events failed (%s: %s) — treating as no "
                "events this tick", self.id, type(exc).__name__, exc,
            )
            return []
        signals = []
        for event in (events or []):
            cb = event.get("cb", "")
            if cb not in self.config.cb_to_pair:
                continue
            shift = event.get("net_shift", 0)
            dovish_thr, hawkish_thr = self._get_thresholds(cb)
            pair, cb_hawkish_side = self.config.cb_to_pair[cb]
            if shift >= hawkish_thr:
                direction = 1 if cb_hawkish_side == "long" else -1
                signals.append({"cb": cb, "pair": pair, "direction": direction,
                                "shift": shift, "doc_id": event.get("doc_id", "")})
            elif shift <= dovish_thr:
                direction = -1 if cb_hawkish_side == "long" else 1
                signals.append({"cb": cb, "pair": pair, "direction": direction,
                                "shift": shift, "doc_id": event.get("doc_id", "")})
        return signals

    def _update_trailing_stops(self, prices: dict[str, Any]) -> list[OrderIntent]:
        exits = []
        for symbol, pos in list(self.open_positions.items()):
            tick = prices.get(symbol)
            if tick is None:
                continue
            current = (tick["bid"] + tick["ask"]) / 2
            pnl_pct = (current - pos.entry_price) / pos.entry_price * pos.direction

            if pnl_pct >= self.config.trailing_trigger_pct:
                if pos.direction > 0:
                    pos.peak_price = max(pos.peak_price or pos.entry_price, current)
                    pos.trailing_stop = pos.peak_price * (1 - self.config.trailing_distance_pct)
                else:
                    pos.peak_price = min(pos.peak_price or pos.entry_price, current)
                    pos.trailing_stop = pos.peak_price * (1 + self.config.trailing_distance_pct)

            exit_reason = None
            if pos.direction > 0:
                if current <= pos.stop_loss:
                    exit_reason = "hard_stop"
                elif pos.trailing_stop and current <= pos.trailing_stop:
                    exit_reason = "trailing_stop"
            else:
                if current >= pos.stop_loss:
                    exit_reason = "hard_stop"
                elif pos.trailing_stop and current >= pos.trailing_stop:
                    exit_reason = "trailing_stop"

            days_held = (datetime.now(UTC) - pos.entry_ts).days
            if days_held >= self.config.holding_days:
                exit_reason = "time_exit"

            if exit_reason:
                logger.info("Exit %s: %s pnl=%.2f%%", symbol, exit_reason, pnl_pct * 100)
                meta = self._emit_snapshot({
                    "trigger": "exit",
                    "exit_reason": exit_reason,
                    "symbol": symbol,
                    "current_price": float(current),
                    "entry_price": float(pos.entry_price),
                    "direction": int(pos.direction),
                    "pnl_pct": float(pnl_pct),
                    "source_cb": pos.source_cb,
                })
                exits.append(OrderIntent(
                    strategy_id=self.id, symbol=symbol,
                    target_position=0, metadata=meta,
                ))
                del self.open_positions[symbol]
        return exits

    async def generate_intents(
        self, prices: dict[str, Any], broker: Any,
    ) -> list[OrderIntent]:
        self._refresh_thresholds()
        intents = self._update_trailing_stops(prices)

        if len(self.open_positions) >= self.config.max_concurrent_positions:
            return intents

        signals = self._check_new_events()
        account = broker.get_account()

        for signal in signals:
            pair = signal["pair"]
            if pair in self.open_positions:
                continue

            tick = prices.get(pair)
            if tick is None:
                continue

            entry_price = tick["ask"] if signal["direction"] > 0 else tick["bid"]
            stop_price = entry_price * (1 - signal["direction"] * self.config.hard_stop_pct)
            stop_distance = abs(entry_price - stop_price)
            size = account.equity * self.config.risk_per_trade_pct / max(stop_distance, 0.0001) * signal["direction"]

            logger.info("Entry %s: %s shift=%.3f size=%.0f", pair, signal["cb"], signal["shift"], size)
            self.open_positions[pair] = OpenPosition(
                symbol=pair, entry_ts=datetime.now(UTC), entry_price=entry_price,
                quantity=size, direction=signal["direction"], stop_loss=stop_price,
                source_cb=signal["cb"],
            )
            meta = self._emit_snapshot({
                "trigger": "entry",
                "cb": signal["cb"],
                "pair": pair,
                "direction": int(signal["direction"]),
                "shift": float(signal["shift"]),
                "doc_id": signal.get("doc_id", ""),
                "entry_price": float(entry_price),
                "stop_price": float(stop_price),
                "size": float(size),
            })
            intents.append(OrderIntent(
                strategy_id=self.id, symbol=pair,
                target_position=size, metadata=meta,
            ))

        return intents
