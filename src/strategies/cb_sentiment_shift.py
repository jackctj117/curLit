"""Strategy 2: CB sentiment shift — event-driven entries on hawkish/dovish extreme diffs."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from src.execution.oms import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class CBSentimentConfig:
    cb_to_pair: dict = field(default_factory=lambda: {
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
    def __init__(self, config: CBSentimentConfig = None, data_provider=None, nlp_provider=None, state_store=None) -> None:
        self.config = config or CBSentimentConfig()
        self.data = data_provider
        self.nlp = nlp_provider
        self.state = state_store
        self.open_positions: dict[str, OpenPosition] = {}
        self._historical_diffs: dict[str, list[float]] = {}
        self._last_refresh: datetime | None = None

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def symbols(self) -> list[str]:
        return list(set(p for p, _ in self.config.cb_to_pair.values()))

    @property
    def signal_interval_seconds(self) -> int:
        return self.config.signal_interval_seconds

    def fit(self, train_data) -> None:
        pass

    def generate_signals(self, data) -> None:
        return None

    def _refresh_thresholds(self) -> None:
        if self.nlp is None:
            return
        now = datetime.now(timezone.utc)
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
        except Exception:
            logger.debug("Threshold refresh skipped — NLP not available")

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

    def _check_new_events(self) -> list[dict]:
        if self.nlp is None:
            return []
        now = datetime.now(timezone.utc)
        try:
            events = self.nlp.get_recent_diff_events(
                since=now - timedelta(hours=2),
                cbs=list(self.config.cb_to_pair.keys()),
            )
        except Exception:
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

    def _update_trailing_stops(self, prices: dict) -> list[OrderIntent]:
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

            days_held = (datetime.now(timezone.utc) - pos.entry_ts).days
            if days_held >= self.config.holding_days:
                exit_reason = "time_exit"

            if exit_reason:
                logger.info("Exit %s: %s pnl=%.2f%%", symbol, exit_reason, pnl_pct * 100)
                exits.append(OrderIntent(strategy_id=self.id, symbol=symbol, target_position=0))
                del self.open_positions[symbol]
        return exits

    async def generate_intents(self, prices: dict, broker) -> list[OrderIntent]:
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
                symbol=pair, entry_ts=datetime.now(timezone.utc), entry_price=entry_price,
                quantity=size, direction=signal["direction"], stop_loss=stop_price,
                source_cb=signal["cb"],
            )
            intents.append(OrderIntent(strategy_id=self.id, symbol=pair, target_position=size))

        return intents
