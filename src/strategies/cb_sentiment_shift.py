"""Strategy 2: CB sentiment shift — event-driven entries on hawkish/dovish extreme diffs."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

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
    trailing_stop: float | None = None
    peak_price: float | None = None


class CBSentimentShiftStrategy:
    def __init__(self, config: CBSentimentConfig = None, data_provider=None, nlp_provider=None, state_store=None) -> None:
        self.config = config or CBSentimentConfig()
        self.data = data_provider
        self.nlp = nlp_provider
        self.state = state_store
        self.open_positions: dict[str, OpenPosition] = {}
        self._historical_diffs = None
        self._last_refresh = None

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def symbols(self) -> list[str]:
        return list(set(p for p, _ in self.config.cb_to_pair.values()))

    def fit(self, train_data) -> None:
        pass

    def generate_signals(self, data) -> None:
        return None

    async def generate_intents(self, prices: dict, broker) -> list[OrderIntent]:
        return []
