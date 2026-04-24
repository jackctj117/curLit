"""Transaction cost model for realistic backtesting."""

from dataclasses import dataclass, field


@dataclass
class CostModel:
    spread_bps: float = 0.5
    slippage_bps: float = 0.3
    per_pair_spreads: dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 0.3,
        "USDJPY": 0.4,
        "GBPUSD": 0.6,
        "USDCAD": 0.5,
        "AUDUSD": 0.6,
        "NZDUSD": 0.7,
        "USDCHF": 0.5,
    })
    cross_spread_default_bps: float = 1.5
    news_spread_multiplier: float = 2.0
    overnight_spread_multiplier: float = 3.0

    @property
    def cost_per_turn(self) -> float:
        return (self.spread_bps + self.slippage_bps) / 10000.0

    def get_cost_bps(self, pair: str) -> float:
        return self.per_pair_spreads.get(pair, self.cross_spread_default_bps) + self.slippage_bps

    def get_cost_per_turn(self, pair: str) -> float:
        return self.get_cost_bps(pair) / 10000.0

    def time_aware_spread(self, pair: str, hour_utc: int, is_news: bool = False) -> float:
        base = self.per_pair_spreads.get(pair, self.cross_spread_default_bps)
        if is_news:
            base *= self.news_spread_multiplier
        elif hour_utc < 1 or hour_utc > 20:
            base *= self.overnight_spread_multiplier
        return base + self.slippage_bps
