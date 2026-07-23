"""Transaction cost model for realistic backtesting.

CL-x50g: added overnight funding/swap drag (``overnight_funding_annual_bps``)
consumed by the walk-forward trades loop. Set it to 0.0 (and use
``WalkForwardConfig.legacy_flat_costs=True``) to reproduce pre-CL-x50g
backtest numbers exactly.
"""

from dataclasses import dataclass, field

# Trading days per year — matches the daily-bar cadence of the walk-forward
# trades loop (funding is charged once per held daily bar).
_TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class CostModel:
    spread_bps: float = 0.5
    slippage_bps: float = 0.3
    per_pair_spreads: dict[str, float] = field(
        default_factory=lambda: {
            "EURUSD": 0.3,
            "USDJPY": 0.4,
            "GBPUSD": 0.6,
            "USDCAD": 0.5,
            "AUDUSD": 0.6,
            "NZDUSD": 0.7,
            "USDCHF": 0.5,
        }
    )
    cross_spread_default_bps: float = 1.5
    news_spread_multiplier: float = 2.0
    overnight_spread_multiplier: float = 3.0
    # CL-x50g: crude FLAT overnight funding/swap cost for held positions,
    # in annualized bps of notional. Charged per trading day on the
    # absolute overnight position in the walk-forward trades loop
    # (annual / 252 per daily bar). 15 bps/yr approximates typical G10
    # swap drag; deliberately direction- and pair-agnostic — refine when
    # real swap-point data lands. Set 0.0 to reproduce pre-CL-x50g numbers.
    overnight_funding_annual_bps: float = 15.0

    @property
    def cost_per_turn(self) -> float:
        return (self.spread_bps + self.slippage_bps) / 10000.0

    @property
    def overnight_funding_daily(self) -> float:
        """Per-trading-day funding cost as a return fraction (CL-x50g)."""
        return self.overnight_funding_annual_bps / 10000.0 / _TRADING_DAYS_PER_YEAR

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
