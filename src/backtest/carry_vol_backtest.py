"""Carry+vol backtest (CL-2hs / A6) — historical simulation of carry_vol_filter strategy.

Loads rates, prices, and a vol-index series, simulates the monthly carry-basket
rebalance with daily vol-filter exposure scaling, and compares filtered vs
unfiltered runs head-to-head. Both runs share the same cost model so the
filter's effect on drawdown and Sharpe is isolated.

Inputs are passed as DataFrames so the backtester stays unit-testable and
DB-agnostic. A thin DB-loading helper (load_from_db) wraps the SQL queries
for the production path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.backtest.analytics import PerformanceAnalytics
from src.backtest.cost_model import CostModel
from src.strategies.carry_vol_filter import (
    _USD_BASE_PAIRS,
    CarryVolFilterConfig,
)

logger = logging.getLogger(__name__)


# Trading days per year — matches the rest of the codebase.
_TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class BacktestRun:
    """One pass of the simulation — filtered or unfiltered."""

    label: str
    returns: pd.Series  # daily fractional returns indexed by date
    equity: pd.Series  # cumulative equity ((1 + r).cumprod())
    exposure_history: pd.Series  # daily exposure multiplier applied
    metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "metrics": dict(self.metrics),
            "n_days": int(len(self.returns)),
        }


@dataclass
class CarryVolBacktestResult:
    """Comparison output of a carry+vol backtest."""

    filtered: BacktestRun
    unfiltered: BacktestRun
    config: CarryVolFilterConfig

    @property
    def sharpe_improvement(self) -> float:
        return float(
            self.filtered.metrics.get("sharpe", 0.0) - self.unfiltered.metrics.get("sharpe", 0.0)
        )

    @property
    def max_dd_reduction(self) -> float:
        # Both max_dd are negative; reduction means filtered is closer to 0.
        return float(
            self.filtered.metrics.get("max_drawdown", 0.0)
            - self.unfiltered.metrics.get("max_drawdown", 0.0)
        )

    def summary(self) -> str:
        return (
            f"Filtered Sharpe={self.filtered.metrics.get('sharpe', 0):.2f}, "
            f"max_dd={self.filtered.metrics.get('max_dd', 0):.2%}, "
            f"CAGR={self.filtered.metrics.get('cagr', 0):.2%}; "
            f"Unfiltered Sharpe={self.unfiltered.metrics.get('sharpe', 0):.2f}, "
            f"max_dd={self.unfiltered.metrics.get('max_dd', 0):.2%}, "
            f"CAGR={self.unfiltered.metrics.get('cagr', 0):.2%}; "
            f"ΔSharpe={self.sharpe_improvement:+.2f}, "
            f"ΔmaxDD={self.max_dd_reduction:+.2%}"
        )


# =============================================================================
# Backtester
# =============================================================================


class CarryVolBacktester:
    """Simulate the carry+vol strategy on historical inputs.

    The simulation is deliberately simpler than the live engine:
        - No bid/ask spreads beyond the cost model's per-turn deduction
        - No latency
        - Positions held continuously (rebalanced once per month)
        - Exposure scaling applied multiplicatively to position-day returns

    This trades realism for clarity. The live engine + paper-vs-live divergence
    detector (G4) is the place to validate execution-quality realism.
    """

    def __init__(
        self,
        config: CarryVolFilterConfig | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self.config = config or CarryVolFilterConfig()
        self.cost_model = cost_model or CostModel()

    def run(
        self,
        rates_df: pd.DataFrame,
        fx_returns_df: pd.DataFrame,
        vol_series: pd.Series,
    ) -> CarryVolBacktestResult:
        """Run the head-to-head simulation.

        Args:
            rates_df: index=date (daily), columns=currencies (subset of G10),
                      values=short rate as a fraction (e.g. 0.04 for 4%).
                      Forward-filled internally — non-trading days OK.
            fx_returns_df: index=date, columns=currencies (non-USD), values=
                           daily fractional return of holding 1 unit of that
                           currency vs USD. Long X uses the column's positive
                           value; short uses the negative.
            vol_series: index=date, values=vol-index level (e.g. CVIX).
        """
        assert not rates_df.empty, "rates_df must be non-empty"
        assert not fx_returns_df.empty, "fx_returns_df must be non-empty"

        # Align all inputs to a common index (intersection).
        common_index = rates_df.index.intersection(fx_returns_df.index)
        if vol_series is not None and not vol_series.empty:
            common_index = common_index.intersection(vol_series.index)
        assert len(common_index) >= 60, f"need >=60 overlapping days, got {len(common_index)}"

        rates = rates_df.reindex(common_index).ffill()
        fx_returns = fx_returns_df.reindex(common_index).ffill().fillna(0.0)
        vol = (
            vol_series.reindex(common_index).ffill()
            if vol_series is not None
            else pd.Series(0.0, index=common_index)
        )

        filtered = self._simulate(
            rates,
            fx_returns,
            vol,
            with_vol_filter=True,
            label="filtered",
        )
        unfiltered = self._simulate(
            rates,
            fx_returns,
            vol,
            with_vol_filter=False,
            label="unfiltered",
        )

        result = CarryVolBacktestResult(
            filtered=filtered,
            unfiltered=unfiltered,
            config=self.config,
        )
        logger.info("Carry+vol backtest complete: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _simulate(
        self,
        rates: pd.DataFrame,
        fx_returns: pd.DataFrame,
        vol: pd.Series,
        with_vol_filter: bool,
        label: str,
    ) -> BacktestRun:
        daily_returns: list[float] = []
        exposure_history: list[float] = []

        # current_basket: {ccy: signed weight}
        current_basket: dict[str, float] = {}
        last_rebalance_month: tuple[int, int] | None = None

        for ts in rates.index:
            # Vol-filter exposure (1.0 when filter is off).
            if with_vol_filter:
                vol_z = self._rolling_vol_z(vol, ts)
                exposure = self._exposure_multiplier(vol_z)
            else:
                exposure = 1.0
            exposure_history.append(exposure)

            # Monthly rebalance — first day in a new (year, month) we see.
            this_month = (int(ts.year), int(ts.month))
            if last_rebalance_month != this_month:
                today_rates = rates.loc[ts].dropna().to_dict()
                new_basket = self._build_basket(today_rates)
                turnover = self._compute_turnover(current_basket, new_basket)
                cost_today = turnover * self.cost_model.cost_per_turn
                current_basket = new_basket
                last_rebalance_month = this_month
            else:
                cost_today = 0.0

            # Daily PnL from holdings.
            day_pnl = 0.0
            if current_basket:
                for ccy, signed_weight in current_basket.items():
                    if ccy not in fx_returns.columns:
                        continue
                    asset_ret = float(fx_returns.at[ts, ccy])
                    day_pnl += signed_weight * asset_ret
                day_pnl *= exposure

            day_pnl -= cost_today
            daily_returns.append(day_pnl)

        ret_series = pd.Series(daily_returns, index=rates.index, name=f"ret_{label}")
        equity = (1.0 + ret_series).cumprod()
        exposure_series = pd.Series(
            exposure_history,
            index=rates.index,
            name=f"exposure_{label}",
        )
        metrics = PerformanceAnalytics.metrics(
            ret_series,
            periods_per_year=_TRADING_DAYS_PER_YEAR,
        )
        return BacktestRun(
            label=label,
            returns=ret_series,
            equity=equity,
            exposure_history=exposure_series,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_basket(self, rates: dict[str, float]) -> dict[str, float]:
        """Rate-ranked basket → {ccy: signed weight}.

        Long-side currencies get +1/top_k weight, short-side get -1/bottom_k.
        Returns empty dict if the spread gate fails or insufficient currencies.
        """
        non_usd = {c: r for c, r in rates.items() if c != "USD"}
        if len(non_usd) < self.config.top_k + self.config.bottom_k:
            return {}
        sorted_ccys = sorted(non_usd.items(), key=lambda x: x[1], reverse=True)
        long_basket = sorted_ccys[: self.config.top_k]
        short_basket = sorted_ccys[-self.config.bottom_k :]
        rate_spread = long_basket[-1][1] - short_basket[0][1]
        if rate_spread < self.config.min_rate_spread - 1e-9:
            return {}
        out: dict[str, float] = {}
        for ccy, _ in long_basket:
            sign = -1.0 if ccy in _USD_BASE_PAIRS else 1.0
            # NOTE: fx_returns are in USD-base convention (return of long X
            # vs USD). When the broker pair is USD/X, holding long-X means
            # _selling_ USD/X — inverted P&L. fx_returns_df should already
            # express returns in long-X convention (test data does this).
            del sign  # for now, fx_returns_df is in long-X convention.
            out[ccy] = +1.0 / self.config.top_k
        for ccy, _ in short_basket:
            out[ccy] = -1.0 / self.config.bottom_k
        return out

    def _compute_turnover(
        self,
        old: dict[str, float],
        new: dict[str, float],
    ) -> float:
        """Sum of absolute weight changes across all currencies."""
        all_ccys = set(old.keys()) | set(new.keys())
        return float(sum(abs(new.get(c, 0.0) - old.get(c, 0.0)) for c in all_ccys))

    def _rolling_vol_z(self, vol: pd.Series, ts: pd.Timestamp) -> float:
        """Z-score of vol at ts vs the trailing lookback window's mean+std.

        Mirrors CarryVolFilterStrategy._compute_vol_z_score but sliced from a
        DataFrame. Returns 0.0 when too little history is available.
        """
        idx = vol.index.get_loc(ts)
        if not isinstance(idx, int):
            return 0.0
        if idx < self.config.vol_lookback_days:
            return 0.0
        window = vol.iloc[idx - self.config.vol_lookback_days : idx]
        if window.empty:
            return 0.0
        sd = float(window.std(ddof=1))
        mean = float(window.mean())
        if sd <= 0:
            delta = float(vol.iloc[idx] - mean)
            if abs(delta) < 1e-9:
                return 0.0
            return 10.0 if delta > 0 else -10.0
        return float((vol.iloc[idx] - mean) / sd)

    def _exposure_multiplier(self, vol_z: float) -> float:
        z = abs(vol_z)
        thresholds = sorted(self.config.vol_z_thresholds.items())
        applied = thresholds[0][1]
        for threshold, mult in thresholds:
            if z >= threshold:
                applied = mult
        return float(applied)
