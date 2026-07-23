"""Regime edge decomposition (G6 / CL-lx4) — where does the edge come from?

A strategy that passes G1+G2 may still derive its entire edge from one
specific market regime — vulnerable to regime change. Decomposing strategy
returns by regime reveals concentration risk that single-period statistics
miss.

Approach (matches reference/14 "Regime Edge Analysis"):

    1. Classify each period into one of five regimes via realized vol +
       trend strength on a benchmark price series:
            CRISIS              top 5% vol periods
            HIGH_VOL_TRENDING   high vol AND strong trend
            HIGH_VOL_CHOPPY     high vol AND weak trend
            LOW_VOL_TRENDING    low vol AND strong trend
            LOW_VOL_CHOPPY      low vol AND weak trend

    2. Per-regime stats: Sharpe, annualized mean+vol, win rate, drawdown,
       count of periods, fraction of time, contribution to total return.

    3. Concentration warnings:
            - any single regime contributes >70% of total return → fragile
            - top 2 regimes contribute >90% → strategy is regime-specific
            - any regime has Sharpe < -0.5 → strong negative-regime exposure

    4. The G8 dashboard reads `is_diversified` (= concentration ≤ 0.7)
       directly into `regime_edge_diversified`, used in the verdict
       composition.

Reference: reference/14_edge_testing.md "Regime Edge Analysis".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# Default classification window sizes — match reference defaults.
_DEFAULT_VOL_WINDOW: int = 20
_DEFAULT_TREND_WINDOW: int = 63

# Crisis-regime threshold: top 5% of realized-vol periods.
_CRISIS_VOL_QUANTILE: float = 0.95

# Concentration warning thresholds.
_CONCENTRATION_HIGH: float = 0.70
_TOP2_HIGH: float = 0.90
_NEG_SHARPE_WARN: float = -0.5

# Per-regime minimum samples — below this a regime's stats are too noisy
# to be informative; we skip with a recorded note.
_MIN_PERIODS_PER_REGIME: int = 20

# Trading days per year for annualization.
_TRADING_DAYS_PER_YEAR: int = 252


class MarketRegime(Enum):
    LOW_VOL_TRENDING = "low_vol_trending"
    HIGH_VOL_TRENDING = "high_vol_trending"
    LOW_VOL_CHOPPY = "low_vol_choppy"
    HIGH_VOL_CHOPPY = "high_vol_choppy"
    CRISIS = "crisis"


@dataclass
class RegimeStats:
    """Per-regime decomposition of strategy returns."""

    regime: MarketRegime
    n_periods: int
    pct_of_time: float
    sharpe: float
    mean_return_annualized: float
    vol_annualized: float
    win_rate: float
    contribution_to_total: float
    max_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "n_periods": self.n_periods,
            "pct_of_time": self.pct_of_time,
            "sharpe": self.sharpe,
            "mean_return_annualized": self.mean_return_annualized,
            "vol_annualized": self.vol_annualized,
            "win_rate": self.win_rate,
            "contribution_to_total": self.contribution_to_total,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class RegimeAnalysisReport:
    """Output of RegimeEdgeAnalyzer.analyze."""

    by_regime: dict[MarketRegime, RegimeStats] = field(default_factory=dict)
    edge_concentration: float = 0.0
    top_2_regime_pct: float = 0.0
    is_diversified: bool = True
    worst_regime: MarketRegime | None = None
    worst_regime_sharpe: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_regime": {r.value: s.to_dict() for r, s in self.by_regime.items()},
            "edge_concentration": self.edge_concentration,
            "top_2_regime_pct": self.top_2_regime_pct,
            "is_diversified": self.is_diversified,
            "worst_regime": self.worst_regime.value if self.worst_regime else None,
            "worst_regime_sharpe": self.worst_regime_sharpe,
            "warnings": list(self.warnings),
        }


# =============================================================================
# Helpers
# =============================================================================


def _sharpe(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    sd = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(returns.mean() / sd * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


# =============================================================================
# Analyzer
# =============================================================================


class RegimeEdgeAnalyzer:
    """Classify periods into regimes and decompose strategy returns.

    Stateless. Construct once, reuse across strategies. Pass the same
    benchmark price series for consistent regime classification.
    """

    def __init__(
        self,
        vol_window: int = _DEFAULT_VOL_WINDOW,
        trend_window: int = _DEFAULT_TREND_WINDOW,
    ) -> None:
        assert vol_window >= 5, f"vol_window must be >= 5, got {vol_window}"
        assert trend_window >= 20, f"trend_window must be >= 20, got {trend_window}"
        self.vol_window = vol_window
        self.trend_window = trend_window

    def classify_regimes(
        self,
        prices: pd.Series,
    ) -> pd.Series:
        """Return a Series of MarketRegime labels (one per timestamp).

        prices: benchmark price series. Function computes realized vol,
        trend strength, and the crisis quantile internally.
        """
        assert not prices.empty, "prices must be non-empty"
        returns = prices.pct_change()
        realized_vol = returns.rolling(self.vol_window).std() * math.sqrt(
            _TRADING_DAYS_PER_YEAR,
        )
        # Median is robust to outliers; using nan-aware median for early NaN window.
        vol_median = float(realized_vol.median())
        high_vol = realized_vol > vol_median

        trend = prices.pct_change(self.trend_window).abs()
        trend_median = float(trend.median())
        trending = trend > trend_median

        crisis_threshold = float(realized_vol.quantile(_CRISIS_VOL_QUANTILE))
        crisis = realized_vol > crisis_threshold

        regimes = pd.Series(
            MarketRegime.LOW_VOL_CHOPPY.value,
            index=prices.index,
            dtype=object,
        )
        # Apply rules in priority order: crisis first, then quadrant.
        regimes[~crisis & high_vol & trending] = MarketRegime.HIGH_VOL_TRENDING.value
        regimes[~crisis & high_vol & ~trending] = MarketRegime.HIGH_VOL_CHOPPY.value
        regimes[~crisis & ~high_vol & trending] = MarketRegime.LOW_VOL_TRENDING.value
        regimes[~crisis & ~high_vol & ~trending] = MarketRegime.LOW_VOL_CHOPPY.value
        regimes[crisis] = MarketRegime.CRISIS.value
        return regimes

    def analyze(
        self,
        strategy_returns: pd.Series,
        benchmark_prices: pd.Series,
    ) -> RegimeAnalysisReport:
        """Decompose strategy_returns by regime, with concentration warnings."""
        assert not strategy_returns.empty, "strategy_returns must be non-empty"
        regimes = self.classify_regimes(benchmark_prices)
        # Align strategy returns to regime index; forward-fill missing.
        regimes = regimes.reindex(strategy_returns.index, method="ffill")

        total_return_sum = float(strategy_returns.sum())
        by_regime: dict[MarketRegime, RegimeStats] = {}

        for regime_label in regimes.dropna().unique():
            try:
                regime_enum = MarketRegime(regime_label)
            except ValueError:
                continue
            mask = regimes == regime_label
            regime_returns = strategy_returns[mask]
            if len(regime_returns) < _MIN_PERIODS_PER_REGIME:
                continue

            contribution = (
                float(regime_returns.sum() / total_return_sum) if total_return_sum != 0 else 0.0
            )

            by_regime[regime_enum] = RegimeStats(
                regime=regime_enum,
                n_periods=int(len(regime_returns)),
                pct_of_time=float(mask.mean()),
                sharpe=_sharpe(regime_returns),
                mean_return_annualized=float(
                    regime_returns.mean() * _TRADING_DAYS_PER_YEAR,
                ),
                vol_annualized=float(
                    regime_returns.std(ddof=1) * math.sqrt(_TRADING_DAYS_PER_YEAR),
                )
                if len(regime_returns) > 1
                else 0.0,
                win_rate=float((regime_returns > 0).mean()),
                contribution_to_total=contribution,
                max_drawdown=_max_drawdown(regime_returns),
            )

        # Concentration analysis. Use absolute-contribution sort to handle
        # negative-contribution regimes correctly (they pull edge down even
        # when summed magnitudes are large).
        contributions = [s.contribution_to_total for s in by_regime.values()]
        max_contribution = max(contributions, default=0.0)
        sorted_desc = sorted(contributions, reverse=True)
        top_2 = sum(sorted_desc[:2]) if sorted_desc else 0.0

        warnings: list[str] = []
        if max_contribution > _CONCENTRATION_HIGH:
            warnings.append(
                f"Edge concentrated: {max_contribution:.0%} of returns from one regime — "
                f"vulnerable to regime change",
            )
        if top_2 > _TOP2_HIGH and len(contributions) > 2:
            warnings.append(
                f"Top 2 regimes contribute {top_2:.0%} — strategy may be regime-specific",
            )

        worst_regime: MarketRegime | None = None
        worst_sharpe: float | None = None
        if by_regime:
            worst_regime = min(by_regime, key=lambda k: by_regime[k].sharpe)
            worst_sharpe = by_regime[worst_regime].sharpe
            if worst_sharpe < _NEG_SHARPE_WARN:
                warnings.append(
                    f"Strategy has strongly negative Sharpe ({worst_sharpe:.2f}) "
                    f"in {worst_regime.value} regime",
                )

        is_diversified = max_contribution <= _CONCENTRATION_HIGH and not (
            top_2 > _TOP2_HIGH and len(contributions) > 2
        )

        return RegimeAnalysisReport(
            by_regime=by_regime,
            edge_concentration=float(max_contribution),
            top_2_regime_pct=float(top_2),
            is_diversified=is_diversified,
            worst_regime=worst_regime,
            worst_regime_sharpe=worst_sharpe,
            warnings=warnings,
        )


__all__ = [
    "MarketRegime",
    "RegimeAnalysisReport",
    "RegimeEdgeAnalyzer",
    "RegimeStats",
]
