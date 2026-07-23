"""Edge decay detection (G7 / CL-675) — detect degrading edge over time.

Edges die. The strategies that worked in 2018 may not work in 2026; the
question is whether ours has degraded enough to act on. Reference/14
"Edge Decay Detection" defines four orthogonal tests; this implementation
follows that pattern with a structured dataclass output (rather than a dict)
so downstream consumers (G8 dashboard, G9 policy via decay_tau) can rely on
typed fields.

Tests:

    1. Mann-Kendall trend on QUARTERLY Sharpes — robust to outliers, sensitive
       to monotone decline. Tau in [-1, +1]; p_value reported.
    2. Mann-Whitney U comparing recent window to historical — non-parametric
       test of "are recent returns drawn from a worse distribution?"
    3. Rolling 6-month Sharpe slope — linear trend; reject H0 of zero slope.
    4. Recent vs historical max drawdown — qualitative: is the current
       drawdown materially worse than anything historical?

Severity escalates based on agreement across tests:

    STRONG_DECAY    tau ≤ -0.7 AND trend_p < 0.01
    MODERATE_DECAY  tau ≤ -0.5 AND mw_p < 0.05
    POSSIBLE_DECAY  tau ≤ -0.3
    NO_DECAY        otherwise

Recommendation maps severity to an action consumed by edge_policy.yaml:
    STRONG → retire_strategy
    MODERATE → reduce_size_50pct_and_investigate
    POSSIBLE → monitor_closely_weekly_review
    NO_DECAY (with slope warning) → check_regime_hypothesis
    NO_DECAY (clean) → no_action_edge_stable

The runner (EdgeRunner) consumes the decay_tau field directly to feed
G9's policy decay-retire precedence.

Reference: reference/14_edge_testing.md "Edge Decay Detection".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any  # noqa: F401  # used in quoted np.ndarray annotation

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# Default minimum history before decay can be assessed. Below this the
# Mann-Kendall on quarterly Sharpes has too few quarters and the rolling
# 6-month slope is unstable.
_DEFAULT_MIN_HISTORY_DAYS: int = 252

# Recent-vs-historical window for Mann-Whitney U. Capped at n/4 so the
# split makes sense for shorter series.
_DEFAULT_RECENT_WINDOW_DAYS: int = 60

# Rolling Sharpe window (6 months ≈ 126 trading days).
_ROLLING_SHARPE_WINDOW: int = 126

# Trading days per year for Sharpe annualization.
_TRADING_DAYS_PER_YEAR: int = 252

# Severity thresholds — calibrated to match reference/14 exactly.
_TAU_STRONG: float = -0.7
_TAU_MODERATE: float = -0.5
_TAU_POSSIBLE: float = -0.3
_TREND_P_STRONG: float = 0.01
_MW_P_MODERATE: float = 0.05
_SLOPE_DECAY_RATE: float = -0.005  # rolling Sharpe drop per day
_SLOPE_DECAY_P: float = 0.05
_DD_AMPLIFY_FACTOR: float = 1.3  # recent dd > 1.3× historical


class DecaySeverity(Enum):
    NO_DECAY = "no_decay"
    POSSIBLE_DECAY = "possible_decay"
    MODERATE_DECAY = "moderate_decay"
    STRONG_DECAY = "strong_decay"


class DecayRecommendation(Enum):
    NO_ACTION_EDGE_STABLE = "no_action_edge_stable"
    CHECK_REGIME_HYPOTHESIS = "check_regime_hypothesis"
    MONITOR_CLOSELY_WEEKLY_REVIEW = "monitor_closely_weekly_review"
    REDUCE_SIZE_50PCT_AND_INVESTIGATE = "reduce_size_50pct_and_investigate"
    RETIRE_STRATEGY = "retire_strategy"


@dataclass
class DecayAssessment:
    """Output of EdgeDecayMonitor.check_decay.

    All fields are populated when evaluated=True. When the input is below
    min_history_days, evaluated=False and the test fields are None — callers
    should treat this as "no signal yet" rather than "no decay".
    """

    evaluated: bool
    days_available: int
    severity: DecaySeverity
    recommendation: DecayRecommendation
    decaying: bool
    quarter_sharpes: list[float] = field(default_factory=list)
    trend_tau: float | None = None
    trend_p_value: float | None = None
    recent_sharpe: float | None = None
    historical_sharpe: float | None = None
    recent_vs_historical_p: float | None = None
    recent_significantly_worse: bool = False
    rolling_sharpe_slope: float | None = None
    rolling_sharpe_slope_p: float | None = None
    recent_max_dd: float | None = None
    historical_max_dd: float | None = None
    dd_worse_than_historical: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "days_available": self.days_available,
            "severity": self.severity.value,
            "recommendation": self.recommendation.value,
            "decaying": self.decaying,
            "quarter_sharpes": list(self.quarter_sharpes),
            "trend_tau": self.trend_tau,
            "trend_p_value": self.trend_p_value,
            "recent_sharpe": self.recent_sharpe,
            "historical_sharpe": self.historical_sharpe,
            "recent_vs_historical_p": self.recent_vs_historical_p,
            "recent_significantly_worse": self.recent_significantly_worse,
            "rolling_sharpe_slope": self.rolling_sharpe_slope,
            "rolling_sharpe_slope_p": self.rolling_sharpe_slope_p,
            "recent_max_dd": self.recent_max_dd,
            "historical_max_dd": self.historical_max_dd,
            "dd_worse_than_historical": self.dd_worse_than_historical,
            "note": self.note,
        }


# =============================================================================
# Helpers
# =============================================================================


def _sharpe(returns: pd.Series, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    if len(returns) == 0:
        return 0.0
    sd = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(returns.mean() / sd * math.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def _slope_significance(y: np.ndarray[Any, Any], slope: float) -> float:
    """Two-sided t-test on the OLS slope of y ~ x where x = arange(n)."""
    n = len(y)
    if n < 3:
        return 1.0
    x = np.arange(n)
    intercept = float(y.mean() - slope * x.mean())
    y_pred = slope * x + intercept
    residuals = y - y_pred
    sse = float(np.sum(residuals**2))
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx == 0:
        return 1.0
    se = math.sqrt(sse / (n - 2)) / math.sqrt(sxx) if n > 2 else 0.0
    if se == 0:
        return 1.0
    t_stat = abs(slope) / se
    return 2 * (1 - float(stats.t.cdf(t_stat, n - 2)))


# =============================================================================
# Monitor
# =============================================================================


class EdgeDecayMonitor:
    """Run the four decay tests and produce a structured DecayAssessment.

    Stateless. Construct once and reuse across strategies.
    """

    def __init__(
        self,
        min_history_days: int = _DEFAULT_MIN_HISTORY_DAYS,
        recent_window_days: int = _DEFAULT_RECENT_WINDOW_DAYS,
    ) -> None:
        assert min_history_days >= 60, "min_history_days must be >= 60"
        assert recent_window_days >= 20, "recent_window_days must be >= 20"
        self.min_history_days = min_history_days
        self.recent_window_days = recent_window_days

    def check_decay(self, returns: pd.Series) -> DecayAssessment:
        """Run all four decay tests; return DecayAssessment with severity."""
        n = len(returns)
        if n < self.min_history_days:
            return DecayAssessment(
                evaluated=False,
                days_available=n,
                severity=DecaySeverity.NO_DECAY,
                recommendation=DecayRecommendation.NO_ACTION_EDGE_STABLE,
                decaying=False,
                note=(f"insufficient history: {n} < {self.min_history_days} days"),
            )

        # ---- Test 1: Mann-Kendall on quarterly Sharpes ----
        quarter_size = n // 4
        quarter_sharpes = [
            _sharpe(returns.iloc[i * quarter_size : (i + 1) * quarter_size]) for i in range(4)
        ]
        try:
            tau_result = stats.kendalltau(range(4), quarter_sharpes)
            tau = float(tau_result.statistic)
            trend_p = float(tau_result.pvalue)
        except Exception:
            logger.exception("kendalltau failed on quarterly Sharpes")
            tau, trend_p = 0.0, 1.0

        # ---- Test 2: Mann-Whitney U recent vs historical ----
        recent_window = min(self.recent_window_days, n // 4)
        recent = returns.iloc[-recent_window:]
        historical = returns.iloc[:-recent_window]
        try:
            mw_result = stats.mannwhitneyu(recent, historical, alternative="less")
            mw_p = float(mw_result.pvalue)
        except ValueError:
            mw_p = 1.0
        recent_significantly_worse = mw_p < _MW_P_MODERATE

        # ---- Test 3: Rolling 6-month Sharpe slope ----
        rolling_sharpe = (
            returns.rolling(_ROLLING_SHARPE_WINDOW)
            .apply(lambda x: _sharpe(pd.Series(x)), raw=False)
            .dropna()
        )
        if len(rolling_sharpe) > 20:
            x = np.arange(len(rolling_sharpe))
            slope, _intercept = np.polyfit(x, rolling_sharpe.values, 1)
            slope_p = _slope_significance(rolling_sharpe.values, float(slope))
        else:
            slope, slope_p = 0.0, 1.0

        # ---- Test 4: Recent drawdown vs historical max ----
        recent_dd = _max_drawdown(recent)
        historical_dd = _max_drawdown(historical)
        dd_worse = (recent_dd < historical_dd * _DD_AMPLIFY_FACTOR) if historical_dd < 0 else False

        # ---- Combine ----
        severity = self._severity(tau, trend_p, mw_p)
        recommendation = self._recommend(tau, trend_p, mw_p, float(slope))
        decaying = (
            (tau < _TAU_MODERATE and trend_p < 0.10)
            or (mw_p < _MW_P_MODERATE)
            or (slope < _SLOPE_DECAY_RATE and slope_p < _SLOPE_DECAY_P)
        )

        return DecayAssessment(
            evaluated=True,
            days_available=n,
            severity=severity,
            recommendation=recommendation,
            decaying=decaying,
            quarter_sharpes=[float(s) for s in quarter_sharpes],
            trend_tau=tau,
            trend_p_value=trend_p,
            recent_sharpe=_sharpe(recent),
            historical_sharpe=_sharpe(historical),
            recent_vs_historical_p=mw_p,
            recent_significantly_worse=recent_significantly_worse,
            rolling_sharpe_slope=float(slope),
            rolling_sharpe_slope_p=slope_p,
            recent_max_dd=recent_dd,
            historical_max_dd=historical_dd,
            dd_worse_than_historical=dd_worse,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _severity(tau: float, trend_p: float, mw_p: float) -> DecaySeverity:
        if tau <= _TAU_STRONG and trend_p < _TREND_P_STRONG:
            return DecaySeverity.STRONG_DECAY
        if tau <= _TAU_MODERATE and mw_p < _MW_P_MODERATE:
            return DecaySeverity.MODERATE_DECAY
        if tau <= _TAU_POSSIBLE:
            return DecaySeverity.POSSIBLE_DECAY
        return DecaySeverity.NO_DECAY

    @staticmethod
    def _recommend(
        tau: float,
        trend_p: float,
        mw_p: float,
        slope: float,
    ) -> DecayRecommendation:
        if tau <= _TAU_STRONG and trend_p < _TREND_P_STRONG:
            return DecayRecommendation.RETIRE_STRATEGY
        if tau <= _TAU_MODERATE and mw_p < _MW_P_MODERATE:
            return DecayRecommendation.REDUCE_SIZE_50PCT_AND_INVESTIGATE
        if tau <= _TAU_POSSIBLE or mw_p < 0.10:
            return DecayRecommendation.MONITOR_CLOSELY_WEEKLY_REVIEW
        if slope < _SLOPE_DECAY_RATE:
            return DecayRecommendation.CHECK_REGIME_HYPOTHESIS
        return DecayRecommendation.NO_ACTION_EDGE_STABLE


__all__ = [
    "DecayAssessment",
    "DecayRecommendation",
    "DecaySeverity",
    "EdgeDecayMonitor",
]
