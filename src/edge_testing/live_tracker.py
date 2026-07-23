"""LiveEdgeTracker (G3 / CL-2py) — continuous live-vs-backtest divergence monitor.

Once a strategy is live, its realized performance must be continuously compared
against the backtest's expectations. If reality diverges from the backtest, we
need to *act* — not just measure. This module enforces that consequence.

Two divergence tests run on every assessment:

    Sharpe Z-score test:
        Under H0 that the true Sharpe equals the backtest expectation, the
        sample Sharpe is approximately Normal with standard error
            σ(Ŝ) ≈ sqrt((1 + Ŝ²/2) / N)
        derived from Lo (2002), "The Statistics of Sharpe Ratios". A negative
        z-score means we underperformed; the magnitude says by how many σ.

    Hit-rate binomial test:
        Under H0 that the true win rate equals the backtest expectation, the
        observed hit count is Binomial(N, p_expected). One-sided lower-tail
        p-value answers "how likely is this hit rate or worse under H0?"

Severity buckets cross-reference both tests with a hard floor on negative
mean returns:

    on_track                          neither test rejects at α
    underperforming                   one test rejects at α
    significantly_underperforming     both tests reject at α
    severely_underperforming          both reject at α/5 OR live mean return < 0

Recommendation maps from severity:

    on_track          → continue
    underperforming   → review
    significantly_… → reduce_size_50pct
    severely_…      → halt_and_investigate

Pre-committed actions in `edge_policy.yaml` (G9 / CL-337) read these
recommendations and execute them automatically. This module is the *trigger*;
the policy file is the *action*.

A minimum live-history threshold (default 60 trading days) gates assessment —
short windows produce too-noisy z-scores to be actionable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.monitoring.metrics import edge_severity

logger = logging.getLogger(__name__)


# Default significance level for the per-test divergence checks. Same as the
# G1 framework; the multiple-testing layer (G2) corrects across strategies.
_DEFAULT_ALPHA: float = 0.05

# Tighter threshold used to escalate from "significantly" to "severely"
# underperforming. Both tests rejecting at α/5 = 0.01 is a strong signal.
_SEVERE_ALPHA_RATIO: float = 5.0

# Minimum live-trading days before we trust the divergence test. Below this,
# the SE on Sharpe is too wide to be informative — we just hold-and-watch.
# 60 days roughly equals 3 calendar months; long enough to span typical
# regime persistence in FX returns.
_DEFAULT_MIN_LIVE_DAYS: int = 60

# Trading days per year for annualizing Sharpe.
_TRADING_DAYS_PER_YEAR: int = 252


class Severity(Enum):
    ON_TRACK = "on_track"
    UNDERPERFORMING = "underperforming"
    SIGNIFICANTLY_UNDERPERFORMING = "significantly_underperforming"
    SEVERELY_UNDERPERFORMING = "severely_underperforming"


# Numeric encoding for the Prometheus gauge — readable thresholds in alerts.
_SEVERITY_RANK: dict[Severity, int] = {
    Severity.ON_TRACK: 0,
    Severity.UNDERPERFORMING: 1,
    Severity.SIGNIFICANTLY_UNDERPERFORMING: 2,
    Severity.SEVERELY_UNDERPERFORMING: 3,
}


class Recommendation(Enum):
    CONTINUE = "continue"
    REVIEW = "review"
    REDUCE_SIZE_50PCT = "reduce_size_50pct"
    HALT_AND_INVESTIGATE = "halt_and_investigate"


_SEVERITY_TO_RECOMMENDATION: dict[Severity, Recommendation] = {
    Severity.ON_TRACK: Recommendation.CONTINUE,
    Severity.UNDERPERFORMING: Recommendation.REVIEW,
    Severity.SIGNIFICANTLY_UNDERPERFORMING: Recommendation.REDUCE_SIZE_50PCT,
    Severity.SEVERELY_UNDERPERFORMING: Recommendation.HALT_AND_INVESTIGATE,
}


@dataclass
class BacktestExpectations:
    """Expected metrics from the strategy's backtest, used as null hypothesis."""

    sharpe: float
    hit_rate: float
    mean_return: float
    vol: float
    n_days_backtest: int = _TRADING_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        assert 0 <= self.hit_rate <= 1, f"hit_rate must be in [0, 1], got {self.hit_rate}"
        assert self.vol >= 0, f"vol must be non-negative, got {self.vol}"
        assert self.n_days_backtest >= 1, "n_days_backtest must be >= 1"


@dataclass
class LiveEdgeAssessment:
    """Output of a single LiveEdgeTracker.assess() call."""

    strategy_id: str
    ts: datetime
    live_n_days: int
    live_sharpe: float
    live_hit_rate: float
    live_mean_return: float
    sharpe_z_score: float
    sharpe_p_value: float
    hit_rate_p_value: float
    severity: Severity
    recommendation: Recommendation
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "ts": self.ts.isoformat(),
            "live_n_days": self.live_n_days,
            "live_sharpe": self.live_sharpe,
            "live_hit_rate": self.live_hit_rate,
            "live_mean_return": self.live_mean_return,
            "sharpe_z_score": self.sharpe_z_score,
            "sharpe_p_value": self.sharpe_p_value,
            "hit_rate_p_value": self.hit_rate_p_value,
            "severity": self.severity.value,
            "recommendation": self.recommendation.value,
            "note": self.note,
        }


# =============================================================================
# Statistics
# =============================================================================


def _sharpe(returns: np.ndarray[Any, Any], periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe (zero risk-free)."""
    if len(returns) == 0:
        return 0.0
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(returns) / sd * math.sqrt(periods_per_year))


def _sharpe_se_lo(sharpe: float, n_obs: int) -> float:
    """Standard error of an annualized Sharpe per Lo (2002).

    Assumes IID returns. For dependent returns, the SE is wider; this is the
    conservative (smaller-SE) approximation that makes the test more sensitive
    — fine for our use because we're looking for *underperformance*, where a
    smaller SE means we reject earlier.
    """
    assert n_obs >= 2, f"need at least 2 observations, got {n_obs}"
    return math.sqrt((1.0 + 0.5 * sharpe**2) / n_obs)


def _sharpe_z_test(
    live_sharpe: float,
    expected_sharpe: float,
    n_obs: int,
) -> tuple[float, float]:
    """Return (z-score, one-sided p-value for live < expected)."""
    se = _sharpe_se_lo(expected_sharpe, n_obs)
    if se == 0.0:
        return 0.0, 1.0
    z = (live_sharpe - expected_sharpe) / se
    # One-sided lower-tail: P(Z <= z) under standard normal.
    p = float(stats.norm.cdf(z))
    return z, p


def _hit_rate_binomial_test(
    n_hits: int,
    n_obs: int,
    expected_hit_rate: float,
) -> float:
    """One-sided lower-tail binomial p-value: P(X <= n_hits | p = expected)."""
    assert 0 <= expected_hit_rate <= 1
    assert 0 <= n_hits <= n_obs
    if n_obs == 0:
        return 1.0
    return float(stats.binom.cdf(n_hits, n_obs, expected_hit_rate))


# =============================================================================
# Tracker
# =============================================================================


class LiveEdgeTracker:
    """Compare live performance to backtest expectations and gate consequences.

    Build one tracker per strategy. Call assess(live_returns) periodically
    (typically nightly) — it computes both divergence tests, decides severity,
    publishes the Prometheus gauge, and returns a structured assessment that
    the edge policy (G9) can act on.
    """

    def __init__(
        self,
        strategy_id: str,
        backtest: BacktestExpectations,
        min_live_days: int = _DEFAULT_MIN_LIVE_DAYS,
        alpha: float = _DEFAULT_ALPHA,
    ) -> None:
        assert strategy_id, "strategy_id must be non-empty"
        assert min_live_days >= 1, "min_live_days must be >= 1"
        assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"
        self.strategy_id = strategy_id
        self.backtest = backtest
        self.min_live_days = min_live_days
        self.alpha = alpha

    def assess(
        self,
        live_returns: pd.Series | np.ndarray[Any, Any],
    ) -> LiveEdgeAssessment:
        """Run both divergence tests, classify severity, publish metric.

        live_returns: T-length array of daily fractional returns since the
        strategy went live. Need at least min_live_days to produce a
        non-trivial assessment.
        """
        arr = np.asarray(live_returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)
        ts = datetime.now(UTC)

        if n < self.min_live_days:
            assessment = LiveEdgeAssessment(
                strategy_id=self.strategy_id,
                ts=ts,
                live_n_days=n,
                live_sharpe=0.0,
                live_hit_rate=0.0,
                live_mean_return=0.0,
                sharpe_z_score=0.0,
                sharpe_p_value=1.0,
                hit_rate_p_value=1.0,
                severity=Severity.ON_TRACK,
                recommendation=Recommendation.CONTINUE,
                note=f"insufficient history: {n} < {self.min_live_days} days",
            )
            self._publish_metric(assessment)
            return assessment

        live_sharpe = _sharpe(arr)
        n_hits = int(np.sum(arr > 0))
        live_hit_rate = float(n_hits / n)
        live_mean_return = float(np.mean(arr))

        z_score, sharpe_p = _sharpe_z_test(
            live_sharpe,
            self.backtest.sharpe,
            n,
        )
        hit_p = _hit_rate_binomial_test(n_hits, n, self.backtest.hit_rate)

        severity = self._classify_severity(
            sharpe_p=sharpe_p,
            hit_p=hit_p,
            live_mean_return=live_mean_return,
        )
        recommendation = _SEVERITY_TO_RECOMMENDATION[severity]

        assessment = LiveEdgeAssessment(
            strategy_id=self.strategy_id,
            ts=ts,
            live_n_days=n,
            live_sharpe=live_sharpe,
            live_hit_rate=live_hit_rate,
            live_mean_return=live_mean_return,
            sharpe_z_score=z_score,
            sharpe_p_value=sharpe_p,
            hit_rate_p_value=hit_p,
            severity=severity,
            recommendation=recommendation,
        )

        logger.info(
            "Live edge %s: severity=%s recommendation=%s "
            "(sharpe %.2f vs %.2f, p=%.3f; hit %.3f vs %.3f, p=%.3f)",
            self.strategy_id,
            severity.value,
            recommendation.value,
            live_sharpe,
            self.backtest.sharpe,
            sharpe_p,
            live_hit_rate,
            self.backtest.hit_rate,
            hit_p,
        )

        self._publish_metric(assessment)
        return assessment

    def _classify_severity(
        self,
        sharpe_p: float,
        hit_p: float,
        live_mean_return: float,
    ) -> Severity:
        sharpe_rejects = sharpe_p < self.alpha
        hit_rejects = hit_p < self.alpha
        severe_alpha = self.alpha / _SEVERE_ALPHA_RATIO
        sharpe_severe = sharpe_p < severe_alpha
        hit_severe = hit_p < severe_alpha

        # Severe gate: both tests reject hard, OR live mean return is negative
        # (mean < 0 is "we're losing money", regardless of test power).
        if (sharpe_severe and hit_severe) or live_mean_return < 0:
            return Severity.SEVERELY_UNDERPERFORMING
        if sharpe_rejects and hit_rejects:
            return Severity.SIGNIFICANTLY_UNDERPERFORMING
        if sharpe_rejects or hit_rejects:
            return Severity.UNDERPERFORMING
        return Severity.ON_TRACK

    def _publish_metric(self, assessment: LiveEdgeAssessment) -> None:
        try:
            edge_severity.labels(strategy_id=self.strategy_id).set(
                _SEVERITY_RANK[assessment.severity],
            )
        except Exception:
            logger.exception("edge_severity gauge update failed")
