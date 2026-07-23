"""Multiple testing correction (G2 / CL-311) — Bonferroni, BH, White's Reality Check.

Picking the best strategy out of N tested is data snooping unless you correct
for the multiplicity of tests. This module supplies three corrections:

    Bonferroni:           p_corrected = min(N × p, 1). Controls family-wise
                          error rate (FWER); very conservative for large N.

    Benjamini-Hochberg:   Controls expected false-discovery rate (FDR) instead
                          of FWER. Less conservative than Bonferroni; the
                          standard choice when you're willing to accept a
                          small fraction of false positives.

    White's Reality Check:Bootstrap-based test of the BEST strategy vs benchmark.
                          Stationary block bootstrap preserves serial correlation,
                          critical for time-series returns. Answers: "is the
                          best-ranked strategy genuinely outperforming or did it
                          win the lottery against the rest?"

Reference: reference/14_edge_testing.md, White (2000) "A Reality Check for Data
Snooping", Benjamini & Hochberg (1995).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NDArray = np.ndarray[Any, Any]

# Block mean length for stationary bootstrap on returns. Same convention as
# null_hypothesis.py — ~1 month captures FX regime persistence.
_BOOTSTRAP_BLOCK_MEAN_LEN: int = 20

# Default bootstrap iterations for White's Reality Check. CV(p) ~ 1/sqrt(N);
# 5000 gives p resolution of ~0.014, enough to distinguish significant from
# borderline at alpha=0.05 without bloating runtime.
_DEFAULT_RC_BOOTSTRAP: int = 5_000


# =============================================================================
# Correction methods
# =============================================================================


def bonferroni(
    p_values: list[float] | _NDArray,
    alpha: float = 0.05,
) -> tuple[_NDArray, _NDArray]:
    """Bonferroni correction.

    Returns (corrected_p_values, reject_mask) where corrected = min(N × p, 1)
    and reject = corrected < alpha.
    """
    assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"
    arr = np.asarray(p_values, dtype=float)
    assert np.all((arr >= 0) & (arr <= 1)), "p-values must be in [0, 1]"
    n = len(arr)
    if n == 0:
        return arr, np.array([], dtype=bool)
    corrected = np.minimum(arr * n, 1.0)
    reject = corrected < alpha
    return corrected, reject


def benjamini_hochberg(
    p_values: list[float] | _NDArray,
    alpha: float = 0.05,
) -> tuple[_NDArray, _NDArray]:
    """Benjamini-Hochberg FDR correction (BH step-up procedure).

    Returns (adjusted_p_values, reject_mask). The adjusted p-value at rank k
    is min over j>=k of (p_(j) × N / j), preserving monotonicity. Reject if
    adjusted < alpha.

    Adjusted p-values are returned in the original input order.
    """
    assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"
    arr = np.asarray(p_values, dtype=float)
    assert np.all((arr >= 0) & (arr <= 1)), "p-values must be in [0, 1]"
    n = len(arr)
    if n == 0:
        return arr, np.array([], dtype=bool)

    # Sort ascending; remember original positions.
    order = np.argsort(arr)
    sorted_p = arr[order]
    ranks = np.arange(1, n + 1)
    raw_adj = sorted_p * n / ranks
    # Monotone non-decreasing adjusted p-values via running min from the right.
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    # Restore original order.
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adj_sorted
    reject = adjusted < alpha
    return adjusted, reject


# =============================================================================
# White's Reality Check
# =============================================================================


@dataclass
class RealityCheckResult:
    """White's Reality Check output."""

    best_strategy: str
    best_excess_mean: float
    p_value: float
    n_strategies: int
    n_bootstrap: int
    bootstrap_distribution: _NDArray = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_strategy": self.best_strategy,
            "best_excess_mean": self.best_excess_mean,
            "p_value": self.p_value,
            "n_strategies": self.n_strategies,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_quantiles": {
                "q05": float(np.quantile(self.bootstrap_distribution, 0.05)),
                "q50": float(np.quantile(self.bootstrap_distribution, 0.50)),
                "q95": float(np.quantile(self.bootstrap_distribution, 0.95)),
            },
        }


def whites_reality_check(
    returns_matrix: pd.DataFrame,
    benchmark_returns: pd.Series,
    n_bootstrap: int = _DEFAULT_RC_BOOTSTRAP,
    block_mean_len: int = _BOOTSTRAP_BLOCK_MEAN_LEN,
    seed: int | None = None,
) -> RealityCheckResult:
    """White's (2000) Reality Check for data snooping.

    Args:
        returns_matrix: T × N DataFrame of strategy returns; columns are
                        strategy ids.
        benchmark_returns: T-length Series; the benchmark each strategy is
                           evaluated against.
        n_bootstrap: number of stationary block bootstrap resamples.
        block_mean_len: stationary bootstrap block mean length in periods.
        seed: optional RNG seed for reproducibility.

    Returns:
        RealityCheckResult with p_value of the BEST strategy.

    p_value interpretation: probability under H0 (no strategy outperforms
    benchmark) of observing a best-strategy excess at least as large as the
    one we did. Small p means the best is unlikely to be a fluke.
    """
    assert not returns_matrix.empty, "returns_matrix must be non-empty"
    assert returns_matrix.shape[1] >= 1, "need at least one strategy column"
    assert n_bootstrap >= 100, f"n_bootstrap must be >= 100, got {n_bootstrap}"

    aligned = returns_matrix.align(benchmark_returns, join="inner", axis=0)
    strategies_df, bench = aligned
    assert isinstance(strategies_df, pd.DataFrame)
    assert isinstance(bench, pd.Series)
    if strategies_df.empty:
        msg = "returns_matrix and benchmark_returns share no overlapping index"
        raise ValueError(msg)

    excess = strategies_df.subtract(bench, axis=0)
    # Drop columns or rows that are all-NaN to avoid degenerate bootstrap.
    excess = excess.dropna(how="all", axis=1).dropna(how="all", axis=0)
    if excess.empty:
        msg = "after alignment, no usable excess returns remain"
        raise ValueError(msg)

    excess_arr = excess.fillna(0.0).to_numpy()
    means = excess_arr.mean(axis=0)
    best_idx = int(np.argmax(means))
    best_strategy = str(excess.columns[best_idx])
    best_excess_mean = float(means[best_idx])

    rng = np.random.default_rng(seed)
    n_periods = excess_arr.shape[0]
    p_geom = 1.0 / block_mean_len

    # Mean-center: under H0, all strategies have zero excess mean. The bootstrap
    # samples should reflect that, so we subtract each strategy's empirical mean
    # before resampling. The max across strategies of the centered bootstrap
    # mean is the H0 distribution of the best statistic.
    centered = excess_arr - means

    bootstrap_max = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        indices = np.empty(n_periods, dtype=np.int64)
        i = int(rng.integers(0, n_periods))
        for k in range(n_periods):
            indices[k] = i
            i = int(rng.integers(0, n_periods)) if rng.random() < p_geom else (i + 1) % n_periods
        sample = centered[indices]
        bootstrap_max[b] = float(np.max(sample.mean(axis=0)))

    # p_value = P(bootstrap max >= observed best excess).
    p_value = float((np.sum(bootstrap_max >= best_excess_mean) + 1) / (n_bootstrap + 1))

    return RealityCheckResult(
        best_strategy=best_strategy,
        best_excess_mean=best_excess_mean,
        p_value=p_value,
        n_strategies=excess.shape[1],
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=bootstrap_max,
    )


# =============================================================================
# Convenience wrapper
# =============================================================================


@dataclass
class MultipleTestingReport:
    """Aggregated correction results across all three methods."""

    p_values_input: list[float]
    bonferroni_corrected: list[float]
    bonferroni_reject: list[bool]
    bh_adjusted: list[float]
    bh_reject: list[bool]
    reality_check: RealityCheckResult | None
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "p_values_input": self.p_values_input,
            "bonferroni_corrected": self.bonferroni_corrected,
            "bonferroni_reject": self.bonferroni_reject,
            "bh_adjusted": self.bh_adjusted,
            "bh_reject": self.bh_reject,
            "reality_check": (self.reality_check.to_dict() if self.reality_check else None),
        }


class MultipleTestingCorrection:
    """Apply Bonferroni + BH + (optionally) White's Reality Check together."""

    def __init__(self, alpha: float = 0.05) -> None:
        assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"
        self.alpha = alpha

    def correct(
        self,
        p_values: list[float] | _NDArray,
        returns_matrix: pd.DataFrame | None = None,
        benchmark_returns: pd.Series | None = None,
        n_bootstrap: int = _DEFAULT_RC_BOOTSTRAP,
        seed: int | None = None,
    ) -> MultipleTestingReport:
        """Apply all corrections; reality check runs only if returns provided.

        p_values: per-strategy individual-test p-values (e.g. from G1 nulls).
        returns_matrix + benchmark_returns: optional inputs to enable Reality Check.
        """
        bonf_corrected, bonf_reject = bonferroni(p_values, alpha=self.alpha)
        bh_adj, bh_reject = benjamini_hochberg(p_values, alpha=self.alpha)

        rc: RealityCheckResult | None = None
        if returns_matrix is not None and benchmark_returns is not None:
            rc = whites_reality_check(
                returns_matrix,
                benchmark_returns,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )

        return MultipleTestingReport(
            p_values_input=list(np.asarray(p_values, dtype=float)),
            bonferroni_corrected=list(bonf_corrected),
            bonferroni_reject=list(bonf_reject),
            bh_adjusted=list(bh_adj),
            bh_reject=list(bh_reject),
            reality_check=rc,
            alpha=self.alpha,
        )
