"""Unit tests — edge_testing.multiple_testing (G2 / CL-311).

Bonferroni + BH + White's Reality Check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.edge_testing.multiple_testing import (
    MultipleTestingCorrection,
    benjamini_hochberg,
    bonferroni,
    whites_reality_check,
)

# =============================================================================
# Bonferroni
# =============================================================================


class TestBonferroni:
    def test_simple_correction(self) -> None:
        # p=[0.01, 0.04, 0.10] with N=3 → corrected = [0.03, 0.12, 0.30]
        corrected, reject = bonferroni([0.01, 0.04, 0.10], alpha=0.05)
        assert corrected[0] == pytest.approx(0.03)
        assert corrected[1] == pytest.approx(0.12)
        assert corrected[2] == pytest.approx(0.30)
        # Only the first survives at alpha=0.05.
        assert reject.tolist() == [True, False, False]

    def test_capped_at_one(self) -> None:
        corrected, _ = bonferroni([0.5, 0.6, 0.7], alpha=0.05)
        assert corrected.max() <= 1.0

    def test_empty_input(self) -> None:
        corrected, reject = bonferroni([], alpha=0.05)
        assert len(corrected) == 0
        assert len(reject) == 0

    def test_invalid_p_value_rejected(self) -> None:
        with pytest.raises(AssertionError):
            bonferroni([0.5, 1.5], alpha=0.05)

    def test_invalid_alpha_rejected(self) -> None:
        with pytest.raises(AssertionError):
            bonferroni([0.5], alpha=1.5)

    def test_more_tests_more_conservative(self) -> None:
        """A given p-value gets harder to reject as N grows."""
        corrected_n3, _ = bonferroni([0.02, 0.5, 0.5], alpha=0.05)
        corrected_n10, _ = bonferroni([0.02] + [0.5] * 9, alpha=0.05)
        # 0.02 corrected: 0.06 (N=3) vs 0.20 (N=10).
        assert corrected_n3[0] < corrected_n10[0]


# =============================================================================
# Benjamini-Hochberg
# =============================================================================


class TestBenjaminiHochberg:
    def test_paper_example(self) -> None:
        """BH (1995) paper example yields known critical p-values."""
        # All p-values < their threshold → all rejected.
        p_values = [0.0001, 0.0004, 0.0019, 0.0095]
        adjusted, reject = benjamini_hochberg(p_values, alpha=0.05)
        # Sanity: every p_i × N / rank_i < 0.05 here, so all reject.
        assert all(reject)

    def test_only_smallest_rejected(self) -> None:
        # p-values where BH rejects only 1.
        p_values = [0.001, 0.10, 0.20, 0.50]
        adjusted, reject = benjamini_hochberg(p_values, alpha=0.05)
        assert reject[0]
        assert not reject[1]
        assert not reject[2]
        assert not reject[3]

    def test_monotone_adjusted_p(self) -> None:
        """Adjusted p-values must be monotone non-decreasing in original p order."""
        p_values = [0.001, 0.005, 0.02, 0.04, 0.30]
        adjusted, _ = benjamini_hochberg(p_values, alpha=0.05)
        # Sort to verify monotonicity in sorted order.
        order = np.argsort(p_values)
        sorted_adj = np.asarray(adjusted)[order]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-12

    def test_less_conservative_than_bonferroni(self) -> None:
        """BH should reject at least as many hypotheses as Bonferroni."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.50]
        _, bonf_reject = bonferroni(p_values, alpha=0.05)
        _, bh_reject = benjamini_hochberg(p_values, alpha=0.05)
        assert int(np.sum(bh_reject)) >= int(np.sum(bonf_reject))

    def test_empty_input(self) -> None:
        adjusted, reject = benjamini_hochberg([], alpha=0.05)
        assert len(adjusted) == 0
        assert len(reject) == 0


# =============================================================================
# White's Reality Check
# =============================================================================


def _build_random_strategies(
    n_periods: int = 800,
    n_strategies: int = 10,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    bench = pd.Series(
        rng.normal(0.0001, 0.01, n_periods),
        index=pd.date_range("2020-01-01", periods=n_periods, freq="B"),
        name="bench",
    )
    cols = {f"strat_{i}": rng.normal(0.0001, 0.01, n_periods) for i in range(n_strategies)}
    df = pd.DataFrame(cols, index=bench.index)
    return df, bench


def _build_one_winning_strategy(
    n_periods: int = 800,
    n_strategies: int = 10,
    seed: int = 0,
    edge_per_period: float = 0.0008,
) -> tuple[pd.DataFrame, pd.Series]:
    """One strategy has a small daily-mean edge over benchmark; others are random."""
    rng = np.random.default_rng(seed)
    bench = pd.Series(
        rng.normal(0.0001, 0.01, n_periods),
        index=pd.date_range("2020-01-01", periods=n_periods, freq="B"),
        name="bench",
    )
    cols: dict[str, np.ndarray] = {}
    for i in range(n_strategies):
        if i == 0:
            cols[f"strat_{i}"] = (
                bench.values
                + edge_per_period
                + rng.normal(
                    0,
                    0.005,
                    n_periods,
                )
            )
        else:
            cols[f"strat_{i}"] = rng.normal(0.0001, 0.01, n_periods)
    df = pd.DataFrame(cols, index=bench.index)
    return df, bench


class TestWhitesRealityCheck:
    def test_random_strategies_yield_high_p(self) -> None:
        # Use 5 strategies (smaller pool → tighter null distribution) and a
        # seed that produces a typical (non-fluke) random outcome.
        df, bench = _build_random_strategies(n_strategies=5, seed=10)
        result = whites_reality_check(df, bench, n_bootstrap=500, seed=11)
        # No real edge → expect p well above alpha; allow some bootstrap noise.
        assert result.p_value > 0.10, f"random strategies fluked p={result.p_value:.3f}"

    def test_winning_strategy_yields_low_p(self) -> None:
        # Edge of 0.003/period is roughly 8 standard errors over 800 periods —
        # decisively detectable above the noise from 9 random competitors.
        df, bench = _build_one_winning_strategy(
            seed=20,
            edge_per_period=0.003,
        )
        result = whites_reality_check(df, bench, n_bootstrap=500, seed=21)
        assert result.p_value < 0.05, f"winning strategy missed; p={result.p_value:.3f}"
        # Winner should be strat_0 (the engineered one).
        assert result.best_strategy == "strat_0"

    def test_p_value_reflects_n_strategies(self) -> None:
        """With more strategies tested, the same observed edge gets a HIGHER p
        (because the chance of any random strategy winning by luck grows).
        """
        # Use a deterministic edge magnitude where neither test rejects, so we
        # can compare p_values across N. Fix benchmark + winner; vary the
        # number of additional random strategies.
        rng = np.random.default_rng(30)
        n = 800
        bench = pd.Series(
            rng.normal(0.0001, 0.01, n),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
            name="bench",
        )
        winner = bench + 0.0005 + rng.normal(0, 0.01, n)
        small = pd.DataFrame(
            {"winner": winner.values, "r1": rng.normal(0.0001, 0.01, n)},
            index=bench.index,
        )
        large_extra = {f"r{i}": rng.normal(0.0001, 0.01, n) for i in range(2, 30)}
        large = pd.concat([small, pd.DataFrame(large_extra, index=bench.index)], axis=1)
        p_small = whites_reality_check(small, bench, n_bootstrap=300, seed=31).p_value
        p_large = whites_reality_check(large, bench, n_bootstrap=300, seed=31).p_value
        # More strategies → harder to reject, so p_large >= p_small (allow tiny
        # bootstrap noise).
        assert p_large >= p_small - 0.05

    def test_empty_returns_rejected(self) -> None:
        with pytest.raises(AssertionError):
            whites_reality_check(
                pd.DataFrame(),
                pd.Series(dtype=float),
                n_bootstrap=100,
                seed=0,
            )

    def test_low_bootstrap_rejected(self) -> None:
        df, bench = _build_random_strategies()
        with pytest.raises(AssertionError):
            whites_reality_check(df, bench, n_bootstrap=10, seed=0)

    def test_index_misalignment_raises(self) -> None:
        df, _ = _build_random_strategies(seed=40)
        # Mismatched bench index → no overlap.
        bad_bench = pd.Series(
            np.zeros(10),
            index=pd.date_range("2030-01-01", periods=10, freq="B"),
        )
        with pytest.raises(ValueError, match="overlapping"):
            whites_reality_check(df, bad_bench, n_bootstrap=100, seed=0)


# =============================================================================
# MultipleTestingCorrection wrapper
# =============================================================================


class TestWrapper:
    def test_runs_all_methods(self) -> None:
        df, bench = _build_random_strategies(seed=50)
        mtc = MultipleTestingCorrection(alpha=0.05)
        report = mtc.correct(
            p_values=[0.001, 0.05, 0.5],
            returns_matrix=df,
            benchmark_returns=bench,
            n_bootstrap=300,
            seed=51,
        )
        assert len(report.bonferroni_corrected) == 3
        assert len(report.bh_adjusted) == 3
        assert report.reality_check is not None

    def test_runs_without_reality_check_when_no_returns(self) -> None:
        mtc = MultipleTestingCorrection(alpha=0.05)
        report = mtc.correct(p_values=[0.01, 0.04, 0.10])
        assert report.reality_check is None
        assert len(report.bonferroni_corrected) == 3

    def test_report_to_dict(self) -> None:
        mtc = MultipleTestingCorrection()
        report = mtc.correct(p_values=[0.01, 0.10])
        d = report.to_dict()
        assert d["alpha"] == 0.05
        assert "bonferroni_corrected" in d
        assert "bh_adjusted" in d
        assert d["reality_check"] is None
