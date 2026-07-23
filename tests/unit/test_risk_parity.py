"""Unit tests — portfolio.risk_parity: weights, diagnostics, rolling fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.portfolio.risk_parity import (
    diagnose_allocation,
    risk_parity_weights,
    rolling_risk_parity_weights,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_returns(n_days: int, n_strategies: int, seed: int = 0) -> pd.DataFrame:
    """Build synthetic daily returns with mild positive drift + heterogeneous vol."""
    rng = np.random.default_rng(seed)
    # Heterogeneous vols from ~0.005 to ~0.025 daily.
    vols = np.linspace(0.005, 0.025, n_strategies)
    data = rng.normal(0.0005, 1.0, size=(n_days, n_strategies)) * vols
    cols = [f"s{i + 1}" for i in range(n_strategies)]
    return pd.DataFrame(data, columns=cols)


def _make_dated_returns(
    n_days: int,
    n_strategies: int,
    seed: int = 0,
) -> pd.DataFrame:
    df = _make_returns(n_days, n_strategies, seed)
    df.index = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return df


# =============================================================================
# Single-strategy and degenerate cases
# =============================================================================


class TestRiskParityWeightsBasic:
    def test_single_strategy_unit_weight(self) -> None:
        returns = _make_returns(252, 1)
        weights = risk_parity_weights(returns)
        assert weights.iloc[0] == pytest.approx(1.0)

    def test_single_strategy_target_vol_scaling(self) -> None:
        returns = _make_returns(252, 1)
        # daily vol ~0.005 → annualized ~0.005*sqrt(252)≈0.079
        weights = risk_parity_weights(returns, target_total_vol=0.10)
        # Single-strategy scaled = target / asset_vol; sign always positive.
        assert weights.iloc[0] > 0

    def test_empty_returns_rejected(self) -> None:
        empty = pd.DataFrame()
        with pytest.raises(AssertionError):
            risk_parity_weights(empty)

    def test_invalid_bounds_rejected(self) -> None:
        returns = _make_returns(252, 3)
        with pytest.raises(AssertionError):
            risk_parity_weights(returns, bounds=(-0.1, 0.40))
        with pytest.raises(AssertionError):
            risk_parity_weights(returns, bounds=(0.40, 0.20))

    def test_invalid_target_vol_rejected(self) -> None:
        returns = _make_returns(252, 3)
        with pytest.raises(AssertionError):
            risk_parity_weights(returns, target_total_vol=-0.1)


# =============================================================================
# Sum-to-1 mode (no target_total_vol)
# =============================================================================


class TestRiskParityWeightsSumToOne:
    def test_weights_sum_to_one(self) -> None:
        returns = _make_returns(400, 4)
        weights = risk_parity_weights(returns)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_respects_bounds(self) -> None:
        # Heterogeneous vol spread → bounds bind on at least one strategy.
        returns = _make_returns(400, 5)
        weights = risk_parity_weights(returns, bounds=(0.05, 0.40))
        assert weights.min() >= 0.05 - 1e-6
        assert weights.max() <= 0.40 + 1e-6

    def test_weights_indexed_by_strategy(self) -> None:
        returns = _make_returns(400, 3)
        weights = risk_parity_weights(returns)
        assert list(weights.index) == ["s1", "s2", "s3"]

    def test_low_vol_strategy_gets_higher_weight(self) -> None:
        """Inverse-vol intuition: lowest-vol strategy gets highest weight."""
        rng = np.random.default_rng(42)
        s1 = rng.normal(0.0, 0.005, 400)  # low vol
        s2 = rng.normal(0.0, 0.020, 400)  # high vol
        returns = pd.DataFrame({"s1": s1, "s2": s2})
        # n=2 → default (0.05, 0.40) infeasible (max sum 0.80); use full range.
        weights = risk_parity_weights(returns, bounds=(0.0, 1.0))
        assert weights["s1"] > weights["s2"]


# =============================================================================
# Equal risk contribution property
# =============================================================================


class TestEqualRiskContribution:
    def test_contributions_approximately_equal(self) -> None:
        """The defining property: per-strategy risk contributions should be ~equal.

        Uses bounds=(0.0, 1.0) so we test the unconstrained equal-contribution
        property; under tight bounds (e.g. 0.40 cap) the optimum is the closest
        feasible point and contributions are only equal in the unconstrained
        directions.
        """
        returns = _make_returns(400, 3, seed=1)
        weights = risk_parity_weights(returns, bounds=(0.0, 1.0))

        cov = returns.cov().values * 252
        diag = diagnose_allocation(weights, cov, returns)
        contribs = diag["contributions"]
        # All contributions within 1% of mean — iterative algorithm converges
        # tightly when bounds aren't binding.
        mean_rc = contribs.mean()
        for rc in contribs:
            assert abs(rc - mean_rc) / abs(mean_rc) < 0.01, (
                f"contribution {rc} far from mean {mean_rc}"
            )

    def test_contributions_sum_to_portfolio_vol(self) -> None:
        """Sum of risk contributions equals portfolio vol (algebraic identity)."""
        returns = _make_returns(400, 4, seed=2)
        weights = risk_parity_weights(returns)
        cov = returns.cov().values * 252
        diag = diagnose_allocation(weights, cov, returns)
        assert diag["contributions"].sum() == pytest.approx(diag["portfolio_vol"])


# =============================================================================
# target_total_vol mode
# =============================================================================


class TestRiskParityTargetVol:
    def test_scaling_hits_target_vol(self) -> None:
        returns = _make_returns(400, 3, seed=3)
        target = 0.12
        weights = risk_parity_weights(returns, target_total_vol=target)
        cov = returns.cov().values * 252
        portfolio_vol = float(np.sqrt(weights.values @ cov @ weights.values))
        assert portfolio_vol == pytest.approx(target, abs=1e-6)

    def test_target_vol_can_produce_leverage(self) -> None:
        """If asset vol < target vol, leveraged sum > 1 is expected."""
        # Build very-low-vol returns; target vol > realized vol.
        rng = np.random.default_rng(4)
        data = rng.normal(0, 0.003, size=(300, 2))  # ~5% annualized each
        returns = pd.DataFrame(data, columns=["s1", "s2"])
        # n=2 → default bounds infeasible; use full range pre-leverage.
        weights = risk_parity_weights(
            returns,
            target_total_vol=0.20,
            bounds=(0.0, 1.0),
        )
        assert weights.sum() > 1.0


# =============================================================================
# Rolling
# =============================================================================


class TestRollingRiskParity:
    def test_returns_dataframe(self) -> None:
        returns = _make_dated_returns(400, 3)
        rolling = rolling_risk_parity_weights(
            returns,
            window_days=252,
            refit_freq_days=21,
        )
        assert isinstance(rolling, pd.DataFrame)
        assert list(rolling.columns) == ["s1", "s2", "s3"]

    def test_refits_at_expected_frequency(self) -> None:
        returns = _make_dated_returns(400, 3)
        rolling = rolling_risk_parity_weights(
            returns,
            window_days=252,
            refit_freq_days=21,
        )
        # Expected refit count: floor((400-252)/21) + 1
        expected = (400 - 252) // 21 + 1
        assert len(rolling) == expected

    def test_insufficient_history_returns_empty(self) -> None:
        returns = _make_dated_returns(50, 3)
        rolling = rolling_risk_parity_weights(returns, window_days=252)
        assert rolling.empty

    def test_each_row_sums_to_one_when_no_target(self) -> None:
        returns = _make_dated_returns(500, 3)
        rolling = rolling_risk_parity_weights(
            returns,
            window_days=252,
            refit_freq_days=42,
            target_vol=None,
        )
        for date, row in rolling.iterrows():
            assert row.sum() == pytest.approx(1.0, abs=1e-6), f"row {date} did not sum to 1"


# =============================================================================
# Hypothesis property tests
# =============================================================================


@given(
    n_strategies=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=20, deadline=None)
def test_weights_always_sum_to_one(n_strategies: int, seed: int) -> None:
    """Sum-to-1 invariant holds across random return generations.

    Uses bounds=(0.0, 1.0) so n=2 cases are feasible (default bounds (0.05, 0.40)
    cap sum at 0.80 for n=2).
    """
    returns = _make_returns(400, n_strategies, seed=seed)
    weights = risk_parity_weights(returns, bounds=(0.0, 1.0))
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


@given(
    n_strategies=st.integers(min_value=3, max_value=6),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=20, deadline=None)
def test_weights_always_within_bounds(n_strategies: int, seed: int) -> None:
    """Per-strategy bounds hold across random return generations.

    Restricted to n_strategies >= 3 because (0.05, 0.40) bounds with sum-to-1
    are infeasible for n=2 (max possible sum 0.80).
    """
    returns = _make_returns(400, n_strategies, seed=seed)
    bounds = (0.05, 0.40)
    weights = risk_parity_weights(returns, bounds=bounds)
    assert weights.min() >= bounds[0] - 1e-6
    assert weights.max() <= bounds[1] + 1e-6


@given(
    target_vol=st.floats(min_value=0.02, max_value=0.40),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=15, deadline=None)
def test_target_vol_achieves_target(target_vol: float, seed: int) -> None:
    """Setting target_total_vol produces a portfolio with that realized vol."""
    returns = _make_returns(400, 3, seed=seed)
    weights = risk_parity_weights(returns, target_total_vol=target_vol)
    cov = returns.cov().values * 252
    realized = float(np.sqrt(weights.values @ cov @ weights.values))
    assert realized == pytest.approx(target_vol, abs=1e-5)
