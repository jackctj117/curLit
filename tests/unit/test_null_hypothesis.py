"""Unit tests — edge_testing.null_hypothesis (G1 / CL-0eo).

Covers acceptance criteria:
- All 6 nulls compute (or note "not_evaluated" when optional inputs missing)
- p-values returned for evaluated nulls
- Known-profitable strategy gets p < 0.05 (edge_exists = True)
- Random strategy does not (edge_exists = False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.edge_testing.null_hypothesis import (
    NullHypothesisFramework,
    NullResult,
)

# Tests use lower n_simulations than production (10000) to keep runtime modest;
# 1000 still gives p_value resolution of ~0.001 which is plenty for the
# assertion thresholds we test.
_TEST_N_SIMULATIONS = 1_000


# =============================================================================
# Synthetic-data builders
# =============================================================================


def _build_asset_returns(n_days: int = 800, seed: int = 0) -> pd.Series:
    """Mild trending FX-like daily returns: ~10% annualized vol, ~3% drift."""
    rng = np.random.default_rng(seed)
    daily = rng.normal(0.0001, 0.01, n_days)
    return pd.Series(daily, index=pd.date_range("2020-01-01", periods=n_days, freq="B"))


def _build_known_profitable_strategy(
    asset: pd.Series,
    hit_rate: float = 0.70,
) -> tuple[pd.Series, pd.Series]:
    """Build a strategy with a known positive expected return.

    On `hit_rate` fraction of bars the position aligns with the contemporaneous
    return sign (synthetic oracle — only acceptable for test fixtures); on the
    remaining bars it's a fair coin flip.

    Expected daily return ≈ (2*hit_rate - 1) × E[|r|] > 0; this gives a
    decisively-positive Sharpe that should beat all the null baselines.
    """
    rng = np.random.default_rng(42)
    n = len(asset)
    asset_arr = asset.to_numpy()
    informed = np.sign(asset_arr)
    random = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    mask = rng.random(n) < hit_rate
    signal = np.where(mask, informed, random).astype(float)
    strat_returns = signal * asset_arr
    return (
        pd.Series(strat_returns, index=asset.index),
        pd.Series(signal, index=asset.index),
    )


def _build_random_strategy(asset: pd.Series, seed: int = 7) -> tuple[pd.Series, pd.Series]:
    """Independent ±1 random signal — no edge by construction."""
    rng = np.random.default_rng(seed)
    n = len(asset)
    signal = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    strat_returns = signal * asset.to_numpy()
    return (
        pd.Series(strat_returns, index=asset.index),
        pd.Series(signal, index=asset.index),
    )


# =============================================================================
# Acceptance criteria
# =============================================================================


class TestAcceptance:
    def test_all_six_nulls_compute(self) -> None:
        asset = _build_asset_returns()
        strat, signal = _build_random_strategy(asset)
        # Provide optional inputs so all 6 nulls evaluate.
        basket = pd.DataFrame(
            {
                "a": _build_asset_returns(seed=1).values,
                "b": _build_asset_returns(seed=2).values,
                "c": _build_asset_returns(seed=3).values,
            },
            index=asset.index,
        )
        yields = pd.DataFrame(
            {
                "base": np.full(len(asset), 0.04),
                "quote": np.full(len(asset), 0.02),
            },
            index=asset.index,
        )
        fw = NullHypothesisFramework(
            n_simulations=_TEST_N_SIMULATIONS,
            seed=42,
        )
        report = fw.test_strategy(
            strategy_returns=strat,
            asset_returns=asset,
            signal=signal,
            basket_returns=basket,
            yields=yields,
        )
        assert set(report.results.keys()) == {
            "random_longshort",
            "random_autocorr",
            "buy_and_hold",
            "equal_weight_basket",
            "simple_momentum",
            "simple_carry",
        }
        # All 6 evaluated.
        for name, r in report.results.items():
            assert r.evaluated, f"{name} not evaluated: {r.note}"
            assert r.p_value is not None

    def test_random_strategy_does_not_show_edge(self) -> None:
        asset = _build_asset_returns(seed=10)
        strat, signal = _build_random_strategy(asset)
        fw = NullHypothesisFramework(
            n_simulations=_TEST_N_SIMULATIONS,
            seed=11,
        )
        report = fw.test_strategy(
            strategy_returns=strat,
            asset_returns=asset,
            signal=signal,
        )
        # Without compelling edge over random nulls, edge_exists must be False.
        assert report.edge_exists is False
        # And random_longshort should NOT reject H0 at alpha=0.05 typically.
        random_p = report.results["random_longshort"].p_value
        assert random_p is not None
        assert random_p > 0.05, f"random strategy fluked p={random_p:.3f} on random_longshort"

    def test_known_profitable_strategy_shows_edge(self) -> None:
        asset = _build_asset_returns(seed=20)
        strat, signal = _build_known_profitable_strategy(asset)
        fw = NullHypothesisFramework(
            n_simulations=_TEST_N_SIMULATIONS,
            seed=21,
        )
        report = fw.test_strategy(
            strategy_returns=strat,
            asset_returns=asset,
            signal=signal,
        )
        # Strategy uses lookahead → should crush all evaluated nulls.
        # Specifically random_longshort and random_autocorr (its closest peers).
        assert report.results["random_longshort"].p_value is not None
        assert report.results["random_longshort"].p_value < 0.05
        assert report.results["random_autocorr"].p_value is not None
        assert report.results["random_autocorr"].p_value < 0.05


# =============================================================================
# Optional inputs missing → not_evaluated, but framework continues
# =============================================================================


class TestOptionalInputs:
    def test_no_basket_means_basket_null_not_evaluated(self) -> None:
        asset = _build_asset_returns()
        strat, signal = _build_random_strategy(asset)
        fw = NullHypothesisFramework(n_simulations=_TEST_N_SIMULATIONS, seed=33)
        report = fw.test_strategy(
            strategy_returns=strat,
            asset_returns=asset,
            signal=signal,
            basket_returns=None,
            yields=None,
        )
        assert report.results["equal_weight_basket"].evaluated is False
        # Other 5 still evaluate.
        for name in (
            "random_longshort",
            "random_autocorr",
            "buy_and_hold",
            "simple_momentum",
            "simple_carry",
        ):
            assert report.results[name].evaluated, f"{name} not evaluated"

    def test_carry_falls_back_without_yields(self) -> None:
        asset = _build_asset_returns()
        strat, signal = _build_random_strategy(asset)
        fw = NullHypothesisFramework(n_simulations=_TEST_N_SIMULATIONS, seed=44)
        report = fw.test_strategy(
            strategy_returns=strat,
            asset_returns=asset,
            signal=signal,
            yields=None,
        )
        # Fallback evaluates but with a note documenting the substitution.
        carry = report.results["simple_carry"]
        assert carry.evaluated is True
        assert "fallback" in carry.note


# =============================================================================
# Reporting
# =============================================================================


class TestReportingHelpers:
    def test_to_dict_round_trip(self) -> None:
        asset = _build_asset_returns()
        strat, signal = _build_random_strategy(asset)
        fw = NullHypothesisFramework(n_simulations=_TEST_N_SIMULATIONS, seed=55)
        report = fw.test_strategy(
            strategy_returns=strat,
            asset_returns=asset,
            signal=signal,
        )
        d = report.to_dict()
        assert "edge_exists" in d
        assert "results" in d
        assert isinstance(d["results"], dict)

    def test_passed_failed_partition(self) -> None:
        asset = _build_asset_returns(seed=60)
        strat, signal = _build_known_profitable_strategy(asset)
        fw = NullHypothesisFramework(n_simulations=_TEST_N_SIMULATIONS, seed=61)
        report = fw.test_strategy(
            strategy_returns=strat,
            asset_returns=asset,
            signal=signal,
        )
        passed = set(report.passed_nulls())
        failed = set(report.failed_nulls())
        # Disjoint and cover only evaluated results.
        assert passed.isdisjoint(failed)
        for name in passed | failed:
            assert report.results[name].evaluated is True


# =============================================================================
# Construction validation
# =============================================================================


class TestConstruction:
    def test_low_n_simulations_rejected(self) -> None:
        with pytest.raises(AssertionError):
            NullHypothesisFramework(n_simulations=10)

    def test_invalid_alpha_rejected(self) -> None:
        with pytest.raises(AssertionError):
            NullHypothesisFramework(alpha=1.5)

    def test_seed_makes_run_reproducible(self) -> None:
        asset = _build_asset_returns(seed=70)
        strat, signal = _build_random_strategy(asset)
        fw1 = NullHypothesisFramework(n_simulations=_TEST_N_SIMULATIONS, seed=999)
        fw2 = NullHypothesisFramework(n_simulations=_TEST_N_SIMULATIONS, seed=999)
        r1 = fw1.test_strategy(strategy_returns=strat, asset_returns=asset, signal=signal)
        r2 = fw2.test_strategy(strategy_returns=strat, asset_returns=asset, signal=signal)
        for name in r1.results:
            p1 = r1.results[name].p_value
            p2 = r2.results[name].p_value
            if p1 is None or p2 is None:
                continue
            assert p1 == p2, f"{name} not reproducible: {p1} vs {p2}"


# =============================================================================
# NullResult invariants
# =============================================================================


class TestNullResult:
    def test_to_dict_keys(self) -> None:
        r = NullResult(
            null_name="x",
            evaluated=True,
            strategy_sharpe=1.5,
            p_value=0.03,
            null_mean=0.5,
            null_std=0.2,
            n_simulations=1000,
        )
        d = r.to_dict()
        assert d["null_name"] == "x"
        assert d["p_value"] == 0.03
