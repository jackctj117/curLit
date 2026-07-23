"""Unit tests — edge_testing.decay_detection (G7 / CL-675)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.edge_testing.decay_detection import (
    DecayRecommendation,
    DecaySeverity,
    EdgeDecayMonitor,
)

# =============================================================================
# Synthetic returns builders
# =============================================================================


def _stable_returns(
    n_days: int = 500, mean: float = 0.0006, vol: float = 0.006, seed: int = 0
) -> pd.Series:
    """Stationary returns with constant mean — should NOT decay."""
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(mean, vol, n_days),
        index=pd.date_range("2024-01-01", periods=n_days, freq="B"),
    )


def _decaying_returns(
    n_days: int = 500,
    vol: float = 0.003,
    seed: int = 0,
) -> pd.Series:
    """Returns whose mean degrades linearly from strongly positive to negative.

    Larger swing + smaller noise so the quarterly Sharpes are reliably
    monotone-decreasing despite finite-sample variance.
    """
    rng = np.random.default_rng(seed)
    # 0.0025 → -0.0015 daily means swing through 4 quarters; small vol so
    # the trend dominates the noise floor.
    means = np.linspace(0.0025, -0.0015, n_days)
    return pd.Series(
        rng.normal(0.0, vol, n_days) + means,
        index=pd.date_range("2024-01-01", periods=n_days, freq="B"),
    )


def _abrupt_drop_returns(
    n_days: int = 500,
    drop_at: int = 400,
    seed: int = 0,
) -> pd.Series:
    """Returns that flip from positive to strongly negative at `drop_at`.

    Should trigger the Mann-Whitney recent-vs-historical test.
    """
    rng = np.random.default_rng(seed)
    arr = np.empty(n_days)
    arr[:drop_at] = rng.normal(0.0008, 0.006, drop_at)
    arr[drop_at:] = rng.normal(-0.002, 0.006, n_days - drop_at)
    return pd.Series(
        arr,
        index=pd.date_range("2024-01-01", periods=n_days, freq="B"),
    )


# =============================================================================
# Insufficient history
# =============================================================================


class TestInsufficientHistory:
    def test_below_min_returns_unevaluated(self) -> None:
        monitor = EdgeDecayMonitor()
        ret = _stable_returns(n_days=100)
        result = monitor.check_decay(ret)
        assert result.evaluated is False
        assert result.severity == DecaySeverity.NO_DECAY
        assert result.recommendation == DecayRecommendation.NO_ACTION_EDGE_STABLE
        assert "insufficient history" in result.note

    def test_evaluates_at_threshold(self) -> None:
        monitor = EdgeDecayMonitor(min_history_days=200)
        ret = _stable_returns(n_days=200)
        result = monitor.check_decay(ret)
        assert result.evaluated is True


# =============================================================================
# Stable returns → no decay
# =============================================================================


class TestNoDecay:
    def test_stable_returns_classified_no_decay(self) -> None:
        monitor = EdgeDecayMonitor()
        ret = _stable_returns(n_days=500, seed=10)
        result = monitor.check_decay(ret)
        # Severity should be NO_DECAY or at most POSSIBLE_DECAY due to noise.
        # Recommendation in that range.
        assert result.severity in {DecaySeverity.NO_DECAY, DecaySeverity.POSSIBLE_DECAY}
        # When stable, decaying should be False.
        if result.severity == DecaySeverity.NO_DECAY:
            assert not result.decaying

    def test_stable_returns_quarter_sharpes_finite(self) -> None:
        monitor = EdgeDecayMonitor()
        ret = _stable_returns(n_days=500, seed=11)
        result = monitor.check_decay(ret)
        assert len(result.quarter_sharpes) == 4
        for s in result.quarter_sharpes:
            assert np.isfinite(s)


# =============================================================================
# Decaying returns
# =============================================================================


class TestDecayingReturns:
    def test_linearly_decaying_triggers_negative_tau(self) -> None:
        monitor = EdgeDecayMonitor()
        ret = _decaying_returns(n_days=500, seed=20)
        result = monitor.check_decay(ret)
        assert result.evaluated is True
        # Quarter Sharpes should be monotonically decreasing → tau < 0.
        assert result.trend_tau is not None
        assert result.trend_tau < 0

    def test_linearly_decaying_severity_at_least_possible(self) -> None:
        monitor = EdgeDecayMonitor()
        ret = _decaying_returns(n_days=500, seed=21)
        result = monitor.check_decay(ret)
        # Should fire something — possible / moderate / strong.
        assert result.severity != DecaySeverity.NO_DECAY

    def test_abrupt_drop_triggers_mw_test(self) -> None:
        monitor = EdgeDecayMonitor()
        ret = _abrupt_drop_returns(n_days=500, drop_at=400, seed=30)
        result = monitor.check_decay(ret)
        # Recent window should look much worse than historical.
        assert result.recent_significantly_worse is True
        # And severity escalated.
        assert result.severity != DecaySeverity.NO_DECAY


# =============================================================================
# Severity / recommendation classification
# =============================================================================


class TestClassification:
    def test_severity_strong_under_extreme_decay(self) -> None:
        # Build returns that produce tau ≈ -1, p < 0.01.
        # Quarterly means strictly decreasing.
        rng = np.random.default_rng(40)
        days = 400
        # Per-quarter means: 0.002, 0.001, 0.0, -0.001 (strict monotone decline).
        means_per_quarter = [0.002, 0.001, 0.0, -0.001]
        n_per_q = days // 4
        arr = np.concatenate([rng.normal(m, 0.005, n_per_q) for m in means_per_quarter])
        ret = pd.Series(
            arr,
            index=pd.date_range("2024-01-01", periods=len(arr), freq="B"),
        )
        monitor = EdgeDecayMonitor()
        result = monitor.check_decay(ret)
        # Strong decay or moderate (tau very negative).
        assert result.severity in {
            DecaySeverity.STRONG_DECAY,
            DecaySeverity.MODERATE_DECAY,
        }

    def test_severity_classifier_is_deterministic(self) -> None:
        # Direct unit test of the static classifier.
        m = EdgeDecayMonitor
        assert m._severity(tau=-0.8, trend_p=0.005, mw_p=0.5) == DecaySeverity.STRONG_DECAY
        assert m._severity(tau=-0.6, trend_p=0.05, mw_p=0.01) == DecaySeverity.MODERATE_DECAY
        assert m._severity(tau=-0.4, trend_p=0.5, mw_p=0.5) == DecaySeverity.POSSIBLE_DECAY
        assert m._severity(tau=-0.1, trend_p=0.5, mw_p=0.5) == DecaySeverity.NO_DECAY

    def test_recommendation_classifier_is_deterministic(self) -> None:
        m = EdgeDecayMonitor
        assert (
            m._recommend(tau=-0.8, trend_p=0.005, mw_p=0.5, slope=0.0)
            == DecayRecommendation.RETIRE_STRATEGY
        )
        assert (
            m._recommend(tau=-0.6, trend_p=0.05, mw_p=0.01, slope=0.0)
            == DecayRecommendation.REDUCE_SIZE_50PCT_AND_INVESTIGATE
        )
        assert (
            m._recommend(tau=-0.4, trend_p=0.5, mw_p=0.5, slope=0.0)
            == DecayRecommendation.MONITOR_CLOSELY_WEEKLY_REVIEW
        )
        # Slope decay without other signals → check regime.
        assert (
            m._recommend(tau=-0.1, trend_p=0.5, mw_p=0.5, slope=-0.01)
            == DecayRecommendation.CHECK_REGIME_HYPOTHESIS
        )
        assert (
            m._recommend(tau=-0.1, trend_p=0.5, mw_p=0.5, slope=0.0)
            == DecayRecommendation.NO_ACTION_EDGE_STABLE
        )


# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    def test_invalid_min_history_rejected(self) -> None:
        with pytest.raises(AssertionError):
            EdgeDecayMonitor(min_history_days=10)

    def test_invalid_recent_window_rejected(self) -> None:
        with pytest.raises(AssertionError):
            EdgeDecayMonitor(recent_window_days=5)

    def test_to_dict_serializable(self) -> None:
        import json

        monitor = EdgeDecayMonitor()
        ret = _stable_returns(n_days=500)
        result = monitor.check_decay(ret)
        # Round-trip JSON to ensure no non-serializable types.
        text = json.dumps(result.to_dict(), default=str)
        parsed = json.loads(text)
        assert parsed["evaluated"] is True
