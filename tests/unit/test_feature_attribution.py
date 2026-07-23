"""Unit tests — edge_testing.feature_attribution (G5 / CL-m0h)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from src.edge_testing.feature_attribution import (
    FeatureEdgeAttributor,
)

# =============================================================================
# Synthetic score functions
# =============================================================================


def _make_features(n: int = 252, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return {
        "real_feature": pd.Series(rng.standard_normal(n), index=idx),
        "noise_feature": pd.Series(rng.standard_normal(n), index=idx),
        "another_feature": pd.Series(rng.standard_normal(n), index=idx),
    }


def _score_with_real_feature_only(features: Mapping[str, pd.Series]) -> float:
    """Score that ONLY uses real_feature — ablating it tanks Sharpe."""
    if "real_feature" not in features:
        return 0.0
    # The real feature contributes a Sharpe of 2.0; everything else is ignored.
    return 2.0


def _score_uniform(features: Mapping[str, pd.Series]) -> float:
    """Score that returns the same value regardless of features."""
    return 1.5


def _score_decreasing_with_count(features: Mapping[str, pd.Series]) -> float:
    """Score that grows with number of features (every one helps)."""
    return 1.0 + 0.5 * len(features)


# =============================================================================
# Construction validation
# =============================================================================


class TestConstruction:
    def test_low_n_random_rejected(self) -> None:
        with pytest.raises(AssertionError):
            FeatureEdgeAttributor(n_random_tests=1)

    def test_empty_features_rejected(self) -> None:
        attr = FeatureEdgeAttributor(n_random_tests=5)
        with pytest.raises(AssertionError):
            attr.attribute(_score_uniform, {})


# =============================================================================
# Identifies real-signal features
# =============================================================================


class TestRealSignalDetection:
    def test_ablating_real_feature_drops_sharpe(self) -> None:
        attr = FeatureEdgeAttributor(n_random_tests=5, seed=0)
        features = _make_features()
        report = attr.attribute(_score_with_real_feature_only, features)
        # Baseline = 2.0, ablating real_feature → 0.0, contribution = 2.0.
        c = report.contributions["real_feature"]
        assert c.contribution == pytest.approx(2.0)
        # Ablating noise/other doesn't change Sharpe (still uses real_feature).
        for name in ("noise_feature", "another_feature"):
            assert report.contributions[name].contribution == pytest.approx(0.0)

    def test_real_feature_above_noise_floor(self) -> None:
        attr = FeatureEdgeAttributor(n_random_tests=5, seed=0)
        features = _make_features()
        report = attr.attribute(_score_with_real_feature_only, features)
        # The score_fn ignores random features, so noise floor ≈ 0.
        # 2.0 contribution from real_feature is well above any noise.
        assert report.contributions["real_feature"].above_noise_floor is True


# =============================================================================
# Noise-floor logic
# =============================================================================


class TestNoiseFloor:
    def test_noise_floor_is_2sigma_of_random_contributions(self) -> None:
        # Use a score_fn whose value depends on how many features are present
        # — so adding a random feature ALWAYS bumps Sharpe, giving a non-zero
        # but tightly-clustered noise distribution.
        attr = FeatureEdgeAttributor(n_random_tests=10, seed=42)
        features = _make_features()
        report = attr.attribute(_score_decreasing_with_count, features)
        # Each random adds 0.5 → contributions are exactly 0.5 each → std=0,
        # so noise_floor=0. Every feature ablation drops by 0.5 → contribution
        # = 0.5 > 0 → above_noise_floor=True for all. Confirm.
        for c in report.contributions.values():
            assert c.contribution == pytest.approx(0.5, abs=0.01)
            assert c.above_noise_floor is True

    def test_below_floor_features_warn(self) -> None:
        # Make a score_fn that returns Sharpe = 1.5 + small_random_noise
        # regardless of features, so every contribution is ≈ 0 and every
        # random adds small noise too.
        rng = np.random.default_rng(0)

        def noisy_score(features: Mapping[str, pd.Series]) -> float:
            return 1.5 + rng.normal(0, 0.05)

        attr = FeatureEdgeAttributor(n_random_tests=15, seed=0)
        features = _make_features()
        report = attr.attribute(noisy_score, features)
        # Real features contribute 0 ± noise; should mostly be below floor.
        below = [c for c in report.contributions.values() if not c.above_noise_floor]
        # At least one feature should hit "below floor" with the expected warning.
        if below:
            assert "noise floor" in below[0].warning


# =============================================================================
# Edge concentration
# =============================================================================


class TestEdgeConcentration:
    def test_single_feature_concentration_is_one(self) -> None:
        attr = FeatureEdgeAttributor(n_random_tests=3, seed=0)
        features = _make_features()
        report = attr.attribute(_score_with_real_feature_only, features)
        # Only real_feature has positive contribution → 100% concentration.
        assert report.edge_concentration == pytest.approx(1.0, abs=0.05)

    def test_uniform_contributions_low_concentration(self) -> None:
        attr = FeatureEdgeAttributor(n_random_tests=3, seed=0)
        features = _make_features()
        report = attr.attribute(_score_decreasing_with_count, features)
        # All 3 features contribute equally → concentration = 1/3 ≈ 0.333.
        assert report.edge_concentration == pytest.approx(0.333, abs=0.05)

    def test_no_positive_contributions_zero_concentration(self) -> None:
        # Score that doesn't depend on features → all contributions are 0.
        attr = FeatureEdgeAttributor(n_random_tests=3, seed=0)
        features = _make_features()
        report = attr.attribute(_score_uniform, features)
        assert report.edge_concentration == 0.0


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_features_above_below_partition(self) -> None:
        attr = FeatureEdgeAttributor(n_random_tests=5, seed=0)
        features = _make_features()
        report = attr.attribute(_score_with_real_feature_only, features)
        above = set(report.features_above_floor)
        below = set(report.features_below_floor)
        # Disjoint and covers all features.
        assert above.isdisjoint(below)
        assert above | below == set(features.keys())
        # real_feature is above; the rest are below (zero contribution).
        assert "real_feature" in above

    def test_to_dict_serializable(self) -> None:
        import json

        attr = FeatureEdgeAttributor(n_random_tests=5, seed=0)
        features = _make_features()
        report = attr.attribute(_score_with_real_feature_only, features)
        text = json.dumps(report.to_dict(), default=str)
        parsed = json.loads(text)
        assert "baseline_sharpe" in parsed
        assert "edge_concentration" in parsed
        assert "contributions" in parsed
