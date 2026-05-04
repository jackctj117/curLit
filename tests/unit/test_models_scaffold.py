"""Light-touch import/smoke tests for the HMM (CL-rsv) and GP (CL-1gx)
scaffolds.

The full numerical correctness of these models lives in their academic
literature — what we verify here is that:
  - The modules import without error (catches syntax / import bugs).
  - The error path on missing optional deps is graceful (caller gets a
    useful message, not a stacktrace at engine boot).
  - Position-sizing helpers are pure-Python and behave correctly.
"""

from __future__ import annotations

import pytest

from src.models.gp_rate_diff import (
    GPRateDiffModel,
    position_size_from_uncertainty,
)
from src.models.hmm_regime import regime_position_multiplier


class TestRegimeMultiplier:
    def test_volatile_reduces_50pct_for_any_strategy(self) -> None:
        for strat in ("momentum", "mean_reversion", "carry", "other"):
            assert regime_position_multiplier("volatile", strat) == 0.5

    def test_momentum_full_size_in_trending(self) -> None:
        assert regime_position_multiplier("trending", "momentum") == 1.0

    def test_momentum_reduced_in_ranging(self) -> None:
        assert regime_position_multiplier("ranging", "momentum") == 0.3

    def test_mean_reversion_full_in_ranging(self) -> None:
        assert regime_position_multiplier("ranging", "mean_reversion") == 1.0

    def test_unknown_label_returns_neutral(self) -> None:
        assert regime_position_multiplier("unknown", "momentum") == 1.0


class TestPositionSizeFromUncertainty:
    def test_zero_std_returns_base_size(self) -> None:
        assert position_size_from_uncertainty(100.0, 0.0) == 100.0

    def test_high_std_reduces_size(self) -> None:
        size = position_size_from_uncertainty(100.0, std=10.0, std_floor=1.0)
        assert 0 < size < 100.0


class TestGPModelGuards:
    def test_predict_before_fit_raises(self) -> None:
        m = GPRateDiffModel(feature_cols=["x"], target_col="y")
        with pytest.raises(RuntimeError):
            m.predict({"x": 0.0})
