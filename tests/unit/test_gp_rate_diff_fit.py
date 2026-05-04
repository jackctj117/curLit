"""Real-fit tests for the GP rate-diff model (CL-1gx).

Synthesizes (spread, fx) data from a known nonlinear function plus
heteroscedastic noise, fits the GP, and verifies:
  1. predictions track the latent function within tolerance,
  2. uncertainty std is larger in noisy regions than clean ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.models.gp_rate_diff import (  # noqa: E402
    GPRateDiffModel,
    position_size_from_uncertainty,
)


def _build_synthetic(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Spread varies in [-2, 2]; fx = mild nonlinear function of spread.
    spread = rng.uniform(-2.0, 2.0, n)
    # Larger noise when |spread| > 1 — mimics regime-dependent noise.
    noise_scale = 0.05 + 0.20 * (np.abs(spread) > 1.0)
    fair = 1.10 + 0.03 * np.tanh(spread)
    fx = fair + rng.normal(0, noise_scale)
    return pd.DataFrame({"spread": spread, "EURUSD": fx})


class TestGPFit:
    def test_predicts_close_to_truth_in_clean_regime(self) -> None:
        df = _build_synthetic()
        model = GPRateDiffModel(feature_cols=["spread"], target_col="EURUSD")
        model.fit(df)
        # Clean regime: |spread| <= 0.5 — noise is ~0.05.
        pred = model.predict({"spread": 0.0})
        # True fair value at spread=0 is 1.10. Allow ~2% tolerance —
        # GP smoothing across the heteroscedastic noise pulls it
        # slightly toward the global mean.
        assert abs(pred.mean - 1.10) < 0.02

    def test_higher_std_in_high_noise_regime(self) -> None:
        df = _build_synthetic()
        model = GPRateDiffModel(feature_cols=["spread"], target_col="EURUSD")
        model.fit(df)
        # spread=0 is the clean center; spread=1.8 is in the noisy regime.
        std_clean = model.predict({"spread": 0.0}).std
        std_noisy = model.predict({"spread": 1.8}).std
        # GP should report larger uncertainty where the data was noisier.
        # We allow a soft factor; with 400 obs, ratio is typically 1.5-3×.
        assert std_noisy > std_clean

    def test_position_size_from_uncertainty_decreases(self) -> None:
        # Real test: scale base position by predicted std.
        df = _build_synthetic()
        model = GPRateDiffModel(feature_cols=["spread"], target_col="EURUSD")
        model.fit(df)
        clean = model.predict({"spread": 0.0})
        noisy = model.predict({"spread": 1.8})
        size_clean = position_size_from_uncertainty(
            base_size=1000.0, std=clean.std, std_floor=clean.std,
        )
        size_noisy = position_size_from_uncertainty(
            base_size=1000.0, std=noisy.std, std_floor=clean.std,
        )
        # Higher uncertainty → smaller position.
        assert size_noisy < size_clean

    def test_predict_batch_returns_one_row_per_input(self) -> None:
        df = _build_synthetic(n=300)
        model = GPRateDiffModel(feature_cols=["spread"], target_col="EURUSD")
        model.fit(df)

        test_df = pd.DataFrame({"spread": [-1.5, 0.0, 1.5]})
        out = model.predict_batch(test_df)
        assert len(out) == 3
        assert "mean" in out.columns and "std" in out.columns
        assert all(out["std"] > 0)
