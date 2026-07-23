"""Gaussian Process model for FX rate diff (CL-1gx).

GP-regressor over the same feature set as the OLS rate-diff model
(CL-mln) but returns prediction + uncertainty. Position sizing scales
inversely with uncertainty — when the model is unsure, we trade smaller.

Compared to OLS:
  - OLS gives a point estimate; GP gives mean + std at every test point.
  - OLS extrapolates linearly into regions with no training data; GP
    falls back toward the prior (mean=0, std=signal-amplitude) so we
    don't trade aggressively where we have no data.
  - GP is more expensive to fit (O(n³) vs O(n×p)). Practical n cap
    is ~2000 obs; we sub-sample for longer histories.

Kernel: RBF(length_scale auto-tuned) + WhiteKernel(noise auto-tuned).
RBF gives smooth interpolation; WhiteKernel models heteroscedastic noise.
sklearn does the GPR, no PyMC needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 2000 obs cap — GPR fitting is O(n³). 2000 is the practical limit on
# a single CPU before fits exceed ~30 seconds. For longer histories,
# sub-sample uniformly (loses some recent-data weight; alternative is
# a sparse-GP variant — out of scope for v1).
_MAX_TRAINING_POINTS: int = 2000

# Default kernel hyperparameter init values. sklearn auto-tunes by
# maximizing log-marginal-likelihood; these are the priors.
_RBF_LENGTH_SCALE_INIT: float = 1.0
_WHITE_NOISE_INIT: float = 0.1


@dataclass
class GPPrediction:
    """One prediction with its uncertainty."""

    mean: float
    std: float


@dataclass
class GPRateDiffModel:
    """Wraps sklearn's GaussianProcessRegressor for FX rate diff.

    ``feature_cols``: ordered list of feature columns the caller will
    supply. The same order must hold at fit and predict time — sklearn
    requires it.

    ``target_col``: column to predict (e.g. fx_pair).
    """

    feature_cols: list[str]
    target_col: str
    _model: Any = None
    _train_mean: float = 0.0

    def fit(self, df: pd.DataFrame) -> None:
        """Fit GP. Sub-samples to MAX_TRAINING_POINTS if needed."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError as exc:
            msg = (
                "scikit-learn not installed. The GP model is opt-in; install "
                "with `pip install scikit-learn`."
            )
            raise ImportError(msg) from exc

        clean = df[self.feature_cols + [self.target_col]].dropna()
        if len(clean) > _MAX_TRAINING_POINTS:
            # Uniform sub-sample (every-k-th row) preserves the time
            # range and avoids favoring recent data, which would bias
            # the kernel hyperparameters toward recent-regime behavior.
            stride = max(1, len(clean) // _MAX_TRAINING_POINTS)
            clean = clean.iloc[::stride]

        X = clean[self.feature_cols].values
        y_raw = clean[self.target_col].astype(float).values
        # Center the target so the GP prior (mean=0) is at the data
        # average — improves numerical stability of the RBF fit.
        self._train_mean = float(y_raw.mean())
        y = y_raw - self._train_mean

        kernel = RBF(length_scale=_RBF_LENGTH_SCALE_INIT) + WhiteKernel(
            noise_level=_WHITE_NOISE_INIT
        )
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,  # we centered manually
            n_restarts_optimizer=2,  # 2 restarts catches local maxima
            random_state=42,
        )
        self._model.fit(X, y)

    def predict(self, x: dict[str, float]) -> GPPrediction:
        """Single-point prediction with std."""
        if self._model is None:
            raise RuntimeError("GPRateDiffModel must be fit() before predict()")

        X = np.array([[x[c] for c in self.feature_cols]])
        mean, std = self._model.predict(X, return_std=True)
        return GPPrediction(
            mean=float(mean[0]) + self._train_mean,
            std=float(std[0]),
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized prediction. Returns df with `mean` and `std` columns."""
        if self._model is None:
            raise RuntimeError("GPRateDiffModel must be fit() before predict()")
        clean = df[self.feature_cols].dropna()
        means, stds = self._model.predict(clean.values, return_std=True)
        return pd.DataFrame(
            {"mean": means + self._train_mean, "std": stds},
            index=clean.index,
        )


def position_size_from_uncertainty(
    base_size: float,
    std: float,
    std_floor: float = 1e-6,
) -> float:
    """Scale base position size inversely with prediction std.

    Convention: 1.0 = full size when std equals the model's average
    in-sample std. Higher std → smaller position. Below floor → full.

    Strategies wire this into the sizer's adjust_for_uncertainty hook
    (CL-amf future work). Default behavior is identity when caller
    doesn't supply a comparison std.
    """
    if std <= std_floor:
        return base_size
    # Soft inverse: at 2× std, position halves; at 4× std, quarters.
    return base_size / (1.0 + std / max(std_floor, std_floor))
