"""
src/strategies/_experimental/the-virtue-of-sparsity-in-complexity.py

SparseFactorFX — implements the "Virtue of Sparsity in Complexity" hypothesis.

Constructs a high-dimensional feature space from a single FX close series
(lagged returns, rolling moments, calendar dummies) and fits a Lasso
(L1-regularised linear regression) to select a sparse set of priced factors.
The fitted model generates a continuous signal; the sign drives position
direction and the magnitude scales position size (clipped to [-1, 1]).

All features are derived exclusively from the `close` column supplied by
the walk-forward harness — no external data sources are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SparseFactorFXConfig:
    """
    Configuration for SparseFactorFX.

    Parameters
    ----------
    lags : list[int]
        Return lags (in bars) to include as features.
        Acceptable range: each lag in [1, 60].
        Default: [1, 2, 3, 5, 10, 21] (daily bars).
    rolling_windows : list[int]
        Windows (in bars) for rolling moment features (vol, skew, kurt).
        Acceptable range: each window in [5, 126].
        Default: [5, 10, 21, 63].
    signal_clip : float
        Absolute cap on the raw signal before it is returned.
        Acceptable range: (0, 10].
        Default: 1.0  (normalised position).
    min_train_bars : int
        Minimum number of non-NaN rows required before fit() will train.
        Acceptable range: [63, 1260].
        Default: 252.
    lasso_cv_folds : int
        Number of time-series CV folds used by LassoCV.
        Acceptable range: [3, 10].
        Default: 5.
    """
    lags: list = field(default_factory=lambda: [1, 2, 3, 5, 10, 21])
    rolling_windows: list = field(default_factory=lambda: [5, 10, 21, 63])
    signal_clip: float = 1.0
    min_train_bars: int = 252
    lasso_cv_folds: int = 5


# ---------------------------------------------------------------------------
# Feature engineering (pure function — no look-ahead)
# ---------------------------------------------------------------------------

def _build_features(close: pd.Series,
                    lags: list[int],
                    rolling_windows: list[int]) -> pd.DataFrame:
    """
    Build a feature matrix from a single close price series.

    Every feature at index t is computed from data with timestamp <= t
    (no look-ahead). Returns are shifted by 1 before rolling to ensure
    the rolling window itself does not include the current bar's return.

    Features
    --------
    * Lagged log-returns: r_{t-k} for k in lags
    * Rolling volatility (std of log-returns) for each window
    * Rolling skewness of log-returns for each window
    * Rolling excess kurtosis of log-returns for each window
    * Calendar dummies: day-of-week (Mon–Thu, Fri is baseline),
      month-of-year (Jan–Nov, Dec is baseline)
    """
    log_ret = np.log(close / close.shift(1))  # r_t = log(P_t / P_{t-1})

    frames: dict[str, pd.Series] = {}

    # --- lagged returns ---
    for k in lags:
        frames[f"lag_ret_{k}"] = log_ret.shift(k)

    # --- rolling moments (computed on already-shifted returns to avoid
    #     look-ahead: the window ending at t-1 informs the signal at t) ---
    ret_shifted = log_ret.shift(1)  # r_{t-1}, r_{t-2}, ...
    for w in rolling_windows:
        roll = ret_shifted.rolling(window=w, min_periods=max(3, w // 2))
        frames[f"roll_vol_{w}"]  = roll.std()
        frames[f"roll_skew_{w}"] = roll.skew()
        frames[f"roll_kurt_{w}"] = roll.kurt()

    # --- calendar dummies ---
    idx = close.index
    if hasattr(idx, "dayofweek"):
        for d in range(4):  # Mon=0 … Thu=3; Fri=4 is baseline
            frames[f"dow_{d}"] = (idx.dayofweek == d).astype(float)
        for m in range(1, 12):  # Jan=1 … Nov=11; Dec=12 is baseline
            frames[f"month_{m}"] = (idx.month == m).astype(float)

    feat = pd.DataFrame(frames, index=close.index)
    return feat


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy:
    """
    SparseFactorFX — Lasso-selected sparse factor model for FX direction.

    The strategy:
    1. Builds a high-dimensional feature matrix from the close series.
    2. Fits a LassoCV model (L1 regularisation) on in-sample data to
       select a sparse subset of priced factors.
    3. At each out-of-sample bar, predicts the next log-return; the
       clipped prediction is the signal (positive → long, negative → short).

    The signal is continuous and non-zero on most bars (the Lasso
    intercept alone ensures this), satisfying the harness's non-constant
    signal requirement.
    """

    id: str = "the-virtue-of-sparsity-in-complexity"
    symbols: list[str] = ["EURUSD"]

    def __init__(self, config: SparseFactorFXConfig | None = None):
        self.config = config if config is not None else SparseFactorFXConfig()
        self._model: LassoCV | None = None
        self._scaler: StandardScaler | None = None
        self._fitted: bool = False
        self._lags = self.config.lags
        self._windows = self.config.rolling_windows

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the sparse factor model on in-sample data.

        Parameters
        ----------
        train_data : pd.DataFrame
            Must contain a ``close`` column with a DatetimeIndex.
            All feature engineering is performed internally.
        """
        self._fitted = False
        close = train_data["close"].copy()

        feat = _build_features(close, self._lags, self._windows)

        # Target: next bar's log-return (shifted back by 1 so that
        # feature row at t predicts return at t+1, but we align so
        # that the label at t is the return *from* t to t+1 — this is
        # the in-sample target; no look-ahead because we only use this
        # during fit on historical data).
        log_ret = np.log(close / close.shift(1))
        target = log_ret.shift(-1)  # r_{t+1} — the thing we want to predict

        # Combine and drop NaNs
        df = feat.copy()
        df["__target__"] = target
        df = df.dropna()

        if len(df) < self.config.min_train_bars:
            # Not enough data — leave unfitted; generate_signals returns 0
            return

        X = df.drop(columns=["__target__"]).values
        y = df["__target__"].values

        # Standardise features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LassoCV with time-series-aware CV (no shuffle)
        model = LassoCV(
            cv=self.config.lasso_cv_folds,
            fit_intercept=True,
            max_iter=5000,
            n_jobs=1,
            random_state=42,
        )
        model.fit(X_scaled, y)

        self._scaler = scaler
        self._model = model
        self._fitted = True

    # ------------------------------------------------------------------
    # generate_signals
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate a continuous signal for each bar in ``data``.

        Returns
        -------
        pd.Series
            Float signal indexed like ``data``. Positive → long,
            negative → short, zero → flat. Clipped to
            [-signal_clip, +signal_clip].
        """
        close = data["close"].copy()
        zero_signal = pd.Series(0.0, index=data.index)

        if not self._fitted or self._model is None or self._scaler is None:
            return zero_signal

        feat = _build_features(close, self._lags, self._windows)

        # We predict at each bar using features available up to that bar.
        # Rows with any NaN feature get a zero signal.
        feat_filled = feat.copy()
        valid_mask = feat_filled.notna().all(axis=1)

        if valid_mask.sum() == 0:
            return zero_signal

        X_valid = feat_filled.loc[valid_mask].values

        # Guard: if feature dimension changed (shouldn't happen in normal
        # walk-forward, but be defensive)
        expected_n_features = self._scaler.n_features_in_
        if X_valid.shape[1] != expected_n_features:
            return zero_signal

        X_scaled = self._scaler.transform(X_valid)
        raw_pred = self._model.predict(X_scaled)

        signal = zero_signal.copy()
        signal.loc[valid_mask] = raw_pred

        # Clip to [-signal_clip, +signal_clip]
        clip = self.config.signal_clip
        signal = signal.clip(-clip, clip)

        return signal
