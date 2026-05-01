# src/strategies/_experimental/price-as-focal-point-prediction-markets-conditional-reflexivity.py
"""
Signal Credibility Index (SCI) Momentum Strategy
=================================================
Implements the hypothesis from:
  "Price as Focal Point: Prediction Markets, Conditional Reflexivity,
   and the Politics of Common Knowledge"

The SCI is a composite regime filter built from three components derived
from a single daily close series:

  1. Variance Ratio (VR): VR(k) = Var(k-period return) / (k * Var(1-period return))
     A VR > 1 indicates positive autocorrelation (trending); VR < 1 indicates
     mean-reversion. We use VR(6) as the primary momentum-quality signal.

  2. Two-sidedness diagnostic: measures whether the return distribution is
     "one-sided" (dominated by moves in one direction) vs. balanced. We
     approximate this as 1 - |skewness| / (1 + |skewness|), normalised so
     that a symmetric distribution scores 1.0 and a heavily skewed one
     scores near 0. High two-sidedness with a trending VR suggests genuine
     coordination rather than a one-sided squeeze.

  3. Trader-concentration proxy: approximated as the ratio of the mean
     absolute return to the rolling standard deviation of returns. When
     this ratio is high, large moves are concentrated (few big days dominate),
     suggesting institutional coordination. We normalise to [0, 1] via a
     rolling percentile rank.

The SCI is the equal-weighted average of the three normalised components.
A momentum signal (sign of the rolling return) is only taken when SCI
exceeds a threshold (default 0.55), otherwise the position is flat.

Config parameters (5 total, within the hard limit):
  - vr_window (int, 6): look-back for variance ratio; range [4, 20]
  - momentum_window (int, 20): look-back for the raw momentum signal; range [5, 60]
  - sci_window (int, 60): rolling window for normalising SCI components; range [30, 120]
  - sci_threshold (float, 0.55): minimum SCI to enter a trade; range [0.40, 0.70]
  - signal_scale (float, 1.0): multiplier on the output signal; range [0.5, 2.0]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SCIMomentumConfig:
    """
    Configuration for the SCI Momentum strategy.

    Parameters
    ----------
    vr_window : int
        Look-back period for the variance ratio VR(k). The variance of the
        k-period return is compared to k times the variance of the 1-period
        return. Acceptable range: [4, 20]. Default: 6.
    momentum_window : int
        Look-back period (in days) for the raw momentum signal (rolling
        return). Acceptable range: [5, 60]. Default: 20.
    sci_window : int
        Rolling window used to compute percentile ranks when normalising
        the SCI sub-components. Acceptable range: [30, 120]. Default: 60.
    sci_threshold : float
        Minimum SCI score required to enter a momentum trade. When SCI is
        below this value the strategy is flat. Acceptable range: [0.40, 0.70].
        Default: 0.55.
    signal_scale : float
        Scalar multiplier applied to the final signal. Acceptable range:
        [0.5, 2.0]. Default: 1.0.
    """
    vr_window: int = 6
    momentum_window: int = 20
    sci_window: int = 60
    sci_threshold: float = 0.55
    signal_scale: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_variance_ratio(returns: pd.Series, k: int, window: int) -> pd.Series:
    """
    Compute a rolling variance ratio VR(k) over `window` observations.

    VR(k) = Var(k-period return) / (k * Var(1-period return))

    Both variances are estimated over the same `window`-day rolling window
    of 1-period returns (the k-period variance is approximated from the
    1-period returns within the window to avoid look-ahead).

    Returns a series of the same length as `returns`, with NaN for the
    first `window + k - 1` observations.
    """
    # 1-period variance (rolling)
    var1 = returns.rolling(window).var()

    # k-period return variance: use overlapping k-period sums within window
    # To avoid look-ahead we shift by 1 before rolling
    k_returns = returns.shift(1).rolling(k).sum()
    var_k = k_returns.rolling(window).var()

    vr = var_k / (k * var1.replace(0, np.nan))
    return vr


def _rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
    """Rolling skewness with a minimum of `window` observations."""
    return returns.rolling(window, min_periods=window // 2).skew()


def _two_sidedness(skew: pd.Series) -> pd.Series:
    """
    Convert rolling skewness to a two-sidedness score in [0, 1].

    A perfectly symmetric distribution (skew=0) scores 1.0.
    A heavily skewed distribution scores near 0.
    Formula: 1 - |skew| / (1 + |skew|)
    """
    abs_skew = skew.abs()
    return 1.0 - abs_skew / (1.0 + abs_skew)


def _concentration_proxy(returns: pd.Series, window: int) -> pd.Series:
    """
    Trader-concentration proxy: mean(|r|) / std(r) over rolling window.

    A high ratio means large moves are concentrated (few big days dominate),
    consistent with institutional coordination.
    """
    mean_abs = returns.abs().rolling(window, min_periods=window // 2).mean()
    std_r = returns.rolling(window, min_periods=window // 2).std()
    return mean_abs / std_r.replace(0, np.nan)


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    For each point t, compute the percentile rank of series[t] within the
    preceding `window` observations (exclusive of t itself to avoid look-ahead).
    Returns values in [0, 1].
    """
    def _rank(arr):
        if len(arr) < 2:
            return np.nan
        val = arr[-1]
        hist = arr[:-1]
        finite = hist[np.isfinite(hist)]
        if len(finite) == 0:
            return np.nan
        return float(np.mean(finite <= val))

    # Use a rolling apply over window+1 so the last element is the current value
    return series.rolling(window + 1, min_periods=max(10, window // 3)).apply(
        _rank, raw=True
    )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class Strategy:
    """
    SCI Momentum Strategy.

    Enters a momentum trade (long or short) only when the Signal Credibility
    Index (SCI) exceeds a threshold, indicating the price move has
    "behavioral traction" consistent with self-fulfilling coordination.
    """

    id: str = "price-as-focal-point-prediction-markets-conditional-reflexivity"
    symbols: list[str] = ["EURUSD"]

    def __init__(self, config: SCIMomentumConfig | None = None):
        if config is None:
            config = SCIMomentumConfig()
        self.config = config
        # State fitted in fit()
        self._vr_mean: float = 1.0
        self._vr_std: float = 0.2
        self._conc_mean: float = 0.8
        self._conc_std: float = 0.2
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit normalisation statistics from in-sample data.

        Computes rolling statistics for the SCI components and stores
        the in-sample mean/std of the variance ratio and concentration
        proxy so that OOS signals are normalised consistently.

        Parameters
        ----------
        train_data : pd.DataFrame
            Must contain a ``close`` column with a DatetimeIndex.
        """
        cfg = self.config
        close = train_data["close"].copy()
        returns = close.pct_change()

        # --- Variance ratio ---
        vr = _rolling_variance_ratio(returns, k=cfg.vr_window, window=cfg.sci_window)
        vr_finite = vr.dropna()
        if len(vr_finite) > 5:
            self._vr_mean = float(vr_finite.mean())
            self._vr_std = float(vr_finite.std()) if vr_finite.std() > 1e-8 else 0.2
        else:
            self._vr_mean = 1.0
            self._vr_std = 0.2

        # --- Concentration proxy ---
        conc = _concentration_proxy(returns, window=cfg.sci_window)
        conc_finite = conc.dropna()
        if len(conc_finite) > 5:
            self._conc_mean = float(conc_finite.mean())
            self._conc_std = float(conc_finite.std()) if conc_finite.std() > 1e-8 else 0.2
        else:
            self._conc_mean = 0.8
            self._conc_std = 0.2

        self._fitted = True

    # ------------------------------------------------------------------
    # generate_signals
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from the SCI-filtered momentum strategy.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain a ``close`` column with a DatetimeIndex.

        Returns
        -------
        pd.Series
            Signal series indexed like ``data``. Values are in
            {-scale, 0, +scale} where scale = config.signal_scale.
            Positive = long, negative = short, zero = flat.
        """
        cfg = self.config
        close = data["close"].copy()
        returns = close.pct_change()

        # ----------------------------------------------------------------
        # Component 1: Variance Ratio → normalised to [0, 1]
        # VR > 1 → trending (good for momentum) → high score
        # We z-score using in-sample stats then map through sigmoid
        # ----------------------------------------------------------------
        vr_raw = _rolling_variance_ratio(
            returns, k=cfg.vr_window, window=cfg.sci_window
        )
        vr_z = (vr_raw - self._vr_mean) / (self._vr_std + 1e-8)
        # Sigmoid: maps z-score to (0, 1); VR > mean → score > 0.5
        vr_score = 1.0 / (1.0 + np.exp(-vr_z))

        # ----------------------------------------------------------------
        # Component 2: Two-sidedness → already in [0, 1]
        # High two-sidedness (symmetric returns) + trending VR = coordination
        # ----------------------------------------------------------------
        skew = _rolling_skewness(returns, window=cfg.sci_window)
        ts_score = _two_sidedness(skew)

        # ----------------------------------------------------------------
        # Component 3: Concentration proxy → normalised to [0, 1]
        # High concentration → institutional coordination → high score
        # ----------------------------------------------------------------
        conc_raw = _concentration_proxy(returns, window=cfg.sci_window)
        conc_z = (conc_raw - self._conc_mean) / (self._conc_std + 1e-8)
        conc_score = 1.0 / (1.0 + np.exp(-conc_z))

        # ----------------------------------------------------------------
        # SCI: equal-weighted average of the three components
        # ----------------------------------------------------------------
        sci = (vr_score + ts_score + conc_score) / 3.0

        # ----------------------------------------------------------------
        # Raw momentum signal: sign of rolling return (shifted to avoid
        # look-ahead — we use the return known at close of day t)
        # ----------------------------------------------------------------
        # Rolling return over momentum_window, shifted so bar t uses
        # data up to and including bar t (no future data)
        mom_return = close.pct_change(cfg.momentum_window)
        # mom_return at index t = (close[t] - close[t-momentum_window]) / close[t-momentum_window]
        # This is valid: at bar t we know close[t].
        raw_signal = np.sign(mom_return)

        # ----------------------------------------------------------------
        # Apply SCI filter: only trade when SCI > threshold
        # ----------------------------------------------------------------
        signal = raw_signal.copy()
        signal[sci <= cfg.sci_threshold] = 0.0
        signal[sci.isna()] = 0.0
        signal[raw_signal.isna()] = 0.0

        # Scale
        signal = signal * cfg.signal_scale

        # Ensure index alignment
        signal = signal.reindex(data.index, fill_value=0.0)
        signal.name = self.id

        return signal
