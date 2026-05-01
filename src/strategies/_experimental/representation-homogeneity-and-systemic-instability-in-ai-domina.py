"""
Strategy: Representation Homogeneity Synchronized Deleveraging
Slug: representation-homogeneity-and-systemic-instability-in-ai-domina

Mechanism:
  The paper argues that AI-driven FX traders share similar learned
  representations, causing synchronized deleveraging during stress.
  This manifests as:
    1. A "compression phase": realized volatility is unusually low
       relative to its long-run baseline (leverage accumulates silently).
    2. A "spike phase": realized volatility suddenly exceeds a threshold
       above the long-run baseline (synchronized deleveraging fires).

  We proxy this two-phase dynamic with a volatility-ratio regime detector:
    - Short-window realized vol  (fast_window, default 5 days)
    - Long-window realized vol   (slow_window, default 63 days)
    - Compression flag: fast_vol / slow_vol < compression_threshold
    - Spike flag:       fast_vol / slow_vol > spike_threshold

  Signal logic (mean-reversion during spike, flat otherwise):
    - When a spike fires AFTER a confirmed compression period, go SHORT
      the pair (expect continued deleveraging / downward pressure on
      risk-on pairs, or mean-reversion of the spike itself).
    - Hold the short for hold_bars bars, then exit.
    - No signal during normal or pure-compression regimes.

  For EURUSD / USDJPY the "risk-off" direction during deleveraging is
  typically USD strength (short EURUSD, long USDJPY). Because the harness
  is single-asset and direction-agnostic, we return -1 on spike entry
  (short the pair) and 0 otherwise. The backtest will determine whether
  this directional assumption holds.

Config parameters (≤ 5 free parameters):
  fast_window          : int   [3, 15]   — short realized-vol window (days)
  slow_window          : int   [42, 126] — long realized-vol window (days)
  compression_threshold: float [0.5, 0.9]— fast/slow ratio below which
                                           compression is flagged
  spike_threshold      : float [1.3, 3.0]— fast/slow ratio above which
                                           a spike is flagged
  hold_bars            : int   [3, 21]   — bars to hold position after
                                           spike entry
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Config:
    """
    Configuration for the Representation Homogeneity Deleveraging strategy.

    Parameters
    ----------
    fast_window : int
        Rolling window (in trading days) for the short-term realized
        volatility estimate. Acceptable range: [3, 15].
    slow_window : int
        Rolling window (in trading days) for the long-term realized
        volatility estimate (the "normal times" baseline).
        Acceptable range: [42, 126].
    compression_threshold : float
        fast_vol / slow_vol ratio below which the market is considered
        to be in a compression (low-vol accumulation) regime.
        Acceptable range: [0.5, 0.9].
    spike_threshold : float
        fast_vol / slow_vol ratio above which the market is considered
        to be in a synchronized-deleveraging spike.
        Acceptable range: [1.3, 3.0].
    hold_bars : int
        Number of bars to hold the short position after a spike entry.
        Acceptable range: [3, 21].
    min_compression_bars : int
        Minimum number of consecutive compression bars required before
        a subsequent spike is considered a valid deleveraging signal.
        Fixed at 5; not a free parameter.
    """
    fast_window: int = 5
    slow_window: int = 63
    compression_threshold: float = 0.75
    spike_threshold: float = 1.6
    hold_bars: int = 10
    # Fixed structural parameter — not counted as a free hyperparameter
    min_compression_bars: int = 5


class Strategy:
    """
    Representation Homogeneity Synchronized Deleveraging Strategy.

    Detects the two-phase volatility signature (compression → spike)
    that the paper associates with AI-trader synchronization and
    exploits the directional move during the spike phase.
    """

    id: str = "representation-homogeneity-and-systemic-instability-in-ai-domina"
    symbols: list[str] = ["EURUSD"]

    def __init__(self, config: Config | None = None) -> None:
        self.config = config if config is not None else Config()
        # Fitted in-sample statistics (populated by fit())
        self._fitted: bool = False
        self._vol_scale: float = 1.0  # in-sample median slow_vol, for sanity checks

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit in-sample parameters from training data.

        The only in-sample calibration performed is computing the median
        long-run realized volatility so that we can sanity-check the
        test-period vol regime. No look-ahead is introduced because we
        only store a scalar summary statistic from the training window.

        Parameters
        ----------
        train_data : pd.DataFrame
            DataFrame with a ``close`` column and a DatetimeIndex.
        """
        cfg = self.config
        close = train_data["close"].astype(float)
        log_ret = np.log(close / close.shift(1))

        slow_vol = (
            log_ret.shift(1)
            .rolling(cfg.slow_window, min_periods=cfg.slow_window // 2)
            .std()
        )
        # Store median in-sample slow vol as a scale reference
        self._vol_scale = float(slow_vol.median())
        if np.isnan(self._vol_scale) or self._vol_scale <= 0:
            self._vol_scale = 1e-4  # fallback; won't affect signal logic
        self._fitted = True

    # ------------------------------------------------------------------
    # generate_signals
    # ------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from close prices.

        Returns a numeric Series aligned to ``data.index``:
          -1 : short the pair (spike / deleveraging detected)
           0 : flat (no signal)

        The signal at index t is computed solely from data at t and
        earlier (no look-ahead).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with a ``close`` column and a DatetimeIndex.

        Returns
        -------
        pd.Series
            Signal series indexed identically to ``data``.
        """
        cfg = self.config
        close = data["close"].astype(float)
        n = len(close)

        # ---- 1. Realized volatility series (no look-ahead) ----
        # shift(1) ensures vol at t uses only returns up to t-1
        log_ret = np.log(close / close.shift(1))
        shifted_ret = log_ret.shift(1)

        fast_vol = shifted_ret.rolling(
            cfg.fast_window, min_periods=max(2, cfg.fast_window // 2)
        ).std()

        slow_vol = shifted_ret.rolling(
            cfg.slow_window, min_periods=cfg.slow_window // 2
        ).std()

        # ---- 2. Vol ratio ----
        # Guard against zero slow_vol
        vol_ratio = fast_vol / slow_vol.replace(0, np.nan)

        # ---- 3. Regime flags ----
        compression_flag = (vol_ratio < cfg.compression_threshold).fillna(False)
        spike_flag = (vol_ratio > cfg.spike_threshold).fillna(False)

        # ---- 4. Rolling count of recent compression bars ----
        # Number of compression bars in the preceding [hold_bars, slow_window] window
        # We use a window of slow_window bars to count compression history
        compression_count = (
            compression_flag.astype(int)
            .shift(1)
            .rolling(cfg.slow_window, min_periods=1)
            .sum()
        )

        # ---- 5. Entry condition ----
        # A valid spike entry requires:
        #   (a) spike_flag is True at t
        #   (b) at least min_compression_bars of compression in the
        #       preceding slow_window bars
        valid_entry = spike_flag & (compression_count >= cfg.min_compression_bars)

        # ---- 6. Build signal with hold logic ----
        # For each valid entry bar, set signal = -1 for hold_bars bars forward.
        # We do this in a forward-fill loop to avoid look-ahead.
        signal_arr = np.zeros(n, dtype=float)
        entry_indices = np.where(valid_entry.values)[0]

        for idx in entry_indices:
            end_idx = min(idx + cfg.hold_bars, n)
            # Only set if not already in a position from a prior entry
            # (first entry wins; subsequent entries extend if they fire
            #  while already in a position — we allow overlap for simplicity)
            signal_arr[idx:end_idx] = -1.0

        signal = pd.Series(signal_arr, index=data.index, name="signal")
        return signal
