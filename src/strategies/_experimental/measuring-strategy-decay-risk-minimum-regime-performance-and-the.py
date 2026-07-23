"""
Strategy: MRP Regime-Filtered FX Momentum
Slug: measuring-strategy-decay-risk-minimum-regime-performance-and-the

Implements a momentum (trend-following) strategy on a single FX pair with a
Minimum Regime Performance (MRP) filter. The filter classifies the current
market regime using rolling volatility and trend-strength features derived
purely from price, then disables the momentum signal when the regime matches
the historically worst-performing regime (lowest risk-adjusted return in-sample).

Protocol constraints:
  - Single-asset, close-only DataFrame input.
  - No look-ahead: all rolling stats use .shift(1) before rolling.
  - At most 5 free hyperparameters (see MRPConfig).
  - No src.* imports.
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
class MRPConfig:
    """
    Configuration for the MRP Regime-Filtered Momentum strategy.

    Parameters
    ----------
    momentum_window : int
        Look-back window (in trading days) for the momentum signal.
        Acceptable range: [10, 120]. Default 20 (≈ 1 month).
    vol_window : int
        Rolling window (in trading days) for realised-volatility estimation
        used in regime classification.
        Acceptable range: [10, 63]. Default 21 (≈ 1 month).
    n_regimes : int
        Number of volatility regimes to identify via quantile-based binning.
        Acceptable range: [2, 4]. Default 3 (low / mid / high vol).
    trend_window : int
        Look-back window (in trading days) for the trend-strength feature
        (absolute value of normalised price change).
        Acceptable range: [5, 63]. Default 10.
    mrp_worst_fraction : float
        Fraction of regimes (by MRP rank) to suppress. E.g. 0.34 suppresses
        the worst third of regime cells.
        Acceptable range: [0.10, 0.50]. Default 0.34.
    """

    momentum_window: int = 20
    vol_window: int = 21
    n_regimes: int = 3
    trend_window: int = 10
    mrp_worst_fraction: float = 0.34


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class Strategy:
    """
    MRP Regime-Filtered FX Momentum Strategy.

    The strategy:
    1. Computes a momentum signal: sign of the rolling return over
       `momentum_window` days (shifted by 1 to avoid look-ahead).
    2. Classifies each bar into a 2-D regime cell defined by:
         - Volatility quantile (n_regimes bins of rolling realised vol)
         - Trend-strength quantile (n_regimes bins of |rolling return| / vol)
    3. In-sample (fit): computes the Sharpe ratio of the momentum strategy
       within each regime cell → MRP table.
    4. Flags the worst `mrp_worst_fraction` of cells (by Sharpe) as
       "suppressed regimes".
    5. OOS (generate_signals): returns the momentum signal unless the current
       bar falls in a suppressed regime, in which case it returns 0.
    """

    id: str = "measuring-strategy-decay-risk-minimum-regime-performance-and-the"
    symbols: list[str] = ["EURUSD"]

    def __init__(self, config: MRPConfig | None = None) -> None:
        self.config = config if config is not None else MRPConfig()
        # Fitted state
        self._vol_quantiles: np.ndarray | None = None  # breakpoints for vol bins
        self._trend_quantiles: np.ndarray | None = None  # breakpoints for trend bins
        self._suppressed_cells: set[tuple[int, int]] = set()  # (vol_bin, trend_bin) pairs
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Derive regime features from a close price series.
        All rolling operations are shifted by 1 day to prevent look-ahead.

        Returns a DataFrame with columns:
          - 'ret'       : 1-day log return (shifted 1)
          - 'mom'       : momentum signal raw value (shifted 1)
          - 'vol'       : rolling realised vol (shifted 1)
          - 'trend_str' : |momentum| / vol, a choppiness proxy (shifted 1)
        """
        cfg = self.config
        log_ret = np.log(close / close.shift(1))

        # Shift by 1 so that at time t we only see data up to t-1
        log_ret_lag = log_ret.shift(1)

        # Momentum: rolling sum of lagged log returns
        mom_raw = log_ret_lag.rolling(
            cfg.momentum_window, min_periods=cfg.momentum_window // 2
        ).sum()

        # Realised vol: rolling std of lagged log returns, annualised
        vol = log_ret_lag.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std() * np.sqrt(
            252
        )
        vol = vol.replace(0, np.nan)

        # Trend strength: |momentum| normalised by vol (high = trending, low = choppy)
        trend_str = mom_raw.abs() / (vol + 1e-10)

        feats = pd.DataFrame(
            {
                "ret": log_ret,  # actual return at t (used for MRP calc in fit)
                "mom": mom_raw,
                "vol": vol,
                "trend_str": trend_str,
            },
            index=close.index,
        )
        return feats

    def _assign_bins(
        self,
        vol: pd.Series,
        trend_str: pd.Series,
        vol_q: np.ndarray,
        trend_q: np.ndarray,
    ) -> pd.DataFrame:
        """
        Assign each bar to a (vol_bin, trend_bin) cell using pre-fitted
        quantile breakpoints.  Returns a DataFrame with integer bin columns.
        """
        n = self.config.n_regimes

        def _digitize(series: pd.Series, breakpoints: np.ndarray) -> pd.Series:
            arr = np.digitize(series.values, breakpoints, right=False)
            arr = np.clip(arr, 0, n - 1)
            result = pd.Series(arr, index=series.index, dtype=int)
            result[series.isna()] = -1  # mark NaN rows
            return result

        vol_bin = _digitize(vol, vol_q)
        trend_bin = _digitize(trend_str, trend_q)
        return pd.DataFrame({"vol_bin": vol_bin, "trend_bin": trend_bin})

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Fit the MRP regime filter from in-sample data.

        Steps:
        1. Compute features (no look-ahead).
        2. Fit quantile breakpoints for vol and trend_str.
        3. Assign each bar to a regime cell.
        4. Compute the Sharpe of the momentum strategy within each cell.
        5. Flag the worst `mrp_worst_fraction` cells as suppressed.
        """
        cfg = self.config
        close = train_data["close"].copy()

        if len(close) < max(cfg.momentum_window, cfg.vol_window) * 2:
            # Not enough data — fall back to no suppression
            self._fitted = False
            return

        feats = self._compute_features(close)
        feats = feats.dropna(subset=["vol", "trend_str", "mom"])

        if feats.empty:
            self._fitted = False
            return

        # Fit quantile breakpoints (exclude endpoints to get n-1 interior cuts)
        n = cfg.n_regimes
        vol_percentiles = np.linspace(0, 100, n + 1)[1:-1]  # e.g. [33.3, 66.7] for n=3
        trend_percentiles = np.linspace(0, 100, n + 1)[1:-1]

        self._vol_quantiles = np.percentile(feats["vol"].dropna(), vol_percentiles)
        self._trend_quantiles = np.percentile(feats["trend_str"].dropna(), trend_percentiles)

        # Assign bins
        bins = self._assign_bins(
            feats["vol"], feats["trend_str"], self._vol_quantiles, self._trend_quantiles
        )
        feats = feats.join(bins)

        # Momentum signal: sign of mom_raw (already lag-adjusted)
        feats["signal"] = np.sign(feats["mom"])
        feats["signal"] = feats["signal"].replace(0, 1)  # break ties long

        # Strategy return at each bar: signal * actual return
        feats["strat_ret"] = feats["signal"] * feats["ret"]

        # Compute MRP (Sharpe) per regime cell
        mrp_table: dict[tuple[int, int], float] = {}
        valid = feats[(feats["vol_bin"] >= 0) & (feats["trend_bin"] >= 0)]

        for (vb, tb), grp in valid.groupby(["vol_bin", "trend_bin"]):
            sr = grp["strat_ret"]
            if len(sr) < 10:
                mrp_table[(int(vb), int(tb))] = np.nan
                continue
            mean_r = sr.mean()
            std_r = sr.std()
            if std_r < 1e-10:
                mrp_table[(int(vb), int(tb))] = 0.0
            else:
                # Annualised Sharpe
                mrp_table[(int(vb), int(tb))] = (mean_r / std_r) * np.sqrt(252)

        # Identify worst cells
        valid_cells = {k: v for k, v in mrp_table.items() if not np.isnan(v)}
        if not valid_cells:
            self._suppressed_cells = set()
            self._fitted = True
            return

        sorted_cells = sorted(valid_cells.items(), key=lambda x: x[1])
        n_suppress = max(1, int(np.ceil(len(sorted_cells) * cfg.mrp_worst_fraction)))
        self._suppressed_cells = {cell for cell, _ in sorted_cells[:n_suppress]}

        self._fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for the supplied data.

        Returns a pd.Series of {-1, 0, +1} values:
          +1 = long EURUSD
          -1 = short EURUSD
           0 = flat (regime suppressed or insufficient data)
        """
        close = data["close"].copy()
        feats = self._compute_features(close)

        # Raw momentum signal
        raw_signal = np.sign(feats["mom"]).replace(0, 1).fillna(0).astype(float)

        if not self._fitted or self._vol_quantiles is None:
            # No filter available — return raw momentum signal
            return raw_signal.rename("signal")

        # Assign regime bins using fitted quantiles
        valid_mask = feats["vol"].notna() & feats["trend_str"].notna()
        bins = self._assign_bins(
            feats["vol"], feats["trend_str"], self._vol_quantiles, self._trend_quantiles
        )

        # Build final signal: suppress if in a worst-regime cell
        signal = raw_signal.copy()
        for idx in data.index:
            if not valid_mask.loc[idx]:
                signal.loc[idx] = 0.0
                continue
            vb = int(bins.loc[idx, "vol_bin"])
            tb = int(bins.loc[idx, "trend_bin"])
            if vb < 0 or tb < 0 or (vb, tb) in self._suppressed_cells:
                signal.loc[idx] = 0.0

        return signal.rename("signal")
