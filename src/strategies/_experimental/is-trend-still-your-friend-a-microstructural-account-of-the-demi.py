"""Tick-size-filtered SMA-crossover trend strategy (joint multi-asset).

Hypothesis slug: is-trend-still-your-friend-a-microstructural-account-of-the-demi

Implements the mechanism from "Is Trend Still Your Friend?: A
Microstructural Account of the Demise of Short-Term Trend-Following":
short-horizon trend profits persist only on instruments whose tick size
is large relative to volatility. The strategy runs a daily SMA(20)/
SMA(50) crossover on a basket of ten major FX pairs, but takes a
position in a pair only on days where its volatility-normalised tick
size, ``tick_size / (realised_vol * price)``, exceeds a threshold fixed
in-sample as the pooled median of that ratio (per the hypothesis brief).

Protocol notes
--------------
* Joint multi-asset mode (CL-40n2 v2): ``generate_signals`` returns a
  DataFrame whose columns are the tradeable pairs in ``symbols`` and
  whose values are per-bar position weights.
* Self-contained: imports limited to stdlib / numpy / pandas.
* No look-ahead: every rolling statistic is computed on ``.shift(1)``-ed
  inputs, so the weight at bar ``t`` uses information through ``t-1``
  only. The filter threshold is fitted on the training window only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Fixed tick sizes (pip values) per the hypothesis brief. These are
# market conventions, not data: 0.01 for JPY-quoted pairs, 0.0001
# otherwise. The median-quantile filter is invariant to any *uniform*
# rescaling of these constants (e.g. pip vs. fractional-pip convention),
# so only the cross-sectional ratios matter.
TICK_SIZE: dict[str, float] = {
    "EURUSD": 0.0001,
    "USDJPY": 0.01,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDCHF": 0.0001,
    "EURJPY": 0.01,
    "EURCHF": 0.0001,
    "GBPJPY": 0.01,
}


@dataclass
class TrendFilteredTickSizeConfig:
    """Configuration for the tick-size-filtered trend strategy.

    Parameters
    ----------
    fast_window : int, default 20
        Fast SMA lookback in daily bars (brief: SMA(20)).
        Acceptable range: [5, 40].
    slow_window : int, default 50
        Slow SMA lookback in daily bars (brief: SMA(50)). Must be
        strictly greater than ``fast_window``.
        Acceptable range: [30, 200].
    vol_window : int, default 60
        Rolling window (daily bars) for realised return volatility used
        to normalise tick size. Acceptable range: [20, 250].
    filter_quantile : float, default 0.5
        Quantile of the pooled in-sample relative-tick-size distribution
        used as the pass threshold. 0.5 is the brief's "median of
        historical values". Acceptable range: [0.3, 0.7].
    """

    fast_window: int = 20
    slow_window: int = 50
    vol_window: int = 60
    filter_quantile: float = 0.5


class Strategy:
    """SMA(20)/SMA(50) trend basket gated by volatility-normalised tick size."""

    id: str = "is-trend-still-your-friend-a-microstructural-account-of-the-demi"

    # Class-level list of full pair codes (read off the class by the
    # harness without instantiation). All are tradeable FX pairs; the
    # strategy trades all of them jointly (DataFrame signals).
    symbols: list[str] = [
        "EURUSD",
        "USDJPY",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDCAD",
        "USDCHF",
        "EURJPY",
        "EURCHF",
        "GBPJPY",
    ]

    # Alias target for the harness's ``close`` column; P&L in joint
    # multi-asset mode is computed per signal-DataFrame column.
    execution_symbol: str = "EURUSD"

    def __init__(self, config: TrendFilteredTickSizeConfig | None = None):
        self.config = config if config is not None else TrendFilteredTickSizeConfig()
        if self.config.slow_window <= self.config.fast_window:
            raise ValueError(
                "slow_window must exceed fast_window "
                f"(got fast={self.config.fast_window}, slow={self.config.slow_window})"
            )
        if not 0.0 < self.config.filter_quantile < 1.0:
            raise ValueError("filter_quantile must lie in (0, 1)")
        # Fitted in-sample pooled threshold for the relative-tick filter.
        # 0.0 (pass-everything) is the pre-fit / degenerate fallback,
        # which degrades gracefully to the unconditional trend baseline.
        self.threshold_: float = 0.0

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _relative_tick(self, data: pd.DataFrame) -> pd.DataFrame:
        """Volatility-normalised tick size per pair, lagged one bar.

        rel_tick[t] = tick / (rolling_vol[t-1] * price[t-1]); every
        value at index t uses information with ts <= t-1 only.
        """
        out: dict[str, pd.Series] = {}
        for pair in self.symbols:
            if pair not in data.columns:
                continue
            px = pd.to_numeric(data[pair], errors="coerce")
            ret = px.pct_change()
            vol = ret.rolling(self.config.vol_window, min_periods=self.config.vol_window).std()
            denom = (vol * px).shift(1)
            denom = denom.where(denom > 0.0)
            out[pair] = TICK_SIZE[pair] / denom
        return pd.DataFrame(out, index=data.index)

    # ------------------------------------------------------------------
    # protocol surface
    # ------------------------------------------------------------------
    def fit(self, train_data: pd.DataFrame) -> None:
        """Fix the filter threshold as the pooled in-sample quantile.

        Pools the lagged relative-tick ratio across all pairs and all
        in-sample bars, then takes ``filter_quantile`` (default: the
        median, per the brief). Uses in-sample data only.
        """
        rel = self._relative_tick(train_data)
        pooled = rel.to_numpy(dtype=float).ravel()
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size == 0:
            # Insufficient in-sample history: degrade to the
            # unconditional trend baseline rather than emit all-zeros.
            self.threshold_ = 0.0
        else:
            self.threshold_ = float(np.quantile(pooled, self.config.filter_quantile))

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Per-pair position weights (joint multi-asset mode).

        weight[pair, t] = sign(SMA_fast - SMA_slow)[t-1 info]
                          * 1{rel_tick[pair, t] > threshold} / N_pairs

        Gross exposure is at most 1. Bars/pairs failing the tick filter
        (or inside the warm-up window) get weight 0.
        """
        cfg = self.config
        rel = self._relative_tick(data)
        traded = [p for p in self.symbols if p in data.columns]
        n = max(len(traded), 1)

        weights: dict[str, pd.Series] = {}
        for pair in traded:
            px = pd.to_numeric(data[pair], errors="coerce")
            lagged = px.shift(1)
            sma_fast = lagged.rolling(cfg.fast_window, min_periods=cfg.fast_window).mean()
            sma_slow = lagged.rolling(cfg.slow_window, min_periods=cfg.slow_window).mean()
            trend = np.sign(sma_fast - sma_slow)
            passes = (rel[pair] > self.threshold_).astype(float)
            weights[pair] = (trend * passes) / n

        return pd.DataFrame(weights, index=data.index).fillna(0.0)
