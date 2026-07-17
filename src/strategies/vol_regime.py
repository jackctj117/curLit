"""Shared volatility-regime helpers (CL-x50g).

Extracted from ``CarryVolFilterStrategy._compute_vol_z_score`` (CL-c77) so
the rate-diff entry filters can reuse the identical CVIX z-score definition
instead of duplicating it. Two entry points:

    - :func:`compute_vol_z_score` — point-in-time value via a DataProvider
      (live path; fails safe to 0.0 on missing data).
    - :func:`rolling_vol_z` — vectorized per-row series for backtests
      (causal: row t only uses observations up to and including t).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def vol_z_from_history(values: Any, lookback_days: int) -> float:
    """Z-score of the latest vol-index value vs its rolling baseline.

    Takes the last ``lookback_days`` observations, uses all but the final
    one as the baseline (mean / std with ddof=1), and scores the final
    observation against that baseline. This is the exact logic previously
    inlined in carry_vol_filter (CL-c77) — moved here unchanged.

    A flat baseline (sd == 0) makes any deviation "infinitely surprising";
    return a synthetic ±10 so downstream exposure/entry filters still kick
    in. Fewer than 2 observations → 0.0 (no opinion).
    """
    recent = np.asarray(values, dtype=float)[-lookback_days:]
    if len(recent) < 2:
        return 0.0
    mean = float(np.mean(recent[:-1]))
    sd = float(np.std(recent[:-1], ddof=1))
    if sd <= 0:
        delta = float(recent[-1] - mean)
        if abs(delta) < 1e-9:
            return 0.0
        return 10.0 if delta > 0 else -10.0
    return float((recent[-1] - mean) / sd)


def compute_vol_z_score(
    data_provider: Any,
    series_id: str,
    lookback_days: int,
    as_of: datetime,
) -> float:
    """Point-in-time vol-index z-score via a DataProvider (live path).

    Falls back to 0.0 (no regime opinion) when the provider is missing,
    the query fails, or history is shorter than 80% of the lookback —
    the strategy degrades safely rather than failing when the vol table
    isn't populated (same contract carry_vol_filter has always had).
    """
    if data_provider is None:
        return 0.0
    # Pull 2× lookback so the rolling stats can warm up.
    start = as_of - timedelta(days=lookback_days * 2)
    try:
        vol_series = data_provider.get_series(series_id, start, as_of)
    except Exception as exc:
        # warning (not exception) — avoids traceback retention in tight
        # signal loops (CL-2yta).
        logger.warning(
            "get_series failed for %s: %s: %s",
            series_id, type(exc).__name__, exc,
        )
        return 0.0
    if vol_series is None or len(vol_series) < lookback_days * 0.8:
        return 0.0
    return vol_z_from_history(np.asarray(vol_series, dtype=float), lookback_days)


def rolling_vol_z(series: pd.Series, lookback_days: int) -> pd.Series:
    """Vectorized per-row vol z-score for backtests — strictly causal.

    Row t's z-score uses only observations up to and including t: the
    baseline is the previous ``lookback_days - 1`` values (series shifted
    by one so the current value never scores against itself), matching
    :func:`vol_z_from_history` applied to each prefix of the series.

    Rows without sufficient warmup (fewer than ~80% of the lookback in the
    baseline window) are NaN — callers should treat NaN as "regime
    unknown" and let the filter pass, mirroring the live helper's 0.0
    fallback on short history. A flat baseline yields ±inf (blocked by any
    finite threshold), the vectorized analogue of the ±10 synthetic z.
    """
    baseline = series.shift(1)
    window = max(2, lookback_days - 1)
    min_periods = max(2, int(lookback_days * 0.8))
    mean = baseline.rolling(window, min_periods=min_periods).mean()
    sd = baseline.rolling(window, min_periods=min_periods).std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (series - mean) / sd
    return z
