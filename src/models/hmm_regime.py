"""HMM-based regime detection (CL-rsv).

3-state Gaussian HMM on macro features. Output is a per-date regime
label that strategies can gate on:

  - State 0 — "trending":     low vol, monotonic moves; momentum works
  - State 1 — "ranging":      moderate vol, mean-reverting; carry/MR
  - State 2 — "volatile":     high vol, regime breaks; reduce exposure

The state-to-label mapping is **not** fixed by the HMM itself —
states emerge from the unsupervised fit. We label them post-fit by
sorting on mean-vol of the volatility feature (lowest = trending,
highest = volatile). This is the canonical convention used in the
regime-switching FX literature (Ang & Bekaert 2002, Guidolin & Timmermann 2007).

Features:
  - VIX                      (US equity vol; risk-off proxy)
  - MOVE                     (Treasury vol; rate-sensitive risk)
  - DXY daily return         (USD breadth)
  - Yield curve slope        (US_10Y - US_2Y; recession leading indicator)

Fit on 10+ years of daily data. Update weekly — HMM parameters drift
slowly; daily refits add no signal but multiply the surface for noise.

Implementation note: hmmlearn is the de-facto Python HMM library.
The dependency lives in pyproject extras.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# 3 states — matches the literature consensus on FX regime granularity.
# 2 states underfits (collapses trending+ranging); 4+ states overfit
# small-N estimation of the transition matrix.
_N_STATES: int = 3

# Min-history floor. HMM EM converges poorly on <500 obs; we want at
# least 2 years of daily data before fit. 504 trading days = 2 years.
_MIN_HISTORY_DAYS: int = 504


@dataclass
class RegimeFit:
    """Output of HMM fit: state labels per date plus model artifacts."""

    states_per_date: pd.Series  # values ∈ {0, 1, 2}
    transition_matrix: np.ndarray[Any, Any]
    means_per_state: np.ndarray[Any, Any]
    covars_per_state: np.ndarray[Any, Any]
    feature_names: list[str] = field(default_factory=list)
    # Index of states sorted by ascending mean of the volatility-proxy
    # feature (VIX). label_map[0]="trending", [1]="ranging", [2]="volatile".
    label_map: dict[int, str] = field(default_factory=dict)

    def label_for_date(self, ts: datetime) -> str:
        ts_ts = pd.Timestamp(ts)
        if ts_ts not in self.states_per_date.index:
            return "unknown"
        state = int(self.states_per_date.loc[ts_ts])
        return self.label_map.get(state, "unknown")


def fit_hmm_regime(
    features: pd.DataFrame,
    n_states: int = _N_STATES,
    random_state: int = 42,
) -> RegimeFit:
    """Fit a Gaussian HMM and return the per-date state labels.

    ``features`` columns must include at least one volatility feature
    (VIX or equivalent) so the post-fit label sort works. Other columns
    are passed through to the HMM as-is.

    Returns a RegimeFit. Raises ValueError when history is too short.
    """
    if len(features) < _MIN_HISTORY_DAYS:
        msg = f"HMM needs at least {_MIN_HISTORY_DAYS} obs for stable EM, got {len(features)}"
        raise ValueError(msg)

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        msg = (
            "hmmlearn not installed. Install with `pip install hmmlearn` "
            "or add to pyproject extras."
        )
        raise ImportError(msg) from exc

    df = features.dropna()
    X = df.values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=random_state,
    )
    model.fit(X)

    raw_states = model.predict(X)

    # Sort states by mean of first feature (assume it's the vol feature
    # — caller is responsible for ordering columns). label_map flips the
    # arbitrary HMM state index to a stable ordinal: 0=lowest-vol →
    # "trending", 2=highest-vol → "volatile".
    state_means_on_vol = model.means_[:, 0]
    state_order = np.argsort(state_means_on_vol)
    label_names = ["trending", "ranging", "volatile"]
    label_map = {int(state_order[i]): label_names[i] for i in range(n_states)}

    return RegimeFit(
        states_per_date=pd.Series(raw_states, index=df.index, dtype=int),
        transition_matrix=model.transmat_,
        means_per_state=model.means_,
        covars_per_state=model.covars_,
        feature_names=list(df.columns),
        label_map=label_map,
    )


def regime_position_multiplier(
    label: str,
    strategy_type: str,
) -> float:
    """Map (regime, strategy_type) → exposure multiplier.

    Strategy_type ∈ {"momentum", "mean_reversion", "carry", "other"}.
    Multiplier is what the risk sizer applies on top of the base size.

    Defaults are conservative — operators tune per their backtest:
      - momentum × trending     = 1.0  (the natural fit)
      - momentum × ranging      = 0.3  (chop kills momentum)
      - momentum × volatile     = 0.0  (too noisy)
      - mean_reversion × ranging = 1.0
      - mean_reversion × trending = 0.4
      - everything in volatile is reduced 50% baseline
    """
    if label == "volatile":
        return 0.5
    if strategy_type == "momentum":
        if label == "trending":
            return 1.0
        if label == "ranging":
            return 0.3
    if strategy_type == "mean_reversion":
        if label == "ranging":
            return 1.0
        if label == "trending":
            return 0.4
    return 1.0
