"""Real-fit tests for HMM regime model (CL-rsv).

Constructs synthetic data with three regimes embedded in it, fits the
HMM, and verifies the recovered states correlate with the ground-truth
regime labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

from src.models.hmm_regime import fit_hmm_regime  # noqa: E402


def _build_three_regime_panel(n_per_regime: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Stitch 3 different-vol regimes into one time series.

    Regime 0 (trending):  low-vol, drifting.
    Regime 1 (ranging):   moderate-vol, mean-zero.
    Regime 2 (volatile):  high-vol, mean-zero.

    Returns (features_df, ground_truth_label_series).
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    labels: list[int] = []

    # Vol features ordered so column 0 IS the volatility proxy that
    # fit_hmm_regime uses to sort states.
    for regime, scale in enumerate([0.5, 1.5, 4.0]):
        n = n_per_regime
        vix = scale + rng.normal(0, scale * 0.1, n)
        move = scale * 1.2 + rng.normal(0, scale * 0.1, n)
        dxy_ret = rng.normal(0, scale * 0.001, n)
        yc = 0.5 + rng.normal(0, 0.1, n)
        parts.append(pd.DataFrame({
            "vix": vix, "move": move, "dxy_ret": dxy_ret, "yc": yc,
        }))
        labels.extend([regime] * n)

    df = pd.concat(parts, ignore_index=True)
    df.index = pd.date_range("2015-01-01", periods=len(df), freq="D")
    truth = pd.Series(labels, index=df.index, name="truth")
    return df, truth


class TestHMMFit:
    def test_recovers_three_distinct_regimes(self) -> None:
        df, truth = _build_three_regime_panel()
        fit = fit_hmm_regime(df)

        # Each label should appear in the fit (HMM didn't collapse to <3).
        assert set(fit.label_map.values()) == {"trending", "ranging", "volatile"}

        # Per-state mean of the vol feature should be sorted: trending
        # state has the smallest mean, volatile has the largest. This
        # is the post-fit invariant that label_map encodes.
        vol_means = fit.means_per_state[:, 0]
        sorted_vol = sorted(vol_means)
        for state_idx, label in fit.label_map.items():
            expected_rank = ["trending", "ranging", "volatile"].index(label)
            assert vol_means[state_idx] == sorted_vol[expected_rank]

    def test_label_for_volatile_regime_matches(self) -> None:
        # We constructed the volatile regime to be the LAST n_per_regime
        # rows. After fit, those rows should be majority labeled "volatile".
        df, truth = _build_three_regime_panel()
        fit = fit_hmm_regime(df)

        last_third_states = fit.states_per_date.iloc[-200:]
        last_labels = [fit.label_map[s] for s in last_third_states]
        # 33% would be random with 3 classes; 50% means the HMM is
        # identifying the regime above noise. Tighter thresholds are
        # flaky on 200-sample synthetic data — production fits use
        # ~2500 daily obs giving much cleaner separation.
        n_volatile = sum(1 for label in last_labels if label == "volatile")
        assert n_volatile >= 100, (
            f"Expected majority of last regime to map to volatile, got "
            f"{n_volatile}/200 (random would be ~67/200)"
        )

    def test_too_short_history_raises(self) -> None:
        # Below the 504-day floor the HMM rejects.
        df = pd.DataFrame({
            "vix": np.random.normal(0, 1, 100),
            "move": np.random.normal(0, 1, 100),
        })
        with pytest.raises(ValueError, match="needs at least"):
            fit_hmm_regime(df)
