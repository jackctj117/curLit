"""Cross-asset correlation monitor — rolling matrices, breakdown/flip detection."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationMonitor:
    def __init__(self, baseline_window: int = 250, recent_window: int = 20) -> None:
        self.baseline = baseline_window
        self.recent = recent_window

    def rolling_correlation_matrix(
        self, returns: pd.DataFrame, windows: list[int] | None = None,
    ) -> dict[int, pd.DataFrame]:
        windows = windows or [20, 60, 250]
        return {w: returns.rolling(w).corr() for w in windows}

    def check_pair(
        self, r1: pd.Series, r2: pd.Series, vix: pd.Series | None = None,
    ) -> dict:
        aligned = pd.concat([r1.rename("a"), r2.rename("b")], axis=1).dropna()
        if len(aligned) < self.baseline:
            return {"regime": "unknown"}

        baseline_corr = float(aligned.iloc[-self.baseline:-self.recent]["a"].corr(aligned.iloc[-self.baseline:-self.recent]["b"]))
        recent_corr = float(aligned.iloc[-self.recent:]["a"].corr(aligned.iloc[-self.recent:]["b"]))
        z = (recent_corr - baseline_corr) / 0.1 if np.isfinite(baseline_corr + recent_corr) else 0.0

        regime = "stable"
        if abs(z) > 2.0 and vix is not None and vix.iloc[-1] > 25:
            regime = "crisis"
        elif abs(z) > 1.5:
            regime = "diverging"

        return {"current": recent_corr, "baseline": baseline_corr, "zscore": z, "regime": regime}
