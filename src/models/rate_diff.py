"""Rate differential model — rolling OLS regression between FX spot and yield spreads."""

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class RateDiffModel:
    def __init__(
        self,
        pair: str = "EURUSD",
        spread_name: str = "US2Y_MINUS_DE2Y",
        window_days: int = 756,
        min_r_squared: float = 0.25,
    ) -> None:
        self.pair = pair
        self.spread_name = spread_name
        self.window_days = window_days
        self.min_r_squared = min_r_squared
        self.result: dict[str, Any] | None = None
        self.last_fit_date: datetime | None = None

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        df = df.dropna()
        if len(df) < 100:
            logger.warning("Insufficient data for model fit (%d rows)", len(df))
            return {}

        X = sm.add_constant(df["spread"])
        y = df["target"]
        ols = sm.OLS(y, X).fit()

        self.result = {
            "alpha": float(ols.params.get("const", 0)),
            "beta": float(ols.params.get("spread", 0)),
            "r_squared": float(ols.rsquared),
            "residual_std": float(ols.resid.std()),
            "n_obs": len(df),
        }
        self.last_fit_date = datetime.utcnow()
        return self.result

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.result is None:
            raise ValueError("Model not fitted")
        return self.result["alpha"] + self.result["beta"] * df["spread"]

    def deviation_zscore(self, df: pd.DataFrame) -> pd.Series:
        fair_value = self.predict(df)
        deviation = df["target"] - fair_value
        return deviation / self.result["residual_std"] if self.result else pd.Series(0, index=df.index)

    @property
    def quality_ok(self) -> bool:
        return self.result is not None and self.result["r_squared"] >= self.min_r_squared
