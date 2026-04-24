"""Strategy 1: Rate differential mean reversion on EUR/USD."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.execution.oms import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class RateDiffMRConfig:
    pair: str = "EURUSD"
    rate_spread_series: str = "US2Y_MINUS_DE2Y"
    lookback_days: int = 252
    entry_z_threshold: float = 1.5
    exit_z_threshold: float = 0.3
    stop_loss_z: float = 3.5
    max_holding_days: int = 30
    volatility_target: float = 0.10
    max_position_pct: float = 0.20
    min_r_squared: float = 0.25
    signal_interval_seconds: int = 3600
    id: str = "eurusd_rate_diff_mr"


class RateDiffMRStrategy:
    def __init__(self, config: RateDiffMRConfig = None, data_provider=None, state_store=None) -> None:
        self.config = config or RateDiffMRConfig()
        self.data = data_provider
        self.state = state_store
        self._model: dict | None = None
        self._last_fit: datetime | None = None
        self._position: float = 0.0
        self._entry_z: float | None = None
        self._entry_ts: datetime | None = None

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def symbols(self) -> list[str]:
        return [self.config.pair]

    def _fit_model(self, df: pd.DataFrame) -> dict:
        import statsmodels.api as sm
        df = df.dropna()
        if len(df) < 100:
            return {}
        df["spread"] = df[self.config.rate_spread_series]
        X = sm.add_constant(df[["spread"]])
        y = df[self.config.pair]
        ols = sm.OLS(y, X).fit()
        self._model = {"alpha": float(ols.params.iloc[0]), "beta": float(ols.params.iloc[1]),
                        "r_squared": float(ols.rsquared), "residual_std": float(ols.resid.std())}
        self._last_fit = datetime.utcnow()
        return self._model

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fit_model(train_data)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self._model is None:
            self._fit_model(data)
        if self._model is None:
            return pd.Series(0.0, index=data.index)
        df = data.copy()
        df["spread"] = df.get(self.config.rate_spread_series, 0)
        df["fair"] = self._model["alpha"] + self._model["beta"] * df["spread"]
        df["deviation"] = df[self.config.pair] - df["fair"]
        df["z"] = df["deviation"] / self._model["residual_std"]
        positions = []
        pos = 0.0
        entry_z = None
        entry_i = None
        for i, z in enumerate(df["z"]):
            if abs(pos) < 0.001:
                if z < -self.config.entry_z_threshold:
                    pos = 1.0
                    entry_z = z
                    entry_i = i
                elif z > self.config.entry_z_threshold:
                    pos = -1.0
                    entry_z = z
                    entry_i = i
            else:
                days = i - entry_i if entry_i is not None else 0
                stop = (pos > 0 and z < -self.config.stop_loss_z) or (pos < 0 and z > self.config.stop_loss_z)
                exit_ok = (pos > 0 and z >= -self.config.exit_z_threshold) or (pos < 0 and z <= self.config.exit_z_threshold)
                if stop or exit_ok or days > self.config.max_holding_days:
                    pos = 0.0
            positions.append(pos)
        return pd.Series(positions, index=df.index, dtype=float)

    async def generate_intents(self, prices: dict, broker) -> list[OrderIntent]:
        return []
