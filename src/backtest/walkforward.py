"""Walk-forward backtesting framework — non-overlapping IS/OOS windows."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

logger = __import__("logging").getLogger(__name__)


class Strategy(Protocol):
    def fit(self, train_data: pd.DataFrame) -> None: ...
    def generate_signals(self, test_data: pd.DataFrame) -> pd.Series: ...


@dataclass
class WalkForwardConfig:
    is_window_days: int = 756
    oos_window_days: int = 63
    step_days: int = 63
    min_history: int = 756


@dataclass
class WalkForwardResult:
    oos_signals: pd.Series
    oos_returns: pd.Series
    trades: pd.DataFrame
    fold_metrics: pd.DataFrame
    params_by_fold: list[dict] = field(default_factory=list)


class WalkForwardRunner:
    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def run(
        self, data: pd.DataFrame, strategy_factory: callable, cost_model,
    ) -> WalkForwardResult:
        cfg = self.config
        folds = []
        oos_signals_all: list[pd.Series] = []
        trades_all: list[pd.DataFrame] = []

        start = cfg.min_history
        while start + cfg.oos_window_days <= len(data):
            train_end = start
            train_start = max(0, train_end - cfg.is_window_days)
            test_end = min(len(data), train_end + cfg.oos_window_days)

            train = data.iloc[train_start:train_end]
            test = data.iloc[train_end:test_end]

            strategy = strategy_factory()
            strategy.fit(train)
            signals = strategy.generate_signals(test)
            oos_signals_all.append(signals)

            df = pd.DataFrame(index=signals.index)
            df["signal"] = signals
            df["position"] = signals.shift(1).fillna(0)
            df["return"] = data.loc[df.index, "close"].pct_change().fillna(0) if "close" in data.columns else pd.Series(0, index=df.index)
            df["strategy_return"] = df["position"] * df["return"]
            df["position_change"] = df["position"].diff().abs().fillna(0)
            df["cost"] = df["position_change"] * cost_model.cost_per_turn
            df["net_return"] = df["strategy_return"] - df["cost"]
            trades_all.append(df)

            folds.append({
                "fold_id": len(folds),
                "train_start": train.index[0], "train_end": train.index[-1],
                "test_start": test.index[0], "test_end": test.index[-1],
                "train_sharpe": self._sharpe(train.get("close", pd.Series())),
                "test_sharpe": self._sharpe(df["net_return"]),
            })
            start += cfg.step_days

        return WalkForwardResult(
            oos_signals=pd.concat(oos_signals_all) if oos_signals_all else pd.Series(dtype=float),
            oos_returns=pd.concat([t["net_return"] for t in trades_all]) if trades_all else pd.Series(dtype=float),
            trades=pd.concat(trades_all) if trades_all else pd.DataFrame(),
            fold_metrics=pd.DataFrame(folds),
        )

    @staticmethod
    def _sharpe(returns: pd.Series, periods: int = 252) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(periods))
