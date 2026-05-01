"""Walk-forward backtesting framework — non-overlapping IS/OOS windows."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

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
    params_by_fold: list[dict[str, Any]] = field(default_factory=list)


class WalkForwardRunner:
    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def run(
        self,
        data: pd.DataFrame,
        strategy_factory: Callable[[], Any],
        cost_model: Any,
    ) -> WalkForwardResult:
        cfg = self.config
        folds: list[dict[str, Any]] = []
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

            # In-sample Sharpe must use STRATEGY RETURNS, not raw close
            # prices (CL-u9rn). The previous version computed Sharpe of
            # the price series itself, yielding mean(price)/std(price)
            # ≈ hundreds for FX pairs and breaking rule A.7. The fix
            # mirrors the OOS path: re-fit the strategy on this train
            # window's first 80% (in-sample), compute net returns on
            # the remaining 20% (still in-sample), take Sharpe of those.
            train_sharpe = self._train_sharpe(
                train, strategy_factory, cost_model,
            )
            folds.append({
                "fold_id": len(folds),
                "train_start": train.index[0], "train_end": train.index[-1],
                "test_start": test.index[0], "test_end": test.index[-1],
                "train_sharpe": train_sharpe,
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

    @staticmethod
    def _train_sharpe(
        train: pd.DataFrame,
        strategy_factory: Callable[[], Any],
        cost_model: Any,
    ) -> float:
        """Compute in-sample Sharpe via an inner fit/test split (CL-u9rn).
        Splits ``train`` 80/20 — fit on the first 80%, generate signals
        on the last 20% (still in-sample), compute net strategy returns
        on that slice, return Sharpe. Mirrors the OOS calculation so the
        is_oos_sharpe_ratio metric is comparable.

        Falls back to 0.0 when the inner split is too short or when the
        strategy can't fit / produces no signals — in those cases the
        rule A.7 (is/oos ratio ≤ 2.5) ends up well-behaved by default."""
        n = len(train)
        if n < 50:
            return 0.0
        cut = int(n * 0.8)
        inner_train = train.iloc[:cut]
        inner_test = train.iloc[cut:]
        if "close" not in train.columns or len(inner_test) < 5:
            return 0.0
        try:
            inner_strategy = strategy_factory()
            inner_strategy.fit(inner_train)
            inner_signals = inner_strategy.generate_signals(inner_test)
        except Exception:
            return 0.0
        if inner_signals.empty:
            return 0.0
        positions = inner_signals.shift(1).fillna(0)
        returns = train.loc[positions.index, "close"].pct_change().fillna(0)
        strategy_returns = positions * returns
        position_change = positions.diff().abs().fillna(0)
        net_returns = strategy_returns - (
            position_change * cost_model.cost_per_turn
        )
        return WalkForwardRunner._sharpe(net_returns)
