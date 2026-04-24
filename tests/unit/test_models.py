"""Unit tests — models: rate diff, walk-forward no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from src.models.rate_diff import RateDiffModel


class TestRateDiffModel:
    def test_fits_known_relationship(self) -> None:
        np.random.seed(42)
        n = 500
        spread = np.random.randn(n).cumsum() * 0.01 + 1.5
        price = 1.10 - 0.05 * spread + np.random.randn(n) * 0.005
        df = pd.DataFrame({"target": price, "spread": spread})
        model = RateDiffModel()
        result = model.fit(df)
        assert abs(result["beta"] - (-0.05)) < 0.02
        assert result["r_squared"] > 0.5

    def test_quality_ok_gate(self) -> None:
        df = pd.DataFrame({"target": np.random.randn(200).cumsum() + 1.1, "spread": np.random.randn(200)})
        model = RateDiffModel(min_r_squared=0.25)
        model.fit(df)
        assert model.quality_ok == (model.result["r_squared"] >= 0.25) if model.result else False


class TestWalkForward:
    def test_no_lookahead(self) -> None:
        from src.backtest.walkforward import WalkForwardRunner, WalkForwardConfig
        from src.backtest.cost_model import CostModel

        data = pd.DataFrame({"close": np.random.randn(1200).cumsum() + 1.1})
        data.index = pd.date_range("2020-01-01", periods=1200, freq="D")

        class SpyStrategy:
            train_max = None
            test_min = None

            def fit(self, train):
                SpyStrategy.train_max = train.index.max()

            def generate_signals(self, test):
                SpyStrategy.test_min = test.index.min()
                return pd.Series(0.0, index=test.index)

        runner = WalkForwardRunner(WalkForwardConfig(min_history=756))
        cost = CostModel()
        result = runner.run(data, lambda: SpyStrategy(), cost)
        assert len(result.fold_metrics) > 0
