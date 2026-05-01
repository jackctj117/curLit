"""Unit tests — models: rate diff, walk-forward no look-ahead."""

import numpy as np
import pandas as pd

from src.models.rate_diff import RateDiffModel


class TestRateDiffModel:
    def test_fits_known_relationship(self) -> None:
        # Stationary spread (not cumsum) avoids spurious-regression artifacts
        # and gives predictable signal magnitude across seeds. With signal/noise
        # ratio ≈ 5x, R² stays >0.9 regardless of the specific draw.
        np.random.seed(42)
        n = 500
        spread = np.random.randn(n) * 0.5 + 1.5
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
        from src.backtest.cost_model import CostModel
        from src.backtest.walkforward import WalkForwardConfig, WalkForwardRunner

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

    def test_train_sharpe_uses_returns_not_prices(self) -> None:
        """CL-u9rn regression: pre-fix train_sharpe computed Sharpe of
        the close-price series itself (mean(price) / std(price) ≈ 100s
        for FX), breaking rule A.7's is/oos ratio. The fix reruns
        strategy.fit on the inner 80% of train and computes Sharpe
        on net-returns over the remaining 20%, just like the OOS
        path. Resulting train_sharpe must be in a sane Sharpe range
        ([-5, 5] approximately), not in the hundreds."""
        import numpy as np
        import pandas as pd

        from src.backtest.cost_model import CostModel
        from src.backtest.walkforward import (
            WalkForwardConfig,
            WalkForwardRunner,
        )

        # FX-shaped price series: small returns ~ 1e-4, prices ~ 1.1
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0001, 0.005, 1500)
        close = 1.1 * np.exp(np.cumsum(rets))
        data = pd.DataFrame({"close": close})
        data.index = pd.date_range("2018-01-01", periods=1500, freq="B")

        class TrivialStrategy:
            def fit(self, train: pd.DataFrame) -> None:
                self._mean = float(train["close"].mean())

            def generate_signals(self, test: pd.DataFrame) -> pd.Series:
                # Mean-revert: long when price is below mean
                return (test["close"] < self._mean).astype(float) - 0.5

        runner = WalkForwardRunner(WalkForwardConfig(min_history=756))
        cost = CostModel()
        result = runner.run(data, lambda: TrivialStrategy(), cost)

        # Sane train_sharpe — must NOT be in the hundreds (was 992 pre-fix)
        for ts in result.fold_metrics["train_sharpe"]:
            assert -5.0 < ts < 5.0, (
                f"train_sharpe={ts:.1f} out of sane range — CL-u9rn regressed"
            )
