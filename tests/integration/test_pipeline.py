"""Integration test — end-to-end pipeline: mock data through ingestion, feature, model, signal."""

import logging
import pytest

logger = logging.getLogger(__name__)


class TestPipeline:
    def test_rate_diff_model_pipeline(self) -> None:
        """Verify rate diff model fits and produces signals on synthetic data."""
        import numpy as np
        import pandas as pd
        from src.models.rate_diff import RateDiffModel

        np.random.seed(42)
        n = 300
        spread = np.random.randn(n).cumsum() * 0.01 + 0.5
        price = 1.10 - 0.05 * spread + np.random.randn(n) * 0.002
        df = pd.DataFrame({"target": price, "spread": spread})

        model = RateDiffModel()
        result = model.fit(df)
        assert result["r_squared"] > 0.3
        z = model.deviation_zscore(df)
        assert abs(z.mean()) < 0.3
        assert abs(z.std() - 1.0) < 0.3

    def test_feature_computation(self) -> None:
        import numpy as np
        import pandas as pd
        from src.features import realized_vol, zscore

        price = pd.Series(np.random.randn(100).cumsum() + 100)
        vol = realized_vol(price, window=20)
        assert vol.notna().sum() > 0
        z = zscore(price, window=20)
        assert abs(z.dropna().mean()) < 0.5
