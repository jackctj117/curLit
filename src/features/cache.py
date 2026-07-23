"""Feature caching — materialize computed features to DB."""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def cache_features(
    engine: Any,
    store: Any,
    symbols: list[str],
    lookback_days: int = 756,
) -> int:
    """Compute all features from store and write to features table."""
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    total = 0

    for sym in symbols:
        df = _load_prices(engine, sym, start, end)
        if df.empty:
            continue
        data = {"price": df["close"]}
        features = store.compute(data)
        for col in features.columns:
            fdf = pd.DataFrame(
                {"ts": features.index, "symbol": sym, "feature_name": col, "value": features[col]}
            )
            fdf.to_sql("features", engine, if_exists="append", index=False)
            total += len(fdf)
    logger.info("Cached %d feature rows for %d symbols", total, len(symbols))
    return total


def _load_prices(
    engine: Any,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    query = "SELECT ts, close FROM prices WHERE symbol = :sym AND ts >= :start AND ts <= :end ORDER BY ts"
    return pd.read_sql(
        query, engine, params={"sym": symbol, "start": start, "end": end}, index_col="ts"
    )
