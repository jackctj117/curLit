"""Data provider — query aligned time-series data from the database."""

import logging
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class DataProvider:
    def __init__(self, engine) -> None:
        self.engine = engine

    def get_aligned_series(
        self, symbols: list[str], start: datetime, end: datetime,
    ) -> pd.DataFrame | None:
        try:
            query = text("""
                SELECT ts, symbol, close
                FROM prices
                WHERE symbol = ANY(:symbols)
                  AND ts >= :start AND ts <= :end
                ORDER BY ts
            """)
            df = pd.read_sql(query, self.engine, params={
                "symbols": symbols, "start": start, "end": end,
            })
            if df.empty:
                return None
            df["ts"] = pd.to_datetime(df["ts"])
            pivot = df.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")
            return pivot
        except Exception:
            logger.debug("Aligned series unavailable (table may not exist yet)")
            return None

    def get_latest_rate_spread(self) -> float | None:
        try:
            query = text("""
                SELECT us.close - de.close AS spread
                FROM prices us
                JOIN prices de ON us.ts = de.ts
                WHERE us.symbol = 'US_2Y' AND de.symbol = 'DE_2Y'
                ORDER BY us.ts DESC LIMIT 1
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()
            return float(result[0]) if result else None
        except Exception:
            return None

    def get_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = text("""
            SELECT ts, symbol, close FROM prices
            WHERE ts >= :start AND ts <= :end
            ORDER BY ts
        """)
        df = pd.read_sql(query, self.engine, params={"start": start, "end": end})
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        return df.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")
