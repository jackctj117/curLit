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
        """Return a DataFrame indexed by date with one column per requested symbol.

        Queries both `prices` (yfinance daily) and `macro_data` (FRED) and
        merges the results. Symbols missing from one table fall through to
        the other (CL-5rtm — pre-fix, prices-only short-circuit dropped any
        FRED-only symbol like DE_10Y). Returns None only if NO data is found
        for any requested symbol.
        """
        # FRED symbol routing for symbols not found in `prices`.
        fred_map = {
            "US_2Y": "DGS2", "US_10Y": "DGS10",
            "DE_2Y": "IRLTLT01DEM156N", "DE_10Y": "IRLTLT01DEM156N",
            "EURUSD": "DEXUSEU", "US_FEDFUNDS": "DFF",
        }

        per_symbol: list[pd.Series] = []

        # Pass 1: prices table.
        try:
            df = pd.read_sql(
                text("""
                    SELECT ts, symbol, close FROM prices
                    WHERE symbol = ANY(:symbols)
                      AND ts >= :start AND ts <= :end
                    ORDER BY ts
                """),
                self.engine,
                params={"symbols": symbols, "start": start, "end": end},
            )
            if not df.empty:
                df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
                pivot = df.pivot_table(
                    index="ts", columns="symbol", values="close", aggfunc="last",
                )
                for col in pivot.columns:
                    per_symbol.append(pivot[col].astype(float))
        except Exception:
            logger.exception("get_aligned_series: prices query failed")

        # Pass 2: macro_data for any symbol still missing AND FRED-mappable.
        # Without this fallback, queries like ['EURUSD', 'US_10Y', 'DE_10Y']
        # drop DE_10Y (FRED-only) because the prices-only result is non-empty.
        found = {s.name for s in per_symbol}
        missing = [s for s in symbols if s not in found and s in fred_map]
        if missing:
            fred_to_caller = {fred_map[s]: s for s in missing}
            try:
                df = pd.read_sql(
                    text("""
                        SELECT DISTINCT ON (observation_date, series_id)
                            observation_date AS ts, series_id, value AS close
                        FROM macro_data
                        WHERE series_id = ANY(:symbols)
                          AND observation_date >= :start AND observation_date <= :end
                        ORDER BY observation_date, series_id, release_date DESC
                    """),
                    self.engine,
                    params={
                        "symbols": list(fred_to_caller.keys()),
                        "start": start.date(), "end": end.date(),
                    },
                )
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"])
                    pivot = df.pivot_table(
                        index="ts", columns="series_id", values="close", aggfunc="last",
                    )
                    # Rename FRED series_id → caller-facing symbol name.
                    pivot.columns = [fred_to_caller[c] for c in pivot.columns]
                    for col in pivot.columns:
                        per_symbol.append(pivot[col].astype(float))
            except Exception:
                logger.exception("get_aligned_series: macro_data query failed")

        if not per_symbol:
            return None
        # Concat with outer-join on the union of indices. Different sources
        # (daily yfinance vs monthly FRED) will be sparse outside their own
        # timestamps; callers can ffill if they want continuous coverage.
        return pd.concat(per_symbol, axis=1).sort_index()

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
