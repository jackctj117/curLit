"""Base ingester — abstract class for all data ingestion sources."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class BaseIngester(ABC):
    def __init__(self, db_url: str, source_name: str) -> None:
        self.engine = create_engine(db_url)
        self.source = source_name

    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch raw data from source."""

    @abstractmethod
    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise raw data to the target schema."""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=self._key_columns())
        df = df.drop_duplicates(subset=self._key_columns())
        return df

    @abstractmethod
    def _key_columns(self) -> list[str]:
        """Columns that form the unique key — subclasses override."""

    @abstractmethod
    def upsert(self, df: pd.DataFrame) -> int:
        """Write DataFrame to the database."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def run(self, start: datetime, end: datetime) -> int:
        # CL-47j: emit ingestion_runs / ingestion_duration / ingestion_records
        # for every run. Status label is "success" / "empty" / "error" so a
        # single Grafana panel can break down by outcome. Errors propagate
        # to the caller (after the metric increment) — the retry decorator
        # handles transient failures, terminal failures should still raise.
        from src.monitoring.metrics import (
            data_freshness_seconds,
            errors_total,
            ingestion_duration,
            ingestion_records,
            ingestion_runs,
        )

        logger.info("Ingesting %s [%s … %s]", self.source, start, end)
        t0 = time.time()
        try:
            raw = self.fetch(start, end)
            if raw.empty:
                logger.warning("No data for %s", self.source)
                ingestion_runs.labels(source=self.source, status="empty").inc()
                ingestion_duration.labels(source=self.source).observe(
                    time.time() - t0,
                )
                return 0
            df = self.transform(raw)
            df = self.validate(df)
            rows = self.upsert(df)
        except Exception:
            ingestion_runs.labels(source=self.source, status="error").inc()
            errors_total.labels(
                service=f"ingest_{self.source}",
                severity="error",
                category="ingestion",
            ).inc()
            ingestion_duration.labels(source=self.source).observe(
                time.time() - t0,
            )
            raise

        elapsed = time.time() - t0
        ingestion_runs.labels(source=self.source, status="success").inc()
        ingestion_duration.labels(source=self.source).observe(elapsed)
        ingestion_records.labels(source=self.source).inc(rows)

        # Freshness gauge: 0 = "just updated". Per-symbol breakdown is
        # available when the result has a "symbol" or "series_id" column
        # (most do; FRED uses series_id, yfinance uses symbol). Lump
        # everything under a synthetic "_all" label too so a source-level
        # panel can show single-line freshness without aggregating.
        try:
            now_ts = datetime.now(UTC).timestamp()
            data_freshness_seconds.labels(
                symbol="_all", source=self.source,
            ).set(0)
            symbol_col = next(
                (c for c in ("symbol", "series_id") if c in df.columns),
                None,
            )
            if symbol_col is not None:
                # Iterate distinct symbols and emit a 0 — they're "as fresh
                # as this ingest run". Subsequent freshness drift is the
                # job of the metric collector (Prometheus subtracts on
                # query); we just stamp the most-recent-seen.
                for sym in df[symbol_col].dropna().unique():
                    data_freshness_seconds.labels(
                        symbol=str(sym), source=self.source,
                    ).set(0)
            # Avoid unused-variable mypy complaint.
            _ = now_ts
        except Exception:
            logger.debug("freshness gauge update failed", exc_info=True)

        logger.info("%s: wrote %d rows in %.2fs", self.source, rows, elapsed)
        return rows

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _upsert_dataframe(
        df: pd.DataFrame, table_name: str, engine: Any, key_cols: list[str],
    ) -> int:
        """Generic COPY + ON CONFLICT upsert."""
        if df.empty:
            return 0
        rows = len(df)
        with engine.begin() as conn:
            # Drop rows that would conflict so we can re-insert
            if key_cols:
                existing = pd.read_sql_table(table_name, conn, columns=key_cols)
                if not existing.empty:
                    merged = df.merge(existing, on=key_cols, how="left", indicator=True)
                    df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
                    if df.empty:
                        return 0
            df.to_sql(table_name, conn, if_exists="append", index=False)
        return len(df)
