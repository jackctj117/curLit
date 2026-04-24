"""Base ingester — abstract class for all data ingestion sources."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

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
        logger.info("Ingesting %s [%s … %s]", self.source, start, end)
        raw = self.fetch(start, end)
        if raw.empty:
            logger.warning("No data for %s", self.source)
            return 0
        df = self.transform(raw)
        df = self.validate(df)
        rows = self.upsert(df)
        logger.info("%s: wrote %d rows", self.source, rows)
        return rows

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _upsert_dataframe(
        df: pd.DataFrame, table_name: str, engine: object, key_cols: list[str],
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
