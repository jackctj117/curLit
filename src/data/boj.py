"""Bank of Japan Time-Series Data provider."""

import logging
from datetime import datetime

import pandas as pd
import httpx

from .base import BaseIngester

logger = logging.getLogger(__name__)


class BoJIngester(BaseIngester):
    BASE_URL = "https://www.stat-search.boj.or.jp/ssi/cgi-bin/famecgi2"

    def __init__(self, db_url: str) -> None:
        super().__init__(db_url, "boj")

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        # BoJ provides CSV exports; simplified placeholder with key series
        logger.info("BoJ ingestion — using stub; extend with actual API integration")
        return pd.DataFrame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return raw
        df = raw.copy()
        df["observation_date"] = pd.to_datetime(df.get("observation_date", pd.Timestamp.utcnow()))
        df["release_date"] = pd.Timestamp.utcnow()
        df["revision"] = 0
        df["source"] = self.source
        return df[["observation_date", "release_date", "series_id", "value", "revision", "source"]]

    def _key_columns(self) -> list[str]:
        return ["observation_date", "release_date", "series_id"]

    def upsert(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        return self._upsert_dataframe(df, "macro_data", self.engine, self._key_columns())
