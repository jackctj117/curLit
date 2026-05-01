"""FRED (Federal Reserve Economic Data) provider with ALFRED vintage support."""

import logging
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import text
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

from .base import BaseIngester

logger = logging.getLogger(__name__)

FRED_SERIES: dict[str, str] = {
    "DFF": "Effective Federal Funds Rate",
    "DGS2": "US Treasury 2Y",
    "DGS10": "US Treasury 10Y",
    "SOFR": "Secured Overnight Financing Rate",
    "DFEDTARL": "Fed Funds Target Lower",
    "DFEDTARU": "Fed Funds Target Upper",
    "CPIAUCSL": "CPI All Urban",
    "CPILFESL": "Core CPI",
    "PCEPI": "PCE Price Index",
    "PCEPILFE": "Core PCE",
    "UNRATE": "Unemployment Rate",
    "PAYEMS": "Nonfarm Payrolls",
    "CES0500000003": "Avg Hourly Earnings",
    "JTSJOL": "Job Openings",
    "GDPC1": "Real GDP",
    "INDPRO": "Industrial Production",
    "RSAFS": "Retail Sales",
    "UMCSENT": "Consumer Sentiment",
    "DTWEXBGS": "Trade Weighted USD Index (broad)",
    "DEXUSEU": "USD/EUR Exchange Rate",
    "DEXJPUS": "JPY/USD Exchange Rate",
    "IRLTLT01DEM156N": "Germany 10Y Bond Yield",
    "IRLTLT01JPM156N": "Japan 10Y Bond Yield",
    "IRLTLT01GBM156N": "UK 10Y Bond Yield",
    # G10 long-term yields used by the carry_vol_filter backtest (CL-lcxd).
    "IRLTLT01CAM156N": "Canada 10Y Bond Yield",
    "IRLTLT01AUM156N": "Australia 10Y Bond Yield",
    "IRLTLT01CHM156N": "Switzerland 10Y Bond Yield",
    "IRLTLT01NZM156N": "New Zealand 10Y Bond Yield",
    "IRLTLT01NOM156N": "Norway 10Y Bond Yield",
    "IRLTLT01SEM156N": "Sweden 10Y Bond Yield",
}


class FREDProvider:
    BASE_URL = "https://api.stlouisfed.org/fred"
    RATE_LIMIT = 0.5

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ["FRED_API_KEY"]
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < 1.0 / self.RATE_LIMIT:
            time.sleep(1.0 / self.RATE_LIMIT - elapsed)
        self._last_request = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self._throttle()
        params = {**params, "api_key": self.api_key, "file_type": "json"}
        resp = httpx.get(f"{self.BASE_URL}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def get_series(
        self,
        series_id: str,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> pd.DataFrame:
        if isinstance(start, str):
            start = datetime.fromisoformat(start[:10])
        if isinstance(end, str):
            end = datetime.fromisoformat(end[:10])
        params: dict[str, Any] = {"series_id": series_id}
        if start:
            params["observation_start"] = start.strftime("%Y-%m-%d")
        if end:
            params["observation_end"] = end.strftime("%Y-%m-%d")
        data = self._get("series/observations", params)
        if not data.get("observations"):
            return pd.DataFrame()
        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.set_index("date")[["value"]]
        df.columns = [series_id]
        return df

    def get_vintage(self, series_id: str, as_of_date: datetime) -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "realtime_start": as_of_date.strftime("%Y-%m-%d"),
            "realtime_end": as_of_date.strftime("%Y-%m-%d"),
        }
        data = self._get("series/observations", params)
        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("date")[["value"]]

    def get_all_releases(self, series_id: str) -> pd.DataFrame:
        data = self._get("series/observations", {
            "series_id": series_id,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
        })
        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["realtime_start"] = pd.to_datetime(df["realtime_start"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df


class FREDIngester(BaseIngester):
    def __init__(self, db_url: str, api_key: str | None = None, series_ids: list[str] | None = None) -> None:
        super().__init__(db_url, "fred")
        self.provider = FREDProvider(api_key)
        self.series_ids = series_ids or list(FRED_SERIES)

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        for sid in self.series_ids:
            df = self.provider.get_series(sid, start, end)
            if df.empty:
                continue
            df = df.reset_index()
            df["series_id"] = sid
            df = df.rename(columns={sid: "value"})
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={"date": "observation_date"})
        # release_date is recorded as the moment we ingested this row, but
        # is not part of the dedup key — otherwise reruns create duplicate
        # observation rows because each rerun gets a fresh utcnow() (CL-mht0).
        # The schema's PK includes release_date for ALFRED-vintage support;
        # the ingester intentionally treats (observation_date, series_id) as
        # the natural dedup key.
        df["release_date"] = pd.Timestamp.utcnow()
        df["revision"] = 0
        df["source"] = self.source
        df = df[["observation_date", "release_date", "series_id", "value", "revision", "source"]]
        return df.dropna(subset=["value"])

    def _key_columns(self) -> list[str]:
        return ["observation_date", "series_id"]

    def upsert(self, df: pd.DataFrame) -> int:
        return self._upsert_dataframe(df, "macro_data", self.engine, self._key_columns())
