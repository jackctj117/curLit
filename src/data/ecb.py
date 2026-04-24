"""ECB Statistical Data Warehouse provider."""

import logging
from datetime import datetime

import pandas as pd
import httpx

from .base import BaseIngester

logger = logging.getLogger(__name__)

ECB_SERIES = {
    "FM.D.U2.EUR.4F.KR.MRR_FR.LEV": "ECB Deposit Facility Rate",
    "FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA": "EURIBOR 3M",
    "ICP.M.U2.N.000000.4.ANR": "Eurozone HICP",
    "MNA.Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N": "Eurozone GDP",
}


class ECBIngester(BaseIngester):
    BASE_URL = "https://sdw-wsrest.ecb.europa.eu/service/data"

    def __init__(self, db_url: str) -> None:
        super().__init__(db_url, "ecb")
        self.series_ids = list(ECB_SERIES)

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        for sid in self.series_ids:
            params = {
                "startPeriod": start.strftime("%Y-%m-%d"),
                "endPeriod": end.strftime("%Y-%m-%d"),
                "format": "jsondata",
            }
            try:
                resp = httpx.get(f"{self.BASE_URL}/{sid}", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                rows = []
                for obs in (data.get("dataSets", [{}])[0].get("series", {}).get("0:0:0:0:0:0", {}).get("observations", {}).items()):
                    idx_str, val = obs
                    idx = int(idx_str)
                    period = data["structure"]["dimensions"]["observation"][0]["values"][idx]["id"]
                    rows.append({"observation_date": pd.Timestamp(period), "value": val[0]})
                if rows:
                    df = pd.DataFrame(rows)
                    df["series_id"] = sid
                    frames.append(df)
            except Exception:
                logger.warning("ECB fetch failed for %s", sid)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df["release_date"] = pd.Timestamp.utcnow()
        df["revision"] = 0
        df["source"] = self.source
        return df[["observation_date", "release_date", "series_id", "value", "revision", "source"]]

    def _key_columns(self) -> list[str]:
        return ["observation_date", "release_date", "series_id"]

    def upsert(self, df: pd.DataFrame) -> int:
        return self._upsert_dataframe(df, "macro_data", self.engine, self._key_columns())
