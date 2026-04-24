"""CME SOFR futures provider — SR1 (1-month) and SR3 (3-month) settlement data."""

import logging
from datetime import date, datetime

import pandas as pd
import httpx

from .base import BaseIngester

logger = logging.getLogger(__name__)

PRODUCT_IDS = {"SR1": "8463", "SR3": "8462"}

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


class CMESOFRProvider:
    SETTLEMENTS_URL = (
        "https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures/Settlements"
        "/{product_id}/FUT"
    )

    def fetch_settlements(self, product: str, trade_date: date | None = None) -> pd.DataFrame:
        url = self.SETTLEMENTS_URL.format(product_id=PRODUCT_IDS[product])
        params = {}
        if trade_date:
            params["tradeDate"] = trade_date.strftime("%m/%d/%Y")
        resp = httpx.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for s in data.get("settlements", []):
            month_code = s["month"]
            price_str = s["settle"]
            if price_str in ("-", "Cab"):
                continue
            price = float(price_str)
            implied_rate = (100.0 - price) / 100.0
            expiry = self._parse_month_code(month_code)
            rows.append({
                "product": product,
                "month_code": month_code,
                "expiry": expiry,
                "settle_price": price,
                "implied_rate": implied_rate,
                "trade_date": data.get("tradeDate"),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_month_code(code: str) -> date:
        parts = code.strip().split()
        month = MONTH_MAP[parts[0].upper()]
        year = 2000 + int(parts[1])
        d = date(year, month, 15)
        while d.weekday() != 2:
            d = date(year, month, d.day + 1)
        return d


class CMESOFRIngester(BaseIngester):
    def __init__(self, db_url: str) -> None:
        super().__init__(db_url, "cme_sofr")
        self.provider = CMESOFRProvider()

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        for product in ["SR1", "SR3"]:
            df = self.provider.fetch_settlements(product)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["ts"] = pd.Timestamp.utcnow()
        df["curve_id"] = "sofr_futures"
        df = df.rename(columns={"expiry_days": "tenor_days"})
        # Compute tenor_days from expiry
        now = pd.Timestamp.utcnow().date()
        df["tenor_days"] = df["expiry"].apply(lambda d: (d - now).days)
        df["rate"] = df["implied_rate"]
        df["source"] = self.source
        return df[["ts", "curve_id", "tenor_days", "rate", "source"]]

    def _key_columns(self) -> list[str]:
        return ["ts", "curve_id", "tenor_days"]

    def upsert(self, df: pd.DataFrame) -> int:
        return self._upsert_dataframe(df, "rate_curves", self.engine, self._key_columns())
