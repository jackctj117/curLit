"""Stooq data provider — free daily OHLC for FX, yields, commodities, indices."""

import logging
from datetime import datetime
from io import StringIO

import pandas as pd
import httpx

from .base import BaseIngester

logger = logging.getLogger(__name__)

SYMBOL_MAP: dict[str, str] = {
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "USDCAD": "usdcad",
    "AUDUSD": "audusd",
    "NZDUSD": "nzdusd",
    "USDCHF": "usdchf",
    "DXY": "^dxy",
    "US_2Y": "2usy.b",
    "US_10Y": "10usy.b",
    "DE_10Y": "10dey.b",
    "GB_10Y": "10gby.b",
    "JP_10Y": "10jpy.b",
    "GOLD": "xauusd",
    "OIL_WTI": "cl.f",
    "COPPER": "hg.f",
    "SPX": "^spx",
    "VIX": "^vix",
}


class StooqProvider:
    BASE_URL = "https://stooq.com/q/d/l/"

    def fetch_daily(
        self, symbol: str, start: datetime, end: datetime, interval: str = "d",
    ) -> pd.DataFrame:
        code = SYMBOL_MAP.get(symbol, symbol.lower())
        params = {
            "s": code,
            "i": interval,
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
        }
        resp = httpx.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df.columns = [c.lower() for c in df.columns]
        df["symbol"] = symbol
        return df


class StooqIngester(BaseIngester):
    def __init__(self, db_url: str, symbols: list[str] | None = None) -> None:
        super().__init__(db_url, "stooq")
        self.provider = StooqProvider()
        self.symbols = symbols or list(SYMBOL_MAP)

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        for sym in self.symbols:
            try:
                df = self.provider.fetch_daily(sym, start, end)
                frames.append(df.reset_index())
            except Exception:
                logger.warning(
                    "Stooq fetch failed for %s", sym, exc_info=True,
                )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={"Date": "ts"})
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["source"] = self.source
        keep = [c for c in ["ts", "symbol", "source", "open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]

    def _key_columns(self) -> list[str]:
        return ["ts", "symbol", "source"]

    def upsert(self, df: pd.DataFrame) -> int:
        return self._upsert_dataframe(df, "prices", self.engine, self._key_columns())
