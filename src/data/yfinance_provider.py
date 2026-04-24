"""Yahoo Finance data provider — free daily OHLCV for FX, yields, commodities."""

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from .base import BaseIngester

logger = logging.getLogger(__name__)

SYMBOL_MAP: dict[str, str] = {
    # FX pairs (yfinance uses XXXYYY=X format)
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "USDCHF=X",
    # Yields (yfinance uses ^TNX-like tickers)
    "US_2Y": "2YY=F",         # US 2Y Treasury futures
    "US_10Y": "^TNX",         # 10-year Treasury
    "DE_2Y": "2YY=F",         # German 2Y — proxy with US (DE not available on yfinance)
    "US_FEDFUNDS": "^IRX",    # 13-week T-bill as proxy
    # Commodities
    "GOLD": "GC=F",
    "OIL_WTI": "CL=F",
    "COPPER": "HG=F",
    # Indices
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
}


class YFinanceProvider:
    """Wrapper around yfinance for batch daily data downloads."""

    def fetch_daily_batch(
        self, symbols: list[str], start: datetime, end: datetime,
    ) -> pd.DataFrame:
        yf_symbols = [SYMBOL_MAP.get(s, s) for s in symbols]
        tickers = yf.download(
            yf_symbols, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True,
        )
        if tickers.empty:
            return pd.DataFrame()

        # Multi-level columns: (Close, SYM), (Open, SYM), etc.
        frames = []
        for sym, yf_sym in zip(symbols, yf_symbols):
            try:
                if isinstance(tickers.columns, pd.MultiIndex):
                    close = tickers[("Close", yf_sym)]
                else:
                    close = tickers["Close"]
                if close.dropna().empty:
                    continue
                df = pd.DataFrame({"ts": close.index, "symbol": sym, "close": close.values})
                frames.append(df)
            except KeyError:
                logger.debug("No data for %s (%s)", sym, yf_sym)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class YFinanceIngester(BaseIngester):
    """Ingester that writes daily OHLCV data to the prices table via yfinance."""

    def __init__(self, db_url: str, symbols: list[str] | None = None) -> None:
        super().__init__(db_url, "yfinance")
        self.provider = YFinanceProvider()
        self.symbols = symbols or list(SYMBOL_MAP)

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        return self.provider.fetch_daily_batch(self.symbols, start, end)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return raw
        df = raw.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["source"] = self.source
        df["open"] = 0.0
        df["high"] = 0.0
        df["low"] = 0.0
        df["volume"] = 0.0
        keep = ["ts", "symbol", "source", "open", "high", "low", "close", "volume"]
        return df[[c for c in keep if c in df.columns]]

    def _key_columns(self) -> list[str]:
        return ["ts", "symbol", "source"]

    def upsert(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        return self._upsert_dataframe(df, "prices", self.engine, self._key_columns())
