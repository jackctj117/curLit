"""CFTC COT (Commitments of Traders) provider — weekly TFF report parsing."""

import logging
from datetime import datetime

import pandas as pd

from .base import BaseIngester

logger = logging.getLogger(__name__)


class CFTCIngester(BaseIngester):
    """Downloads and parses the weekly Traders in Financial Futures (TFF) report."""

    REPORT_URL = "https://www.cftc.gov/dea/newcot/f_disagg.txt"

    def __init__(self, db_url: str) -> None:
        super().__init__(db_url, "cftc_cot")

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        df = pd.read_csv(
            self.REPORT_URL,
            skiprows=2,  # skip header rows
            header=None,
            sep=r"\s*,\s*|\s{2,}",
            engine="python",
        )
        # COT TFF format (simplified columns)
        cols = [
            "market_and_exchange", "symbol", "contract_name",
            "cfc_code", "cot_date",
            "dealer_long", "dealer_short", "dealer_spread",
            "asset_mgr_long", "asset_mgr_short", "asset_mgr_spread",
            "lev_fund_long", "lev_fund_short", "lev_fund_spread",
            "other_report_long", "other_report_short", "other_report_spread",
            "non_report_long", "non_report_short", "non_report_spread",
        ]
        if len(df.columns) >= len(cols):
            df.columns = cols[: len(df.columns)]
        df["cot_date"] = pd.to_datetime(df["cot_date"], format="%y%m%d")
        return df

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        rows = []
        categories = {
            "dealer": ("dealer_long", "dealer_short"),
            "asset_mgr": ("asset_mgr_long", "asset_mgr_short"),
            "lev_funds": ("lev_fund_long", "lev_fund_short"),
            "other_report": ("other_report_long", "other_report_short"),
            "non_report": ("non_report_long", "non_report_short"),
        }
        for _, r in raw.iterrows():
            for cat, (lc, sc) in categories.items():
                if lc not in r or sc not in r:
                    continue
                rows.append({
                    "report_date": r["cot_date"],
                    "symbol": str(r.get("symbol", "")),
                    "category": cat,
                    "longs": int(r[lc]) if pd.notna(r[lc]) else 0,
                    "shorts": int(r[sc]) if pd.notna(r[sc]) else 0,
                    "spreads": 0,
                    "open_interest": (int(r[lc]) + int(r[sc])) if pd.notna(r[lc]) and pd.notna(r[sc]) else 0,
                })
        return pd.DataFrame(rows)

    def _key_columns(self) -> list[str]:
        return ["report_date", "symbol", "category"]

    def upsert(self, df: pd.DataFrame) -> int:
        return self._upsert_dataframe(df, "cot_positioning", self.engine, self._key_columns())
