"""NLP data provider — query CB sentiment and diff events from the database."""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


class NLPDataProvider:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def get_recent_diff_events(self, since: datetime, cbs: list[str]) -> list[dict[str, Any]]:
        try:
            query = text("""
                SELECT ts, cb, doc_id, prev_doc_id, net_shift,
                       added_hawkish, removed_hawkish, change_ratio
                FROM cb_diff_events
                WHERE ts >= :since AND cb = ANY(:cbs)
                ORDER BY ts DESC
            """)
            df = pd.read_sql(query, self.engine, params={"since": since, "cbs": cbs})
            return df.to_dict("records")
        except Exception:
            logger.debug("No diff events available (table may not exist yet)")
            return []

    def get_historical_diff_scores(self, cbs: list[str], lookback_years: int = 5) -> pd.DataFrame:
        since = datetime.utcnow() - timedelta(days=lookback_years * 365)
        try:
            query = text("""
                SELECT ts, cb, net_shift
                FROM cb_diff_events
                WHERE ts >= :since AND cb = ANY(:cbs)
                ORDER BY ts
            """)
            return pd.read_sql(query, self.engine, params={"since": since, "cbs": cbs})
        except Exception:
            logger.debug("Historical diff scores unavailable")
            return pd.DataFrame()
