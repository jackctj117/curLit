"""Compute G10 equal-weighted 20-day realized vol → fx_volatility (CL-6h6).

For each trading day with at least 21 closes for each G10 pair, computes
log-return std × sqrt(252) per pair, then averages across pairs to get
G10_RV20. Writes to fx_volatility (CL-43l) as ``index_name='G10_RV20'``.

Idempotent — uses INSERT … ON CONFLICT to update existing rows.

Cadence: run nightly after yfinance ingest. Light enough to be a cron
one-liner; cost is dominated by the read query, not the math.

Usage:
  .venv/bin/python -m scripts.compute_g10_realized_vol [--lookback-days 365]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)


# G10 USD pairs — yfinance ticker conventions in our prices table.
# Excludes USD itself (no USD/USD pair). Subset matches what's actually
# ingested today; the script silently skips pairs with no data so adding
# more here is safe.
G10_PAIRS: tuple[str, ...] = (
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "USDNOK",
    "USDSEK",
)

WINDOW: int = 20  # 20 trading days ≈ 1 month
ANNUALIZATION: float = 252.0


def _load_closes(engine: Any, start: datetime, end: datetime) -> pd.DataFrame:
    """Wide DataFrame: index=date, columns=pair, values=close."""
    df = pd.read_sql(
        text("""
            SELECT ts, symbol, close FROM prices
            WHERE symbol = ANY(:syms)
              AND ts >= :start AND ts <= :end
            ORDER BY ts
        """),
        engine,
        params={"syms": list(G10_PAIRS), "start": start, "end": end},
    )
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")


def compute_g10_rv20(closes: pd.DataFrame) -> pd.Series:
    """Equal-weighted G10 RV20: per-pair rolling std of log returns,
    then mean across pairs. Skips dates where any pair lacks ``WINDOW+1``
    prior observations (NaN propagates from the rolling window)."""
    if closes.empty:
        return pd.Series(dtype=float, name="G10_RV20")
    log_ret = np.log(closes / closes.shift(1))
    per_pair_rv = log_ret.rolling(WINDOW).std(ddof=1) * np.sqrt(ANNUALIZATION)
    g10 = per_pair_rv.mean(axis=1, skipna=False).dropna()
    g10.name = "G10_RV20"
    return g10


def upsert_volatility(engine: Any, index_name: str, series: pd.Series) -> int:
    """INSERT … ON CONFLICT idempotent write to fx_volatility."""
    if series.empty:
        return 0
    rows = [
        {"d": ts.date(), "name": index_name, "v": float(v)}
        for ts, v in series.items()
        if pd.notna(v)
    ]
    with engine.begin() as conn:
        # Postgres-only ON CONFLICT — the production DB. The function
        # also runs on sqlite in tests using INSERT OR REPLACE; we
        # detect dialect at write time.
        dialect = engine.dialect.name
        if dialect == "postgresql":
            stmt = text("""
                INSERT INTO fx_volatility (date, index_name, value)
                VALUES (:d, :name, :v)
                ON CONFLICT (date, index_name) DO UPDATE SET value = EXCLUDED.value
            """)
        else:
            stmt = text(
                "INSERT OR REPLACE INTO fx_volatility "
                "(date, index_name, value) VALUES (:d, :name, :v)",
            )
        for row in rows:
            conn.execute(stmt, row)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    p = argparse.ArgumentParser(description="Compute G10 RV20 → fx_volatility.")
    p.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="How far back to recompute (idempotent, so safe to overlap)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.runtime.run_engine import _build_db_engine  # noqa: PLC0415

    engine = _build_db_engine()

    end = datetime.utcnow()
    # Pad lookback by WINDOW+5 trading days so the rolling window has
    # enough warm-up to emit a value on the first requested date.
    start = end - timedelta(days=args.lookback_days + WINDOW + 5)

    closes = _load_closes(engine, start, end)
    if closes.empty:
        logger.warning("No G10 closes in [%s, %s] — nothing to write", start, end)
        return 1

    rv = compute_g10_rv20(closes)
    written = upsert_volatility(engine, "G10_RV20", rv)
    logger.info(
        "Wrote %d rows of G10_RV20 (range %s → %s)",
        written,
        rv.index.min() if not rv.empty else None,
        rv.index.max() if not rv.empty else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
