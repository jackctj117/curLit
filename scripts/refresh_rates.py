"""Refresh the foreign/derived daily rate series (CL-gr8o).

Runs :class:`src.data.foreign_rates.ForeignRatesIngester` to populate the
five series the rate_diff / carry_vol strategies gate on — DE2Y,
US2Y_MINUS_DE2Y, USD_3M_OIS, EUR_3M_ESTR_OIS and the CVIX proxy — then
logs per-series row counts + last observation dates from macro_data.

Sources: Bundesbank SDMX (DE2Y), FRED (DGS2 spread leg + SOFR90DAYAVG),
ECB Data Portal (3M compounded €STR), own prices table (CVIX proxy) — see
the module docstring in ``src/data/foreign_rates.py`` for the honest
proxy-composition notes.

Meant for a daily cron (after the fx_daily price ingest, so the CVIX
proxy sees fresh closes). Default backfill window is 550 calendar days
(~380 trading days) — cheap, idempotent (insert-only dedup on
observation_date+series_id), and deep enough for the strategies' 250-row
minimums with slack.

Usage:
    .venv/bin/python scripts/refresh_rates.py            # one refresh
    .venv/bin/python scripts/refresh_rates.py --loop 86400
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, text  # noqa: E402

logger = logging.getLogger(__name__)


def _build_db_url() -> str:
    """Mirror the engine/other-scripts Postgres credential resolution."""
    explicit = os.environ.get("DATABASE_URL")
    if explicit:
        return explicit
    user = os.environ.get("POSTGRES_USER", "fx")
    password = os.environ.get("POSTGRES_PASSWORD", "changeme")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "fx")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _one_cycle(db_url: str, backfill_days: int) -> int:
    from src.data.foreign_rates import (  # noqa: PLC0415
        SERIES_SOURCES,
        ForeignRatesIngester,
    )

    end = datetime.now(UTC).replace(tzinfo=None)
    start = end - timedelta(days=backfill_days)
    ingester = ForeignRatesIngester(db_url)
    written = ingester.run(start, end)

    engine = create_engine(db_url)
    lines = []
    with engine.connect() as conn:
        for sid in SERIES_SOURCES:
            n, last = conn.execute(
                text("SELECT COUNT(*), MAX(observation_date) FROM macro_data "
                     "WHERE series_id = :s"),
                {"s": sid},
            ).one()
            lines.append(f"  {sid:18} rows={n:5} last={last}")
    logger.info("foreign_rates refresh: wrote %d new rows", written)
    for line in lines:
        logger.info("%s", line)
    print(f"foreign_rates: wrote {written} new rows")
    for line in lines:
        print(line)
    return written


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(
        description="Refresh foreign/derived daily rate series (CL-gr8o).",
    )
    parser.add_argument(
        "--once", action="store_true", default=True,
        help="Run a single refresh and exit (default).",
    )
    parser.add_argument(
        "--loop", type=int, metavar="SECONDS", default=0,
        help="Refresh every SECONDS forever (e.g. 86400 for daily).",
    )
    parser.add_argument(
        "--backfill-days", type=int, default=550,
        help="Calendar-day backfill window per refresh (default 550).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_url = _build_db_url()
    if args.loop > 0:
        while True:
            try:
                _one_cycle(db_url, args.backfill_days)
            except Exception:
                # Loop mode is a daemon — log and try again next interval.
                logger.exception("foreign_rates refresh cycle failed")
            time.sleep(args.loop)
    _one_cycle(db_url, args.backfill_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
