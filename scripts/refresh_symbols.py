"""Refresh the US-listed symbol universe (CL-tzug).

Fetches the two free NASDAQ Trader public files (nasdaqlisted.txt +
otherlisted.txt), parses + upserts them into the ``symbols`` table via
:class:`src.data.symbols.SymbolUniverse`, and logs the inserted/updated/
skipped counts plus the ``last_refreshed`` stamp and a by-exchange breakdown.

The NASDAQ Trader files update every trading day, so this is meant to run on
a daily schedule. TODO: wire into the Airflow ``event_ingestion`` DAG (or a
standalone daily cron job) so the universe stays current without manual runs.

Usage:
    .venv/bin/python -m scripts.refresh_symbols            # --once (default)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from sqlalchemy import create_engine, text

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


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(
        description="Refresh the US-listed symbol universe from NASDAQ Trader.",
    )
    parser.add_argument(
        "--once", action="store_true", default=True,
        help="Run a single refresh and exit (default; only mode supported).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.data.symbols import SymbolUniverse  # noqa: PLC0415

    engine = create_engine(_build_db_url())
    universe = SymbolUniverse(engine)

    counts = universe.refresh()

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM symbols")).scalar()
        by_exchange = conn.execute(text(
            "SELECT exchange, COUNT(*) FROM symbols "
            "GROUP BY exchange ORDER BY COUNT(*) DESC",
        )).fetchall()
        etf_count = conn.execute(text(
            "SELECT COUNT(*) FROM symbols WHERE is_etf",
        )).scalar()
        last_refreshed = conn.execute(text(
            "SELECT MAX(last_refreshed) FROM symbols",
        )).scalar()

    logger.info("symbols refresh complete: %s", counts)
    logger.info("total symbols in table: %s", total)
    for exchange, n in by_exchange:
        logger.info("  %-6s %d", exchange, n)
    logger.info("  ETFs (flagged): %d", etf_count)
    logger.info("last_refreshed: %s", last_refreshed)

    print(
        f"symbols: inserted={counts['inserted']} updated={counts['updated']} "
        f"skipped={counts['skipped']} total={total} etfs={etf_count}",
    )
    for exchange, n in by_exchange:
        print(f"  {exchange}: {n}")
    print(f"last_refreshed: {last_refreshed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
