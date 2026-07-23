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
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

# Make `src` importable when launched as a file (harmless under -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    parser = argparse.ArgumentParser(
        description="Refresh the US-listed symbol universe from NASDAQ Trader.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="Run a single refresh and exit (default; only mode supported).",
    )
    parser.add_argument(
        "--no-sec",
        action="store_true",
        help="Skip the SEC EDGAR name/CIK enrichment step (CL-9xha).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.data.symbols import SymbolUniverse  # noqa: PLC0415

    engine = create_engine(build_db_url())
    universe = SymbolUniverse(engine)

    counts = universe.refresh()

    # SEC EDGAR name/CIK overlay (CL-9xha). Best-effort: a SEC fetch failure
    # logs and returns zeros without disturbing the NASDAQ-sourced rows.
    sec_counts: dict[str, int] | None = None
    if not args.no_sec:
        sec_counts = universe.refresh_sec_names()

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM symbols")).scalar()
        by_exchange = conn.execute(
            text(
                "SELECT exchange, COUNT(*) FROM symbols GROUP BY exchange ORDER BY COUNT(*) DESC",
            )
        ).fetchall()
        etf_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM symbols WHERE is_etf",
            )
        ).scalar()
        last_refreshed = conn.execute(
            text(
                "SELECT MAX(last_refreshed) FROM symbols",
            )
        ).scalar()
        sec_covered = None
        if sec_counts is not None:
            sec_covered = conn.execute(
                text(
                    "SELECT COUNT(*) FROM symbols WHERE sec_name IS NOT NULL",
                )
            ).scalar()

    logger.info("symbols refresh complete: %s", counts)
    if sec_counts is not None:
        logger.info("SEC enrichment: %s (sec_name populated on %s rows)", sec_counts, sec_covered)
    logger.info("total symbols in table: %s", total)
    for exchange, n in by_exchange:
        logger.info("  %-6s %d", exchange, n)
    logger.info("  ETFs (flagged): %d", etf_count)
    logger.info("last_refreshed: %s", last_refreshed)

    print(
        f"symbols: inserted={counts['inserted']} updated={counts['updated']} "
        f"skipped={counts['skipped']} total={total} etfs={etf_count}",
    )
    if sec_counts is not None:
        print(
            f"sec: matched={sec_counts['matched']} "
            f"unmatched={sec_counts['unmatched']} covered={sec_covered}",
        )
    for exchange, n in by_exchange:
        print(f"  {exchange}: {n}")
    print(f"last_refreshed: {last_refreshed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
