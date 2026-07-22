"""CLI for Polymarket market-history seeding (CL-3t4j v2).

Pulls per-market implied-probability time-series from Polymarket's
Gamma/CLOB API for every market listed in
``configs/polymarket_markets.yaml`` and upserts to the existing
``prices`` table. Strategies declared with ``symbols=['EURUSD',
'POLY:<slug>']`` then read both via ``DataProvider.get_aligned_series``
without any other plumbing.

Usage:
    python -m scripts.seed_polymarket_history \\
        --start 2024-01-01 --end 2026-05-01

Designed to run alongside ``scripts/seed_historical_data.py`` —
either chain it manually after the FRED/yfinance ingest, or add
to a daily cron.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.data.db_env import build_db_url
from src.data.polymarket import (
    DEFAULT_CONFIG_PATH,
    PolymarketHistoryIngester,
    load_market_config,
)


def _parse_iso_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(
        description=(
            "Seed Polymarket implied-probability history into the prices "
            "table. Strategies use these via 'POLY:<slug>' symbols."
        ),
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="Path to polymarket_markets.yaml",
    )
    parser.add_argument(
        "--start", required=True,
        type=_parse_iso_date,
        help="ISO date for the lower bound (e.g. 2024-01-01)",
    )
    parser.add_argument(
        "--end", required=True,
        type=_parse_iso_date,
        help="ISO date for the upper bound (e.g. 2026-05-01)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After seeding, query Postgres for per-symbol counts + "
             "latest probability. Useful as the operator sanity-check "
             "step from the runbook.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    markets = load_market_config(Path(args.config))
    if not markets:
        print(f"no markets in {args.config}", file=sys.stderr)
        return 2

    placeholder_count = sum(
        1 for m in markets if str(m.get("token_id", "")).startswith("PLACEHOLDER_")
    )
    if placeholder_count:
        print(
            f"WARNING: {placeholder_count} of {len(markets)} markets have "
            f"PLACEHOLDER_ token_ids — these will fetch nothing. Replace "
            f"with real CLOB token_ids from "
            f"https://clob.polymarket.com/markets/<condition_id>",
            file=sys.stderr,
        )

    db_url = build_db_url()
    ingester = PolymarketHistoryIngester(db_url=db_url, markets=markets)
    rows = ingester.run(start=args.start, end=args.end)
    print(f"polymarket: wrote {rows} rows for {len(markets)} markets")

    # Verification (step 5 of the operator workflow): query Postgres
    # for current per-symbol counts and the latest probability so the
    # operator can sanity-check at a glance after seeding.
    if args.verify:
        from sqlalchemy import create_engine, text  # noqa: PLC0415
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT symbol,
                       COUNT(*) AS n,
                       MAX(ts) AS latest_ts,
                       (
                           SELECT close FROM prices p2
                           WHERE p2.symbol = p1.symbol
                           ORDER BY ts DESC LIMIT 1
                       ) AS latest_prob
                FROM prices p1
                WHERE symbol LIKE 'POLY:%'
                GROUP BY symbol
                ORDER BY n DESC
            """)).fetchall()
        if not result:
            print("(verification: no POLY:* rows in prices table)")
        else:
            print(f"\nPOLY:* rows in prices ({len(result)} symbols):")
            print(f"  {'symbol':<50s} {'n':>5s}  {'latest':>10s}  {'prob':>6s}")
            for row in result:
                print(
                    f"  {str(row[0]):<50s} {row[1]:>5d}  "
                    f"{str(row[2])[:10]:>10s}  "
                    f"{float(row[3]):>6.3f}",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
