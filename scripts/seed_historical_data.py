#!/usr/bin/env python3
"""Historical data seeding orchestrator (CL-4bu).

Wraps the single-source ingesters in src/data/ into one CLI so we can
seed 2015-onward history into Postgres in one command. Each ingester
already implements BaseIngester.run(start, end) → fetch/transform/
validate/upsert; this script chains them with date ranges, error
handling, and a summary report.

Usage:
    .venv/bin/python scripts/seed_historical_data.py \\
        --start 2015-01-01 --end 2026-04-26 \\
        [--dry-run] [--source fred,yfinance,cftc,cme_sofr]

Environment:
    POSTGRES_HOST/USER/PASSWORD/DB or DATABASE_URL  (required)
    FRED_API_KEY                                    (required for fred source)

Rerunning is safe — ingesters use upsert semantics; existing rows are
skipped, only new rows get inserted.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Make `src.*` imports work when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cftc import CFTCIngester
from src.data.cme_sofr import CMESOFRIngester
from src.data.db_env import build_db_url
from src.data.fred import FREDIngester
from src.data.yfinance_provider import YFinanceIngester

logger = logging.getLogger(__name__)


# Default date range: 11 years of history is enough for walk-forward
# validation (multiple full rate cycles) without ballooning the FRED
# API budget. Override via --start/--end.
DEFAULT_START = datetime(2015, 1, 1, tzinfo=UTC)


@dataclass
class SourceResult:
    name: str
    rows: int = 0
    elapsed_sec: float = 0.0
    status: str = "ok"
    error: str = ""


def _run_source(
    name: str,
    factory: Callable[[], object],
    start: datetime,
    end: datetime,
    dry_run: bool,
) -> SourceResult:
    """Run one ingester end-to-end with timing and error capture."""
    t0 = time.time()
    if dry_run:
        logger.info(
            "[%s] DRY-RUN — would seed [%s … %s]",
            name,
            start.date(),
            end.date(),
        )
        return SourceResult(
            name=name,
            rows=0,
            elapsed_sec=0.0,
            status="dry-run",
        )
    try:
        ingester = factory()
        rows = ingester.run(start, end)
    except KeyError as exc:
        # Missing env var (e.g. FRED_API_KEY) — fail loud but isolate to
        # this source.
        return SourceResult(
            name=name,
            rows=0,
            elapsed_sec=time.time() - t0,
            status="env-missing",
            error=f"missing env var: {exc}",
        )
    except Exception as exc:
        logger.exception("[%s] ingest failed", name)
        return SourceResult(
            name=name,
            rows=0,
            elapsed_sec=time.time() - t0,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    return SourceResult(
        name=name,
        rows=int(rows),
        elapsed_sec=time.time() - t0,
        status="ok",
    )


def _factories(db_url: str) -> dict[str, Callable[[], object]]:
    """Map source name → zero-arg constructor."""
    return {
        "fred": lambda: FREDIngester(db_url),
        "yfinance": lambda: YFinanceIngester(db_url),
        "cftc": lambda: CFTCIngester(db_url),
        "cme_sofr": lambda: CMESOFRIngester(db_url),
    }


def _print_summary(results: list[SourceResult]) -> None:
    """Pretty-print a one-line-per-source summary."""
    width_name = max(len(r.name) for r in results)
    print()
    print(f"{'source':<{width_name}}  {'status':<11}  {'rows':>10}  {'elapsed':>10}  detail")
    print("-" * (width_name + 60))
    total_rows = 0
    for r in results:
        elapsed = f"{r.elapsed_sec:.1f}s"
        detail = r.error if r.error else ""
        print(f"{r.name:<{width_name}}  {r.status:<11}  {r.rows:>10}  {elapsed:>10}  {detail}")
        total_rows += r.rows
    print("-" * (width_name + 60))
    print(f"{'TOTAL':<{width_name}}  {'':<11}  {total_rows:>10}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed historical market data into Postgres.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START.strftime("%Y-%m-%d"),
        help=f"Start date YYYY-MM-DD (default {DEFAULT_START.date()})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        help="Comma-separated source names, or 'all' (default). "
        "Available: fred, yfinance, cftc, cme_sofr.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without fetching or writing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    if end <= start:
        logger.error("end (%s) must be after start (%s)", end, start)
        return 2

    db_url = build_db_url()
    factories = _factories(db_url)

    if args.source == "all":
        sources = list(factories.keys())
    else:
        sources = [s.strip() for s in args.source.split(",") if s.strip()]
        unknown = set(sources) - set(factories)
        if unknown:
            logger.error("unknown sources: %s. Available: %s", sorted(unknown), sorted(factories))
            return 2

    logger.info(
        "Seeding %s from %s to %s%s",
        ",".join(sources),
        start.date(),
        end.date(),
        " (dry-run)" if args.dry_run else "",
    )

    results: list[SourceResult] = []
    for name in sources:
        result = _run_source(name, factories[name], start, end, args.dry_run)
        results.append(result)

    _print_summary(results)

    # Non-zero exit if any source genuinely errored — env-missing and
    # dry-run are reported but don't fail the whole run.
    n_errors = sum(1 for r in results if r.status == "error")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
