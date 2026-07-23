"""Cron entry point for daily broker reconciliation (CL-unlt).

Runs once per day (after midnight UTC, before any morning trading) to
diff yesterday's OANDA transactions against the internal trade journal.

Usage:
  .venv/bin/python -m scripts.run_daily_reconciliation [--date YYYY-MM-DD]
                                                       [--practice]
                                                       [--quiet-success]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import UTC, datetime

import httpx

from src.dotenv_bootstrap import load_project_env
from src.execution.broker_reconciliation import run_daily_reconciliation

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    load_project_env()

    p = argparse.ArgumentParser(
        description="Diff OANDA fills vs internal trade journal.",
    )
    p.add_argument(
        "--date",
        default=None,
        help="UTC date to reconcile (default: today). Format YYYY-MM-DD.",
    )
    p.add_argument(
        "--practice",
        action="store_true",
        help="Use OANDA practice host instead of fxtrade",
    )
    p.add_argument(
        "--quiet-success",
        action="store_true",
        help="Print nothing on a clean day (cron-friendly)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    api_key = os.environ.get("OANDA_API_KEY")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not api_key or not account_id:
        print(
            "OANDA_API_KEY / OANDA_ACCOUNT_ID not set — cannot reconcile",
            file=sys.stderr,
        )
        return 2

    base = "https://api-fxpractice.oanda.com" if args.practice else "https://api-fxtrade.oanda.com"
    client = httpx.Client(
        base_url=base,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )

    date = (
        datetime.fromisoformat(args.date).replace(tzinfo=UTC)
        if args.date
        else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    )

    from src.runtime.run_engine import _build_db_engine

    engine = _build_db_engine()

    report = run_daily_reconciliation(engine, client, account_id, date=date)

    if report.is_clean:
        if not args.quiet_success:
            print(
                f"Reconciliation clean for {date.date()}: {report.matched} matched fills",
            )
        return 0

    print(
        f"Reconciliation found {len(report.mismatches)} mismatches for {date.date()}:",
        file=sys.stderr,
    )
    for m in report.mismatches:
        print(f"  [{m.kind}] {m.detail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
