"""Intraday OANDA quote poller daemon (CL-dz71).

Polls OANDA pricing for every tradable event instrument (from the playbooks)
and writes fresh mids into ``intraday_quotes`` so event confluence Gate B can
confirm real intraday moves instead of expiring on stale daily closes.

Usage:
    .venv/bin/python scripts/intraday_pricer.py --once
    .venv/bin/python scripts/intraday_pricer.py --loop 120     # every 120s

Requires OANDA_API_KEY + OANDA_ACCOUNT_ID in the environment (same creds the
oanda-practice engine uses); OANDA_PRACTICE=true selects the practice host.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine

# Make `src`/`scripts` importable when launched as a file
# (scripts/intraday_pricer.py), matching scripts/event_pipeline.py. Harmless
# under `python -m scripts.intraday_pricer`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)


def _instruments(explicit: str | None) -> list[str]:
    if explicit:
        return [s.strip() for s in explicit.split(",") if s.strip()]
    from src.events.playbooks import (  # noqa: PLC0415
        DEFAULT_PLAYBOOKS_PATH,
        all_tradable_instruments,
        load_playbooks,
    )
    return sorted(all_tradable_instruments(load_playbooks(DEFAULT_PLAYBOOKS_PATH)))


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(
        description="Poll OANDA pricing into intraday_quotes (CL-dz71).",
    )
    parser.add_argument("--once", action="store_true",
                        help="Poll once and exit.")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None,
                        help="Poll every SECONDS (daemon mode).")
    parser.add_argument("--instruments", default=None,
                        help="Comma-separated OANDA ids (default: playbook set).")
    parser.add_argument("--retention-hours", type=int, default=24,
                        help="Prune quotes older than this each cycle.")
    parser.add_argument("--no-candles", action="store_true",
                        help="Skip the daily-candle vol backfill (CL-lb03).")
    parser.add_argument("--candles-count", type=int, default=60,
                        help="Daily candles to backfill per unmapped instrument.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = os.environ.get("OANDA_API_KEY", "")
    account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
    practice = os.environ.get("OANDA_PRACTICE", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if not api_key or not account_id:
        logger.error("OANDA_API_KEY / OANDA_ACCOUNT_ID not set — cannot poll")
        return 2

    from src.data.intraday_pricer import IntradayPricer  # noqa: PLC0415

    instruments = _instruments(args.instruments)
    engine = create_engine(build_db_url())
    pricer = IntradayPricer(
        engine, instruments, api_key, account_id, practice=practice,
        retention_hours=args.retention_hours,
    )
    logger.info(
        "intraday pricer: %d instruments, practice=%s, retention=%dh",
        len(instruments), practice, args.retention_hours,
    )

    # Daily-candle vol backfill (CL-lb03) for the unmapped instruments — the
    # ones with no canonical daily series, run once at startup and once per
    # UTC-day rollover (daily bars don't change intraday).
    from datetime import UTC, datetime  # noqa: PLC0415

    from src.data.oanda_candles import (  # noqa: PLC0415
        refresh_daily_candles,
        unmapped_tradables,
    )
    candle_instruments = [] if args.no_candles else unmapped_tradables(instruments)
    last_candle_date: Any = None

    def _maybe_refresh_candles() -> None:
        nonlocal last_candle_date
        if not candle_instruments:
            return
        today = datetime.now(UTC).date()
        if today == last_candle_date:
            return
        refresh_daily_candles(
            engine, candle_instruments, api_key, account_id,
            practice=practice, count=args.candles_count,
        )
        last_candle_date = today

    if candle_instruments:
        logger.info("daily-candle backfill for %d unmapped instruments: %s",
                    len(candle_instruments), ", ".join(candle_instruments))

    if args.loop:
        logger.info("looping every %ds (Ctrl-C to stop)", args.loop)
        try:
            while True:
                _maybe_refresh_candles()
                pricer.poll_once()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("intraday pricer: stopped")
        return 0

    _maybe_refresh_candles()
    counts = pricer.poll_once()
    print(
        f"intraday: wrote={counts['written']} pruned={counts['pruned']} "
        f"instruments={counts['instruments']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
