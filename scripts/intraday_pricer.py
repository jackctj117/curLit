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

from sqlalchemy import create_engine

# Make `src`/`scripts` importable when launched as a file
# (scripts/intraday_pricer.py), matching scripts/event_pipeline.py. Harmless
# under `python -m scripts.intraday_pricer`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def _build_db_url() -> str:
    explicit = os.environ.get("DATABASE_URL")
    if explicit:
        return explicit
    user = os.environ.get("POSTGRES_USER", "fx")
    password = os.environ.get("POSTGRES_PASSWORD", "changeme")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "fx")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


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
    engine = create_engine(_build_db_url())
    pricer = IntradayPricer(
        engine, instruments, api_key, account_id, practice=practice,
        retention_hours=args.retention_hours,
    )
    logger.info(
        "intraday pricer: %d instruments, practice=%s, retention=%dh",
        len(instruments), practice, args.retention_hours,
    )

    if args.loop:
        logger.info("looping every %ds (Ctrl-C to stop)", args.loop)
        try:
            while True:
                pricer.poll_once()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("intraday pricer: stopped")
        return 0

    counts = pricer.poll_once()
    print(
        f"intraday: wrote={counts['written']} pruned={counts['pruned']} "
        f"instruments={counts['instruments']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
