"""Backfill historical intraday quotes for the event study (CL-b425).

The CL-z95p event study prices assessed geo_events against ``intraday_quotes``,
but the live feed only exists from the pricer daemon's first run — so every
event seen before that was unmeasurable (4,631 of 5,971 candidate legs on the
first study run). This one-shot script fills the gap from OANDA's own M5
candle history (same venue, bid/ask/mid closes, no-lookahead end-stamping —
see ``src.data.oanda_candles.parse_mba_candles``):

  window start = min(geo_events.seen_at) − 1h
  window end   = first LIVE quote ts (the clip: the live region stays live)

Idempotent (upsert on ts/symbol/source), read-only on geo_events, additive on
intraday_quotes under source='oanda_m5_backfill'. Re-run safe.

Usage:
    .venv/bin/python scripts/backfill_event_quotes.py [--granularity M5]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dotenv_bootstrap import load_project_env  # noqa: E402

logger = logging.getLogger(__name__)

#: The live pricer's source tag — its earliest row is the backfill's end clip.
LIVE_SOURCE = "oanda"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--granularity", default="M5", help="OANDA candle granularity (M1/M5/M15)")
    args = parser.parse_args(argv)

    load_project_env()
    from sqlalchemy import create_engine, text  # noqa: PLC0415

    from src.data.db_env import build_db_url  # noqa: PLC0415
    from src.data.oanda_candles import backfill_intraday_quotes  # noqa: PLC0415
    from src.events.playbooks import all_tradable_instruments, load_playbooks  # noqa: PLC0415
    from src.research.event_study import parse_ts  # noqa: PLC0415

    api_key = os.environ.get("OANDA_API_KEY")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not api_key or not account_id:
        logger.error("OANDA_API_KEY / OANDA_ACCOUNT_ID missing — cannot backfill")
        return 1
    practice = os.environ.get("OANDA_PRACTICE", "true").lower() != "false"

    engine = create_engine(build_db_url())
    with engine.connect() as conn:
        earliest_event = parse_ts(
            conn.execute(text("SELECT min(seen_at) FROM geo_events")).scalar()
        )
        live_start = parse_ts(
            conn.execute(
                text("SELECT min(ts) FROM intraday_quotes WHERE source = :s"),
                {"s": LIVE_SOURCE},
            ).scalar()
        )
    if earliest_event is None or live_start is None:
        logger.error(
            "cannot compute window (earliest_event=%s live_start=%s) — nothing to do",
            earliest_event,
            live_start,
        )
        return 1

    start = earliest_event - timedelta(hours=1)
    end = live_start
    if start >= end:
        logger.info("live coverage already reaches the earliest event — nothing to backfill")
        return 0

    instruments = sorted(all_tradable_instruments(load_playbooks("configs/event_playbooks.yaml")))
    logger.info(
        "backfilling %d instruments, %s -> %s (%.1f days), granularity %s",
        len(instruments),
        start.isoformat(),
        end.isoformat(),
        (end - start).total_seconds() / 86400.0,
        args.granularity,
    )
    counts = backfill_intraday_quotes(
        engine,
        instruments,
        api_key,
        account_id,
        start=start,
        end=end,
        granularity=args.granularity,
        practice=practice,
    )
    logger.info(
        "backfill complete: %d instruments, %d rows written",
        counts["instruments"],
        counts["rows"],
    )
    return 0 if counts["rows"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
