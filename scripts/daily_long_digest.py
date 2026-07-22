"""Morning LONG-positions Telegram digest daemon (CL-lpai).

Sends ONE Telegram message per trading morning (default 09:15 ET,
LONG_DIGEST_TIME_ET) listing every LONG holding: OANDA trades with
positive units + all Alpaca option positions. Venue read failures render
as explicit "(unavailable)" sections — never a silently flat digest.

Usage:
    .venv/bin/python scripts/daily_long_digest.py --once   # send now if due
    .venv/bin/python scripts/daily_long_digest.py --loop 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

STATE_PATH = Path(os.environ.get("LONG_DIGEST_STATE", "data/long_digest_state.json"))


def _load_last_sent() -> str | None:
    try:
        return str(json.loads(STATE_PATH.read_text()).get("last_sent_ny_date"))
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("long digest: unreadable state %s — treating as never "
                       "sent", STATE_PATH, exc_info=True)
        return None


def _fetch_oanda_positions() -> list | None:
    key = os.environ.get("OANDA_API_KEY", "")
    acct = os.environ.get("OANDA_ACCOUNT_ID", "")
    if not key or not acct:
        return None
    try:
        from src.execution.oanda_broker import OandaBroker  # noqa: PLC0415
        return OandaBroker(key, acct, practice=True).get_positions()
    except Exception:
        logger.warning("long digest: OANDA positions unavailable",
                       exc_info=True)
        return None


def _fetch_alpaca_positions() -> list | None:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or not secret:
        return None
    try:
        from src.execution.alpaca_options import AlpacaOptionsClient  # noqa: PLC0415
        return AlpacaOptionsClient(key, secret).list_option_positions()
    except Exception:
        logger.warning("long digest: Alpaca positions unavailable",
                       exc_info=True)
        return None


def run_once(now: datetime | None = None, *, force: bool = False) -> bool:
    """Send the digest if due; True when a message went out."""
    from src.events._util import atomic_write_json  # noqa: PLC0415
    from src.monitoring.long_digest import (  # noqa: PLC0415
        _NY,
        build_long_digest,
        should_send,
    )
    from src.research.notifications import notify_operator  # noqa: PLC0415

    now = now or datetime.now(UTC)
    send_time = os.environ.get("LONG_DIGEST_TIME_ET", "09:15")
    if not force and not should_send(now, _load_last_sent(), send_time):
        return False

    body = build_long_digest(
        _fetch_oanda_positions(), _fetch_alpaca_positions(), now,
    )
    result = notify_operator("☀️ Morning LONG positions", body, html=True)
    if not result.any_succeeded:
        logger.warning("long digest: send failed — will retry next cycle")
        return False
    atomic_write_json(
        STATE_PATH,
        {"last_sent_ny_date": now.astimezone(_NY).date().isoformat()},
    )
    logger.info("long digest: sent for %s",
                now.astimezone(_NY).date().isoformat())
    return True


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(description="Morning LONG digest.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="send immediately, ignore time/dedup gates")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.loop:
        logger.info("long digest: looping every %ds (send at %s ET)",
                    args.loop, os.environ.get("LONG_DIGEST_TIME_ET", "09:15"))
        try:
            while True:
                try:
                    run_once()
                except Exception:
                    logger.exception("long digest: cycle failed — retrying")
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("long digest: stopped")
        return 0

    sent = run_once(force=args.force)
    print("sent" if sent else "not due (use --force to send now)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
