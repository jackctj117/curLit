"""Reddit monitor daemon (CL-okww).

Polls the tiered subreddit watchlist (configs/reddit_watchlist.yaml) and
ingests theme-matched posts as NEW geo_events rows, feeding the existing
triage → impact-agent pipeline. Keyless (public Reddit JSON, descriptive
User-Agent); tier cadence keeps request volume tiny.

Usage:
    .venv/bin/python scripts/reddit_monitor.py --once
    .venv/bin/python scripts/reddit_monitor.py --loop 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)

WATCHLIST_PATH = "configs/reddit_watchlist.yaml"
#: Gentle gap between subreddit requests (public-API politeness).
REQUEST_GAP_SEC = 2.0


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(description="Reddit → geo_events monitor.")
    parser.add_argument("--once", action="store_true", help="One cycle, exit.")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None,
                        help="Poll every SECONDS (daemon mode).")
    parser.add_argument("--watchlist", default=WATCHLIST_PATH)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.data.reddit_monitor import (  # noqa: PLC0415
        due_this_cycle,
        fetch_posts,
        ingest_reddit_posts,
        load_reddit_watchlist,
        oauth_from_env,
    )
    from src.events.playbooks import (  # noqa: PLC0415
        DEFAULT_PLAYBOOKS_PATH,
        load_playbooks,
    )

    subs = load_reddit_watchlist(args.watchlist)
    playbooks = load_playbooks(DEFAULT_PLAYBOOKS_PATH)
    engine = create_engine(build_db_url())
    oauth = oauth_from_env()
    if oauth is None:
        logger.warning(
            "REDDIT_CLIENT_ID/SECRET not set — falling back to the public JSON "
            "endpoint, which Reddit 403-blocks for most clients. Create a "
            "'script' app at https://www.reddit.com/prefs/apps and set the "
            "creds in .env for reliable polling.",
        )
    logger.info("reddit monitor: %d subreddits (%d tier-1), %d playbooks, "
                "auth=%s", len(subs), sum(1 for s in subs if s.tier == 1),
                len(playbooks), "oauth" if oauth else "public(blocked-risk)")

    def _cycle(cycle_n: int) -> None:
        polled = ingested = 0
        for sub in subs:
            if not due_this_cycle(sub, cycle_n):
                continue
            posts = fetch_posts(sub, oauth=oauth)
            polled += 1
            if posts:
                result = ingest_reddit_posts(engine, sub, posts, playbooks)
                if result.ingested or result.skipped_no_theme:
                    logger.info(result.summary_line(sub.name))
                ingested += result.ingested
            time.sleep(REQUEST_GAP_SEC)
        logger.info("reddit cycle %d complete: polled=%d ingested=%d",
                    cycle_n, polled, ingested)

    if args.loop:
        logger.info("looping every %ds (Ctrl-C to stop)", args.loop)
        cycle_n = 0
        try:
            while True:
                _cycle(cycle_n)
                cycle_n += 1
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("reddit monitor: stopped")
        return 0

    _cycle(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
