"""Outcome-scoring daemon (CL-6axf).

Scores the forward returns of surfaced trade ideas into ``idea_outcomes`` — the
track record the reflective loop learns from. Run daily.

Usage:
    .venv/bin/python scripts/score_outcomes.py --once
    .venv/bin/python scripts/score_outcomes.py --loop 86400   # daily
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


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(description="Score trade-idea outcomes.")
    parser.add_argument("--once", action="store_true", help="Score once, exit.")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None,
                        help="Score every SECONDS (daemon mode).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.events.outcome_tracker import score_open_ideas  # noqa: PLC0415

    engine = create_engine(build_db_url())

    def _run() -> None:
        counts = score_open_ideas(engine)
        print(
            f"outcomes: open={counts['open']} win={counts['win']} "
            f"loss={counts['loss']} flat={counts['flat']} "
            f"no_data={counts['no_data']}",
        )

    if args.loop:
        logger.info("outcome scoring: looping every %ds (Ctrl-C to stop)", args.loop)
        try:
            while True:
                _run()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("outcome scoring: stopped")
        return 0

    _run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
