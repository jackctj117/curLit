"""Reflective self-tuning review (CL-g8jl).

Reviews the finalised trade-idea outcomes and proposes conservative config
tuning for the operator to approve. Advisory only — never edits config.

Usage:
    .venv/bin/python scripts/reflect.py --once            # print the review
    .venv/bin/python scripts/reflect.py --once --telegram  # + push to Telegram
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(description="Reflective outcome review.")
    parser.add_argument("--once", action="store_true", default=True)
    parser.add_argument("--telegram", action="store_true",
                        help="Also push the proposal to the operator's Telegram.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.events.reflective_review import ReflectiveReviewer  # noqa: PLC0415

    engine = create_engine(build_db_url())
    result = ReflectiveReviewer(engine).review()

    print(f"status={result.status} sample={result.sample}")
    print(json.dumps(result.summary.get("overall", {}), indent=2))
    if result.proposal:
        print(json.dumps(result.proposal, indent=2))

    if args.telegram:
        from src.research.notifications import notify_operator  # noqa: PLC0415
        res = notify_operator("Reflective review", result.to_telegram(), html=True)
        print(f"telegram: ok={res.telegram_succeeded}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
