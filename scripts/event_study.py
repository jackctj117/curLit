"""Event-study CLI (CL-z95p) — READ-ONLY measurement of the event pipeline.

Renders :mod:`src.research.event_study` against the live Postgres DB. Issues
SELECTs only: it never writes, transitions, or trades.

Usage:
    .venv/bin/python scripts/event_study.py                       # 30d, stdout
    .venv/bin/python scripts/event_study.py --days 7
    .venv/bin/python scripts/event_study.py --days 30 --options
    .venv/bin/python scripts/event_study.py --options --out data/research/event_study.md

``--out`` writes the report to a file. Point it at ``data/research/`` — the
whole top-level ``data/`` tree is gitignored, so generated reports stay out of
the repo. NEVER ``git add`` a rendered report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)

#: Suggested output directory — untracked (`/data/` is in .gitignore).
DEFAULT_OUT_DIR = Path("data/research")


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    parser = argparse.ArgumentParser(
        description="Event study over geo_events + intraday_quotes (read-only).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Look-back window in days (default 30). NOTE: intraday_quotes is a "
        "rolling buffer pruned to ~24h — the report states its ACTUAL coverage.",
    )
    parser.add_argument(
        "--options",
        action="store_true",
        help="Also render the realized options P&L attribution section.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Write the report to PATH instead of stdout. Use {DEFAULT_OUT_DIR}/ "
        "(gitignored) — never commit a generated report.",
    )
    parser.add_argument(
        "--playbooks",
        type=Path,
        default=Path("configs/event_playbooks.yaml"),
        help="Playbook config for the theme→tier map.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from src.events.playbooks import load_playbooks  # noqa: PLC0415
    from src.research.event_study import (  # noqa: PLC0415
        EventStudyConfig,
        build_report,
        run_event_study,
        run_options_attribution,
    )

    if args.days <= 0:
        parser.error("--days must be positive")

    # Fail LOUD on a corrupt playbook config — a silently-empty theme→tier map
    # would render every event as "unmatched" and quietly misattribute the
    # generic/specific split.
    playbooks = load_playbooks(args.playbooks)

    engine = create_engine(build_db_url())
    config = EventStudyConfig()
    result = run_event_study(engine, config, days=args.days, playbooks=playbooks)
    options = (
        run_options_attribution(engine, days=args.days, min_bucket_n=config.min_bucket_n)
        if args.options
        else None
    )
    report = build_report(result, options)

    if args.out is None:
        print(report)
        return 0

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path} ({len(report):,} chars)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
