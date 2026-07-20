"""Current-events pipeline entrypoint (CL-6iu7).

Usage:
    .venv/bin/python scripts/event_pipeline.py --ingest --assess --once
    .venv/bin/python scripts/event_pipeline.py --ingest --loop 900
    .venv/bin/python scripts/event_pipeline.py --assess --limit 10

``--ingest`` polls GDELT (one themed query per playbook) and inserts
NEW rows into geo_events; ``--assess`` runs the Event Impact Agent over
NEW rows (capped per run). ``--loop N`` repeats every N seconds —
suitable for a systemd timer / cron / Airflow later; ``--once`` (the
default) runs a single cycle.

GDELT updates ~every 15 minutes, so looping faster than ~900s only
re-fetches the same articles (they dedup away harmlessly).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("event_pipeline")


def _db_url() -> str:
    return (
        f"postgresql+psycopg2://"
        f"{os.environ.get('POSTGRES_USER', 'fx')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'fx')}"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GDELT ingest + LLM impact assessment for geo_events",
    )
    p.add_argument("--ingest", action="store_true", help="Poll GDELT for new events")
    p.add_argument("--assess", action="store_true", help="Assess NEW rows via the impact agent")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one cycle (default)")
    mode.add_argument(
        "--loop", type=int, metavar="SECONDS", default=None,
        help="Repeat every N seconds until interrupted",
    )
    p.add_argument(
        "--lookback-minutes", type=int, default=60,
        help="GDELT ingest window ending now (default 60; overlap dedups away)",
    )
    p.add_argument(
        "--limit", type=int, default=20,
        help="Max NEW rows assessed per cycle (default 20)",
    )
    p.add_argument(
        "--playbooks", default="configs/event_playbooks.yaml",
        help="Path to the event playbook config",
    )
    p.add_argument(
        "--model", default=os.environ.get("EVENT_IMPACT_MODEL", ""),
        help="Impact agent model override (default: agent default)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return p


def _cycle(args: argparse.Namespace) -> None:
    if args.ingest:
        from src.data.gdelt import GdeltIngester  # noqa: PLC0415

        ingester = GdeltIngester(_db_url(), playbooks_path=args.playbooks)
        end = datetime.now(UTC)
        start = end - timedelta(minutes=args.lookback_minutes)
        rows = ingester.run(start, end)
        logger.info("ingest: %d new geo_events rows", rows)

    if args.assess:
        from sqlalchemy import create_engine  # noqa: PLC0415

        from src.events.impact_agent import (  # noqa: PLC0415
            DEFAULT_MODEL,
            EventImpactAgent,
        )

        agent = EventImpactAgent(
            engine=create_engine(_db_url()),
            model=args.model or DEFAULT_MODEL,
            playbooks_path=args.playbooks,
        )
        results = agent.assess_new_events(limit=args.limit)
        for result in results:
            logger.info("%s", result.summary_line())
        assessed = sum(1 for r in results if r.status == "ASSESSED")
        logger.info(
            "assess: %d processed (%d assessed, %d dismissed)",
            len(results), assessed, len(results) - assessed,
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.ingest and not args.assess:
        _build_parser().error("nothing to do: pass --ingest and/or --assess")

    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    if args.loop is None:
        _cycle(args)
        return 0

    logger.info("looping every %ds (Ctrl-C to stop)", args.loop)
    try:
        while True:
            t0 = time.time()
            try:
                _cycle(args)
            except Exception:
                # A failed cycle (GDELT hiccup, DB blip) must not kill
                # the loop — log and try again next interval.
                logger.exception("cycle failed; continuing")
            elapsed = time.time() - t0
            time.sleep(max(0.0, args.loop - elapsed))
    except KeyboardInterrupt:
        logger.info("interrupted — exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
