"""Truth Social event-study daemon (CL-s9as) — RESEARCH ONLY.

Each cycle: ingest new public posts (trumpstruth.org archive RSS) →
classify unlabeled posts (Haiku tier) → measure matured relevant posts
against liquid instruments. Produces DATA, never orders or alerts.

Usage:
    .venv/bin/python scripts/truth_monitor.py --once
    .venv/bin/python scripts/truth_monitor.py --loop 300
Analysis: scripts/truth_report.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine

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


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(description="Truth Social event study.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if os.environ.get("TRUTH_STUDY_ENABLED", "1").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        logger.error("TRUTH_STUDY_ENABLED is off — exiting")
        return 3

    from sqlalchemy import text  # noqa: PLC0415

    from src.data.truth_reactions import measure_pending  # noqa: PLC0415
    from src.data.truth_social import ingest_posts  # noqa: PLC0415
    from src.events.truth_classifier import classify_pending  # noqa: PLC0415

    engine = create_engine(_build_db_url())

    def _backlogs() -> tuple[int, int]:
        with engine.connect() as conn:
            unclassified = conn.execute(text("""
                SELECT COUNT(*) FROM truth_posts p
                LEFT JOIN truth_classifications c ON c.post_id = p.post_id
                WHERE c.post_id IS NULL
            """)).scalar() or 0
            unmeasured = conn.execute(text("""
                SELECT COUNT(*) FROM truth_classifications c
                WHERE c.is_market_relevant
                  AND NOT EXISTS (SELECT 1 FROM truth_market_reactions r
                                  WHERE r.post_id = c.post_id)
            """)).scalar() or 0
        return int(unclassified), int(unmeasured)

    def _run() -> None:
        new = ingest_posts(engine)
        labeled = classify_pending(engine)
        measured = measure_pending(engine)
        unclassified, unmeasured = _backlogs()
        # Backlog watch: at 10 classifications/cycle a backlog >50 means
        # >25 min of lag — visible drift toward silently-stale labels.
        if unclassified > 50:
            logger.warning("truth study: classification backlog %d — "
                           "drain rate may be too slow", unclassified)
        print(f"truth study: ingested={new} classified={labeled} "
              f"measured={measured} backlog_unclassified={unclassified} "
              f"backlog_unmeasured={unmeasured}")

    if args.loop:
        logger.info("truth study: looping every %ds", args.loop)
        try:
            while True:
                try:
                    _run()
                except Exception:
                    logger.exception("truth study: cycle failed — retrying")
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("truth study: stopped")
        return 0

    _run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
