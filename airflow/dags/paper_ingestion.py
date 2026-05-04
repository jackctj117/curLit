"""Airflow DAG — nightly paper ingestion (CL-28j bridge).

Runs scripts/ingest_papers.py against the configured paper_streams.yaml
feeds. Each fetched paper:
  1. Inserts a row into research_papers (ON CONFLICT DO NOTHING)
  2. Calls RelevanceScorer.score_and_update so the dashboard sees a
     real relevance_score on first read
  3. LLM-extracts to data/research/extracts/*.md for the Idea Agent

Schedule: 03:00 UTC nightly. Paper feeds publish on irregular cadences
(arXiv: continuous, NBER: weekday afternoons, BIS: monthly batches);
nightly catches everything within ~1 day of publication, which is fine
for FX research on weekly+ horizons. Weekend runs still help — NBER
publishes Friday afternoons, this catches them Saturday morning.

A failed feed (broken URL, network blip, malformed RSS) doesn't fail
the DAG — IngestRunner logs the failure into the summary. We treat
the DAG as best-effort: the dashboard tolerates partial-day data.

Required env (loaded via dotenv_bootstrap inside the script):
  ANTHROPIC_API_KEY or DEEPSEEK_API_KEY (for paper_extractor LLM)
  POSTGRES_HOST / USER / PASSWORD / DB
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)


def ingest_papers() -> None:
    """Run the ingester. Returns nothing; Airflow records pass/fail
    via exception propagation."""
    # Import inside the task so the DAG file itself stays light
    # (Airflow reads DAG files frequently — keep imports cheap).
    from scripts.ingest_papers import main as ingest_main

    rc = ingest_main([
        "--config", "configs/paper_streams.yaml",
        "--research-config", "configs/research_agents.yaml",
    ])
    if rc != 0:
        msg = f"scripts.ingest_papers exited with code {rc}"
        raise RuntimeError(msg)


def backfill_scores() -> None:
    """Re-score papers whose relevance_score is at the schema default.

    The post-insert score_and_update path covers fresh papers, but
    historical rows or papers inserted before the scorer existed
    have score=0. This task picks them up so the dashboard sees a
    real triage queue. Cheap (no LLM); just keyword matching.
    """
    import os

    from sqlalchemy import create_engine, text

    from src.research.relevance_scorer import RelevanceScorer

    db_url = os.environ.get("DATABASE_URL") or (
        f"postgresql+psycopg2://"
        f"{os.environ.get('POSTGRES_USER', 'fx')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/"
        f"{os.environ.get('POSTGRES_DB', 'fx')}"
    )
    engine = create_engine(db_url)
    scorer = RelevanceScorer()

    # Pull rows with relevance_score == 0 (the schema default — these
    # are unscored, not "scored to 0"). 1000-row cap so a single run
    # can't OOM on a fresh load. The next nightly catches the rest.
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT paper_id, title, abstract, source, authors
            FROM research_papers
            WHERE relevance_score = 0
            LIMIT 1000
        """)).fetchall()

    rescored = 0
    for paper_id, title, abstract, source, authors in rows:
        author_list = (
            [str(a) for a in authors] if isinstance(authors, list) else []
        )
        # The scorer wants Paper-shape attributes; build a minimal
        # surrogate that satisfies score_paper's getattr-based access.
        surrogate = type("_S", (), {
            "title": title or "",
            "abstract": abstract or "",
            "authors": author_list,
            "category": "",
            "source_label": source or "",
        })()
        scorer.score_and_update(engine, paper_id, surrogate)
        rescored += 1
    logger.info("Backfilled relevance scores for %d papers", rescored)


try:
    from src.monitoring.airflow_callbacks import (
        on_dag_failure,
        on_dag_success,
    )
except Exception:  # noqa: BLE001
    on_dag_success = on_dag_failure = None  # type: ignore[assignment]


default_args = {
    "owner": "curlit",
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
    "on_success_callback": on_dag_success,
    "on_failure_callback": on_dag_failure,
}


with DAG(
    "paper_ingestion",
    default_args=default_args,
    description="Nightly paper-stream ingest into research_papers",
    schedule_interval="0 3 * * *",   # 03:00 UTC daily
    start_date=datetime(2026, 5, 1),
    catchup=False,
    max_active_runs=1,
    tags=["curlit", "research", "ingestion"],
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest_papers",
        python_callable=ingest_papers,
    )
    t_backfill = PythonOperator(
        task_id="backfill_scores",
        python_callable=backfill_scores,
    )

    t_ingest >> t_backfill
