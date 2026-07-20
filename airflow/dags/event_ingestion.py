"""Airflow DAG — resilient event ingestion + relative-volume scan (CL-i4sr).

Every 10 minutes:
  * ingest_gdelt — poll GDELT for new playbook-themed events (the same
    GdeltIngester call scripts/event_pipeline.py --ingest makes; the
    60-min lookback overlaps run-to-run and dedups away harmlessly)
  * scan_rvol   — key-free relative-volume scan over the playbook
    equity watch universe (src.scanners.relative_volume), persisting
    to volume_spikes for the Telegram digest's Watch-line marks

============================================================================
NO LLM ASSESS TASK — ON PURPOSE. The Event Impact Agent shells out to
the `claude` CLI, which lives on the HOST (operator subscription), not
in this Airflow container. Assessment (and the Telegram digest) runs
via the host-side daemon:

    scripts/event_pipeline.py --assess --loop

(see docs/BOOT.md). This DAG only makes ingestion + scanning resilient
to that daemon being down; do NOT add an assess task here.
============================================================================

Infra (same pattern as fx_daily.py): the container mounts src/
read-only at /opt/curlit/src and reaches the HOST Postgres via
host.docker.internal (DATABASE_URL env). The playbook config is
mounted at /opt/curlit/configs (see docker-compose.yml airflow
volumes); EVENT_PLAYBOOKS_PATH overrides. yfinance is baked into the
image (airflow/Dockerfile) — scan_rvol degrades to a skip-with-log if
it is ever removed.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{os.environ.get('POSTGRES_USER', 'fx')}:{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/{os.environ.get('POSTGRES_DB', 'fx')}",
)

#: GDELT ingest window per run. The DAG fires every 10 min, so
#: consecutive windows overlap heavily — geo_events dedups on the URL
#: hash, and the overlap is what makes missed/late runs harmless.
LOOKBACK_MINUTES = 60


def _playbooks_path() -> str:
    """Resolve configs/event_playbooks.yaml inside the container.

    EVENT_PLAYBOOKS_PATH wins; then the compose mount
    (/opt/curlit/configs); then a repo-relative path for anyone running
    the DAG code outside the container. Skip-with-log if absent — a
    missing config mount shouldn't page as a red task forever.
    """
    candidates = [
        os.environ.get("EVENT_PLAYBOOKS_PATH", ""),
        "/opt/curlit/configs/event_playbooks.yaml",
        "configs/event_playbooks.yaml",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    msg = (
        "event_playbooks.yaml not found (checked EVENT_PLAYBOOKS_PATH, "
        "/opt/curlit/configs, ./configs) — mount ./configs to "
        "/opt/curlit/configs:ro in docker-compose.yml"
    )
    raise AirflowSkipException(msg)


def ingest_gdelt() -> None:
    """Poll GDELT into geo_events — same call as event_pipeline --ingest."""
    from src.data.gdelt import GdeltIngester

    ingester = GdeltIngester(DB_URL, playbooks_path=_playbooks_path())
    end = datetime.now(UTC)
    start = end - timedelta(minutes=LOOKBACK_MINUTES)
    rows = ingester.run(start, end)
    logger.info("ingest_gdelt: %d new geo_events rows", rows)


def scan_rvol() -> None:
    """RVOL scan over the equity watch universe → volume_spikes.

    yfinance IS in airflow/Dockerfile; if a future image drops it,
    skip loudly instead of failing the DAG every 10 minutes.
    """
    try:
        import yfinance  # noqa: F401
    except ImportError as exc:
        msg = (
            "yfinance not installed in this Airflow image — RVOL scan "
            "skipped; add yfinance to airflow/Dockerfile to restore it"
        )
        raise AirflowSkipException(msg) from exc

    from src.scanners.relative_volume import RelativeVolumeScanner

    scanner = RelativeVolumeScanner(DB_URL, playbooks_path=_playbooks_path())
    rows = scanner.scan()
    unusual = sum(1 for r in rows if r.is_unusual)
    logger.info("scan_rvol: %d tickers scanned, %d unusual", len(rows), unusual)


try:
    from src.monitoring.airflow_callbacks import (
        on_dag_failure,
        on_dag_success,
    )
except Exception:  # noqa: BLE001 — Airflow workers may have stale PYTHONPATH
    on_dag_success = on_dag_failure = None  # type: ignore[assignment]

default_args = {
    "owner": "curlit",
    "retries": 1,  # the 10-min cadence is itself the retry loop
    "retry_delay": timedelta(minutes=2),
    "on_success_callback": on_dag_success,
    "on_failure_callback": on_dag_failure,
}

with DAG(
    "event_ingestion",
    default_args=default_args,
    description="GDELT event ingestion + equity RVOL scan (every 10 min)",
    schedule_interval="*/10 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["curlit", "ingestion", "events"],
) as dag:

    # No dependency edge on purpose: a GDELT 429 storm must not block
    # the volume scan, and vice versa.
    t_ingest = PythonOperator(task_id="ingest_gdelt", python_callable=ingest_gdelt)
    t_scan = PythonOperator(task_id="scan_rvol", python_callable=scan_rvol)
