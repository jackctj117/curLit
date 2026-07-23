"""Airflow → Prometheus callback shims (CL-47j).

Airflow exposes per-DAG ``on_success_callback`` / ``on_failure_callback``
hooks. This module wires those to ``fx_dag_runs_total`` so a single
Grafana panel shows ingestion-DAG success rate without us having to
scrape Airflow's own metrics endpoint (which requires plugin install
and adds another moving part).

Usage in a DAG:

    from src.monitoring.airflow_callbacks import (
        on_dag_success, on_dag_failure,
    )

    default_args = {
        ...,
        "on_success_callback": on_dag_success,
        "on_failure_callback": on_dag_failure,
    }

The callbacks are no-ops when the metrics module isn't importable
(e.g. Airflow workers without the curLit src on PYTHONPATH); they log
once and continue. Failure of an observability hook should never
break the DAG run itself.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _record(dag_id: str, status: str) -> None:
    try:
        from src.monitoring.metrics import dag_runs

        dag_runs.labels(dag_id=dag_id, status=status).inc()
    except Exception:
        logger.warning(
            "Airflow callback could not emit dag_runs metric (dag=%s status=%s)",
            dag_id,
            status,
            exc_info=True,
        )


def on_dag_success(context: dict[str, Any]) -> None:
    """Airflow hook fired on DAG success. ``context`` is the standard
    Airflow task_instance context — the ``dag`` key holds the DAG model."""
    dag = context.get("dag")
    dag_id = getattr(dag, "dag_id", "unknown")
    _record(dag_id, "success")


def on_dag_failure(context: dict[str, Any]) -> None:
    dag = context.get("dag")
    dag_id = getattr(dag, "dag_id", "unknown")
    _record(dag_id, "failure")
