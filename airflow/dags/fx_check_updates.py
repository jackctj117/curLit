"""Airflow DAG — weekly update check, bd issue creation, Telegram notification."""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def check_updates() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("check_updates — stub (wire to scripts/check_updates.py)")


try:
    from src.monitoring.airflow_callbacks import on_dag_failure, on_dag_success
except Exception:  # noqa: BLE001
    on_dag_success = on_dag_failure = None  # type: ignore[assignment]

default_args = {
    "owner": "curlit", "retries": 1,
    "on_success_callback": on_dag_success,
    "on_failure_callback": on_dag_failure,
}

with DAG(
    "fx_check_updates",
    default_args=default_args,
    description="Weekly dependency update check with bd issue creation",
    schedule_interval="0 12 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["curlit", "maintenance"],
) as dag:

    PythonOperator(task_id="check_updates", python_callable=check_updates)
