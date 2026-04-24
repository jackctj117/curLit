"""Airflow DAG — weekly update check, bd issue creation, Pushover notification."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def check_updates() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("check_updates — stub (wire to scripts/check_updates.py)")


default_args = {"owner": "curlit", "retries": 1}

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
