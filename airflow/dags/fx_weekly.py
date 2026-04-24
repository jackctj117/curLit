"""
Airflow DAG — weekly COT ingestion (Friday after CFTC release).
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def ingest_cot() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("ingest_cot — stub (wire to CFTCIngester)")


def compute_cot_features() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("compute_cot_features — stub (wire to feature computation)")


default_args = {
    "owner": "curlit",
    "retries": 3,
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    "fx_weekly_cot",
    default_args=default_args,
    description="Weekly CFTC COT ingestion and feature computation",
    schedule_interval="30 20 * * 5",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["curlit", "cot"],
) as dag:

    t_ingest = PythonOperator(task_id="ingest_cot", python_callable=ingest_cot)
    t_features = PythonOperator(task_id="compute_cot_features", python_callable=compute_cot_features)

    t_ingest >> t_features
