"""
Skeleton Airflow DAG — health check only.
This will be extended in CL-pck to run the full ingestion pipeline.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def health_check():
    """Verify Airflow is operational."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("curLit Airflow health check passed")
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


default_args = {
    "owner": "curlit",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "fx_health_check",
    default_args=default_args,
    description="curLit health check — verifies Airflow operational",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["curlit", "health"],
) as dag:

    health = PythonOperator(
        task_id="health_check",
        python_callable=health_check,
    )
