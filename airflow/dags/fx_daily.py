"""
Airflow DAG — daily ingestion pipeline.
Fetches prices, macro data, and SOFR futures, then runs validation.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def ingest_prices() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("ingest_prices — stub (wire to StooqIngester)")


def ingest_macro() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("ingest_macro — stub (wire to FRED + ECB + BoJ + BoE ingesters)")


def ingest_sofr() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("ingest_sofr — stub (wire to CMESOFRIngester)")


def validate_all() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("validate_all — stub (wire to data.validation)")


default_args = {
    "owner": "curlit",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "fx_daily_pipeline",
    default_args=default_args,
    description="Daily FX data ingestion and validation",
    schedule_interval="0 23 * * 1-5",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["curlit", "ingestion"],
) as dag:

    t_prices = PythonOperator(task_id="ingest_prices", python_callable=ingest_prices)
    t_macro = PythonOperator(task_id="ingest_macro", python_callable=ingest_macro)
    t_sofr = PythonOperator(task_id="ingest_sofr", python_callable=ingest_sofr)
    t_validate = PythonOperator(task_id="validate_all", python_callable=validate_all)

    [t_prices, t_macro, t_sofr] >> t_validate
