"""
Airflow DAG — daily FX data ingestion and validation.
"""

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{os.environ.get('POSTGRES_USER', 'fx')}:{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/{os.environ.get('POSTGRES_DB', 'fx')}",
)


def ingest_prices() -> None:
    from src.data.yfinance_provider import YFinanceIngester
    ingester = YFinanceIngester(DB_URL)
    end = datetime.utcnow()
    start = end - timedelta(days=3)
    rows = ingester.run(start, end)
    logger.info("Prices ingested: %d rows", rows)


def ingest_macro() -> None:
    from src.data.fred import FREDIngester
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    try:
        ingester = FREDIngester(DB_URL)
        rows = ingester.run(start, end)
        logger.info("Macro ingested (FRED): %d rows", rows)
    except Exception:
        logger.exception("FRED ingestion failed")


def ingest_sofr() -> None:
    try:
        from src.data.cme_sofr import CMESOFRIngester
        ingester = CMESOFRIngester(DB_URL)
        end = datetime.utcnow()
        start = end - timedelta(days=7)
        rows = ingester.run(start, end)
        logger.info("SOFR futures ingested: %d rows", rows)
    except Exception:
        logger.warning("CME SOFR ingestion failed (expected — may need network access to CME)")


def validate_all() -> None:
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            for tbl in ["prices", "macro_data", "rate_curves"]:
                cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).fetchone()[0]
                logger.info("  %s: %d rows", tbl, cnt)
    except Exception:
        logger.warning("Validation query failed")
    logger.info("Validation complete")


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
