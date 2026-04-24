"""Airflow DAG — CB scraper run every 6 hours + sentiment processing."""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


def run_scrapers() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("run_scrapers — stub (wire to Fed/ECB/BoE/BoJ/BoC scrapers)")


def preprocess() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("preprocess — stub (wire to TextPreprocessor)")


def score_lexicon() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("score_lexicon — stub (wire to LexiconScorer)")


def compute_diffs() -> None:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("compute_diffs — stub (wire to StatementDiffer)")


default_args = {"owner": "curlit", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    "cb_scrapers",
    default_args=default_args,
    description="Central bank document scraping and sentiment analysis",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["curlit", "nlp"],
) as dag:

    t_scrape = PythonOperator(task_id="run_scrapers", python_callable=run_scrapers)
    t_clean = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_score = PythonOperator(task_id="score_lexicon", python_callable=score_lexicon)
    t_diff = PythonOperator(task_id="compute_diffs", python_callable=compute_diffs)

    t_scrape >> t_clean >> t_score >> t_diff
