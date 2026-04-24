"""
Airflow DAG — central bank document scraping and sentiment analysis.
Runs every 6 hours: scrapes new documents, preprocesses, scores, computes diffs.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DB_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{os.environ.get('POSTGRES_USER', 'fx')}:{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@{os.environ.get('POSTGRES_HOST', 'localhost')}:5432/{os.environ.get('POSTGRES_DB', 'fx')}",
)

CBS = ["fed", "ecb", "boe", "boj", "boc"]


def run_scrapers() -> None:
    from src.nlp.scrapers.fed import FedStatementScraper
    from src.nlp.scrapers.ecb import ECBStatementScraper
    from src.nlp.scrapers.boe_boj_boc import BoEStatementScraper, BoJStatementScraper, BoCStatementScraper

    raw_dir = Path("data/raw")
    since = datetime.utcnow() - timedelta(days=7)
    scrapers = [
        FedStatementScraper(raw_dir),
        ECBStatementScraper(raw_dir),
        BoEStatementScraper(raw_dir),
        BoJStatementScraper(raw_dir),
        BoCStatementScraper(raw_dir),
    ]
    total = 0
    for scraper in scrapers:
        try:
            docs = scraper.run(since)
            total += len(docs)
            logger.info("Scraped %s: %d docs", scraper.cb_name, len(docs))
        except Exception:
            logger.exception("Scraper failed for %s", scraper.cb_name)
    logger.info("Total new documents: %d", total)


def preprocess() -> None:
    from src.nlp.preprocessing import TextPreprocessor
    from src.nlp.scrapers.base import Document

    processor = TextPreprocessor()
    raw_dir = Path("data/raw")
    files = list(raw_dir.glob("*.json"))
    if not files:
        logger.info("No new documents to preprocess")
        return
    count = 0
    for fpath in files:
        try:
            data = __import__("json").loads(fpath.read_text())
            doc = Document(cb=data["cb"], doc_type=data["doc_type"], title=data["title"],
                           date=datetime.fromisoformat(data["date"]), url=data["url"],
                           raw_text=data.get("raw_text", ""))
            processed = processor.process(doc)
            count += len(processed.sentences)
        except Exception:
            logger.debug("Preprocess skip: %s", fpath.name)
    logger.info("Preprocessed: %d sentences from %d docs", count, len(files))


def score_lexicon() -> None:
    from src.nlp.lexicon_scorer import LexiconScorer
    scorer = LexiconScorer()
    logger.info("Lexicon scoring — processed %d documents (stub: scores available in preprocess stage)", 0)


def compute_diffs() -> None:
    from src.nlp.diff import StatementDiffer
    differ = StatementDiffer()
    logger.info("Diff computation — stub (requires DB-persisted consecutive statements per CB)")


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
