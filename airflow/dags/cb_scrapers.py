"""
Airflow DAG — central bank document scraping and sentiment analysis.
Runs every 6 hours: scrapes new documents, preprocesses, scores, computes diffs.
"""

import json
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

RAW_DIR = Path("data/raw")


def run_scrapers() -> None:
    from src.nlp.scrapers.boe_boj_boc import (
        BoCStatementScraper,
        BoEStatementScraper,
        BoJStatementScraper,
    )
    from src.nlp.scrapers.ecb import ECBStatementScraper
    from src.nlp.scrapers.fed import FedStatementScraper

    since = datetime.utcnow() - timedelta(days=7)
    scrapers = [
        FedStatementScraper(RAW_DIR),
        ECBStatementScraper(RAW_DIR),
        BoEStatementScraper(RAW_DIR),
        BoJStatementScraper(RAW_DIR),
        BoCStatementScraper(RAW_DIR),
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

    files = list(RAW_DIR.glob("*.json"))
    if not files:
        logger.info("No new documents to preprocess")
        return

    from sqlalchemy import create_engine, text
    engine = create_engine(DB_URL)
    processor = TextPreprocessor()
    count = 0
    for fpath in files:
        try:
            data = json.loads(fpath.read_text())
            doc = Document(cb=data["cb"], doc_type=data["doc_type"], title=data["title"],
                            date=datetime.fromisoformat(data["date"]), url=data["url"],
                            raw_text=data.get("raw_text", ""))
            processed = processor.process(doc)
            # Persist to DB
            with engine.begin() as conn:
                for idx, sent in enumerate(processed.sentences):
                    conn.execute(text("""
                        INSERT INTO cb_sentiment (ts, doc_id, cb, doc_type, sentence_idx, sentence)
                        VALUES (:ts, :doc_id, :cb, :doc_type, :idx, :sentence)
                        ON CONFLICT (doc_id, sentence_idx) DO NOTHING
                    """), {"ts": doc.date, "doc_id": doc.doc_id, "cb": doc.cb,
                            "doc_type": doc.doc_type, "idx": idx, "sentence": sent})
            count += len(processed.sentences)
        except Exception:
            logger.debug("Preprocess skip: %s", fpath.name)
    logger.info("Preprocessed: %d sentences", count)


def score_lexicon() -> None:
    from sqlalchemy import create_engine, text

    from src.nlp.lexicon_scorer import LexiconScorer

    engine = create_engine(DB_URL)
    scorer = LexiconScorer()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT doc_id, sentence_idx, sentence FROM cb_sentiment WHERE lex_hawkish IS NULL LIMIT 5000"
        )).fetchall()

    scored = 0
    with engine.begin() as conn:
        for doc_id, idx, sentence in rows:
            s = scorer.score_text(sentence)
            conn.execute(text("""
                UPDATE cb_sentiment SET lex_hawkish=:h, lex_dovish=:d, lex_net=:n
                WHERE doc_id=:did AND sentence_idx=:i
            """), {"h": s.hawkish_count, "d": s.dovish_count, "n": s.net_score,
                    "did": doc_id, "i": idx})
            scored += 1
    logger.info("Lexicon scored: %d sentences", scored)


def compute_diffs() -> None:
    from sqlalchemy import create_engine, text

    from src.nlp.diff import StatementDiffer

    engine = create_engine(DB_URL)
    differ = StatementDiffer()

    # Get distinct CBs with multiple statements
    with engine.connect() as conn:
        cbs = conn.execute(text("SELECT DISTINCT cb FROM cb_sentiment")).fetchall()

    diffs = 0
    for (cb,) in cbs:
        with engine.connect() as conn:
            docs = conn.execute(text("""
                SELECT DISTINCT doc_id, ts FROM cb_diff_events
                WHERE cb = :cb UNION
                SELECT DISTINCT doc_id, ts FROM (SELECT doc_id, MAX(ts) as ts FROM cb_sentiment
                WHERE cb = :cb GROUP BY doc_id) sub
                ORDER BY ts DESC LIMIT 10
            """), {"cb": cb}).fetchall()

        if len(docs) < 2:
            continue

        with engine.begin() as conn:
            for i in range(1, len(docs)):
                curr_id, _ = docs[i]
                prev_id, _ = docs[i-1]
                existing = conn.execute(text(
                    "SELECT 1 FROM cb_diff_events WHERE doc_id = :did"
                ), {"did": curr_id}).fetchone()
                if existing:
                    continue
                curr_sents = conn.execute(text(
                    "SELECT sentence FROM cb_sentiment WHERE doc_id=:did ORDER BY sentence_idx"
                ), {"did": curr_id}).fetchall()
                prev_sents = conn.execute(text(
                    "SELECT sentence FROM cb_sentiment WHERE doc_id=:did ORDER BY sentence_idx"
                ), {"did": prev_id}).fetchall()
                if not curr_sents or not prev_sents:
                    continue
                diff = differ.diff(
                    [r[0] for r in curr_sents], [r[0] for r in prev_sents],
                )
                conn.execute(text("""
                    INSERT INTO cb_diff_events (ts, cb, doc_id, prev_doc_id, net_shift,
                        added_hawkish, removed_hawkish, change_ratio)
                    VALUES (:ts, :cb, :did, :pid, :ns, :ah, :rh, :cr)
                """), {"ts": datetime.utcnow(), "cb": cb, "did": curr_id,
                        "pid": prev_id, "ns": diff.net_shift,
                        "ah": diff.added_hawkish_score, "rh": diff.removed_hawkish_score,
                        "cr": diff.raw_diff_ratio})
                diffs += 1
    logger.info("Diff computed: %d pairs", diffs)


try:
    from src.monitoring.airflow_callbacks import on_dag_failure, on_dag_success
except Exception:  # noqa: BLE001
    on_dag_success = on_dag_failure = None  # type: ignore[assignment]

default_args = {
    "owner": "curlit", "retries": 1, "retry_delay": timedelta(minutes=5),
    "on_success_callback": on_dag_success,
    "on_failure_callback": on_dag_failure,
}

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
