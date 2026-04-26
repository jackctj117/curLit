#!/usr/bin/env python3
"""Seed CB statement pipeline (CL-53s5).

Runs the existing scrapers + preprocessor + lexicon scorer + diff analyzer
end-to-end to populate the `cb_sentiment` and `cb_diff_events` tables in
Postgres. The cb_sentiment_shift backtest reads from those tables.

Phase 1: Fed only (cleanest URL pattern, ~96 statements 2015-2026). ECB
/ BoE / BoJ / BoC scrapers exist but their archives are messier — file
follow-up tickets to extend.

Usage:
    .venv/bin/python scripts/seed_cb_statements.py \\
        [--since 2015-01-01] [--cbs fed] [--raw-dir data/cb_raw]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Make src.* imports work when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.nlp.diff import StatementDiffer
from src.nlp.lexicon_scorer import LexiconScorer
from src.nlp.preprocessing import TextPreprocessor
from src.nlp.scrapers.base import Document
from src.nlp.scrapers.fed import FedStatementScraper
from src.runtime.run_engine import _build_db_engine

logger = logging.getLogger(__name__)


# Each scraper is keyed by CB code so --cbs takes simple short names.
SCRAPERS: dict[str, type] = {
    "fed": FedStatementScraper,
}


def _load_raw_doc(path: Path) -> Document | None:
    """Reconstruct Document from a saved JSON file (matches base.save_document)."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        logger.exception("failed to load %s", path)
        return None
    return Document(
        cb=d["cb"], doc_type=d["doc_type"], title=d["title"],
        date=datetime.fromisoformat(d["date"]),
        url=d["url"], speaker=d.get("speaker"),
        raw_html="", raw_text=d["raw_text"],
        metadata=d.get("metadata", {}),
    )


def _upsert_sentences(
    engine: Any, processed: Any, scores: list[Any],
) -> int:
    """Write per-sentence rows to cb_sentiment.

    On conflict (doc_id, sentence_idx) we DO NOTHING — re-runs are no-ops.
    """
    rows: list[dict[str, Any]] = []
    for idx, (sent, score) in enumerate(zip(processed.sentences, scores)):
        rows.append({
            "ts": processed.date,
            "doc_id": processed.doc_id,
            "cb": processed.cb,
            "doc_type": processed.doc_type,
            "sentence_idx": idx,
            "sentence": sent,
            "lex_hawkish": score.hawkish_count,
            "lex_dovish": score.dovish_count,
            "lex_net": float(score.net_score),
        })
    if not rows:
        return 0
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO cb_sentiment
                  (ts, doc_id, cb, doc_type, sentence_idx, sentence,
                   lex_hawkish, lex_dovish, lex_net)
                VALUES
                  (:ts, :doc_id, :cb, :doc_type, :sentence_idx, :sentence,
                   :lex_hawkish, :lex_dovish, :lex_net)
                ON CONFLICT (doc_id, sentence_idx) DO NOTHING
            """),
            rows,
        )
    return len(rows)


def _upsert_diff_event(
    engine: Any, cb: str, current: Any, previous: Any, diff: Any,
) -> int:
    """Write one diff event row to cb_diff_events."""
    n_prev = max(len(previous.sentences), 1)
    change_ratio = (
        len(diff.added_sentences) + len(diff.removed_sentences)
    ) / n_prev
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO cb_diff_events
                  (ts, cb, doc_id, prev_doc_id, net_shift,
                   added_hawkish, removed_hawkish, change_ratio)
                VALUES (:ts, :cb, :doc_id, :prev_doc_id, :net_shift,
                        :added_hawkish, :removed_hawkish, :change_ratio)
                ON CONFLICT (doc_id) DO NOTHING
            """),
            {
                "ts": current.date,
                "cb": cb,
                "doc_id": current.doc_id,
                "prev_doc_id": previous.doc_id,
                "net_shift": float(diff.net_shift),
                "added_hawkish": float(diff.added_hawkish_score),
                "removed_hawkish": float(diff.removed_hawkish_score),
                "change_ratio": float(change_ratio),
            },
        )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed CB statements + scoring + diff events into Postgres.",
    )
    parser.add_argument("--since", type=str, default="2015-01-01")
    parser.add_argument(
        "--cbs", type=str, default="fed",
        help=f"Comma list. Available: {sorted(SCRAPERS)}",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/cb_raw"),
        help="Where to cache scraped JSON. Re-runs skip already-fetched docs.",
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip scraping; reprocess whatever's already in raw-dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    since = datetime.strptime(args.since, "%Y-%m-%d")
    cbs = [c.strip() for c in args.cbs.split(",") if c.strip()]
    unknown = [c for c in cbs if c not in SCRAPERS]
    if unknown:
        logger.error("Unknown CBs %s. Available: %s", unknown, sorted(SCRAPERS))
        return 2

    args.raw_dir.mkdir(parents=True, exist_ok=True)

    engine = _build_db_engine()
    preprocessor = TextPreprocessor()
    scorer = LexiconScorer()
    differ = StatementDiffer(lexicon_scorer=scorer)

    summary: dict[str, dict[str, Any]] = {}

    for cb in cbs:
        t0 = time.time()
        scraper_cls = SCRAPERS[cb]
        cb_raw = args.raw_dir / cb
        cb_raw.mkdir(parents=True, exist_ok=True)
        scraper = scraper_cls(raw_dir=cb_raw)

        # 1. Scrape (skipped if --skip-scrape; existing files cached).
        if not args.skip_scrape:
            logger.info("[%s] scraping since %s", cb, since.date())
            docs = scraper.run(since=since)
            logger.info("[%s] scraped %d new statements", cb, len(docs))

        # 2. Load all cached raw docs (whether just scraped or pre-existing).
        raw_paths = sorted(cb_raw.glob("*.json"))
        loaded: list[Any] = []
        for p in raw_paths:
            doc = _load_raw_doc(p)
            if doc is not None and doc.date >= since:
                loaded.append(doc)
        logger.info("[%s] loaded %d cached docs", cb, len(loaded))

        # 3. Preprocess + score each → cb_sentiment table.
        processed_by_doc: list[Any] = []
        n_sentences = 0
        for doc in loaded:
            try:
                proc = preprocessor.process(doc)
            except Exception:
                logger.exception("[%s] preprocess failed: %s", cb, doc.doc_id)
                continue
            scores = scorer.score_sentences(proc.sentences)
            n_sentences += _upsert_sentences(engine, proc, scores)
            processed_by_doc.append(proc)
        logger.info("[%s] upserted %d sentence rows", cb, n_sentences)

        # 4. Diff each consecutive pair (sorted by date) → cb_diff_events.
        processed_by_doc.sort(key=lambda p: p.date)
        n_events = 0
        for prev, curr in zip(processed_by_doc, processed_by_doc[1:]):
            try:
                diff = differ.diff(curr.sentences, prev.sentences)
            except Exception:
                logger.exception(
                    "[%s] diff failed: %s vs %s", cb, prev.doc_id, curr.doc_id,
                )
                continue
            n_events += _upsert_diff_event(engine, cb, curr, prev, diff)
        logger.info("[%s] upserted %d diff events", cb, n_events)

        summary[cb] = {
            "n_docs": len(processed_by_doc),
            "n_sentences": n_sentences,
            "n_events": n_events,
            "elapsed_sec": round(time.time() - t0, 1),
        }

    # ---- print summary ----
    print()
    print(f"{'cb':<6}  {'docs':>5}  {'sentences':>10}  {'events':>7}  {'elapsed':>9}")
    print("-" * 50)
    for cb, s in summary.items():
        print(
            f"{cb:<6}  {s['n_docs']:>5}  {s['n_sentences']:>10}  "
            f"{s['n_events']:>7}  {s['elapsed_sec']:>8}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
