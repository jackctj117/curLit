"""CLI entry for paper-stream ingestion (CL-2klj).

Usage:
    .venv/bin/python scripts/ingest_papers.py [--feed=<name>]
                                              [--config=<path>]
                                              [--dry-run]

Reads ``configs/paper_streams.yaml``, runs each configured feed
through the IngestRunner (fetch → dedup → LLM extract → persist), and
writes a summary to stdout. Designed to be cron-driven (idempotent —
already-extracted papers are skipped via hash-based dedup).

``--dry-run`` lists the papers that WOULD be extracted (i.e. not yet
in the store) without calling the LLM. Useful for previewing a run
before consuming budget.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

from src.research.config import load_config
from src.research.ingest import (
    DEFAULT_EXTRACT_ROOT,
    ExtractStore,
    IngestRunner,
    PaperExtractor,
    build_fetcher,
    load_feed_configs,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the paper-stream ingester")
    p.add_argument(
        "--config",
        default="configs/paper_streams.yaml",
        help="Path to feed config YAML",
    )
    p.add_argument(
        "--research-config",
        default="configs/research_agents.yaml",
        help="Path to research agent config (loads paper_extractor agent)",
    )
    p.add_argument(
        "--feed",
        default=None,
        help="Run only this named feed (default: all configured feeds)",
    )
    p.add_argument(
        "--extract-root",
        default=str(DEFAULT_EXTRACT_ROOT),
        help="Where to write extracts",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List new papers without extracting",
    )
    p.add_argument(
        "--no-db",
        action="store_true",
        help=(
            "Skip writing rows to research_papers (CL-28j bridge). "
            "Useful when iterating on extractor prompts."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Auto-load .env so the LLM extractor's API keys are available
    # without first sourcing the file. Explicit env vars still win.
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    feeds = load_feed_configs(args.config)
    if args.feed:
        feeds = [f for f in feeds if f.name == args.feed]
        if not feeds:
            print(f"feed {args.feed!r} not found in {args.config}", file=sys.stderr)
            return 2

    store = ExtractStore(root=Path(args.extract_root))

    if args.dry_run:
        print(f"DRY RUN — extract root: {store.root}")
        total_seen = total_new = 0
        for feed in feeds:
            fetcher = build_fetcher(feed.adapter)
            papers = fetcher.fetch(feed)
            new = [p for p in papers if not store.has(p)]
            total_seen += len(papers)
            total_new += len(new)
            print(f"\n{feed.name}: {len(papers)} fetched, {len(new)} new")
            for p in new[:10]:
                print(f"  - {p.year} {p.title[:80]} ({len(p.authors)} authors)")
            if len(new) > 10:
                print(f"  ... and {len(new) - 10} more")
        print(
            f"\nDRY RUN total: {total_seen} fetched, {total_new} new across {len(feeds)} feed(s)",
        )
        return 0

    research_config = load_config(args.research_config)
    # from_config returns the Agent base type; cast to the subclass.
    extractor = cast(
        PaperExtractor,
        PaperExtractor.from_config(
            name="paper_extractor",
            research_config=research_config,
        ),
    )

    # CL-28j bridge: optionally write each paper to research_papers
    # so the triage dashboard sees them. Default is on; pass
    # --no-db to skip (e.g. when iterating on extractor prompts and
    # don't want to pollute the table).
    db_engine = None
    relevance_scorer = None
    if not args.no_db:
        try:
            from src.research.relevance_scorer import RelevanceScorer
            from src.runtime.run_engine import _build_db_engine

            db_engine = _build_db_engine()
            relevance_scorer = RelevanceScorer()
        except Exception:
            logging.exception(
                "DB engine + scorer build failed — running disk-only "
                "(papers won't appear in triage dashboard)",
            )

    runner = IngestRunner(
        extractor=extractor,
        store=store,
        db_engine=db_engine,
        relevance_scorer=relevance_scorer,
    )
    summary = runner.run(feeds)

    print(
        f"Ingest summary: feeds={summary.feeds_total} "
        f"failed_feeds={summary.feeds_failed} "
        f"papers_seen={summary.papers_seen} "
        f"skipped_dup={summary.papers_skipped_duplicate} "
        f"extracted={summary.papers_extracted} "
        f"extract_failed={summary.papers_extract_failed} "
        f"db_inserted={summary.papers_db_inserted} "
        f"db_failed={summary.papers_db_failed}",
    )
    for path in summary.extract_paths:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
