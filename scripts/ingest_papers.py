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
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
            f"\nDRY RUN total: {total_seen} fetched, "
            f"{total_new} new across {len(feeds)} feed(s)",
        )
        return 0

    research_config = load_config(args.research_config)
    # from_config returns the Agent base type; cast to the subclass.
    extractor = cast(PaperExtractor, PaperExtractor.from_config(
        name="paper_extractor", research_config=research_config,
    ))
    runner = IngestRunner(extractor=extractor, store=store)
    summary = runner.run(feeds)

    print(
        f"Ingest summary: feeds={summary.feeds_total} "
        f"failed_feeds={summary.feeds_failed} "
        f"papers_seen={summary.papers_seen} "
        f"skipped_dup={summary.papers_skipped_duplicate} "
        f"extracted={summary.papers_extracted} "
        f"extract_failed={summary.papers_extract_failed}",
    )
    for path in summary.extract_paths:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
