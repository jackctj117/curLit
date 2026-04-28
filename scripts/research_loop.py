"""CLI entry for the autonomous research loop (CL-x561).

Usage:
    .venv/bin/python -m scripts.research_loop [--config=path]
                                              [--feeds=path]

Composes the research pipeline (ingest → idea → implement → debate →
verdict) from configs/research_agents.yaml + configs/paper_streams.yaml
and runs one full pass. Idempotent — already-processed extracts /
hypotheses / candidates are skipped on subsequent runs via the state
file at ``data/research/state.json``.

Designed for cron. Per-run summary at ``data/research/runs/{ts}.json``
gives the operator a quick "what did this run do" view without needing
to diff state.json across runs.

Side-effects (paper-shadow registration, Pushover/Telegram alerting on
PROMOTE/ESCALATE) are tracked as separate beads — this loop populates
the verdict log; the side-effect agents read it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

from src.research.agents.idea import IdeaGenerator
from src.research.agents.implementer import Implementer
from src.research.config import load_config
from src.research.ingest import (
    ExtractStore,
    IngestRunner,
    PaperExtractor,
    load_feed_configs,
)
from src.research.loop import (
    DEFAULT_CANDIDATE_DIR,
    DEFAULT_HYPOTHESIS_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_STATE_PATH,
    ResearchLoop,
)
from src.research.orchestrator import DebateOrchestrator
from src.research.verdict import parse_rules


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the autonomous research loop")
    p.add_argument(
        "--config", default="configs/research_agents.yaml",
        help="Research agent + debate config",
    )
    p.add_argument(
        "--feeds", default="configs/paper_streams.yaml",
        help="Paper feed config",
    )
    p.add_argument("--debate", default="promotion_review")
    p.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--hypothesis-dir", default=str(DEFAULT_HYPOTHESIS_DIR))
    p.add_argument("--candidate-dir", default=str(DEFAULT_CANDIDATE_DIR))
    p.add_argument(
        "-v", "--verbose", action="store_true", help="DEBUG-level logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    research_config = load_config(args.config)
    feed_configs = load_feed_configs(args.feeds)

    extractor = cast(PaperExtractor, PaperExtractor.from_config(
        name="paper_extractor", research_config=research_config,
    ))
    idea_agent = cast(IdeaGenerator, IdeaGenerator.from_config(
        name="idea_generator", research_config=research_config,
    ))
    implementer = cast(Implementer, Implementer.from_config(
        name="implementer", research_config=research_config,
    ))
    extract_store = ExtractStore()
    ingest_runner = IngestRunner(extractor=extractor, store=extract_store)
    orchestrator = DebateOrchestrator(
        research_config=research_config, debate_name=args.debate,
    )

    rules_path = research_config.debates[args.debate].rules_path
    loop = ResearchLoop(
        ingest_runner=ingest_runner,
        idea_agent=idea_agent,
        implementer=implementer,
        debate_orchestrator=orchestrator,
        rules_loader=lambda: parse_rules(rules_path),
        feed_configs=feed_configs,
        extract_store=extract_store,
        state_path=Path(args.state),
        runs_dir=Path(args.runs_dir),
        hypothesis_dir=Path(args.hypothesis_dir),
        candidate_dir=Path(args.candidate_dir),
    )

    summary = loop.run()
    print(
        f"Run done — extracts_new={summary.extracts_new} "
        f"ideas_proposed={summary.ideas_proposed} "
        f"declined={summary.ideas_declined} "
        f"implemented={summary.candidates_implemented} "
        f"impl_rejected={summary.candidates_rejected} "
        f"debates={summary.debates_run} "
        f"PROMOTE={summary.verdicts_promote} "
        f"REJECT={summary.verdicts_reject} "
        f"ESCALATE={summary.verdicts_escalate} "
        f"errors={len(summary.errors)}",
    )
    if summary.errors:
        print("\nErrors:", file=sys.stderr)
        for e in summary.errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
