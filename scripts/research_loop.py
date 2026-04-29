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
import os
import sys
from pathlib import Path
from typing import cast

from src.research.agents.idea import IdeaGenerator
from src.research.agents.implementer import Implementer
from src.research.backtest_runner import make_backtest_runner
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
        "--backtest-start", default="2018-01-01",
        help="Backtest window start (ISO date)",
    )
    p.add_argument(
        "--backtest-end", default="2024-12-31",
        help="Backtest window end (ISO date)",
    )
    p.add_argument(
        "--no-backtest", action="store_true",
        help=(
            "Skip the real backtest_runner. Implementer ships with "
            "empty backtest_metrics, verdict engine ESCALATEs everything "
            "for missing-metric. Useful for dev / dry-run."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Walk every phase with stubbed LLM calls + stubbed HTTP + "
            "stubbed backtest. No real API calls, no network, no "
            "money risk. Preflight before CL-77rq (the real smoke). "
            "Implies skip_git=True + skip_pr=True on the registrar."
        ),
    )
    p.add_argument(
        "--auto-approve", action="store_true",
        help=(
            "Auto-flip PENDING_OPERATOR_APPROVAL → APPROVED and "
            "PENDING_DEPLOY_CONFIRMATION → DEPLOY_APPROVED before each "
            "phase. Useful with --dry-run for a single-invocation walk "
            "through every phase. Never use in production — it bypasses "
            "the operator gates."
        ),
    )
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
    # Auto-load .env so the operator doesn't need to source it before
    # invoking the script. Explicit env vars (systemd / shell exports)
    # still win over the file's contents.
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    # Dry-run pollutes nothing: every output path gets re-rooted under
    # a tmp dir so the production tree stays clean. Used by the
    # extract store, hypothesis dir, candidate dir, transcript root,
    # registrar code+yaml paths, and state file.
    dry_run_tmp: Path | None = None
    if args.dry_run:
        # Install the stubs BEFORE loading the research config so the
        # config validator sees the dry-run drivers under the
        # canonical provider names.
        from src.research.dry_run import install_dry_run_driver  # noqa: PLC0415
        install_dry_run_driver()
        # Make sure agent-config validators don't fail on missing API
        # keys for the canonical providers — DryRunDriver doesn't need
        # them, but the config layer reads them at construction time.
        for env_name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY"):
            os.environ.setdefault(env_name, "dry-run-fake-key")
        # Re-root every output path so the dry-run never touches the
        # real repo. Operator can override individual paths with the
        # explicit --state / --hypothesis-dir / --candidate-dir flags
        # if they want to inspect outputs in a stable location.
        import tempfile  # noqa: PLC0415
        dry_run_tmp = Path(tempfile.mkdtemp(prefix="research-dry-run-"))
        defaults_used = {
            "state": args.state == str(DEFAULT_STATE_PATH),
            "runs": args.runs_dir == str(DEFAULT_RUNS_DIR),
            "hyp": args.hypothesis_dir == str(DEFAULT_HYPOTHESIS_DIR),
            "cand": args.candidate_dir == str(DEFAULT_CANDIDATE_DIR),
        }
        if defaults_used["state"]:
            args.state = str(dry_run_tmp / "state.json")
        if defaults_used["runs"]:
            args.runs_dir = str(dry_run_tmp / "runs")
        if defaults_used["hyp"]:
            args.hypothesis_dir = str(dry_run_tmp / "hypotheses")
        if defaults_used["cand"]:
            args.candidate_dir = str(dry_run_tmp / "candidates")
        print(
            f"DRY RUN: stubbed LLM + HTTP + backtest; outputs under "
            f"{dry_run_tmp}",
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
    if args.dry_run and dry_run_tmp is not None:
        from src.research.dry_run import stub_http_get  # noqa: PLC0415
        extract_store = ExtractStore(root=dry_run_tmp / "extracts")
        ingest_runner = IngestRunner(
            extractor=extractor, store=extract_store, http_get=stub_http_get,
        )
        orchestrator = DebateOrchestrator(
            research_config=research_config, debate_name=args.debate,
            transcript_root=dry_run_tmp / "debates",
        )
    else:
        extract_store = ExtractStore()
        ingest_runner = IngestRunner(extractor=extractor, store=extract_store)
        orchestrator = DebateOrchestrator(
            research_config=research_config, debate_name=args.debate,
        )

    # Wire the real backtest_runner unless --no-backtest / --dry-run.
    # Construction is lazy because the DataProvider needs a Postgres
    # engine and we don't want to require it for --dry-run / --no-backtest.
    from collections.abc import Callable as _Callable  # noqa: PLC0415
    from typing import Any as _Any  # noqa: PLC0415
    backtest_runner: _Callable[[Path], dict[str, _Any]] | None = None
    if args.dry_run:
        from src.research.dry_run import stub_backtest_runner  # noqa: PLC0415
        backtest_runner = stub_backtest_runner
    elif not args.no_backtest:
        from src.data.provider import DataProvider  # noqa: PLC0415
        from src.runtime.run_engine import _build_db_engine  # noqa: PLC0415
        backtest_runner = make_backtest_runner(
            data_provider=DataProvider(_build_db_engine()),
            start=args.backtest_start,
            end=args.backtest_end,
        )

    # Dry runs use a no-op registrar with paths re-rooted under the
    # tmp dir so the production strategies dir + portfolio yaml stay
    # untouched.
    registrar = None
    if args.dry_run and dry_run_tmp is not None:
        from src.research.promote import PromoteRegistrar  # noqa: PLC0415
        # Seed a minimal portfolio YAML for the registrar to mutate.
        portfolio_yaml = dry_run_tmp / "live_portfolio.yaml"
        portfolio_yaml.write_text(
            "strategies: []\ninitial_weights: {}\n",
        )
        registrar = PromoteRegistrar(
            experimental_dir=dry_run_tmp / "_experimental",
            production_dir=dry_run_tmp / "production",
            portfolio_yaml=portfolio_yaml,
            skip_git=True, skip_pr=True,
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
        experimental_code_dir=(
            dry_run_tmp / "_experimental" if dry_run_tmp is not None
            else Path("src/strategies/_experimental")
        ),
        backtest_runner=backtest_runner,
        registrar=registrar,
        auto_approve=args.auto_approve,
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
