"""Hop-graph vs baseline shadow replay over captured events (CL-ynuh).

SHADOW ONLY: reads frozen capture JSON files (the niche_shadow CapturedInput
format), writes one JSON report, and never touches trade_ideas, a ledger, a
broker, the production database or the production .env.

Use --validate-only for an offline cohort manifest without model or memory
access. Duplicate event identities/content are rejected before any calls.
Twenty distinct supplied events and completed paired coverage are engineering
sample checks only: they do not establish authenticity, independent events,
equal research budgets, semantic evidence quality, profitability or permission
to publish ideas. Those limitations remain explicit in the comparison report.

Model calls happen ONLY when an operator passes --allow-model-calls. The
hop-graph traversal, entailment and the shared critic all go through
strict subscription transport with no_tools=True (usd_cost remains unknown,
not zero). The optional equivalent-tool baseline uses the same approved Sonnet
model. All roles share one hard per-event invocation budget. API baselines and
unapproved model overrides are rejected before calls.

Example (operator, research-only environment):

    .venv/bin/python scripts/niche_hopgraph_shadow.py \
        --captures /path/to/captures/ --output /path/to/report.json \
        --baseline claude-subscription --max-model-calls-per-event 3 \
        --memory-sqlite /path/to/hopgraph_memory.sqlite \
        --wall-timeout-seconds 120 --allow-model-calls
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.events._util import ThreadLocalClient  # noqa: E402
from src.events.hop_graph_memory import NicheEdgeMemory  # noqa: E402
from src.events.hop_graph_shadow import compare_events, validate_capture_cohort  # noqa: E402
from src.events.niche_shadow import CapturedInput, ResearchModel  # noqa: E402
from src.events.shadow_deadline import (  # noqa: E402
    Deadline,
    DeadlineClient,
    Diagnostics,
    supervise,
)
from src.events.subscription_shadow_budget import (  # noqa: E402
    DEFAULT_EVENT_CALL_LIMIT,
    SUBSCRIPTION_MODEL,
    SubscriptionResearchModel,
)

logger = logging.getLogger("scripts.niche_hopgraph_shadow")

MIGRATION_026 = Path(__file__).resolve().parents[1] / "migrations" / "026_niche_edges.sql"


@dataclass
class ShadowClients:
    """Injected in tests; built from get_client("claude-code") otherwise."""

    traversal: Any
    entailment: Any
    critic: Any
    baseline: ResearchModel | None


def load_captures(paths: list[Path]) -> list[CapturedInput]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not files:
        raise ValueError("no capture files found")
    captures = [CapturedInput.capture(json.loads(f.read_text())) for f in files]
    validate_capture_cohort(captures)
    return captures


def sqlite_memory(path: Path) -> NicheEdgeMemory:
    """A local sqlite edge memory; the shadow never writes the production DB."""
    from migrations.run import _strip_sql_comments  # noqa: PLC0415
    from sqlalchemy import create_engine, text  # noqa: PLC0415

    engine = create_engine(f"sqlite:///{path}")
    sql = _strip_sql_comments(MIGRATION_026.read_text()).replace("TIMESTAMPTZ", "TEXT")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return NicheEdgeMemory(engine)


def _live_clients(args: argparse.Namespace) -> ShadowClients:  # pragma: no cover - operator only
    baseline: ResearchModel | None = None
    if args.baseline == "claude-subscription":
        baseline = SubscriptionResearchModel(ThreadLocalClient())
    return ShadowClients(
        traversal=ThreadLocalClient(),
        entailment=ThreadLocalClient(),
        critic=ThreadLocalClient(),
        baseline=baseline,
    )


def main(argv: list[str] | None = None, *, clients: ShadowClients | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traversal-model", default=None)
    parser.add_argument("--entailment-model", default=None)
    parser.add_argument("--critic-model", default=None)
    parser.add_argument("--no-critic", action="store_true", help="Leaves every lead unapproved")
    parser.add_argument("--baseline", choices=("none", "claude-subscription"), default="none")
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument(
        "--max-model-calls-per-event",
        type=int,
        default=DEFAULT_EVENT_CALL_LIMIT,
        help="Hard shared CLI invocation limit across both arms and every role",
    )
    parser.add_argument(
        "--wall-timeout-seconds",
        type=float,
        default=None,
        help="Required for real calls: total experiment deadline, plus bounded cleanup",
    )
    parser.add_argument("--memory-sqlite", type=Path, default=None)
    parser.add_argument("--playbooks", type=Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Write offline cohort validation only; no model, retrieval, or memory access",
    )
    parser.add_argument(
        "--allow-model-calls",
        action="store_true",
        help="Explicitly allow approved subscription-only CLI calls; no API fallback",
    )
    args = parser.parse_args(argv)
    for model in (
        args.traversal_model,
        args.entailment_model,
        args.critic_model,
        args.baseline_model,
    ):
        if model not in (None, SUBSCRIPTION_MODEL):
            parser.error(f"subscription-only shadow requires {SUBSCRIPTION_MODEL}")
    if args.max_model_calls_per_event <= 0:
        parser.error("--max-model-calls-per-event must be positive")
    if args.wall_timeout_seconds is not None and (
        not math.isfinite(args.wall_timeout_seconds) or args.wall_timeout_seconds <= 0
    ):
        parser.error("--wall-timeout-seconds must be finite and positive")
    captures = load_captures(args.captures)
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    if args.validate_only:
        validation = {
            "mode": "offline_capture_validation",
            "cohort": validate_capture_cohort(captures),
            "model_calls": 0,
            "comparison_run": False,
            "quality_acceptance": "not_evaluated",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(validation, stream, indent=2, allow_nan=False)
        return 0
    if not args.allow_model_calls:
        parser.error(
            f"{len(captures)} capture(s) validated; --allow-model-calls is required to run"
        )
    if clients is None and args.wall_timeout_seconds is None:
        parser.error("--wall-timeout-seconds is required for real model calls")
    if args.wall_timeout_seconds is not None:
        return supervise(
            lambda deadline, diagnostics: _run_comparison(
                args, captures, clients, deadline, diagnostics
            ),
            output=args.output,
            seconds=args.wall_timeout_seconds,
        )
    report = _run_comparison(args, captures, clients)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False, default=str)
    print(
        f"Hop-graph shadow report written to {args.output}; no orders submitted, "
        "nothing merged into trade_ideas."
    )
    return 0


def _run_comparison(
    args: argparse.Namespace,
    captures: list[CapturedInput],
    clients: ShadowClients | None,
    deadline: Deadline | None = None,
    diagnostics: Diagnostics | None = None,
) -> dict[str, Any]:
    clients = clients or _live_clients(args)
    if deadline is not None:
        assert diagnostics is not None
        diagnostics.append(
            "captured_inputs",
            input_hashes=[capture.input_hash for capture in captures],
            event_ids=[capture.payload()["event"].get("id") for capture in captures],
        )
        clients = ShadowClients(
            traversal=DeadlineClient(clients.traversal, deadline, diagnostics, "traversal"),
            entailment=DeadlineClient(clients.entailment, deadline, diagnostics, "entailment"),
            critic=DeadlineClient(clients.critic, deadline, diagnostics, "critic"),
            baseline=(
                SubscriptionResearchModel(
                    DeadlineClient(clients.baseline.client, deadline, diagnostics, "baseline")
                )
                if isinstance(clients.baseline, SubscriptionResearchModel)
                else clients.baseline
            ),
        )
    from src.events.playbooks import load_playbooks  # noqa: PLC0415

    playbooks = load_playbooks(args.playbooks) if args.playbooks else load_playbooks()
    memory = sqlite_memory(args.memory_sqlite) if args.memory_sqlite else None
    traversal_model = SUBSCRIPTION_MODEL
    entailment_model = SUBSCRIPTION_MODEL
    logger.info(
        "hop-graph shadow: %d captures traversal=%s entailment=%s baseline=%s",
        len(captures),
        traversal_model,
        entailment_model,
        args.baseline,
    )
    return compare_events(
        captures,
        traversal_client=clients.traversal,
        entailment_client=clients.entailment,
        baseline_model=clients.baseline if args.baseline != "none" else None,
        critic=None,
        traversal_model=traversal_model,
        entailment_model=entailment_model,
        memory=memory,
        playbooks=playbooks,
        subscription_call_limit=args.max_model_calls_per_event,
        subscription_critic_client=None if args.no_critic else clients.critic,
    )


if __name__ == "__main__":
    raise SystemExit(main())
