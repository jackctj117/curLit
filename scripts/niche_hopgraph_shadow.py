"""Hop-graph vs baseline shadow replay over captured events (CL-ynuh).

SHADOW ONLY: reads frozen capture JSON files (the niche_shadow CapturedInput
format), writes one JSON report, and never touches trade_ideas, a ledger, a
broker, the production database or the production .env.

Model calls happen ONLY when an operator passes --allow-model-calls. The
hop-graph traversal, entailment and the shared critic all go through
get_client("claude-code") with no_tools=True (subscription CLI: usd_cost is
reported unknown, not zero). The optional baseline arm uses the existing
equivalent-tool harness with an API model (MOONSHOT_API_KEY or
ANTHROPIC_API_KEY must already be in the environment).

Example (operator, research-only environment):

    .venv/bin/python scripts/niche_hopgraph_shadow.py \
        --captures /path/to/captures/ --output /path/to/report.json \
        --baseline moonshot --baseline-model YOUR_KIMI_MODEL \
        --memory-sqlite /path/to/hopgraph_memory.sqlite --allow-model-calls
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.events.adversarial_critic import AdversarialCritic  # noqa: E402
from src.events.hop_graph import default_traversal_model  # noqa: E402
from src.events.hop_graph_memory import NicheEdgeMemory  # noqa: E402
from src.events.hop_graph_shadow import compare_events  # noqa: E402
from src.events.hop_graph_verify import default_entailment_model  # noqa: E402
from src.events.niche_shadow import CapturedInput, ResearchModel  # noqa: E402

logger = logging.getLogger("scripts.niche_hopgraph_shadow")

MIGRATION_025 = Path(__file__).resolve().parents[1] / "migrations" / "025_niche_edges.sql"


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
    return [CapturedInput.capture(json.loads(f.read_text())) for f in files]


def sqlite_memory(path: Path) -> NicheEdgeMemory:
    """A local sqlite edge memory; the shadow never writes the production DB."""
    from migrations.run import _strip_sql_comments  # noqa: PLC0415
    from sqlalchemy import create_engine, text  # noqa: PLC0415

    engine = create_engine(f"sqlite:///{path}")
    sql = _strip_sql_comments(MIGRATION_025.read_text()).replace("TIMESTAMPTZ", "TEXT")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return NicheEdgeMemory(engine)


def _live_clients(args: argparse.Namespace) -> ShadowClients:  # pragma: no cover - operator only
    from src.research.llm.client import get_client  # noqa: PLC0415

    baseline: ResearchModel | None = None
    if args.baseline == "moonshot":
        from src.events.niche_shadow import MoonshotResearchModel  # noqa: PLC0415

        key = os.environ.get("MOONSHOT_API_KEY")
        if not key:
            raise SystemExit("MOONSHOT_API_KEY missing; do not copy production .env")
        baseline = MoonshotResearchModel(args.baseline_model, key)
    elif args.baseline == "claude":
        from src.events.niche_shadow import ClaudeResearchModel  # noqa: PLC0415

        baseline = ClaudeResearchModel(args.baseline_model)
    return ShadowClients(
        traversal=get_client("claude-code"),
        entailment=get_client("claude-code"),
        critic=get_client("claude-code"),
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
    parser.add_argument("--baseline", choices=("none", "moonshot", "claude"), default="none")
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--memory-sqlite", type=Path, default=None)
    parser.add_argument("--playbooks", type=Path, default=None)
    parser.add_argument(
        "--allow-model-calls",
        action="store_true",
        help="Explicitly allow model calls (subscription CLI and, for a baseline, a billed API)",
    )
    args = parser.parse_args(argv)
    captures = load_captures(args.captures)
    if args.baseline != "none" and not args.baseline_model:
        parser.error("--baseline-model is required with a baseline arm")
    if not args.allow_model_calls:
        parser.error(
            f"{len(captures)} capture(s) validated; --allow-model-calls is required to run"
        )
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    clients = clients or _live_clients(args)
    from src.events.playbooks import load_playbooks  # noqa: PLC0415

    playbooks = load_playbooks(args.playbooks) if args.playbooks else load_playbooks()
    critic = (
        None
        if args.no_critic
        else AdversarialCritic(client=clients.critic, model=args.critic_model, enabled=True)
    )
    memory = sqlite_memory(args.memory_sqlite) if args.memory_sqlite else None
    traversal_model = args.traversal_model or default_traversal_model()
    entailment_model = args.entailment_model or default_entailment_model()
    logger.info(
        "hop-graph shadow: %d captures traversal=%s entailment=%s baseline=%s",
        len(captures),
        traversal_model,
        entailment_model,
        args.baseline,
    )
    report = compare_events(
        captures,
        traversal_client=clients.traversal,
        entailment_client=clients.entailment,
        baseline_model=clients.baseline if args.baseline != "none" else None,
        critic=critic,
        traversal_model=traversal_model,
        entailment_model=entailment_model,
        memory=memory,
        playbooks=playbooks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False, default=str)
    print(
        f"Hop-graph shadow report written to {args.output}; no orders submitted, "
        "nothing merged into trade_ideas."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
