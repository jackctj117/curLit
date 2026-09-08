"""Opt-in model comparison over a supplied capture; never loads production .env."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.events.adversarial_critic import AdversarialCritic  # noqa: E402
from src.events.niche_shadow import (  # noqa: E402
    CapturedInput,
    ClaudeResearchModel,
    MoonshotResearchModel,
    ResearchModel,
    blinded_candidates,
    compare_captured,
)
from src.research.llm.client import get_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kimi-model", required=True)
    parser.add_argument("--claude-model", required=True)
    parser.add_argument("--critic-model", required=True)
    parser.add_argument(
        "--claude-first", action="store_true", help="Alternate arm order across events"
    )
    parser.add_argument(
        "--allow-model-calls",
        action="store_true",
        help="Explicitly allow billed research calls; no broker access is used",
    )
    args = parser.parse_args()
    capture = CapturedInput.capture(json.loads(args.snapshot.read_text()))
    if not args.allow_model_calls:
        parser.error("capture validated; --allow-model-calls is required to contact providers")
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        parser.error("MOONSHOT_API_KEY missing; do not copy production .env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY missing; the equivalent-budget comparison uses the API")
    # Refuse to overwrite previous experiments. Restrict private trial provenance.
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    models: list[ResearchModel] = [
        MoonshotResearchModel(args.kimi_model, key),
        ClaudeResearchModel(args.claude_model),
    ]
    report = compare_captured(
        capture,
        list(reversed(models)) if args.claude_first else models,
        critic=AdversarialCritic(
            client=get_client("claude"), model=args.critic_model, enabled=True
        ),
    )
    report["code_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()
    report["dirty_worktree"] = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    )
    for filename, payload in (
        ("private_report.json", report),
        ("blinded_candidates.json", blinded_candidates(report)),
    ):
        with (args.output_dir / filename).open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
    print("Shadow comparison recorded; no orders submitted. Review failures and missing evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
