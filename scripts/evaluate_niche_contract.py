"""Paid opt-in native Kimi output-contract comparison, frozen tools only (CL-uofe).

Only MOONSHOT_API_KEY is required. No .env loading, database, broker, order or
live retrieval capability. This is an engineering trial, not model selection or
profitability evidence. Raw discoveries are scored before any critic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from scripts.alpaca_recovery_report import write_private

from src.events.kimi_tool_agent import KimiToolAgent
from src.events.niche_scoring import evidence_score, parse_niche_ideas, verify_ideas
from src.events.niche_shadow import CapturedInput, FrozenTools
from src.events.research_evidence import timestamp


class FrozenNative(KimiToolAgent):
    def __init__(self, captured: CapturedInput, *, references: bool, model: str) -> None:
        self.frozen = FrozenTools(captured)
        super().__init__(
            universe=self.frozen,
            tools=None,
            model=model,
            max_iterations=8,
            max_tokens=4096,
            max_prompt_chars=100_000,
            passage_references=references,
        )

    def _dispatch(self, name: str, args: Any) -> dict[str, Any]:
        return self.frozen.dispatch(name, dict(args))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allow-model-calls", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_calls:
        parser.error("explicit --allow-model-calls required")
    raw = json.loads(args.snapshot.read_text())
    captured = CapturedInput.capture(raw.get("capture", raw))
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    payload = captured.payload()
    as_of = timestamp(payload["captured_at"])
    assert as_of is not None
    write_private(args.output_dir / "capture.json", payload)
    trials = []
    for references in (False, True):
        agent = FrozenNative(captured, references=references, model=args.model)
        outcome = agent.discover_result(payload["event"])
        ideas = parse_niche_ideas(outcome.text)
        for idea in ideas:
            idea.discovery_status, idea.sources = outcome.status, list(outcome.sources)
        verified = verify_ideas(ideas, agent.frozen)
        for idea in verified:
            evidence_score(idea, payload["market_data"], as_of)
        trial: dict[str, Any] = {
            "passage_references": references,
            "outcome": outcome.to_dict(),
            "candidates": [idea.to_trade_idea() for idea in ideas],
            "metrics": {
                "parsed": len(ideas),
                "identity_verified": len(verified),
                "source_backed": sum(i.evidence_status == "source_backed" for i in ideas),
                "eligible": sum(i.research_eligible for i in ideas),
            },
            "semantic_review": "not_performed",
            "billed_cost_usd": None,
        }
        write_private(args.output_dir / f"{'passages' if references else 'baseline'}.json", trial)
        trials.append(trial)
        print(
            json.dumps(
                {
                    "passage_references": references,
                    "status": outcome.status,
                    "reason": outcome.reason,
                    **trial["metrics"],
                }
            )
        )
    write_private(
        args.output_dir / "report.json",
        {
            "mode": "shadow_only",
            "evaluation": "single_event_engineering_not_profitability",
            "input_hash": captured.input_hash,
            "trials": trials,
            "bounds_per_arm": {
                "calls": 8,
                "tools": 24,
                "requested_output_tokens": 32768,
                "prompt_chars_per_call": 100000,
            },
            "code_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path("src/events/kimi_tool_agent.py"),
                    Path("src/events/passage_contract.py"),
                )
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
