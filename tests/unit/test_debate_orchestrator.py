"""Unit tests for the debate orchestrator (CL-ew4a).

Uses fake agents — the orchestrator takes an ``agent_factory`` callback,
so tests inject mock Agent instances whose ``run`` returns canned
per-call responses. No LLM driver involvement, no monkeypatching.

Covers:
  * parallel round dispatches all participants with the same context
  * sequential round threads each participant's output into the next
  * per_agent_async round parses smart-questions, dispatches via the
    resolver, accumulates open questions on ESCALATE/FAILED
  * transcript persistence — markdown header + per-entry blocks +
    jsonl ledger
  * final positions extracted from each reviewer's last response
  * rebuttal cap enforcement
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.research.agents.base import Agent, AgentResponse
from src.research.agents.resolver import QuestionResolver
from src.research.agents.reviewer import Position
from src.research.config import (
    AgentConfig,
    DebateConfig,
    ProviderConfig,
    ResearchConfig,
    RoundConfig,
    RoundType,
)
from src.research.llm.client import Driver, LLMResponse, register_driver
from src.research.orchestrator import (
    MAX_REBUTTAL_ROUNDS,
    DebateOrchestrator,
    _slim_candidate_report,
    _slim_transcript,
)

# Provider registration ------------------------------------------------------
# The config schema validates provider names against registered drivers; we
# only need the registration to satisfy that check — the orchestrator never
# actually calls the driver in these tests because we inject mock agents.

class _NullDriver(Driver):
    name = "null"

    def complete(
        self,
        messages: Any,  # noqa: ARG002
        model: str,
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            text="", model=model, provider="null",
            input_tokens=0, output_tokens=0, usd_cost=0.0, elapsed_sec=0.0,
        )


register_driver("null", _NullDriver)


# Mock agent ----------------------------------------------------------------


@dataclass
class _FakeAgent:
    """Drop-in for Agent that returns canned per-call responses.

    ``responses`` is a list of strings — each ``run`` call pops the next
    one and returns it as the agent's response. Calls beyond the list
    return the last response on repeat (so the test doesn't have to
    pre-compute the exact number of internal calls).
    """

    name: str
    role: str
    responses: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)
    _idx: int = 0

    def run(
        self,
        user_prompt: str,
        context_files: dict[str, str] | None = None,
        extra_system: str | None = None,
    ) -> AgentResponse:
        self.calls.append({
            "user_prompt": user_prompt,
            "context_files": dict(context_files or {}),
            "extra_system": extra_system,
        })
        text = (
            self.responses[self._idx]
            if self._idx < len(self.responses)
            else self.responses[-1]
        )
        if self._idx < len(self.responses) - 1:
            self._idx += 1
        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            text=text,
            model="canned",
            provider="null",
            input_tokens=10,
            output_tokens=20,
            usd_cost=0.001,
            elapsed_sec=0.01,
        )


def _make_factory(agents: dict[str, _FakeAgent]) -> Callable[[str], Agent]:
    """Build an agent_factory closure that returns the prepared fakes."""

    def factory(name: str) -> Agent:
        if name not in agents:
            msg = f"no fake agent registered for {name!r}"
            raise KeyError(msg)
        return agents[name]  # type: ignore[return-value]

    return factory


# Config builder ------------------------------------------------------------


def _build_config(
    rules_path: Path,
    rounds: list[RoundConfig],
    participants: tuple[str, ...] = ("bull", "bear"),
) -> ResearchConfig:
    return ResearchConfig(
        providers={"null": ProviderConfig(api_key_env="UNUSED", default_model="m1")},
        agents={
            "bull": AgentConfig(
                provider="null", role="bull",
                prompt_path=str(rules_path), model=None,
            ),
            "bear": AgentConfig(
                provider="null", role="bear",
                prompt_path=str(rules_path), model=None,
            ),
            "resolver_agent": AgentConfig(
                provider="null", role="resolver",
                prompt_path=str(rules_path), model=None,
            ),
        },
        debates={
            "promotion_review": DebateConfig(
                participants=list(participants),
                rules_path=str(rules_path),
                rounds=rounds,
            ),
        },
    )


@pytest.fixture
def rules_file(tmp_path: Path) -> Path:
    f = tmp_path / "REVIEW_RULES.md"
    f.write_text("# Stub rules\n\nA.1: stub\nB.2: stub\n")
    return f


@pytest.fixture
def transcript_root(tmp_path: Path) -> Path:
    return tmp_path / "debates"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_unknown_debate_raises(self, rules_file: Path) -> None:
        cfg = _build_config(
            rules_file,
            [RoundConfig(name="r1", type=RoundType.PARALLEL)],
        )
        with pytest.raises(KeyError, match="ghost"):
            DebateOrchestrator(
                research_config=cfg, debate_name="ghost",
            )

    def test_default_factory_uses_agent_from_config(
        self, rules_file: Path,
    ) -> None:
        # Just verify the orchestrator wires a default factory — we don't
        # actually run a debate (would hit live API).
        cfg = _build_config(
            rules_file,
            [RoundConfig(name="r1", type=RoundType.PARALLEL)],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
        )
        assert orch.debate.participants == ["bull", "bear"]
        assert orch.debate_name == "promotion_review"


# ---------------------------------------------------------------------------
# Parallel round
# ---------------------------------------------------------------------------


class TestParallelRound:
    def test_dispatches_all_participants_with_same_prompt(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        bull = _FakeAgent("bull", "bull", ["**FINAL_POSITION**: PROMOTE"])
        bear = _FakeAgent("bear", "bear", ["**FINAL_POSITION**: REJECT"])
        cfg = _build_config(
            rules_file,
            [RoundConfig(name="initial_positions", type=RoundType.PARALLEL)],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            transcript_root=transcript_root,
        )
        result = orch.run_debate(
            strategy_slug="strat-001",
            candidate_report_text="metrics: sharpe=0.7",
        )
        # Both agents called exactly once
        assert len(bull.calls) == 1
        assert len(bear.calls) == 1
        # Same context fed to both
        for agent in (bull, bear):
            ctx = agent.calls[0]["context_files"]
            assert "REVIEW_RULES.md" in ctx
            assert "candidate_report" in ctx
            assert "metrics: sharpe=0.7" in ctx["candidate_report"]
            # Round 1: no prior transcript yet
            assert "debate_transcript_so_far" not in ctx
        # Final positions parsed correctly
        assert result.final_positions == {
            "bull": Position.PROMOTE,
            "bear": Position.REJECT,
        }
        assert result.transcript_path.exists()


# ---------------------------------------------------------------------------
# Sequential round
# ---------------------------------------------------------------------------


class TestSequentialRound:
    def test_second_agent_sees_first_response_in_transcript(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        bull = _FakeAgent(
            "bull", "bull",
            ["BULL_INITIAL_R1", "BULL_REBUTTAL line"],
        )
        bear = _FakeAgent(
            "bear", "bear",
            ["BEAR_INITIAL_R1", "BEAR_REBUTTAL line"],
        )
        cfg = _build_config(
            rules_file,
            [
                RoundConfig(name="initial_positions", type=RoundType.PARALLEL),
                RoundConfig(name="rebuttal", type=RoundType.SEQUENTIAL),
            ],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            transcript_root=transcript_root,
        )
        orch.run_debate(
            strategy_slug="strat-002",
            candidate_report_text="metrics",
        )
        # Round 2 calls are calls[1] for each agent
        bull_round2_ctx = bull.calls[1]["context_files"]
        bear_round2_ctx = bear.calls[1]["context_files"]
        # Bull (first in declared order) sees the Round-1 outputs but
        # NOT Bear's Round-2 rebuttal yet (hasn't happened)
        assert "BULL_INITIAL_R1" in bull_round2_ctx["debate_transcript_so_far"]
        assert "BEAR_INITIAL_R1" in bull_round2_ctx["debate_transcript_so_far"]
        assert "BEAR_REBUTTAL line" not in bull_round2_ctx[
            "debate_transcript_so_far"]
        # Bear (second) DOES see Bull's just-emitted rebuttal
        assert "BULL_REBUTTAL line" in bear_round2_ctx[
            "debate_transcript_so_far"]


# ---------------------------------------------------------------------------
# Per-agent async (smart-questions)
# ---------------------------------------------------------------------------


_GOOD_QUESTION = """```yaml
question_id: q-2-1
asker: bull
decision_blocker: |
  Whether to PASS rule B.2 — regime decomposition isn't in the report
what_i_tried_first: |
  - "Read backtest output JSON: no regime field present"
  - "Grepped src/backtest/walkforward.py: no regime tagging"
specific_evidence_needed: |
  A direct line reference in the strategy code showing whether regime
  tagging is computed at all
acceptance: |
  A line citation. 'Yes with line' = PASS B.2; 'No' = FAIL B.2.
routing: human
```"""


_BAD_QUESTION = """```yaml
question_id: q-2-2
asker: bull
decision_blocker: I am not sure
what_i_tried_first: just looked
specific_evidence_needed: I am not sure
routing: human
acceptance: yes
```"""


class TestPerAgentAsyncRound:
    def test_human_routing_records_open_question(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        bull = _FakeAgent("bull", "bull", [_GOOD_QUESTION])
        bear = _FakeAgent("bear", "bear", ["No questions."])
        cfg = _build_config(
            rules_file,
            [RoundConfig(
                name="smart_questions", type=RoundType.PER_AGENT_ASYNC,
            )],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            transcript_root=transcript_root,
        )
        result = orch.run_debate(
            strategy_slug="strat-003",
            candidate_report_text="metrics",
        )
        # Human-routed question → ESCALATE → open question
        assert len(result.open_questions) == 1
        assert "q-2-1" in result.open_questions[0]
        assert len(result.resolved_questions) == 1

    def test_malformed_question_does_not_become_open_question_first_pass(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        # A question that fails validation gets REFORMULATE (not
        # ESCALATE). Open-question accumulation only happens on
        # ESCALATE/FAILED.
        bull = _FakeAgent("bull", "bull", [_BAD_QUESTION])
        bear = _FakeAgent("bear", "bear", ["No questions"])
        cfg = _build_config(
            rules_file,
            [RoundConfig(
                name="smart_questions", type=RoundType.PER_AGENT_ASYNC,
            )],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            transcript_root=transcript_root,
        )
        result = orch.run_debate(
            strategy_slug="strat-004",
            candidate_report_text="metrics",
        )
        # First reformulate doesn't escalate yet
        assert result.open_questions == []
        # But the resolution outcome is recorded
        assert len(result.resolved_questions) == 1
        from src.research.agents.resolver import ResolutionStatus
        assert (
            result.resolved_questions[0].result.status
            == ResolutionStatus.REFORMULATE
        )

    def test_other_agent_routing_calls_target_agent(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        # Bull asks bear via other_agent routing.
        question = _GOOD_QUESTION.replace(
            "routing: human", "routing: other_agent",
        )
        bull = _FakeAgent("bull", "bull", [question])
        # Iteration order: bull's emit (1 bull call) → bull's question
        # dispatched to bear via other_agent (1st bear call) → bear's
        # own emit (2nd bear call). Order the bear cans accordingly.
        bear = _FakeAgent(
            "bear", "bear",
            [
                "Bear's answer to bull's question.",  # dispatch result
                "Bear's emit: no questions.",         # bear's own emit
            ],
        )
        cfg = _build_config(
            rules_file,
            [RoundConfig(
                name="smart_questions", type=RoundType.PER_AGENT_ASYNC,
            )],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            transcript_root=transcript_root,
        )
        result = orch.run_debate(
            strategy_slug="strat-005",
            candidate_report_text="metrics",
        )
        # No open questions — the other agent answered
        assert result.open_questions == []
        assert len(result.resolved_questions) == 1
        from src.research.agents.resolver import ResolutionStatus
        assert (
            result.resolved_questions[0].result.status
            == ResolutionStatus.RESOLVED
        )
        # Bear's answer should be in the resolution
        assert "Bear's answer" in result.resolved_questions[0].result.answer

    def test_external_resolver_dispatch_restored_after_round(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        # The orchestrator monkeypatches resolver.agent_dispatch during
        # the per_agent_async round. It must restore the original after.
        def sentinel(target: str, _question: str) -> str:
            return f"sentinel:{target}"

        resolver = QuestionResolver(agent_dispatch=sentinel)
        bull = _FakeAgent("bull", "bull", ["No questions"])
        bear = _FakeAgent("bear", "bear", ["No questions"])
        cfg = _build_config(
            rules_file,
            [RoundConfig(
                name="smart_questions", type=RoundType.PER_AGENT_ASYNC,
            )],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            resolver=resolver,
            transcript_root=transcript_root,
        )
        orch.run_debate(strategy_slug="strat-006", candidate_report_text="m")
        assert resolver.agent_dispatch is sentinel


# ---------------------------------------------------------------------------
# Transcript persistence
# ---------------------------------------------------------------------------


class TestTranscriptPersistence:
    def test_writes_md_and_jsonl_files(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        bull = _FakeAgent("bull", "bull", ["**FINAL_POSITION**: PROMOTE"])
        bear = _FakeAgent("bear", "bear", ["**FINAL_POSITION**: REJECT"])
        cfg = _build_config(
            rules_file,
            [RoundConfig(name="r1", type=RoundType.PARALLEL)],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            transcript_root=transcript_root,
        )
        result = orch.run_debate(
            strategy_slug="strat-007",
            candidate_report_text="metrics",
        )
        debate_dir = transcript_root / "strat-007"
        md = debate_dir / "transcript.md"
        jsonl = debate_dir / "transcript.jsonl"
        assert md.exists() and jsonl.exists()
        assert result.transcript_path == md

        md_text = md.read_text()
        assert "Debate transcript — promotion_review — strat-007" in md_text
        assert "**FINAL_POSITION**: PROMOTE" in md_text
        assert "**FINAL_POSITION**: REJECT" in md_text
        # Summary footer
        assert "**bull**: PROMOTE" in md_text
        assert "**bear**: REJECT" in md_text

        jsonl_lines = jsonl.read_text().strip().split("\n")
        assert len(jsonl_lines) == 2
        for line in jsonl_lines:
            row = json.loads(line)
            assert row["round_name"] == "r1"
            assert row["agent_name"] in ("bull", "bear")
            assert "content" in row
            assert "usd_cost" in row

    def test_open_questions_appear_in_summary_footer(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        bull = _FakeAgent("bull", "bull", [_GOOD_QUESTION])
        bear = _FakeAgent("bear", "bear", ["No questions"])
        cfg = _build_config(
            rules_file,
            [RoundConfig(
                name="smart_questions", type=RoundType.PER_AGENT_ASYNC,
            )],
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            transcript_root=transcript_root,
        )
        result = orch.run_debate(
            strategy_slug="strat-008", candidate_report_text="m",
        )
        md_text = result.transcript_path.read_text()
        assert "Open questions" in md_text
        assert "q-2-1" in md_text


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


class TestCaps:
    def test_extra_rebuttal_rounds_skipped_above_cap(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        # Build a debate with one parallel + (cap+2) rebuttal rounds.
        rounds = [RoundConfig(name="initial", type=RoundType.PARALLEL)]
        rounds.extend(
            RoundConfig(name="rebuttal", type=RoundType.SEQUENTIAL)
            for _ in range(MAX_REBUTTAL_ROUNDS + 2)
        )
        bull = _FakeAgent("bull", "bull", ["B"])
        bear = _FakeAgent("bear", "bear", ["B"])
        cfg = _build_config(rules_file, rounds)
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({"bull": bull, "bear": bear}),
            transcript_root=transcript_root,
        )
        result = orch.run_debate(
            strategy_slug="strat-009", candidate_report_text="m",
        )
        # Initial round + capped rebuttal rounds = 1 + MAX_REBUTTAL_ROUNDS
        # Each round produces 2 entries (one per participant).
        expected_rounds = 1 + MAX_REBUTTAL_ROUNDS
        expected_entries = expected_rounds * 2
        assert len(result.transcript_entries) == expected_entries


# ---------------------------------------------------------------------------
# Final position extraction
# ---------------------------------------------------------------------------


class TestFinalPositions:
    def test_only_reviewer_roles_get_positions(
        self, rules_file: Path, transcript_root: Path,
    ) -> None:
        # Add a non-reviewer participant (role='resolver'). It must not
        # appear in final_positions.
        bull = _FakeAgent("bull", "bull", ["**FINAL_POSITION**: PROMOTE"])
        bear = _FakeAgent("bear", "bear", ["**FINAL_POSITION**: REJECT"])
        helper = _FakeAgent("resolver_agent", "resolver", ["I help"])
        cfg = _build_config(
            rules_file,
            [RoundConfig(name="r1", type=RoundType.PARALLEL)],
            participants=("bull", "bear", "resolver_agent"),
        )
        orch = DebateOrchestrator(
            research_config=cfg, debate_name="promotion_review",
            agent_factory=_make_factory({
                "bull": bull, "bear": bear, "resolver_agent": helper,
            }),
            transcript_root=transcript_root,
        )
        result = orch.run_debate(
            strategy_slug="strat-010", candidate_report_text="m",
        )
        assert set(result.final_positions.keys()) == {"bull", "bear"}
        assert result.final_positions["bull"] == Position.PROMOTE
        assert result.final_positions["bear"] == Position.REJECT


# ---------------------------------------------------------------------------
# Context-trimming helpers (token budget for tier-1 rate limits)
# ---------------------------------------------------------------------------


class TestSlimCandidateReport:
    def test_drops_bulk_keys(self) -> None:
        big = json.dumps({
            "schema_version": 1,
            "strategy_slug": "x",
            "oos_metrics": {"sharpe": 0.7, "n_trades": 50},
            "sharpe_ci_95": {"low": 0.2, "high": 1.2},
            "edge_concentration": 0.4,
            "regime_diversified": True,
            "decay_severity": "NONE",
            # Bulk to drop:
            "backtest_metrics": {"verbose": "x" * 500},
            "fold_metrics": [{"fold_id": i} for i in range(20)],
            "_metrics_provenance": {"all": "x" * 200},
            "generated_at": "2026-01-01",
            "hypothesis_path": "y",
            "provenance": {"agent": "z" * 100},
        })
        slim = _slim_candidate_report(big)
        assert "backtest_metrics" not in slim
        assert "fold_metrics" not in slim
        assert "_metrics_provenance" not in slim
        assert "generated_at" not in slim
        assert "provenance" not in slim
        # Verdict-relevant keys preserved
        assert "oos_metrics" in slim
        assert "sharpe_ci_95" in slim
        assert "edge_concentration" in slim
        assert "regime_diversified" in slim
        assert "decay_severity" in slim
        assert "strategy_slug" in slim
        # Materially smaller
        assert len(slim) < len(big) * 0.5

    def test_passes_through_non_json(self) -> None:
        text = "this is just markdown, not JSON"
        assert _slim_candidate_report(text) == text


class TestSlimTranscript:
    def test_empty_returns_empty(self) -> None:
        assert _slim_transcript("") == ""

    def test_compresses_long_blocks(self) -> None:
        # A real-shape transcript block from the orchestrator's writer
        long_block = (
            "## Round: rebuttal — agent: bull_reviewer (bull)\n"
            "*ts=2026-04-30T10:30:00 model=claude-sonnet-4-6 "
            "in/out_tokens=4000/2000 cost=$0.0420 elapsed=3.10s*\n\n"
            + "This is a very verbose Bull rebuttal " * 200 + "\n"
            + "**FINAL_POSITION**: PROMOTE\n"
        )
        slim = _slim_transcript(long_block + "\n---\n" + long_block)
        # Each block summarized to a header + position + capped digest
        assert "FINAL_POSITION" in slim
        assert "PROMOTE" in slim
        # Materially shorter than the input
        assert len(slim) < len(long_block) * 0.5
