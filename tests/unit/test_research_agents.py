"""Unit tests for the agent base class + reviewers (CL-w2b0, CL-a411).

Mocks the LLM so no live API calls. Validates:
  * Agent.from_config wires provider + model + prompt correctly
  * run() composes context-file blocks with XML delimiters
  * extra_system appends correctly
  * Reviewer.review wraps run() and parses the position keyword
  * parse_position handles the formats the prompt asks for
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.research.agents.base import Agent, AgentResponse
from src.research.agents.reviewer import (
    BearReviewer,
    BullReviewer,
    Position,
    Reviewer,
    parse_position,
)
from src.research.config import ResearchConfig
from src.research.llm.client import (
    Driver,
    LLMResponse,
    Message,
    register_driver,
)

# =============================================================================
# Mock LLM driver — returns canned text + records calls
# =============================================================================


class _RecordingDriver(Driver):
    """Mock that returns a configurable canned response and records the
    incoming messages for assertions."""

    name = "recording"

    def __init__(self, api_key: str = "fake", canned_text: str = "stub") -> None:
        super().__init__(api_key)
        self.canned_text = canned_text
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, model, max_tokens=4096, temperature=0.0, **kwargs):  # noqa: ANN001
        self.calls.append({
            "messages": [Message(role=m.role, content=m.content) for m in messages],
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return LLMResponse(
            text=self.canned_text, model=model, provider=self.name,
            input_tokens=10, output_tokens=20, usd_cost=0.001, elapsed_sec=0.01,
        )


# Register the recording driver once at import time so config validation
# (which checks provider names against _DRIVERS) accepts "recording".
register_driver("recording", _RecordingDriver)


@pytest.fixture
def tmp_rules(tmp_path: Path) -> Path:
    f = tmp_path / "REVIEW_RULES.md"
    f.write_text("# stub rules")
    return f


@pytest.fixture
def tmp_prompt(tmp_path: Path) -> Path:
    f = tmp_path / "agent_prompt.md"
    f.write_text("You are a test agent. Be brief.")
    return f


@pytest.fixture
def base_config_dict(tmp_rules: Path, tmp_prompt: Path) -> dict[str, Any]:
    return {
        "providers": {
            "recording": {
                "api_key_env": "RECORDING_KEY",
                "default_model": "rec-1",
            },
        },
        "agents": {
            "test_agent": {
                "provider": "recording",
                "role": "test",
                "prompt_path": str(tmp_prompt),
                "max_tokens": 1024,
                "temperature": 0.0,
            },
        },
        "debates": {
            "review": {
                "participants": ["test_agent"],
                "rules_path": str(tmp_rules),
                "rounds": [{"name": "r1", "type": "parallel"}],
            },
        },
    }


@pytest.fixture
def configured_agent(
    base_config_dict: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Agent:
    monkeypatch.setenv("RECORDING_KEY", "fake-key")
    cfg = ResearchConfig(**base_config_dict)
    return Agent.from_config(name="test_agent", research_config=cfg)


# =============================================================================
# Agent.from_config
# =============================================================================


class TestAgentConstruction:
    def test_from_config_loads_prompt_from_disk(
        self, configured_agent: Agent, tmp_prompt: Path,
    ) -> None:
        assert configured_agent.system_prompt == tmp_prompt.read_text()

    def test_from_config_resolves_model_via_provider_default(
        self, configured_agent: Agent,
    ) -> None:
        # No explicit model on the agent → falls back to provider default
        assert configured_agent.model == "rec-1"
        assert configured_agent.client.provider == "recording"

    def test_unknown_agent_raises(self, base_config_dict: dict[str, Any]) -> None:
        cfg = ResearchConfig(**base_config_dict)
        with pytest.raises(KeyError, match="not declared in research config"):
            Agent.from_config(name="ghost", research_config=cfg)

    def test_missing_prompt_file_raises(
        self, base_config_dict: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RECORDING_KEY", "fake-key")
        # Replace prompt path with a non-existent one — but config schema
        # doesn't validate prompt_path existence (intentionally), so we
        # construct then fail at from_config
        base_config_dict["agents"]["test_agent"]["prompt_path"] = "/nope.md"
        cfg = ResearchConfig(**base_config_dict)
        with pytest.raises(FileNotFoundError, match="system prompt not found"):
            Agent.from_config(name="test_agent", research_config=cfg)


# =============================================================================
# Prompt composition
# =============================================================================


class TestPromptComposition:
    def test_run_passes_system_and_user_to_driver(
        self, configured_agent: Agent,
    ) -> None:
        configured_agent.run(user_prompt="hello world")
        driver = configured_agent.client.driver
        assert isinstance(driver, _RecordingDriver)
        msgs = driver.calls[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"
        assert "hello world" in msgs[1].content

    def test_context_files_injected_with_xml_delimiters(
        self, configured_agent: Agent,
    ) -> None:
        configured_agent.run(
            user_prompt="answer below",
            context_files={"rules": "Rule A.1: foo", "report": "metrics: 0.5"},
        )
        driver = configured_agent.client.driver
        assert isinstance(driver, _RecordingDriver)
        user_text = driver.calls[0]["messages"][1].content
        assert "<context label='rules'>" in user_text
        assert "Rule A.1: foo" in user_text
        assert "<context label='report'>" in user_text
        # User prompt comes after the context blocks
        assert user_text.index("answer below") > user_text.index("Rule A.1")

    def test_extra_system_appended_to_base(
        self, configured_agent: Agent,
    ) -> None:
        configured_agent.run(
            user_prompt="x",
            extra_system="ROUND 1 INSTRUCTIONS",
        )
        driver = configured_agent.client.driver
        assert isinstance(driver, _RecordingDriver)
        sys_text = driver.calls[0]["messages"][0].content
        assert "You are a test agent" in sys_text
        assert "ROUND 1 INSTRUCTIONS" in sys_text


# =============================================================================
# parse_position
# =============================================================================


class TestParsePosition:
    def test_explicit_final_position_keyword(self) -> None:
        text = "Some preamble\n\n**FINAL_POSITION**: PROMOTE\n\nrationale"
        assert parse_position(text) == Position.PROMOTE

    def test_unbolded_final_position(self) -> None:
        text = "FINAL_POSITION: REJECT"
        assert parse_position(text) == Position.REJECT

    def test_bare_bold_keyword_at_end(self) -> None:
        text = "section 1 ...\nsection 2 ...\n\n**ABSTAIN**\n"
        assert parse_position(text) == Position.ABSTAIN

    def test_last_position_wins_when_multiple(self) -> None:
        # An earlier exploratory mention shouldn't override the final.
        text = (
            "Initial draft: maybe **PROMOTE**.\n\n"
            "Round 3 rebuttal led me to update.\n\n"
            "**FINAL_POSITION**: REJECT"
        )
        assert parse_position(text) == Position.REJECT

    def test_no_keyword_defaults_to_abstain(self) -> None:
        text = "I am uncertain about everything."
        assert parse_position(text) == Position.ABSTAIN

    def test_lowercase_keyword_normalizes(self) -> None:
        text = "**FINAL_POSITION**: promote"
        assert parse_position(text) == Position.PROMOTE


# =============================================================================
# Reviewer subclasses
# =============================================================================


@pytest.fixture
def reviewer_config(
    base_config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> ResearchConfig:
    monkeypatch.setenv("RECORDING_KEY", "fake-key")
    return ResearchConfig(**base_config_dict)


class TestReviewer:
    def test_bull_returns_promote_when_canned_says_so(
        self, reviewer_config: ResearchConfig,
    ) -> None:
        # Construct the agent with a canned response that ends in PROMOTE.
        agent = BullReviewer.from_config(
            name="test_agent", research_config=reviewer_config,
        )
        # Swap in a driver with the canned response we want
        agent.client.driver.canned_text = (  # type: ignore[union-attr]
            "## Section A\nA.1: PASS — Sharpe=0.7 from report.\n\n"
            "**FINAL_POSITION**: PROMOTE"
        )
        result = agent.review(
            candidate_report="{stub}",
            review_rules="# rules",
            round_name="initial_positions",
        )
        assert result.position == Position.PROMOTE
        assert "Sharpe=0.7" in result.raw_text

    def test_bear_returns_reject_when_canned_says_so(
        self, reviewer_config: ResearchConfig,
    ) -> None:
        agent = BearReviewer.from_config(
            name="test_agent", research_config=reviewer_config,
        )
        agent.client.driver.canned_text = (  # type: ignore[union-attr]
            "C.1: FAIL — found ts > current_eval_ts at line 142.\n\n"
            "**FINAL_POSITION**: REJECT"
        )
        result = agent.review(
            candidate_report="{stub}",
            review_rules="# rules",
            round_name="initial_positions",
        )
        assert result.position == Position.REJECT

    def test_review_passes_round_instruction_to_system_prompt(
        self, reviewer_config: ResearchConfig,
    ) -> None:
        agent = BullReviewer.from_config(
            name="test_agent", research_config=reviewer_config,
        )
        agent.client.driver.canned_text = "**FINAL_POSITION**: ABSTAIN"  # type: ignore[union-attr]
        agent.review(
            candidate_report="{}", review_rules="# r",
            round_name="rebuttal",
        )
        sys_text = agent.client.driver.calls[0]["messages"][0].content  # type: ignore[union-attr]
        assert "Round 3 — rebuttal" in sys_text

    def test_review_passes_prior_transcript_when_given(
        self, reviewer_config: ResearchConfig,
    ) -> None:
        agent = BullReviewer.from_config(
            name="test_agent", research_config=reviewer_config,
        )
        agent.client.driver.canned_text = "**FINAL_POSITION**: PROMOTE"  # type: ignore[union-attr]
        agent.review(
            candidate_report="{}", review_rules="# r",
            round_name="rebuttal",
            prior_transcript="Bear said: C.1 FAIL line 142",
        )
        user_text = agent.client.driver.calls[0]["messages"][1].content  # type: ignore[union-attr]
        assert "Bear said: C.1 FAIL line 142" in user_text
        assert "<context label='debate_transcript_so_far'>" in user_text

    def test_review_returns_agent_response_with_token_counts(
        self, reviewer_config: ResearchConfig,
    ) -> None:
        agent = BullReviewer.from_config(
            name="test_agent", research_config=reviewer_config,
        )
        agent.client.driver.canned_text = "**FINAL_POSITION**: PROMOTE"  # type: ignore[union-attr]
        result = agent.review(
            candidate_report="{}", review_rules="# r",
            round_name="initial_positions",
        )
        assert isinstance(result.response, AgentResponse)
        assert result.response.input_tokens == 10
        assert result.response.output_tokens == 20
        assert result.response.usd_cost == pytest.approx(0.001)


class TestRoleSpecificity:
    def test_bull_and_bear_are_distinct_subclasses(self) -> None:
        # Sanity: they are different classes so they can be type-checked
        # independently in the orchestrator (e.g. routing).
        assert BullReviewer is not BearReviewer
        assert issubclass(BullReviewer, Reviewer)
        assert issubclass(BearReviewer, Reviewer)


class TestRealPromptFiles:
    """The committed system-prompt files at configs/research_prompts/
    must exist and be loadable. This guards against the orchestrator
    finding a missing prompt at runtime."""

    def test_bull_prompt_loads(self) -> None:
        path = Path("configs/research_prompts/bull_reviewer.md")
        assert path.exists(), "bull prompt missing"
        text = path.read_text()
        assert "Bull Reviewer" in text
        assert "PROMOTE_CASE" in text
        assert "FINAL_POSITION" in text

    def test_bear_prompt_loads(self) -> None:
        path = Path("configs/research_prompts/bear_reviewer.md")
        assert path.exists(), "bear prompt missing"
        text = path.read_text()
        assert "Bear Reviewer" in text
        assert "REJECT_CASE" in text
        assert "FINAL_POSITION" in text

    def test_bear_prompt_is_adversarial(self) -> None:
        """The asymmetric Bear prompt is the safeguard against two-LLM
        convergence — verify it explicitly tells Bear to be adversarial."""
        text = Path("configs/research_prompts/bear_reviewer.md").read_text()
        assert "adversarial" in text.lower()
        # Hunting list should include the failure-mode keywords
        for keyword in ("look-ahead", "overfitting", "regime concentration"):
            assert keyword.lower() in text.lower(), f"missing {keyword}"

    def test_evidence_first_principle_loaded_in_both_prompts(self) -> None:
        """EVIDENCE_FIRST.md is the operating principle — both reviewer
        prompts must reference it so agents inherit the discipline."""
        for prompt_file in (
            "configs/research_prompts/bull_reviewer.md",
            "configs/research_prompts/bear_reviewer.md",
        ):
            text = Path(prompt_file).read_text()
            assert "EVIDENCE_FIRST.md" in text, (
                f"{prompt_file} doesn't reference EVIDENCE_FIRST.md"
            )

    def test_evidence_first_doc_exists_with_principle(self) -> None:
        """The doctrine itself — must contain the operator's standing
        guidance about pulling data before risking money."""
        path = Path("docs/research/EVIDENCE_FIRST.md")
        assert path.exists(), "EVIDENCE_FIRST.md missing"
        text = path.read_text()
        # Key concept markers — vibes-not-edge framing + data-first action
        assert "trading a vibe" in text.lower()
        assert "pull the data first" in text.lower()
        # Examples of public data sources that should be in the canon
        for example in ("Dubai Land Department", "CBUAE", "family-office"):
            assert example.lower() in text.lower(), f"missing example {example!r}"
