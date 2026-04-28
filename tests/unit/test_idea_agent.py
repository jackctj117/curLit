"""Unit tests for the Idea Generator agent (CL-n0m8).

Mocks the LLM with a canned-response driver. No live calls.

Covers:
  * parse_position extracts PROPOSED / DECLINED, defaults DECLINED
  * default_slug derives a kebab-case slug from extract H1 title
  * validate_brief flags missing sections + missing prediction fields
    + missing extract reference
  * ideate() writes hypothesis md when PROPOSED
  * ideate() returns DECLINED early on DECLINED keyword
  * ideate() returns DECLINED when validation fails
  * ideate() raises FileNotFoundError on missing extract
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.research.agents.idea import (
    IdeaGenerator,
    IdeaStatus,
    default_slug,
    parse_position,
    validate_brief,
)
from src.research.config import (
    AgentConfig,
    ProviderConfig,
    ResearchConfig,
)
from src.research.llm.client import (
    Driver,
    LLMResponse,
    register_driver,
)


class _CannedDriver(Driver):
    name = "canned-idea"

    def __init__(self, api_key: str = "x", canned_text: str = "stub") -> None:
        super().__init__(api_key)
        self.canned_text = canned_text
        self.calls: list[Any] = []

    def complete(
        self,
        messages: Any,
        model: str,
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            text=self.canned_text, model=model, provider=self.name,
            input_tokens=10, output_tokens=20, usd_cost=0.0001, elapsed_sec=0.01,
        )


register_driver("canned-idea", _CannedDriver)


# Fixtures ----------------------------------------------------------------


@pytest.fixture
def idea_prompt(tmp_path: Path) -> Path:
    f = tmp_path / "idea_prompt.md"
    f.write_text("You are a stub idea generator for tests.")
    return f


@pytest.fixture
def extract_file(tmp_path: Path) -> Path:
    f = tmp_path / "extracts" / "abc123.md"
    f.parent.mkdir(parents=True)
    f.write_text(
        "# Carry-trade returns under regime switches\n\n"
        "- **authors**: Alice Doe, Bob Roe\n"
        "- **year**: 2026\n"
        "- **paper_hash**: `abc123`\n\n"
        "---\n\n"
        "## Methodology\nWe study carry-trade Sharpe across regimes.\n"
        "## Findings\nRegime-conditional carry persists in EM.\n",
    )
    return f


@pytest.fixture
def cfg(idea_prompt: Path, monkeypatch: pytest.MonkeyPatch) -> ResearchConfig:
    monkeypatch.setenv("CANNED_IDEA_KEY", "fake")
    return ResearchConfig(
        providers={
            "canned-idea": ProviderConfig(
                api_key_env="CANNED_IDEA_KEY", default_model="m1",
            ),
        },
        agents={
            "idea_generator": AgentConfig(
                provider="canned-idea",
                role="idea_generator",
                prompt_path=str(idea_prompt),
                model=None,
            ),
        },
        debates={},  # type: ignore[arg-type]
    )


def _make_idea(cfg: ResearchConfig, canned: str) -> IdeaGenerator:
    a = IdeaGenerator.from_config(name="idea_generator", research_config=cfg)
    a.client.driver.canned_text = canned  # type: ignore[attr-defined]
    return a


# --------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------- #


class TestParsePosition:
    def test_proposed(self) -> None:
        assert (
            parse_position("**FINAL_POSITION**: PROPOSED") == IdeaStatus.PROPOSED
        )

    def test_declined(self) -> None:
        assert (
            parse_position("**FINAL_POSITION**: DECLINED") == IdeaStatus.DECLINED
        )

    def test_no_keyword_defaults_declined(self) -> None:
        # Caution-default — never PROPOSED without an explicit keyword.
        assert parse_position("ambiguous text") == IdeaStatus.DECLINED

    def test_last_match_wins(self) -> None:
        text = (
            "Initial draft might be PROPOSED.\n\n"
            "**FINAL_POSITION**: DECLINED"
        )
        assert parse_position(text) == IdeaStatus.DECLINED


class TestDefaultSlug:
    def test_derives_from_h1(self, tmp_path: Path) -> None:
        f = tmp_path / "x.md"
        f.write_text("# Carry-Trade Returns Under Regime Switches\n\nbody")
        slug = default_slug(f)
        assert slug == "carry-trade-returns-under-regime-switches"

    def test_falls_back_to_filename_when_no_h1(self, tmp_path: Path) -> None:
        f = tmp_path / "abc123.md"
        f.write_text("no heading here\n")
        slug = default_slug(f)
        assert slug == "idea-abc123"

    def test_caps_length(self, tmp_path: Path) -> None:
        long_title = "a " * 200
        f = tmp_path / "x.md"
        f.write_text(f"# {long_title}\n")
        slug = default_slug(f)
        assert len(slug) <= 64

    def test_missing_file_falls_back(self, tmp_path: Path) -> None:
        slug = default_slug(tmp_path / "nope.md")
        assert slug.startswith("idea-")


class TestValidateBrief:
    def test_valid_brief_returns_no_issues(self, tmp_path: Path) -> None:
        extract = tmp_path / "extracts" / "abc.md"
        extract.parent.mkdir(parents=True)
        extract.write_text("# stub\n")
        text = _valid_brief_text(extract)
        assert validate_brief(text, extract) == []

    def test_missing_section_flagged(self, tmp_path: Path) -> None:
        extract = tmp_path / "extracts" / "abc.md"
        extract.parent.mkdir(parents=True)
        extract.write_text("# stub\n")
        # Drop the "Abandon condition" section
        text = _valid_brief_text(extract).replace(
            "## Abandon condition\n",
            "## Something else\n",
        )
        issues = validate_brief(text, extract)
        assert any("Abandon condition" in i for i in issues)

    def test_missing_prediction_field_flagged(self, tmp_path: Path) -> None:
        extract = tmp_path / "extracts" / "abc.md"
        extract.parent.mkdir(parents=True)
        extract.write_text("# stub\n")
        text = _valid_brief_text(extract).replace(
            "predicted_sharpe_range", "sharp_range",
        )
        issues = validate_brief(text, extract)
        assert any("predicted_sharpe_range" in i for i in issues)

    def test_missing_extract_reference_flagged(self, tmp_path: Path) -> None:
        extract = tmp_path / "extracts" / "abc.md"
        extract.parent.mkdir(parents=True)
        extract.write_text("# stub\n")
        # Replace the references section with one that doesn't cite it
        text = _valid_brief_text(extract).replace(
            f"- {extract}", "- some other reference"
        ).replace(extract.name, "different.md")
        issues = validate_brief(text, extract)
        assert any("does not cite the source extract" in i for i in issues)


def _valid_brief_text(extract_path: Path) -> str:
    """Compose a syntactically-valid brief for the validator tests."""
    return (
        "# Hypothesis: regime-conditional carry\n\n"
        "## Source extract\n"
        f"- **path**: `{extract_path}`\n"
        "- **paper title**: Carry-trade returns\n"
        "- **paper hash**: `abc123`\n\n"
        "## Change to baseline\n"
        "Replaces the existing carry strategy with regime-conditional gating.\n\n"
        "## Prediction\n"
        "- `predicted_sharpe_range`: [0.4, 0.8]\n"
        "- `predicted_hit_rate_range`: [0.55, 0.62]\n"
        "- `expected_n_trades_per_year`: 60\n"
        "- `regime_dependence`: regime-agnostic\n"
        "- `time_to_signal`: daily\n\n"
        "## Abandon condition\n"
        "- OOS Sharpe < 0 over any rolling 6-month window\n\n"
        "## Data requirements\n"
        "- FRED IRLTLT01DEM156N\n\n"
        "## References\n"
        f"- {extract_path}\n\n"
        "**FINAL_POSITION**: PROPOSED\n"
    )


# --------------------------------------------------------------------- #
# ideate() integration
# --------------------------------------------------------------------- #


class TestIdeateIntegration:
    def test_proposed_writes_hypothesis_file(
        self,
        cfg: ResearchConfig,
        extract_file: Path,
        tmp_path: Path,
    ) -> None:
        canned = _valid_brief_text(extract_file)
        idea = _make_idea(cfg, canned)
        result = idea.ideate(
            extract_path=extract_file,
            strategy_slug="test-hyp",
            hypothesis_dir=tmp_path / "hypotheses",
        )
        assert result.status == IdeaStatus.PROPOSED
        assert result.hypothesis_path is not None
        assert result.hypothesis_path.exists()
        # File contents = the LLM brief verbatim
        assert result.hypothesis_path.read_text() == canned

    def test_declined_short_circuits_no_file(
        self,
        cfg: ResearchConfig,
        extract_file: Path,
        tmp_path: Path,
    ) -> None:
        canned = (
            "Cannot derive a falsifiable thesis from this extract.\n\n"
            "**FINAL_POSITION**: DECLINED\n\n"
            "The extract is too thin — full text required."
        )
        idea = _make_idea(cfg, canned)
        result = idea.ideate(
            extract_path=extract_file,
            strategy_slug="declined-hyp",
            hypothesis_dir=tmp_path / "hypotheses",
        )
        assert result.status == IdeaStatus.DECLINED
        assert result.hypothesis_path is None
        assert "too thin" in result.reason or "full text" in result.reason

    def test_validation_failure_returns_declined(
        self,
        cfg: ResearchConfig,
        extract_file: Path,
        tmp_path: Path,
    ) -> None:
        # Position says PROPOSED, but the brief is missing required
        # sections. Should be downgraded to DECLINED with issues listed.
        canned = (
            "# Hypothesis: incomplete\n\n"
            "## Source extract\nblah\n\n"
            "**FINAL_POSITION**: PROPOSED\n"
        )
        idea = _make_idea(cfg, canned)
        result = idea.ideate(
            extract_path=extract_file,
            strategy_slug="incomplete",
            hypothesis_dir=tmp_path / "hypotheses",
        )
        assert result.status == IdeaStatus.DECLINED
        assert result.hypothesis_path is None
        assert result.validation_issues
        assert any(
            "Change to baseline" in i for i in result.validation_issues
        )

    def test_default_slug_used_when_none_passed(
        self,
        cfg: ResearchConfig,
        extract_file: Path,
        tmp_path: Path,
    ) -> None:
        # Extract H1 = "Carry-trade returns under regime switches"
        canned = _valid_brief_text(extract_file)
        idea = _make_idea(cfg, canned)
        result = idea.ideate(
            extract_path=extract_file,
            hypothesis_dir=tmp_path / "hypotheses",
        )
        assert result.status == IdeaStatus.PROPOSED
        assert result.strategy_slug.startswith("carry-trade-returns")

    def test_backlog_summary_passed_through(
        self,
        cfg: ResearchConfig,
        extract_file: Path,
        tmp_path: Path,
    ) -> None:
        canned = _valid_brief_text(extract_file)
        idea = _make_idea(cfg, canned)
        idea.ideate(
            extract_path=extract_file,
            strategy_slug="t",
            backlog_summary="existing: foo, bar",
            hypothesis_dir=tmp_path / "hypotheses",
        )
        # The driver recorded the prompt — backlog_summary should be in
        # the user message
        msgs = idea.client.driver.calls[0]  # type: ignore[attr-defined]
        user_msg = msgs[1].content
        assert "hypothesis_backlog_summary" in user_msg
        assert "existing: foo, bar" in user_msg

    def test_missing_extract_raises(
        self,
        cfg: ResearchConfig,
        tmp_path: Path,
    ) -> None:
        idea = _make_idea(cfg, "anything")
        with pytest.raises(FileNotFoundError, match="extract not found"):
            idea.ideate(
                extract_path=tmp_path / "nope.md",
                strategy_slug="ghost",
                hypothesis_dir=tmp_path / "hypotheses",
            )


# --------------------------------------------------------------------- #
# Real prompt file
# --------------------------------------------------------------------- #


class TestRealPromptFile:
    def test_idea_generator_prompt_exists(self) -> None:
        path = Path("configs/research_prompts/idea_generator.md")
        assert path.exists()
        text = path.read_text()
        # Doctrine reference
        assert "EVIDENCE_FIRST.md" in text
        # Required sections named so the LLM emits them
        for section in (
            "Source extract",
            "Change to baseline",
            "Prediction",
            "Abandon condition",
            "Data requirements",
            "References",
        ):
            assert section in text, f"prompt missing section: {section!r}"
        # Position keywords
        assert "PROPOSED" in text and "DECLINED" in text
