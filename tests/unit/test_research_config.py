"""Unit tests for the research-pipeline config schema (CL-rzf7).

Validates that the YAML loader catches the common error classes early
so misconfiguration fails loudly at boot rather than mid-loop:
  * unknown provider name (typo / missing register_driver)
  * agent referencing undeclared provider
  * debate referencing undeclared agent
  * missing rules_path file
  * default model resolution
  * env-var key resolution
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.research.config import (
    ResearchConfig,
    RoundType,
    load_config,
)


@pytest.fixture
def tmp_rules(tmp_path: Path) -> Path:
    """Create a stub rules file the DebateConfig validator can find."""
    f = tmp_path / "REVIEW_RULES.md"
    f.write_text("# stub rules")
    return f


@pytest.fixture
def valid_config_dict(tmp_rules: Path) -> dict:
    """Minimal valid config covering all major surfaces."""
    return {
        "providers": {
            "claude": {
                "api_key_env": "ANTHROPIC_API_KEY",
                "default_model": "claude-opus-4-7",
            },
            "deepseek": {
                "api_key_env": "DEEPSEEK_API_KEY",
                "default_model": "deepseek-chat",
            },
        },
        "agents": {
            "bull": {
                "provider": "claude",
                "role": "bull",
                "prompt_path": "configs/research_prompts/bull.md",
            },
            "bear": {
                "provider": "claude",
                "role": "bear",
                "prompt_path": "configs/research_prompts/bear.md",
                "model": "claude-sonnet-4-6",  # explicit override
            },
            "idea": {
                "provider": "deepseek",
                "role": "idea",
                "prompt_path": "configs/research_prompts/idea.md",
            },
        },
        "debates": {
            "promotion_review": {
                "participants": ["bull", "bear"],
                "rules_path": str(tmp_rules),
                "rounds": [
                    {"name": "round1", "type": "parallel"},
                    {"name": "round2", "type": "per_agent_async", "sub_loop": "list_unknowns"},
                    {"name": "round3", "type": "sequential"},
                    {"name": "round4", "type": "parallel"},
                ],
            },
        },
    }


# =============================================================================
# Happy path
# =============================================================================


class TestValidConfig:
    def test_load_valid_config(self, valid_config_dict: dict) -> None:
        cfg = ResearchConfig(**valid_config_dict)
        assert set(cfg.providers) == {"claude", "deepseek"}
        assert set(cfg.agents) == {"bull", "bear", "idea"}
        assert "promotion_review" in cfg.debates

    def test_round_types_parsed(self, valid_config_dict: dict) -> None:
        cfg = ResearchConfig(**valid_config_dict)
        rounds = cfg.debates["promotion_review"].rounds
        assert rounds[0].type == RoundType.PARALLEL
        assert rounds[1].type == RoundType.PER_AGENT_ASYNC
        assert rounds[2].type == RoundType.SEQUENTIAL
        assert rounds[3].type == RoundType.PARALLEL

    def test_resolve_model_uses_default_when_unset(
        self,
        valid_config_dict: dict,
    ) -> None:
        cfg = ResearchConfig(**valid_config_dict)
        # bull has no explicit model → falls back to claude provider's default
        assert cfg.resolve_model("bull") == "claude-opus-4-7"
        # bear has explicit model → that value
        assert cfg.resolve_model("bear") == "claude-sonnet-4-6"
        # idea uses deepseek default
        assert cfg.resolve_model("idea") == "deepseek-chat"

    def test_resolve_api_key_reads_env(
        self,
        valid_config_dict: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        cfg = ResearchConfig(**valid_config_dict)
        assert cfg.resolve_api_key("bull") == "test-key-123"
        assert cfg.resolve_api_key("idea") == ""  # not set


# =============================================================================
# Cross-validation: unknown references fail loud
# =============================================================================


class TestCrossValidation:
    def test_unknown_provider_fails(self, valid_config_dict: dict) -> None:
        valid_config_dict["providers"]["mystery"] = {
            "api_key_env": "MYSTERY_KEY",
            "default_model": "mystery-1",
        }
        with pytest.raises(ValueError, match="not in registered LLM drivers"):
            ResearchConfig(**valid_config_dict)

    def test_agent_referencing_unknown_provider_fails(
        self,
        valid_config_dict: dict,
    ) -> None:
        valid_config_dict["agents"]["broken"] = {
            "provider": "nonexistent",
            "role": "broken",
            "prompt_path": "x.md",
        }
        with pytest.raises(ValueError, match="references unknown provider"):
            ResearchConfig(**valid_config_dict)

    def test_debate_referencing_unknown_agent_fails(
        self,
        valid_config_dict: dict,
    ) -> None:
        valid_config_dict["debates"]["promotion_review"]["participants"].append(
            "ghost_agent",
        )
        with pytest.raises(ValueError, match="references unknown agent"):
            ResearchConfig(**valid_config_dict)

    def test_missing_rules_path_fails(self, valid_config_dict: dict) -> None:
        valid_config_dict["debates"]["promotion_review"]["rules_path"] = "/nope/does/not/exist.md"
        with pytest.raises(ValueError, match="does not exist"):
            ResearchConfig(**valid_config_dict)


# =============================================================================
# YAML round-trip
# =============================================================================


class TestLoadFromYaml:
    def test_load_config_yaml_round_trip(
        self,
        valid_config_dict: dict,
        tmp_path: Path,
    ) -> None:
        yaml_path = tmp_path / "research_agents.yaml"
        yaml_path.write_text(yaml.safe_dump(valid_config_dict))
        cfg = load_config(yaml_path)
        assert "bull" in cfg.agents
        assert cfg.debates["promotion_review"].verdict_engine == "rule_based"

    def test_load_config_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_load_config_non_dict_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="did not parse as a dict"):
            load_config(bad)


# =============================================================================
# The committed example config loads
# =============================================================================


class TestRealConfig:
    def test_committed_config_loads(self) -> None:
        """The configs/research_agents.yaml that ships with the repo
        must validate against the schema. This is the integration check
        for the YAML the operator actually edits."""
        cfg = load_config("configs/research_agents.yaml")
        assert "claude" in cfg.providers
        assert "claude-code" in cfg.providers
        assert cfg.providers["claude"].default_model == "claude-fable-5"
        assert cfg.providers["claude-code"].default_model == "claude-fable-5"
        # All-Claude routing since 2026-07-17 (operator dropped xAI):
        # every agent rides the subscription driver.
        assert all(a.provider == "claude-code" for a in cfg.agents.values())
        assert "bull_reviewer" in cfg.agents
        assert "bear_reviewer" in cfg.agents
        assert "promotion_review" in cfg.debates
        # The 4-round structure: initial_positions, smart_questions,
        # rebuttal, final_position
        rounds = cfg.debates["promotion_review"].rounds
        assert len(rounds) == 4
        assert rounds[1].type == RoundType.PER_AGENT_ASYNC
        assert rounds[1].sub_loop == "list_unknowns_then_resolve"
