"""Config schema + loader for the multi-agent research pipeline (CL-rzf7).

Reads ``configs/research_agents.yaml`` and validates it via pydantic.
Returns ``ResearchConfig`` containing fully-resolved ``ProviderConfig``,
``AgentConfig``, and ``DebateConfig`` objects ready to plug into the
orchestrator (CL-ew4a).

Adding a new agent or a new debate type requires only a YAML edit — no
code changes anywhere else. That's the entire point of the schema:
debates are composed at config load time from a registry of declared
agents.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# Registered LLM providers — must match the keys in
# src/research/llm/client.py:_DRIVERS. Future providers register via
# register_driver() at import time and the validator picks them up
# implicitly (we re-derive the set on each load).
def _known_providers() -> set[str]:
    """Inspect the LLM client registry. Late binding so test code that
    registers a custom driver before loading config gets accepted."""
    from src.research.llm.client import _DRIVERS  # noqa: PLC0415
    return set(_DRIVERS.keys())


class RoundType(StrEnum):
    """How a debate round is executed.

    parallel: every participant runs simultaneously with the same prompt.
              Used for Round 1 (initial positions) and Round 4 (final).
    sequential: participants run in declared order, each seeing the
                running transcript. Used for Round 3 (rebuttal).
    per_agent_async: each participant runs its own sub-loop independently
                     and the round completes when all sub-loops finish.
                     Used for Round 2 (smart-question stage where each
                     agent surfaces unknowns and waits for resolution).
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    PER_AGENT_ASYNC = "per_agent_async"


class ProviderConfig(BaseModel):
    """One LLM provider entry."""

    api_key_env: str = Field(
        ...,
        description="Env var name to read the API key from",
    )
    default_model: str = Field(
        ...,
        description="Model to use if an agent doesn't specify one",
    )

    @field_validator("api_key_env")
    @classmethod
    def _api_key_env_uppercase(cls, v: str) -> str:
        return v.strip()


class AgentConfig(BaseModel):
    """One agent declaration. The orchestrator builds the agent at
    runtime by combining (provider, model, prompt_path)."""

    provider: str = Field(..., description="Must match a key in providers.")
    model: str | None = Field(
        None,
        description="Specific model — falls back to provider's default_model.",
    )
    prompt_path: str = Field(
        ...,
        description="Relative path to the system-prompt markdown file.",
    )
    role: str = Field(
        ...,
        description="Free-form label (e.g. 'idea_generator', 'bull', 'bear').",
    )
    max_tokens: int = Field(default=4096, ge=128, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class RoundConfig(BaseModel):
    """One round inside a debate."""

    name: str = Field(..., description="Round name — appears in transcripts.")
    type: RoundType
    # For per_agent_async rounds, each agent runs its own sub-loop using
    # this prompt template (e.g. 'list_unknowns_then_resolve').
    sub_loop: str | None = None


class DebateConfig(BaseModel):
    """One debate type declaration. The orchestrator (CL-ew4a) loads
    this and runs the rounds in order, dispatching to the participants
    via the agents registry."""

    participants: list[str] = Field(..., min_length=1)
    rules_path: str = Field(
        ...,
        description="Path to the REVIEW_RULES.md doc the participants read",
    )
    rounds: list[RoundConfig] = Field(..., min_length=1)
    verdict_engine: str = Field(
        default="rule_based",
        description=(
            "Which verdict-engine implementation to use. Currently only "
            "'rule_based' is wired (CL-ath2)."
        ),
    )

    @field_validator("rules_path")
    @classmethod
    def _rules_path_exists(cls, v: str) -> str:
        if not Path(v).exists():
            msg = f"rules_path {v!r} does not exist"
            raise ValueError(msg)
        return v


class ResearchConfig(BaseModel):
    """Full config: providers + agents + debates. Cross-validated."""

    providers: dict[str, ProviderConfig]
    agents: dict[str, AgentConfig]
    debates: dict[str, DebateConfig]

    @model_validator(mode="after")
    def _cross_validate(self) -> ResearchConfig:
        # Every provider name must match a registered LLM driver.
        known = _known_providers()
        for name, p in self.providers.items():
            if name not in known:
                msg = (
                    f"provider {name!r} not in registered LLM drivers "
                    f"({sorted(known)}). Did you forget register_driver()?"
                )
                raise ValueError(msg)
            # The default model must have a known cost row; warn (not error)
            # if it doesn't — unknown-model cost is logged $0 (see client.py)
            from src.research.llm.client import _PRICING_USD_PER_MTOK  # noqa: PLC0415
            if p.default_model not in _PRICING_USD_PER_MTOK:
                logger.warning(
                    "provider %r default_model %r missing from pricing table — "
                    "cost will log as $0 for this model",
                    name, p.default_model,
                )

        # Every agent's provider must be in providers.
        for agent_name, a in self.agents.items():
            if a.provider not in self.providers:
                msg = (
                    f"agent {agent_name!r} references unknown provider "
                    f"{a.provider!r}; available: {sorted(self.providers)}"
                )
                raise ValueError(msg)

        # Every debate's participants must be in agents.
        for debate_name, d in self.debates.items():
            for p_name in d.participants:
                if p_name not in self.agents:
                    msg = (
                        f"debate {debate_name!r} references unknown agent "
                        f"{p_name!r}; available: {sorted(self.agents)}"
                    )
                    raise ValueError(msg)

        return self

    def resolve_model(self, agent_name: str) -> str:
        """Get the effective model for an agent — explicit if set, else
        the agent's provider default."""
        a = self.agents[agent_name]
        return a.model or self.providers[a.provider].default_model

    def resolve_api_key(self, agent_name: str) -> str:
        """Read the API key for an agent's provider from env. Empty
        string if not set — the LLM client raises at construction time."""
        a = self.agents[agent_name]
        env_var = self.providers[a.provider].api_key_env
        return os.environ.get(env_var, "")


def load_config(path: Path | str) -> ResearchConfig:
    """Load + validate the YAML config at ``path``. Raises ValueError
    with a precise message on validation failure — fail-loud at boot,
    not on first call mid-loop."""
    p = Path(path)
    if not p.exists():
        msg = f"research config not found at {p}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        msg = f"config at {p} did not parse as a dict"
        raise ValueError(msg)
    return ResearchConfig(**raw)
