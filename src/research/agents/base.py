"""Agent base class — shared scaffolding for all research agents.

Every agent in this package shares the same shape:
  1. A system prompt (markdown file at ``AgentConfig.prompt_path``)
  2. An ``LLMClient`` constructed from the agent's provider config
  3. Optional access to the ``KnowledgeRetriever`` (CL-bcr4)
  4. A standardized ``run(user_prompt, context_files=None) -> AgentResponse``

Concrete agents (Bull, Bear, Idea, Implementer, Resolver) differ in:
  * Their system prompt (the doctrine they internalize)
  * How they post-process the LLM response (e.g. Bull/Bear extract a
    structured PROMOTE_CASE / REJECT_CASE)
  * Whether they use knowledge tools

The agent does NOT decide outcomes. It produces evidence + a position;
the verdict engine (CL-ath2) maps evidence to PROMOTE/REJECT/ESCALATE
deterministically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.research.config import AgentConfig, ResearchConfig
from src.research.knowledge import KnowledgeRetriever
from src.research.llm.client import LLMClient, LLMResponse, Message, get_client

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """One agent invocation's full output.

    The orchestrator persists this to the debate transcript so the
    verdict engine + post-mortem reviewer can audit what each agent
    said and at what cost.
    """

    agent_name: str
    role: str
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    usd_cost: float
    elapsed_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    """Provider-agnostic agent. Construct via ``Agent.from_config(...)``
    so the LLM client + knowledge retriever wiring is uniform across
    concrete subclasses."""

    def __init__(
        self,
        name: str,
        config: AgentConfig,
        client: LLMClient,
        model: str,
        system_prompt: str,
        retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.retriever = retriever

    @property
    def role(self) -> str:
        return self.config.role

    @classmethod
    def from_config(
        cls,
        name: str,
        research_config: ResearchConfig,
        retriever: KnowledgeRetriever | None = None,
    ) -> Agent:
        """Build an agent from the loaded research config.

        Reads the agent's provider, resolves the model + API key, loads
        the system prompt from disk. Fails loud at construction if any
        required piece is missing — better than a mid-loop surprise.
        """
        if name not in research_config.agents:
            msg = f"agent {name!r} not declared in research config"
            raise KeyError(msg)
        agent_cfg = research_config.agents[name]

        prompt_path = Path(agent_cfg.prompt_path)
        if not prompt_path.exists():
            msg = (
                f"agent {name!r} system prompt not found at {prompt_path} — "
                f"create the markdown file or update prompt_path"
            )
            raise FileNotFoundError(msg)
        system_prompt = prompt_path.read_text()

        api_key = research_config.resolve_api_key(name)
        model = research_config.resolve_model(name)
        client = get_client(provider=agent_cfg.provider, api_key=api_key)

        return cls(
            name=name, config=agent_cfg, client=client, model=model,
            system_prompt=system_prompt, retriever=retriever,
        )

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def run(
        self,
        user_prompt: str,
        context_files: dict[str, str] | None = None,
        extra_system: str | None = None,
    ) -> AgentResponse:
        """Single-turn agent call. Returns ``AgentResponse``.

        ``context_files`` is a mapping of label → file content that the
        orchestrator wants the agent to see (e.g. the candidate report,
        REVIEW_RULES.md, the debate transcript so far). They get
        injected into the user prompt as labeled blocks so the agent
        can cite them precisely.

        ``extra_system`` is appended to the agent's base system prompt —
        used for round-specific instructions that change between calls
        (e.g. "this is Round 1 — produce your initial position").
        """
        system_text = self.system_prompt
        if extra_system:
            system_text = f"{system_text}\n\n---\n{extra_system}"

        full_user_prompt = self._compose_user_prompt(user_prompt, context_files or {})

        messages = [
            Message(role="system", content=system_text),
            Message(role="user", content=full_user_prompt),
        ]

        resp: LLMResponse = self.client.complete(
            messages=messages,
            model=self.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            text=resp.text,
            model=resp.model,
            provider=resp.provider,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            usd_cost=resp.usd_cost,
            elapsed_sec=resp.elapsed_sec,
        )

    @staticmethod
    def _compose_user_prompt(
        user_prompt: str,
        context_files: dict[str, str],
    ) -> str:
        """Stitch context-file blocks before the user prompt with clear
        section headers so the agent can cite them by label.

        Each block is XML-tagged so the LLM has a hard delimiter — much
        more reliable than relying on the model to parse markdown
        headers correctly when the content itself is markdown.
        """
        if not context_files:
            return user_prompt

        parts: list[str] = []
        for label, content in context_files.items():
            parts.append(f"<context label={label!r}>\n{content}\n</context>")
        parts.append(user_prompt)
        return "\n\n".join(parts)
