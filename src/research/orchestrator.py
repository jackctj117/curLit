"""Debate orchestrator (CL-ew4a) — drives the multi-agent debate that
decides whether a candidate strategy is promoted.

Loads a debate config (e.g. ``promotion_review``) from
``configs/research_agents.yaml`` and executes its rounds in order:

  * **parallel**         — every participant runs simultaneously with the
                           same prompt. Used for Round 1 (initial
                           positions) and Round 4 (final position).
  * **sequential**       — participants run in declared order, each
                           seeing the running transcript. Used for
                           Round 3 (rebuttal).
  * **per_agent_async**  — each participant runs its own sub-loop:
                           emit smart-questions → resolver dispatches →
                           answers feed back into the agent's context.
                           Used for Round 2.

For each round the orchestrator appends to a debate transcript
(``docs/research/debates/{slug}/transcript.md``) and a structured ledger
(``transcript.jsonl``) so the verdict engine + the human reviewer can
audit every claim and cost.

Hard caps enforced here (over and above whatever the config says):
  * max 3 rebuttal rounds
  * max 8 smart-questions per agent (the resolver also enforces)
  * max 30s wall time per code_tool resolution (resolver enforces)

The orchestrator is decoupled from concrete agent construction via an
``agent_factory`` callback — the default builds via ``Agent.from_config``
but tests inject mock agents. Same for the ``QuestionResolver`` instance
which the caller pre-configures with whatever code_tools / agent
dispatch the deployment supports.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.agents.base import Agent, AgentResponse
from src.research.agents.resolver import (
    QuestionResolver,
    ResolutionResult,
    ResolutionStatus,
    Routing,
    SmartQuestion,
    parse_questions,
)
from src.research.agents.reviewer import Position, parse_position
from src.research.config import DebateConfig, ResearchConfig, RoundType

logger = logging.getLogger(__name__)


# Hard ceilings — these win over any config value.
MAX_REBUTTAL_ROUNDS: int = 3


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TranscriptEntry:
    """One agent invocation persisted to the debate ledger."""

    timestamp: str
    round_name: str
    agent_name: str
    role: str
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    usd_cost: float
    elapsed_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedQuestion:
    """One smart-question + the resolver's outcome — kept for transcript."""

    question: SmartQuestion
    result: ResolutionResult


@dataclass
class DebateResult:
    """Output of ``run_debate``. Verdict engine consumes this."""

    debate_name: str
    strategy_slug: str
    transcript_path: Path
    transcript_entries: list[TranscriptEntry]
    final_positions: dict[str, Position]
    open_questions: list[str]
    resolved_questions: list[ResolvedQuestion]
    total_cost_usd: float
    total_elapsed_sec: float


# Type alias for the orchestrator's agent factory dependency.
AgentFactory = Callable[[str], Agent]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class DebateOrchestrator:
    """Runs one named debate end-to-end against a candidate report.

    Construct once per debate run. The orchestrator owns the transcript
    file handle for the duration of ``run_debate``. Reuse across runs is
    safe — state (transcript, positions) is per-call, not per-instance.
    """

    def __init__(
        self,
        research_config: ResearchConfig,
        debate_name: str,
        resolver: QuestionResolver | None = None,
        agent_factory: AgentFactory | None = None,
        transcript_root: Path | str = "docs/research/debates",
    ) -> None:
        if debate_name not in research_config.debates:
            msg = (
                f"debate {debate_name!r} not declared in research config; "
                f"available: {sorted(research_config.debates)}"
            )
            raise KeyError(msg)
        self.research_config = research_config
        self.debate_name = debate_name
        self.debate: DebateConfig = research_config.debates[debate_name]
        self.transcript_root = Path(transcript_root)
        self.resolver = resolver or QuestionResolver()
        self._agent_factory: AgentFactory = (
            agent_factory or self._default_agent_factory
        )

    def _default_agent_factory(self, name: str) -> Agent:
        return Agent.from_config(name=name, research_config=self.research_config)

    def run_debate(
        self,
        strategy_slug: str,
        candidate_report_text: str,
        review_rules_text: str | None = None,
    ) -> DebateResult:
        """Execute the full debate against one candidate. Returns a
        ``DebateResult`` ready for the verdict engine.

        ``candidate_report_text`` is the markdown / JSON the reviewers
        cite from. ``review_rules_text`` defaults to reading the file at
        the debate's ``rules_path`` (already validated to exist by the
        config loader).
        """
        run_started = time.time()
        rules_text = review_rules_text or Path(self.debate.rules_path).read_text()

        agents: dict[str, Agent] = {
            name: self._agent_factory(name) for name in self.debate.participants
        }
        transcript_dir = self.transcript_root / strategy_slug
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_md_path = transcript_dir / "transcript.md"
        transcript_jsonl_path = transcript_dir / "transcript.jsonl"
        # Truncate prior runs — one debate, one transcript.
        transcript_md_path.write_text(self._transcript_header(strategy_slug))
        transcript_jsonl_path.write_text("")

        entries: list[TranscriptEntry] = []
        resolved_questions: list[ResolvedQuestion] = []
        open_questions: list[str] = []
        last_round_responses: dict[str, AgentResponse] = {}
        rebuttal_rounds_seen = 0

        for round_cfg in self.debate.rounds:
            if round_cfg.name == "rebuttal":
                rebuttal_rounds_seen += 1
                if rebuttal_rounds_seen > MAX_REBUTTAL_ROUNDS:
                    logger.warning(
                        "Skipping extra rebuttal round %r — cap %d already hit",
                        round_cfg.name, MAX_REBUTTAL_ROUNDS,
                    )
                    continue

            transcript_so_far = self._render_transcript_so_far(entries)

            if round_cfg.type == RoundType.PARALLEL:
                round_responses = self._run_parallel(
                    agents=agents,
                    round_name=round_cfg.name,
                    candidate_report_text=candidate_report_text,
                    review_rules_text=rules_text,
                    transcript_so_far=transcript_so_far,
                )
            elif round_cfg.type == RoundType.SEQUENTIAL:
                round_responses = self._run_sequential(
                    agents=agents,
                    round_name=round_cfg.name,
                    candidate_report_text=candidate_report_text,
                    review_rules_text=rules_text,
                    initial_transcript=transcript_so_far,
                    entries=entries,
                )
            elif round_cfg.type == RoundType.PER_AGENT_ASYNC:
                round_responses, round_resolved, round_open = (
                    self._run_per_agent_async(
                        agents=agents,
                        round_name=round_cfg.name,
                        candidate_report_text=candidate_report_text,
                        review_rules_text=rules_text,
                        transcript_so_far=transcript_so_far,
                    )
                )
                resolved_questions.extend(round_resolved)
                open_questions.extend(round_open)
            else:
                msg = f"unknown round type: {round_cfg.type!r}"
                raise ValueError(msg)

            for resp in round_responses.values():
                entry = self._make_entry(round_cfg.name, resp)
                entries.append(entry)
                self._persist_entry(
                    transcript_md_path, transcript_jsonl_path, entry,
                )
            last_round_responses = round_responses

        final_positions = self._extract_final_positions(
            entries=entries,
            last_round_responses=last_round_responses,
            agents=agents,
        )

        # Footer with positions + open questions for the human reader.
        self._append_summary(
            transcript_md_path,
            final_positions=final_positions,
            open_questions=open_questions,
            total_cost=sum(e.usd_cost for e in entries),
        )

        return DebateResult(
            debate_name=self.debate_name,
            strategy_slug=strategy_slug,
            transcript_path=transcript_md_path,
            transcript_entries=entries,
            final_positions=final_positions,
            open_questions=open_questions,
            resolved_questions=resolved_questions,
            total_cost_usd=sum(e.usd_cost for e in entries),
            total_elapsed_sec=time.time() - run_started,
        )

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        agents: dict[str, Agent],
        round_name: str,
        candidate_report_text: str,
        review_rules_text: str,
        transcript_so_far: str,
    ) -> dict[str, AgentResponse]:
        """All participants run with the same prompt simultaneously.

        Uses a thread pool — LLM calls are I/O-bound, the GIL doesn't
        bite, and the providers handle concurrency on their end. If a
        provider rate-limits, the per-call retry inside the LLM client
        absorbs the bounce; concurrent calls don't compound the issue.
        """
        responses: dict[str, AgentResponse] = {}
        with ThreadPoolExecutor(max_workers=max(len(agents), 1)) as pool:
            futures = {
                pool.submit(
                    self._invoke_round,
                    agent=agent,
                    round_name=round_name,
                    candidate_report_text=candidate_report_text,
                    review_rules_text=review_rules_text,
                    transcript_so_far=transcript_so_far,
                ): name
                for name, agent in agents.items()
            }
            for fut in futures:
                name = futures[fut]
                responses[name] = fut.result()
        return responses

    def _run_sequential(
        self,
        agents: dict[str, Agent],
        round_name: str,
        candidate_report_text: str,
        review_rules_text: str,
        initial_transcript: str,
        entries: list[TranscriptEntry],
    ) -> dict[str, AgentResponse]:
        """Participants run in declared order; each sees the running
        transcript including the prior agent's response from this round.
        Used for the rebuttal round so each agent reads the other's
        Round 1 case + can engage with the running rebuttal chain."""
        responses: dict[str, AgentResponse] = {}
        running = initial_transcript
        for name, agent in agents.items():
            resp = self._invoke_round(
                agent=agent,
                round_name=round_name,
                candidate_report_text=candidate_report_text,
                review_rules_text=review_rules_text,
                transcript_so_far=running,
            )
            responses[name] = resp
            # Append this response to the running transcript so the next
            # participant in the chain reads it.
            tentative_entry = self._make_entry(round_name, resp)
            running = self._render_transcript_so_far(entries + [tentative_entry])
        return responses

    def _run_per_agent_async(
        self,
        agents: dict[str, Agent],
        round_name: str,
        candidate_report_text: str,
        review_rules_text: str,
        transcript_so_far: str,
    ) -> tuple[dict[str, AgentResponse], list[ResolvedQuestion], list[str]]:
        """Each participant runs its own sub-loop: emit smart-questions,
        resolver dispatches, answers feed back into the participant's
        next response. The round completes when every participant's
        sub-loop has terminated.

        For v1 the orchestrator handles routing decisions itself:
          * ``other_agent`` — target = the OTHER participant in the
            debate. The resolver invokes ``agent_dispatch`` (which the
            orchestrator wires to the in-process agent registry).
          * ``code_tool``   — dispatched to whichever tool the resolver
            already has registered. If no tools registered, the question
            FAILS and goes to open_questions.
          * ``human``       — ESCALATE immediately, accumulates as an
            open question for the verdict engine.

        Returns (responses, resolved_questions, open_question_strings).
        """
        # Wire resolver to dispatch other_agent questions back into the
        # agent registry the orchestrator owns.
        original_dispatch = self.resolver.agent_dispatch
        self.resolver.agent_dispatch = lambda target, qtext: (
            self._dispatch_to_agent(agents, target, qtext)
        )

        try:
            responses: dict[str, AgentResponse] = {}
            resolved: list[ResolvedQuestion] = []
            open_q: list[str] = []

            for name, agent in agents.items():
                # 1) Ask the agent to emit smart-questions for this candidate.
                emit_resp = self._invoke_round(
                    agent=agent,
                    round_name=round_name,
                    candidate_report_text=candidate_report_text,
                    review_rules_text=review_rules_text,
                    transcript_so_far=transcript_so_far,
                    extra_instruction=(
                        "Emit your smart-questions per docs/research/"
                        "SMART_QUESTIONS.md as fenced YAML blocks. Each "
                        "block must have all six required fields. If you "
                        "have no genuine blockers, emit zero blocks and "
                        "say so explicitly."
                    ),
                )
                # 2) Parse + dispatch each emitted question.
                questions = parse_questions(emit_resp.text)
                for q in questions:
                    # Force asker to match the agent we asked, not whatever
                    # the agent self-labeled. Prevents one agent borrowing
                    # another's quota.
                    q.asker = name
                    other_participants = [
                        a for a in self.debate.participants if a != name
                    ]
                    target_agent = (
                        other_participants[0] if other_participants else None
                    )
                    result = self._dispatch_one_question(q, target_agent)
                    resolved.append(ResolvedQuestion(question=q, result=result))
                    if result.status in (
                        ResolutionStatus.ESCALATE, ResolutionStatus.FAILED,
                    ):
                        open_q.append(
                            f"{q.question_id} (asker={q.asker}, "
                            f"routing={q.routing.value}): "
                            f"{result.escalate_reason or 'no answer obtainable'}"
                        )
                # The agent's smart-question emission IS the round's
                # response we record. The resolved-question chain is
                # surfaced separately in the result.
                responses[name] = emit_resp
            return responses, resolved, open_q
        finally:
            self.resolver.agent_dispatch = original_dispatch

    def _dispatch_one_question(
        self,
        q: SmartQuestion,
        target_agent: str | None,
    ) -> ResolutionResult:
        """Route one smart-question through the resolver. The resolver
        enforces caps + validates; the orchestrator just supplies the
        target_agent for OTHER_AGENT routing."""
        kwargs: dict[str, Any] = {}
        if q.routing == Routing.OTHER_AGENT:
            kwargs["target_agent"] = target_agent
        # CODE_TOOL questions: in v1 we don't auto-extract a tool name
        # from the question text. The asker can specify one in the
        # acceptance / specific_evidence_needed fields, but until we have
        # a registered tool catalog the resolver returns FAILED for
        # code_tool with no tool_name — recorded as an open question.
        return self.resolver.resolve(q, **kwargs)

    def _dispatch_to_agent(
        self,
        agents: dict[str, Agent],
        target_name: str,
        question_text: str,
    ) -> str:
        """Wire-up for resolver.agent_dispatch. Routes a smart-question to
        the target agent's ``run`` and returns just the answer text."""
        if target_name not in agents:
            msg = (
                f"resolver tried to dispatch to agent {target_name!r} but "
                f"only {sorted(agents)} are participants"
            )
            raise KeyError(msg)
        resp = agents[target_name].run(
            user_prompt=(
                f"You are answering a smart-question from another agent. "
                f"Be specific, cite by line/metric, and don't speculate.\n\n"
                f"{question_text}"
            ),
        )
        return resp.text

    # ------------------------------------------------------------------
    # Single-agent invocation
    # ------------------------------------------------------------------

    def _invoke_round(
        self,
        agent: Agent,
        round_name: str,
        candidate_report_text: str,
        review_rules_text: str,
        transcript_so_far: str,
        extra_instruction: str | None = None,
    ) -> AgentResponse:
        """Single agent.run call with the round's standard context wiring."""
        context_files: dict[str, str] = {
            "REVIEW_RULES.md": review_rules_text,
            "candidate_report": candidate_report_text,
        }
        if transcript_so_far:
            context_files["debate_transcript_so_far"] = transcript_so_far
        instruction = (
            f"This is round {round_name!r} of the {self.debate_name!r} "
            f"debate. Produce the output specified by your system prompt "
            f"for this round."
        )
        if extra_instruction:
            instruction = f"{instruction}\n\n{extra_instruction}"
        return agent.run(
            user_prompt=instruction,
            context_files=context_files,
        )

    # ------------------------------------------------------------------
    # Position extraction
    # ------------------------------------------------------------------

    def _extract_final_positions(
        self,
        entries: list[TranscriptEntry],
        last_round_responses: dict[str, AgentResponse],
        agents: dict[str, Agent],
    ) -> dict[str, Position]:
        """Map each participant to the position they declared in their
        most recent response. Only reviewer-role agents (bull/bear) get
        positions; other roles don't declare PROMOTE/REJECT/ABSTAIN."""
        positions: dict[str, Position] = {}
        reviewer_roles = {"bull", "bear"}
        for name, agent in agents.items():
            if agent.role not in reviewer_roles:
                continue
            resp = last_round_responses.get(name)
            if resp is None:
                positions[name] = Position.ABSTAIN
                continue
            positions[name] = parse_position(resp.text)
        return positions

    # ------------------------------------------------------------------
    # Transcript persistence
    # ------------------------------------------------------------------

    def _make_entry(
        self, round_name: str, resp: AgentResponse,
    ) -> TranscriptEntry:
        return TranscriptEntry(
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            round_name=round_name,
            agent_name=resp.agent_name,
            role=resp.role,
            content=resp.text,
            model=resp.model,
            provider=resp.provider,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            usd_cost=resp.usd_cost,
            elapsed_sec=resp.elapsed_sec,
            metadata=dict(resp.metadata),
        )

    def _persist_entry(
        self,
        md_path: Path,
        jsonl_path: Path,
        entry: TranscriptEntry,
    ) -> None:
        # Markdown: human-readable per-entry block.
        with md_path.open("a") as f:
            f.write(self._render_entry_md(entry))
        # JSONL: structured ledger the verdict engine + tests parse.
        with jsonl_path.open("a") as f:
            f.write(json.dumps(asdict(entry), default=str) + "\n")

    def _render_entry_md(self, entry: TranscriptEntry) -> str:
        return (
            f"\n## Round: {entry.round_name} — agent: {entry.agent_name} "
            f"({entry.role})\n"
            f"*ts={entry.timestamp} model={entry.model} "
            f"in/out_tokens={entry.input_tokens}/{entry.output_tokens} "
            f"cost=${entry.usd_cost:.4f} elapsed={entry.elapsed_sec:.2f}s*\n\n"
            f"{entry.content}\n\n---\n"
        )

    def _render_transcript_so_far(self, entries: list[TranscriptEntry]) -> str:
        """Render the running transcript that gets fed back into agents
        as context. Same format as the persisted markdown so the agent
        sees what the human reviewer will see."""
        if not entries:
            return ""
        return "\n".join(self._render_entry_md(e) for e in entries)

    def _transcript_header(self, strategy_slug: str) -> str:
        return (
            f"# Debate transcript — {self.debate_name} — {strategy_slug}\n\n"
            f"Started: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
            f"Participants: {', '.join(self.debate.participants)}\n"
            f"Rounds: {', '.join(r.name for r in self.debate.rounds)}\n"
        )

    def _append_summary(
        self,
        md_path: Path,
        final_positions: dict[str, Position],
        open_questions: list[str],
        total_cost: float,
    ) -> None:
        lines = ["\n## Debate summary\n"]
        for name, pos in final_positions.items():
            lines.append(f"- **{name}**: {pos.value}\n")
        if open_questions:
            lines.append("\n### Open questions\n")
            for oq in open_questions:
                lines.append(f"- {oq}\n")
        lines.append(f"\n**Total cost**: ${total_cost:.4f}\n")
        with md_path.open("a") as f:
            f.write("".join(lines))
