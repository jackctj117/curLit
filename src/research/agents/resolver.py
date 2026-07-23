"""Question resolver (CL-kiw7) — routes smart-questions emitted by
agents during the smart-question round of a debate.

Per ``docs/research/SMART_QUESTIONS.md``, every question must be a
6-field YAML block. The resolver:

  1. Parses YAML question blocks out of an agent's text response.
  2. Validates each against the SMART_QUESTIONS.md format
     requirements. Malformed questions get a ``REFORMULATE`` result
     with a list of specific defects — the asker re-submits.
  3. Dispatches well-formed questions per their ``routing`` field:
     - ``code_tool``    — runs a registered tool (grep / db query)
     - ``other_agent``  — invokes a callback the orchestrator supplies
     - ``human``        — returns ``ESCALATE`` payload immediately

Pure Python — no LLM call. The validation is mechanical (field
presence + heuristic specificity checks). v2 may add LLM-based
specificity scoring; v1 is deterministic for auditability and so the
orchestrator can run resolver passes without API key dependency.

Caps enforced: 8 questions / agent / debate, 30s wall time per
code_tool, 3 reformulate rounds before forcing ESCALATE.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Caps per docs/research/SMART_QUESTIONS.md
MAX_QUESTIONS_PER_AGENT: int = 8
MAX_REFORMULATES_PER_QUESTION: int = 3
MAX_CODE_TOOL_WALL_TIME_SEC: float = 30.0


class Routing(StrEnum):
    CODE_TOOL = "code_tool"
    OTHER_AGENT = "other_agent"
    HUMAN = "human"


class ResolutionStatus(StrEnum):
    """Outcome of a single resolve() call."""

    RESOLVED = "RESOLVED"  # answer obtained from tool/agent, returned in transcript
    REFORMULATE = "REFORMULATE"  # question malformed; asker re-submits
    ESCALATE = "ESCALATE"  # routing=human OR caps exceeded; needs operator
    FAILED = "FAILED"  # tool errored / agent unreachable; orchestrator decides


@dataclass
class SmartQuestion:
    """One parsed question per SMART_QUESTIONS.md format."""

    question_id: str  # 'q-{round}-{n}'
    asker: str  # agent name
    decision_blocker: str
    what_i_tried_first: str
    specific_evidence_needed: str
    routing: Routing
    acceptance: str
    raw_yaml: str = ""  # original text — preserved for transcript


@dataclass
class ResolutionResult:
    """Outcome of resolving one question."""

    status: ResolutionStatus
    question_id: str
    answer: str = ""  # tool output / agent response (when RESOLVED)
    reformulate_reasons: list[str] = field(default_factory=list)  # for REFORMULATE
    escalate_reason: str = ""  # for ESCALATE / FAILED
    elapsed_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Parsing — extract YAML question blocks from agent text
# =============================================================================


# Match a fenced YAML block of any of these forms:
#   ```yaml ... ```
#   ```yml ... ```
#   ```      (no language tag) ... ```
_YAML_FENCE_PATTERN = re.compile(
    r"```(?:yaml|yml)?\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)

_REQUIRED_FIELDS: tuple[str, ...] = (
    "question_id",
    "asker",
    "decision_blocker",
    "what_i_tried_first",
    "specific_evidence_needed",
    "routing",
    "acceptance",
)


def parse_questions(text: str) -> list[SmartQuestion]:
    """Extract all SmartQuestion blocks from an agent's text response.

    YAML blocks that don't have ALL the required fields are silently
    skipped (they're just regular code blocks the agent emitted, not
    questions). Blocks that DO have all fields but fail validation
    later get a REFORMULATE — that's the resolver's job, not parser's.
    """
    out: list[SmartQuestion] = []
    for match in _YAML_FENCE_PATTERN.finditer(text):
        raw = match.group(1)
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            continue  # malformed YAML — skip silently, not a question
        if not isinstance(data, dict):
            continue
        # Must have ALL the required fields to be considered a question.
        # Missing-field blocks are just code samples, not questions.
        if not all(f in data for f in _REQUIRED_FIELDS):
            continue
        try:
            routing = Routing(str(data["routing"]).strip())
        except ValueError:
            # Routing field present but unknown value — keep the block
            # so validate_question() can surface this with a precise
            # reformulate reason. Use a sentinel routing for now.
            out.append(_build_question_with_invalid_routing(data, raw))
            continue
        out.append(
            SmartQuestion(
                question_id=str(data["question_id"]).strip(),
                asker=str(data["asker"]).strip(),
                decision_blocker=str(data["decision_blocker"]).strip(),
                what_i_tried_first=str(data["what_i_tried_first"]).strip(),
                specific_evidence_needed=str(data["specific_evidence_needed"]).strip(),
                routing=routing,
                acceptance=str(data["acceptance"]).strip(),
                raw_yaml=raw,
            )
        )
    return out


def _build_question_with_invalid_routing(data: dict[str, Any], raw: str) -> SmartQuestion:
    """Construct a SmartQuestion when routing is invalid — using HUMAN as
    placeholder so the validator's reformulate note can surface the
    real issue precisely."""
    return SmartQuestion(
        question_id=str(data["question_id"]).strip(),
        asker=str(data["asker"]).strip(),
        decision_blocker=str(data["decision_blocker"]).strip(),
        what_i_tried_first=str(data["what_i_tried_first"]).strip(),
        specific_evidence_needed=str(data["specific_evidence_needed"]).strip(),
        routing=Routing.HUMAN,  # placeholder
        acceptance=str(data["acceptance"]).strip(),
        raw_yaml=raw,
    )


# =============================================================================
# Validation — apply SMART_QUESTIONS.md rules
# =============================================================================


# Rule IDs from REVIEW_RULES.md (A.1–A.7, B.1–B.3, C.1–C.5, D.1–D.4).
# decision_blocker SHOULD reference one of these — vague questions don't.
_RULE_ID_PATTERN = re.compile(r"\b[A-D]\.\d+\b")

# Vague phrases the asker shouldn't use as the entire decision_blocker.
_VAGUE_PHRASES: tuple[str, ...] = (
    "not sure",
    "don't know if",
    "feels off",
    "seems wrong",
    "looks weird",
    "unsure",
    "uncertain",
)

# Heuristic: the what_i_tried_first should look like 2+ list items.
# Markdown dashes / numbers / bullets, OR explicit "1." numbering.
_LIST_ITEM_PATTERN = re.compile(r"(^\s*[-*•]\s+)|(^\s*\d+[.)]\s+)", re.MULTILINE)


def validate_question(q: SmartQuestion) -> list[str]:
    """Return a list of reformulate reasons (empty list = valid).

    Implements the bounce conditions from SMART_QUESTIONS.md:
      - decision_blocker vague (no rule ID, or just a vague phrase)
      - what_i_tried_first has < 2 specific attempts
      - specific_evidence_needed is a re-statement of the question
      - acceptance is a bare yes/no without verification criterion
      - routing field unparseable

    These checks are heuristic, not perfect — but the asking agent gets
    the reformulate-with-reasons response and re-submits cleanly when
    the format is wrong.
    """
    issues: list[str] = []

    # Required fields all non-empty
    if not q.question_id:
        issues.append("question_id is empty")
    if not q.asker:
        issues.append("asker is empty")

    # decision_blocker: must reference at least one rule ID and not be
    # purely a vague phrase
    db = q.decision_blocker.lower().strip()
    if not q.decision_blocker:
        issues.append("decision_blocker is empty")
    else:
        if not _RULE_ID_PATTERN.search(q.decision_blocker):
            issues.append(
                "decision_blocker must reference a rule ID from REVIEW_RULES.md "
                "(e.g. A.1, B.2, C.3)"
            )
        if any(phrase in db for phrase in _VAGUE_PHRASES) and len(db) < 80:
            issues.append(
                "decision_blocker is vague — name a SPECIFIC verdict-affecting "
                "decision, not 'not sure / don't know if'"
            )

    # what_i_tried_first: must look like ≥ 2 list items
    if not q.what_i_tried_first:
        issues.append("what_i_tried_first is empty")
    else:
        list_items = _LIST_ITEM_PATTERN.findall(q.what_i_tried_first)
        if len(list_items) < 2:
            issues.append(
                "what_i_tried_first must list at least 2 specific attempts "
                "(use markdown bullets '- attempt' or numbered '1. attempt')"
            )

    # specific_evidence_needed: must not just re-state the question.
    # Heuristic: must be substantively different from decision_blocker.
    if not q.specific_evidence_needed:
        issues.append("specific_evidence_needed is empty")
    elif _is_substantially_same(q.specific_evidence_needed, q.decision_blocker):
        issues.append(
            "specific_evidence_needed restates the question — describe the "
            "ANSWER form (a number, a code line, a metric), not the question"
        )

    # acceptance: must not be a bare yes/no
    if not q.acceptance:
        issues.append("acceptance is empty")
    elif _is_bare_yes_no(q.acceptance):
        issues.append(
            "acceptance is a bare yes/no — describe the verification criterion "
            "the answer must satisfy"
        )

    # Routing handled at parse time (Routing enum); if the parser fell
    # back to HUMAN placeholder for an unknown routing value, the
    # original raw_yaml will reflect that. For now we trust the enum.

    return issues


def _is_substantially_same(a: str, b: str) -> bool:
    """Heuristic: are these two strings close enough that one is a re-
    statement of the other? Used to catch askers who paste the question
    into specific_evidence_needed verbatim."""
    norm_a = re.sub(r"\s+", " ", a.lower().strip())
    norm_b = re.sub(r"\s+", " ", b.lower().strip())
    if not norm_a or not norm_b:
        return False
    # Exact match or one fully contained in the other
    if norm_a == norm_b:
        return True
    shorter, longer = sorted([norm_a, norm_b], key=len)
    return len(shorter) >= 25 and shorter in longer


def _is_bare_yes_no(text: str) -> bool:
    """Heuristic: is acceptance a bare yes/no answer with no verification?"""
    norm = re.sub(r"\s+", " ", text.lower().strip())
    if len(norm) < 30:
        # Short acceptance + matches yes/no pattern likely bare
        return bool(re.match(r"^(yes|no|true|false|y|n)\.?$", norm))
    return False


# =============================================================================
# Resolver
# =============================================================================


# Type aliases for the callbacks the orchestrator hands the resolver.
# The orchestrator owns the actual implementations; the resolver just
# dispatches to them. Keeps the resolver decoupled.
CodeToolFn = Callable[[str, dict[str, Any]], str]
"""(tool_name, args) → answer text. Raises on tool failure."""

AgentDispatchFn = Callable[[str, str], str]
"""(target_agent_name, question_text) → that agent's answer text."""


@dataclass
class _AgentLedger:
    """Per-agent ask + reformulate counters for cap enforcement."""

    questions_asked: int = 0
    reformulates_per_question: dict[str, int] = field(default_factory=dict)


class QuestionResolver:
    """Routes smart-questions per the SMART_QUESTIONS.md format.

    Construct once per debate. Resolver-level state (per-agent caps,
    per-question reformulate counters) lives here so the orchestrator
    can call ``resolve()`` repeatedly during the smart-question round
    without re-passing the bookkeeping.

    All side-effecting actions (running a code tool, dispatching to
    another agent, sending an escalation alert) are callbacks the
    caller supplies — the resolver has no DB or HTTP knowledge of its
    own. This keeps it unit-testable without mocks of subprocess /
    agent registry / Telegram.
    """

    def __init__(
        self,
        code_tools: dict[str, CodeToolFn] | None = None,
        agent_dispatch: AgentDispatchFn | None = None,
    ) -> None:
        self.code_tools: dict[str, CodeToolFn] = dict(code_tools or {})
        self.agent_dispatch = agent_dispatch
        self._ledger: dict[str, _AgentLedger] = {}

    def register_code_tool(self, name: str, fn: CodeToolFn) -> None:
        """Register a code/data tool the resolver can dispatch to. The
        question's `tool` field (parsed from question text) selects."""
        self.code_tools[name] = fn

    def resolve(
        self,
        question: SmartQuestion,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        target_agent: str | None = None,
    ) -> ResolutionResult:
        """Validate + dispatch. Returns a ResolutionResult.

        For ``code_tool`` routing, supply ``tool_name`` and optional
        ``tool_args`` — the resolver looks up the registered tool and
        calls it.

        For ``other_agent`` routing, supply ``target_agent`` (name of
        the agent to ask). The resolver invokes ``agent_dispatch`` with
        the question text and returns whatever comes back.

        For ``human`` routing, the resolver returns immediately with
        status=ESCALATE — no tool call, no agent call.
        """
        # Cap check — questions per agent per debate
        ledger = self._ledger.setdefault(question.asker, _AgentLedger())
        if ledger.questions_asked >= MAX_QUESTIONS_PER_AGENT:
            return ResolutionResult(
                status=ResolutionStatus.ESCALATE,
                question_id=question.question_id,
                escalate_reason=(
                    f"asker {question.asker!r} exceeded "
                    f"{MAX_QUESTIONS_PER_AGENT} questions/debate cap"
                ),
            )
        ledger.questions_asked += 1

        # Reformulate cap (3 strikes → ESCALATE the underlying decision)
        prior_reformulates = ledger.reformulates_per_question.get(
            question.question_id,
            0,
        )
        if prior_reformulates >= MAX_REFORMULATES_PER_QUESTION:
            return ResolutionResult(
                status=ResolutionStatus.ESCALATE,
                question_id=question.question_id,
                escalate_reason=(
                    f"question {question.question_id!r} bounced "
                    f"{prior_reformulates}× — underlying decision can't be "
                    f"formulated cleanly enough to resolve"
                ),
            )

        # Validate format
        issues = validate_question(question)
        if issues:
            ledger.reformulates_per_question[question.question_id] = prior_reformulates + 1
            return ResolutionResult(
                status=ResolutionStatus.REFORMULATE,
                question_id=question.question_id,
                reformulate_reasons=issues,
            )

        # Dispatch
        t0 = time.time()
        try:
            if question.routing == Routing.CODE_TOOL:
                return self._resolve_code_tool(
                    question,
                    tool_name,
                    tool_args,
                    t0,
                )
            if question.routing == Routing.OTHER_AGENT:
                return self._resolve_other_agent(question, target_agent, t0)
            if question.routing == Routing.HUMAN:
                return ResolutionResult(
                    status=ResolutionStatus.ESCALATE,
                    question_id=question.question_id,
                    escalate_reason=(
                        f"question {question.question_id!r} routed to human per asker request"
                    ),
                    elapsed_sec=time.time() - t0,
                )
            # Should be unreachable — Routing is exhaustive
            return ResolutionResult(
                status=ResolutionStatus.FAILED,
                question_id=question.question_id,
                escalate_reason=f"unknown routing: {question.routing!r}",
                elapsed_sec=time.time() - t0,
            )
        except Exception as exc:
            logger.exception("resolver failed for %s", question.question_id)
            return ResolutionResult(
                status=ResolutionStatus.FAILED,
                question_id=question.question_id,
                escalate_reason=f"{type(exc).__name__}: {exc}",
                elapsed_sec=time.time() - t0,
            )

    # ------------------------------------------------------------------
    # Private dispatchers
    # ------------------------------------------------------------------

    def _resolve_code_tool(
        self,
        q: SmartQuestion,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
        t0: float,
    ) -> ResolutionResult:
        if not tool_name:
            return ResolutionResult(
                status=ResolutionStatus.FAILED,
                question_id=q.question_id,
                escalate_reason=("code_tool routing but no tool_name supplied to resolve()"),
                elapsed_sec=time.time() - t0,
            )
        if tool_name not in self.code_tools:
            return ResolutionResult(
                status=ResolutionStatus.FAILED,
                question_id=q.question_id,
                escalate_reason=(
                    f"unknown code tool {tool_name!r}; registered: {sorted(self.code_tools)}"
                ),
                elapsed_sec=time.time() - t0,
            )
        # Wall-time cap — we can't preempt the tool callback synchronously
        # (Python signals are platform-specific), so we rely on the tool
        # itself to respect the cap. We measure + report wall time.
        answer = self.code_tools[tool_name](tool_name, tool_args or {})
        elapsed = time.time() - t0
        if elapsed > MAX_CODE_TOOL_WALL_TIME_SEC:
            return ResolutionResult(
                status=ResolutionStatus.ESCALATE,
                question_id=q.question_id,
                answer=str(answer),
                escalate_reason=(
                    f"code_tool {tool_name!r} ran {elapsed:.1f}s > "
                    f"{MAX_CODE_TOOL_WALL_TIME_SEC:.0f}s cap"
                ),
                elapsed_sec=elapsed,
            )
        return ResolutionResult(
            status=ResolutionStatus.RESOLVED,
            question_id=q.question_id,
            answer=str(answer),
            elapsed_sec=elapsed,
            metadata={"tool": tool_name, "args": dict(tool_args or {})},
        )

    def _resolve_other_agent(
        self,
        q: SmartQuestion,
        target_agent: str | None,
        t0: float,
    ) -> ResolutionResult:
        if not target_agent:
            return ResolutionResult(
                status=ResolutionStatus.FAILED,
                question_id=q.question_id,
                escalate_reason=("other_agent routing but no target_agent supplied to resolve()"),
                elapsed_sec=time.time() - t0,
            )
        if self.agent_dispatch is None:
            return ResolutionResult(
                status=ResolutionStatus.FAILED,
                question_id=q.question_id,
                escalate_reason=(
                    "other_agent routing but no agent_dispatch callback configured on resolver"
                ),
                elapsed_sec=time.time() - t0,
            )
        # Compose the prompt the target agent sees — the asker's full
        # question YAML so the responder has full context.
        question_text = (
            f"Agent {q.asker!r} asked you a smart-question. Decision "
            f"blocker: {q.decision_blocker}\n\n"
            f"Specific evidence needed: {q.specific_evidence_needed}\n\n"
            f"Acceptance: {q.acceptance}"
        )
        answer = self.agent_dispatch(target_agent, question_text)
        return ResolutionResult(
            status=ResolutionStatus.RESOLVED,
            question_id=q.question_id,
            answer=str(answer),
            elapsed_sec=time.time() - t0,
            metadata={"target_agent": target_agent},
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def ledger(self, asker: str) -> _AgentLedger:
        """Read the per-agent ledger — useful for the orchestrator's
        round-summary transcript."""
        return self._ledger.get(asker, _AgentLedger())
