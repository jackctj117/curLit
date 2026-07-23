"""Reviewer agents (Bull + Bear) — share scaffolding, differ in role
and how they declare their final position.

Both reviewers:
  * Read the candidate report (metrics + per-fold breakdown)
  * Read REVIEW_RULES.md
  * Optionally query the KnowledgeRetriever for historical precedent
  * Produce a structured PROMOTE_CASE / REJECT_CASE markdown document
  * Declare a position keyword the verdict engine parses

The asymmetric prompts are the safeguard against two-LLM convergence:
Bull is told to make the strongest PROMOTE case with cited evidence;
Bear is told adversarially to find why it will fail. Neither is told
to "review fairly" — that's the verdict engine's job.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.research.agents.base import Agent, AgentResponse

logger = logging.getLogger(__name__)


class Position(StrEnum):
    """Final position keywords. The verdict engine parses these."""

    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


@dataclass
class ReviewerResult:
    """Bull/Bear output with the parsed final position attached.

    The verdict engine (CL-ath2) reads ``position`` directly without
    re-parsing the agent's text — keeps the LLM output contract
    auditable but the verdict logic deterministic.
    """

    response: AgentResponse
    position: Position
    raw_text: str  # full markdown the agent produced


# Match a final-position declaration anywhere in the text. Tolerant of
# the markdown bold formatting the prompt asks for, the explicit
# FINAL_POSITION: prefix, and trailing punctuation.
_POSITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\*\*FINAL_POSITION\*\*\s*:\s*(PROMOTE|REJECT|ABSTAIN)", re.IGNORECASE),
    re.compile(r"FINAL[_\s]POSITION\s*:\s*\*?\*?(PROMOTE|REJECT|ABSTAIN)\*?\*?", re.IGNORECASE),
    re.compile(r"\*\*(PROMOTE|REJECT|ABSTAIN)\*\*", re.IGNORECASE),
]


def parse_position(text: str) -> Position:
    """Extract the agent's declared position from its markdown output.

    Tries patterns in priority order: explicit FINAL_POSITION marker
    first, then the bare bold keyword as fallback. Within a pattern,
    takes the LAST match (the operative final declaration after any
    earlier exploratory mentions). Returns ABSTAIN as the safe default
    if no pattern matches — caution-default per REVIEW_RULES.md.
    """
    for pattern in _POSITION_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return Position(matches[-1].group(1).upper())
    logger.warning("No final-position keyword found in reviewer output — defaulting to ABSTAIN")
    return Position.ABSTAIN


class Reviewer(Agent):
    """Common scaffolding for Bull + Bear. Subclasses provide the role
    name (system prompt loads from configs/research_prompts/{role}.md)
    and may override ``review`` for role-specific post-processing."""

    def review(
        self,
        candidate_report: str,
        review_rules: str,
        round_name: str,
        prior_transcript: str | None = None,
    ) -> ReviewerResult:
        """Run one review round.

        ``round_name`` selects the round-specific instructions appended
        to the base system prompt. The base prompt covers all three
        round formats; this just nudges the agent toward the current
        round's output shape.
        """
        round_instruction = self._round_instruction(round_name)
        context_files: dict[str, Any] = {
            "REVIEW_RULES.md": review_rules,
            "candidate_report.json": candidate_report,
        }
        if prior_transcript is not None:
            context_files["debate_transcript_so_far"] = prior_transcript

        resp = self.run(
            user_prompt=(
                f"This is round {round_name!r}. Produce the output specified "
                f"by your system prompt for this round."
            ),
            context_files=context_files,
            extra_system=round_instruction,
        )
        position = parse_position(resp.text)
        return ReviewerResult(response=resp, position=position, raw_text=resp.text)

    @staticmethod
    def _round_instruction(round_name: str) -> str:
        """Round-specific reminder appended to the system prompt. Short
        because the system prompt already covers each round's format —
        this just tells the agent which round it's in."""
        return {
            "initial_positions": (
                "Round 1 — initial positions. Produce the structured "
                "PROMOTE_CASE / REJECT_CASE markdown with one rule per line "
                "in sections A through D, then declare your **FINAL_POSITION**."
            ),
            "rebuttal": (
                "Round 3 — rebuttal. The other reviewer's case is in the "
                "context. Engage specifically with each cited claim, then "
                "declare your **revised final position**."
            ),
            "final_position": (
                "Round 4 — final position. Declare **FINAL_POSITION** on a "
                "single line: PROMOTE, REJECT, or ABSTAIN. One sentence "
                "rationale only."
            ),
        }.get(round_name, "")


class BullReviewer(Reviewer):
    """Argues PROMOTE with evidence cited from REVIEW_RULES.md."""


class BearReviewer(Reviewer):
    """Adversarial; argues REJECT, hunts for failure modes (look-ahead,
    overfitting, regime concentration, sample-size insufficiency,
    unrealistic costs). The asymmetric system prompt is what makes
    Bear/Bull genuinely adversarial rather than mutually agreeing."""
