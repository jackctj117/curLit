"""Adversarial red-team critic for niche ideas (CL-3v56).

The niche pass generates high-torque, multi-hop ideas; this is the opposing
half — a hostile analyst whose only job is to DESTROY each surviving idea
before it reaches the operator. It attacks on the ways narrative trades
actually fail: already priced-in / the move already happened, liquidity or
borrow traps, a mundane alternative explanation, the base rate of similar
events fizzling, a thesis-invalidating fact, or an overcrowded consensus.

Only ideas that SURVIVE the attack are surfaced; refuted ones are dropped, and
survivors carry the strongest surviving counter-argument so the operator sees
the bear case, not just the bull one.

Deliberately an INDEPENDENT adversary: it runs on the claude-code subscription
(free) even when the generator ran on the Kimi API — a different model
attacking a different model's ideas, which is where adversarial review gets its
teeth. One batched call per event (all survivors at once) keeps it cheap.

FAIL-OPEN: any transport/parse failure leaves the ideas untouched — a critic
outage must not silently wipe every idea. OPT-IN via ``NICHE_CRITIC_ENABLED``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.events._util import env_flag
from src.events.impact_agent import extract_json_object
from src.research.llm import Message
from src.research.llm.client import LLMClient

logger = logging.getLogger(__name__)

#: Sonnet-class is plenty for a focused critique; overridable via
#: NICHE_CRITIC_MODEL.
DEFAULT_CRITIC_MODEL = "claude-sonnet-4-6"

VALID_VERDICTS = frozenset({"confirmed", "weakened", "refuted"})


@dataclass(frozen=True)
class CritiqueVerdict:
    """The red team's ruling on one idea."""

    ticker: str
    verdict: str  # confirmed | weakened | refuted
    strongest_attack: str
    adjusted_confidence: float | None
    survives: bool  # False only for 'refuted'


_SYSTEM_PROMPT = """\
You are the RED TEAM for an event-driven trading desk. You receive niche trade
ideas another analyst already generated and liked. Your ONLY job is to try to
DESTROY each one. Be skeptical, specific, and honest — a plausible-but-wrong
idea that reaches the operator costs real money.

Attack each idea on whichever of these actually bite:
- Already priced in / the move already happened (if the name is already up or
  down a lot on this news, chasing is a trap).
- Liquidity or borrow trap (too thin to trade, hard/expensive to short).
- A mundane ALTERNATIVE explanation that makes the thesis unnecessary.
- Base rate: events of this type usually fizzle / mean-revert.
- A thesis-INVALIDATING fact (the company's real exposure is smaller than
  claimed, it's hedged, the chain link is weaker than stated).
- Overcrowded consensus (everyone already has this on; no edge left).

Then rule on each idea:
- "refuted"  — a fatal flaw; it should NOT be surfaced.
- "weakened" — survives but with a real caveat; lower the confidence.
- "confirmed" — the attack failed; the idea holds up.

Respond with ONE JSON object and NOTHING else:
{
  "verdicts": [
    {"ticker": "<same ticker>",
     "verdict": "confirmed" | "weakened" | "refuted",
     "strongest_attack": "<the single best argument against it, one line>",
     "adjusted_confidence": <float 0.0-1.0: your post-attack confidence>}
  ]
}
Include exactly one verdict per idea, echoing the ticker. Be willing to refute —
if most of these are weak, say so.
"""


class AdversarialCritic:
    """Batched red-team pass over an event's surviving niche ideas."""

    def __init__(
        self,
        client: LLMClient | None = None,
        model: str | None = None,
        enabled: bool | None = None,
        max_tokens: int = 1200,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            from src.research.llm import get_client  # noqa: PLC0415
            self.client = get_client("claude-code")
        self.model = model or os.environ.get(
            "NICHE_CRITIC_MODEL", DEFAULT_CRITIC_MODEL,
        )
        self.enabled = (
            enabled if enabled is not None
            else env_flag("NICHE_CRITIC_ENABLED", default=False)
        )
        self.max_tokens = max_tokens

    def _user_prompt(
        self, ideas: list[Any], event_row: Mapping[str, Any],
    ) -> str:
        lines = [
            f"EVENT: {event_row.get('headline')}",
            f"THEME: {event_row.get('theme') or 'unmatched'}",
            "",
            "IDEAS TO ATTACK:",
        ]
        for idea in ideas:
            lines.append(
                f"- {idea.ticker} ({idea.company_name}) | {idea.action} | "
                f"hop {idea.hop_count} | torque: {idea.torque_reason} | "
                f"thesis: {idea.rationale}",
            )
        lines.append("\nAttack each and return the verdicts JSON only.")
        return "\n".join(lines)

    def critique(
        self, ideas: list[Any], event_row: Mapping[str, Any],
    ) -> dict[str, CritiqueVerdict]:
        """One batched red-team call → ``{ticker: CritiqueVerdict}``.

        FAILS OPEN: a transport/parse failure returns ``{}`` so the caller
        keeps every idea (no verdict → survives). Never raises.
        """
        if not ideas:
            return {}
        try:
            resp = self.client.complete(
                messages=[
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(role="user",
                            content=self._user_prompt(ideas, event_row)),
                ],
                model=self.model,
                max_tokens=self.max_tokens,
            )
            payload = extract_json_object(resp.text)
        except Exception as exc:
            logger.warning(
                "red-team critic failed — failing OPEN (all %d survive): %s",
                len(ideas), str(exc)[:200],
            )
            return {}

        raw = payload.get("verdicts")
        if not isinstance(raw, list):
            return {}
        verdicts: dict[str, CritiqueVerdict] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).strip().upper()
            verdict = str(entry.get("verdict", "")).strip().lower()
            if not ticker or verdict not in VALID_VERDICTS:
                continue
            raw_conf = entry.get("adjusted_confidence")
            adj: float | None = None
            if raw_conf is not None:
                try:
                    adj = max(0.0, min(1.0, float(raw_conf)))
                except (TypeError, ValueError):
                    adj = None
            verdicts[ticker] = CritiqueVerdict(
                ticker=ticker,
                verdict=verdict,
                strongest_attack=str(entry.get("strongest_attack", "")).strip(),
                adjusted_confidence=adj,
                survives=verdict != "refuted",
            )
        return verdicts

    def apply(
        self, ideas: list[Any], event_row: Mapping[str, Any],
    ) -> list[Any]:
        """Critique ``ideas`` and return the survivors, mutated in place:
        refuted ideas are dropped; weakened/confirmed survivors carry the
        strongest surviving attack (``red_team_note``) and a possibly-lowered
        confidence. A ticker with no verdict survives untouched (fail-open)."""
        if not ideas:
            return ideas
        verdicts = self.critique(ideas, event_row)
        if not verdicts:
            return ideas  # fail-open / nothing to apply
        survivors: list[Any] = []
        dropped = 0
        for idea in ideas:
            v = verdicts.get(idea.ticker.upper())
            if v is None:
                survivors.append(idea)  # no ruling → keep
                continue
            if not v.survives:
                dropped += 1
                logger.info(
                    "red-team: REFUTED %s (%s) — %s",
                    idea.ticker, idea.company_name, v.strongest_attack[:100],
                )
                continue
            idea.red_team_note = v.strongest_attack
            idea.red_team_verdict = v.verdict
            if v.adjusted_confidence is not None:
                idea.confidence = min(idea.confidence, v.adjusted_confidence)
            survivors.append(idea)
        logger.info(
            "red-team: event id=%s — %d critiqued, %d survived, %d refuted",
            event_row.get("id"), len(ideas), len(survivors), dropped,
        )
        return survivors
