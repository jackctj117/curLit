"""Fast event triage (CL-cunh) — a cheap Haiku pre-filter in front of the
expensive Event Impact Agent.

The impact agent fires a ~250-line system prompt plus a full structured
assessment (trade ideas, concrete levels, fade candidates) on EVERY new
``geo_event`` via Sonnet — including the opinion columns, market recaps, and
stale/retrospective articles the prompt itself says to score neutral. That is
wasted subscription quota and latency.

This triage tier scores a whole BATCH of headlines in ONE small, fast Haiku
call (relevance 0-10 + tradable) and lets the impact agent skip the full
assessment for the clearly-irrelevant ones. Only escalated events pay for the
expensive Sonnet pass — so 20 headlines cost one cheap Haiku call plus a
handful of full assessments, not 20 full assessments.

Design rules (mirror the impact agent's hard-won "never silently drop a real
event" lesson):

* FAIL OPEN — if the triage call raises or returns garbage, every event
  escalates (empty verdict map), so a triage outage can never lose events.
* CONSERVATIVE bar — only clearly-irrelevant events (relevance below
  ``min_relevance``, default 4) are held back; the real urgency/confidence
  gate still lives in the full assessment + confluence downstream.
* OPT-IN — disabled unless ``EVENT_TRIAGE_ENABLED`` is truthy, so existing
  callers keep their exact behavior until the operator turns it on.
* Anthropic-only — runs Haiku on the SAME ``claude-code`` subscription client
  as the impact agent (no new provider, no API billing, no Grok). Haiku 4.5 is
  the fast/cheap tier; correctness on "is this a real market-moving event?" is
  well within its range.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from src.events._util import env_flag
from src.research.llm import Message, get_client
from src.research.llm.client import LLMClient

logger = logging.getLogger(__name__)

#: Fast/cheap tier for batch relevance scoring; matches the client pricing key
#: so cost logging is accurate. Override via EVENT_TRIAGE_MODEL.
DEFAULT_TRIAGE_MODEL = "claude-haiku-4-5-20251001"

#: Minimum triage relevance (0-10) to escalate to the full assessment. A LOW
#: bar on purpose — this only drops obvious junk; the true urgency>=7 gate is
#: downstream. Override via EVENT_TRIAGE_MIN_RELEVANCE.
DEFAULT_MIN_RELEVANCE = 4

_TRIAGE_SYSTEM_PROMPT = """\
You are a FAST triage filter for a geopolitical / macro trading system. You
receive a batch of news headlines (already ~15+ minutes old, from GDELT or an
X watchlist). For EACH headline decide, in one pass, whether it is worth a
full, expensive downstream analysis.

Score each headline:
- relevance (integer 0-10): is this a concrete, NEW, potentially market-moving
  macro / geopolitical event? HIGH (7-10): war or military escalation,
  sanctions / export bans, supply disruption (mines, chips, shipping,
  chokepoints), coups / nationalization, central-bank or major policy shocks,
  chokepoint closures. LOW (0-3): opinion / analysis columns, retrospectives,
  market recaps, routine or scheduled updates, vague commentary, human-interest,
  or anything already old news.
- tradable (true/false): could a trader plausibly act on this today?

Be decisive and fast. When genuinely unsure, score relevance 4+ (let the full
analysis decide) — do not spend words deliberating.

Respond with ONE JSON array and NOTHING else — no markdown fences, no prose.
One object per input id, echoing the same integer ids:
[{"id": <int>, "relevance": <int 0-10>, "tradable": <true|false>, "reason": "<= 8 words"}]
"""


@dataclass(frozen=True)
class TriageVerdict:
    """One headline's fast triage outcome."""

    event_id: int
    relevance: int
    tradable: bool
    reason: str
    escalate: bool  # True → run the full impact assessment


def extract_json_array(raw_text: str) -> list[Any]:
    """Pull the first balanced top-level JSON array out of LLM text.

    Tolerates markdown fences and surrounding prose. Raises ``ValueError``
    when nothing parseable is found (caller fails open).
    """
    import json  # noqa: PLC0415 — local keeps module import surface minimal

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("[")
    if start < 0:
        msg = "no JSON array in triage output"
        raise ValueError(msg)
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                obj = json.loads(candidate)
                if not isinstance(obj, list):
                    msg = "top-level JSON is not an array"
                    raise ValueError(msg)
                return obj
    msg = "unbalanced JSON array in triage output"
    raise ValueError(msg)


class EventTriage:
    """Batch relevance scorer that gates the expensive impact assessment."""

    def __init__(
        self,
        client: LLMClient | None = None,
        model: str | None = None,
        min_relevance: int | None = None,
        enabled: bool | None = None,
        max_tokens: int = 900,
    ) -> None:
        self.client = client if client is not None else get_client("claude-code")
        self.model = model or os.environ.get(
            "EVENT_TRIAGE_MODEL",
            DEFAULT_TRIAGE_MODEL,
        )
        if min_relevance is not None:
            self.min_relevance = min_relevance
        else:
            try:
                self.min_relevance = int(
                    os.environ.get(
                        "EVENT_TRIAGE_MIN_RELEVANCE",
                        str(DEFAULT_MIN_RELEVANCE),
                    ),
                )
            except ValueError:
                self.min_relevance = DEFAULT_MIN_RELEVANCE
        self.enabled = (
            enabled if enabled is not None else env_flag("EVENT_TRIAGE_ENABLED", default=False)
        )
        self.max_tokens = max_tokens

    def _user_prompt(self, rows: list[dict[str, Any]]) -> str:
        lines = ["Score each headline. Return the JSON array only.", ""]
        for row in rows:
            theme = row.get("theme") or "unmatched"
            headline = str(row.get("headline", "")).strip()
            lines.append(f"[{row.get('id')}] theme={theme} :: {headline}")
        return "\n".join(lines)

    def score_batch(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[int, TriageVerdict]:
        """Score a batch of NEW-event rows → ``{event_id: TriageVerdict}``.

        FAILS OPEN: any transport/parse failure returns ``{}`` so every event
        escalates. An event id missing from the response is also absent from
        the map (→ the caller escalates it). Never raises.
        """
        if not rows:
            return {}
        try:
            resp = self.client.complete(
                messages=[
                    Message(role="system", content=_TRIAGE_SYSTEM_PROMPT),
                    Message(role="user", content=self._user_prompt(rows)),
                ],
                model=self.model,
                max_tokens=self.max_tokens,
            )
            items = extract_json_array(resp.text)
        except Exception as exc:
            logger.warning(
                "event triage failed — failing OPEN, all %d escalate: %s",
                len(rows),
                str(exc)[:200],
            )
            return {}

        verdicts: dict[int, TriageVerdict] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                eid = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                relevance = max(
                    0,
                    min(10, int(round(float(item.get("relevance", 0))))),
                )
            except (TypeError, ValueError):
                relevance = 10  # unparseable score → escalate (fail safe)
            tradable = bool(item.get("tradable", True))
            reason = str(item.get("reason", "")).strip()[:120]
            verdicts[eid] = TriageVerdict(
                event_id=eid,
                relevance=relevance,
                tradable=tradable,
                reason=reason,
                escalate=relevance >= self.min_relevance,
            )
        return verdicts
