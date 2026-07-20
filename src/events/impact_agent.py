"""Event Impact Agent (CL-6iu7) — turns a NEW ``geo_events`` headline
into a structured tradability assessment.

Flow per event:
  1. Look up the row's matched playbook (theme set by the GDELT
     ingester) for pre-researched instrument mappings + direction hints.
  2. One strict-JSON LLM call (existing ``claude-code`` driver — rides
     the operator's subscription, no API billing).
  3. Defensive parse: extract JSON from prose/fences, schema-validate,
     clamp numeric ranges, drop unreachable instruments, force equities
     to watch-only.
  4. Write ``assessment`` + status ASSESSED (or DISMISSED with a
     rationale when the output is unusable).

The agent proposes; it never trades. The consumer half (confirmation +
strategy, sibling work) decides what to do with ASSESSED rows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.events.playbooks import (
    DEFAULT_PLAYBOOKS_PATH,
    INSTRUMENT_RE,
    Playbook,
    all_tradable_instruments,
    load_playbooks,
)
from src.research.llm import Message, get_client
from src.research.llm.client import LLMClient

logger = logging.getLogger(__name__)

#: Sonnet-class is plenty for single-headline classification and keeps
#: per-event subscription load small; override via EVENT_IMPACT_MODEL.
DEFAULT_MODEL = "claude-sonnet-4-6"

VALID_EVENT_DIRECTIONS = frozenset({"bullish", "bearish", "neutral"})
VALID_HORIZONS = frozenset({"minutes", "hours", "days"})
VALID_AFFECTED_KINDS = frozenset({"oanda", "fx", "equity_watch", "polymarket"})
VALID_AFFECTED_DIRECTIONS = frozenset({"long", "short", "watch"})

_SYSTEM_PROMPT = """\
You are the Event Impact Agent for an FX/CFD trading system. You receive
one news headline (from GDELT, so it is already ~15+ minutes old) plus a
pre-researched playbook for its matched theme. Your job is honest
interpretation, not speed: decide whether this specific headline is a
tradable macro event, which instruments it touches, and how urgent it is.

Rules:
- Respond with ONE JSON object and NOTHING else. No markdown fences, no
  commentary.
- Only propose instruments from the playbook mapping, or other clearly
  reachable OANDA CFD/FX symbols (format like EUR_USD, BCO_USD,
  XAU_USD, SPX500_USD).
- Equity tickers are ALERT-ONLY: kind "equity_watch", direction "watch".
  Never mark an equity long or short.
- Be skeptical: opinion pieces, retrospectives, market-recap articles,
  and stale news deserve direction "neutral", low urgency, and an empty
  or watch-only affected list. Confidence above 0.7 requires a concrete,
  new, physical or policy event.
- Be selective in "affected": list only instruments THIS specific event
  genuinely moves — do not copy the entire playbook mapping.
- "direction" is the overall risk impulse of the event itself
  (bullish/bearish for risk assets, neutral if unclear); per-instrument
  direction lives in "affected".

JSON schema (all keys required):
{
  "core_event": "<one sentence: what actually happened>",
  "direction": "bullish" | "bearish" | "neutral",
  "urgency": <int 1-10>,
  "horizon": "minutes" | "hours" | "days",
  "confidence": <float 0.0-1.0>,
  "affected": [
    {"instrument": "<symbol>",
     "kind": "oanda" | "fx" | "equity_watch" | "polymarket",
     "direction": "long" | "short" | "watch",
     "reason": "<one line>"}
  ],
  "rationale": "<2-3 sentences: why this assessment>"
}
"""


@dataclass
class AssessmentResult:
    """Per-event outcome, for the entrypoint's one-line summaries."""

    event_id: int
    headline: str
    theme: str | None
    status: str  # ASSESSED | DISMISSED
    assessment: dict[str, Any]

    def summary_line(self) -> str:
        if self.status != "ASSESSED":
            reason = self.assessment.get("rationale", "")
            return (
                f"event id={self.event_id} DISMISSED ({reason[:80]}) "
                f"| {self.headline[:70]}"
            )
        a = self.assessment
        affected = ",".join(
            f"{x['instrument']}:{x['direction']}" for x in a.get("affected", [])
        ) or "-"
        return (
            f"event id={self.event_id} ASSESSED theme={self.theme} "
            f"dir={a.get('direction')} urg={a.get('urgency')} "
            f"hor={a.get('horizon')} conf={a.get('confidence'):.2f} "
            f"affected=[{affected}] | {self.headline[:70]}"
        )


# ---------------------------------------------------------------------- #
# Defensive parsing
# ---------------------------------------------------------------------- #


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Pull the first balanced top-level JSON object out of LLM text.

    Tolerates markdown fences and prose around the object. Raises
    ``ValueError`` when nothing parseable is found.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence line and any trailing fence.
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    if start < 0:
        msg = "no JSON object in LLM output"
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
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    msg = f"JSON object failed to parse: {exc}"
                    raise ValueError(msg) from exc
                if not isinstance(obj, dict):
                    msg = "top-level JSON is not an object"
                    raise ValueError(msg)
                return obj
    msg = "unbalanced JSON object in LLM output"
    raise ValueError(msg)


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(float(value)))))


def _clamp_float(value: Any, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def normalise_assessment(
    payload: dict[str, Any],
    headline: str,
    playbook: Playbook | None,
    fallback_tradables: set[str],
) -> dict[str, Any]:
    """Schema-validate + clamp a parsed LLM payload into the shared
    ``assessment`` shape. Raises ``ValueError`` on unrecoverable schema
    violations (→ row is DISMISSED); silently drops bad ``affected``
    entries and clamps out-of-range numbers.
    """
    direction = str(payload.get("direction", "")).strip().lower()
    if direction not in VALID_EVENT_DIRECTIONS:
        msg = f"invalid direction {direction!r}"
        raise ValueError(msg)

    horizon = str(payload.get("horizon", "")).strip().lower().rstrip("s") + "s"
    if horizon not in VALID_HORIZONS:
        msg = f"invalid horizon {payload.get('horizon')!r}"
        raise ValueError(msg)

    try:
        urgency = _clamp_int(payload.get("urgency"), 1, 10)
        confidence = _clamp_float(payload.get("confidence"), 0.0, 1.0)
    except (TypeError, ValueError) as exc:
        msg = f"non-numeric urgency/confidence: {exc}"
        raise ValueError(msg) from exc

    core_event = str(payload.get("core_event", "")).strip() or headline
    rationale = str(payload.get("rationale", "")).strip()

    raw_affected = payload.get("affected")
    if raw_affected is None:
        raw_affected = []
    if not isinstance(raw_affected, list):
        msg = "'affected' is not a list"
        raise ValueError(msg)

    playbook_instruments = (
        {i.instrument for i in playbook.instruments} if playbook else set()
    )
    affected: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in raw_affected:
        if not isinstance(entry, dict):
            continue
        instrument = str(entry.get("instrument", "")).strip()
        kind = str(entry.get("kind", "")).strip().lower()
        adir = str(entry.get("direction", "")).strip().lower()
        reason = str(entry.get("reason", "")).strip()
        if not instrument or kind not in VALID_AFFECTED_KINDS:
            logger.debug("dropping affected entry (bad kind): %r", entry)
            continue
        if adir not in VALID_AFFECTED_DIRECTIONS:
            adir = "watch"
        if kind == "equity_watch":
            adir = "watch"  # equities are alert-only, always
        if kind in ("oanda", "fx"):
            instrument = instrument.upper().replace("/", "_").replace("-", "_")
            if not INSTRUMENT_RE.match(instrument):
                logger.debug("dropping unreachable tradable: %r", entry)
                continue
            # Whitelist gate: playbook mapping first, any playbook's
            # tradable second. An OANDA-shaped symbol outside both is
            # demoted to watch rather than dropped — the shape says
            # reachable, but nobody pre-researched it.
            known = instrument in playbook_instruments or instrument in fallback_tradables
            if not known and adir != "watch":
                logger.debug(
                    "demoting un-vetted tradable %s to watch", instrument,
                )
                adir = "watch"
        if instrument in seen:
            continue
        seen.add(instrument)
        affected.append({
            "instrument": instrument,
            "kind": kind,
            "direction": adir,
            "reason": reason,
        })

    return {
        "core_event": core_event,
        "direction": direction,
        "urgency": urgency,
        "horizon": horizon,
        "confidence": confidence,
        "affected": affected,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------- #
# The agent
# ---------------------------------------------------------------------- #


class EventImpactAgent:
    """Assess NEW ``geo_events`` rows via the existing LLM stack."""

    def __init__(
        self,
        engine: Engine,
        client: LLMClient | None = None,
        model: str = DEFAULT_MODEL,
        playbooks_path: Path | str = DEFAULT_PLAYBOOKS_PATH,
        max_tokens: int = 2000,
    ) -> None:
        self.engine = engine
        self.client = client if client is not None else get_client("claude-code")
        self.model = model
        self.max_tokens = max_tokens
        self.playbooks = load_playbooks(playbooks_path)
        self._fallback_tradables = all_tradable_instruments(self.playbooks)

    # -- prompt ---------------------------------------------------------

    def _playbook_context(self, theme: str | None) -> str:
        pb = self.playbooks.get(theme or "")
        if pb is None:
            lines = [
                "No playbook matched this headline's theme. Only propose",
                "instruments from this cross-theme whitelist (or clearly",
                "reachable OANDA symbols), and lean neutral/low-confidence:",
                ", ".join(sorted(self._fallback_tradables)),
            ]
            return "\n".join(lines)
        lines = [
            f"Matched playbook: {pb.key} — {pb.name}",
            pb.description,
            "Instrument mapping (direction = historical hint, not an order):",
        ]
        for inst in pb.instruments:
            lines.append(
                f"  - {inst.instrument} [{inst.kind}] hint={inst.direction}: "
                f"{inst.rationale}"
            )
        return "\n".join(lines)

    def _user_prompt(self, row: dict[str, Any]) -> str:
        return (
            f"HEADLINE: {row['headline']}\n"
            f"URL: {row.get('url') or 'n/a'}\n"
            f"SEEN AT (UTC): {row.get('seen_at')}\n"
            f"THEME: {row.get('theme') or 'unmatched'}\n\n"
            f"{self._playbook_context(row.get('theme'))}\n\n"
            "Assess this event now. Respond with the JSON object only."
        )

    # -- assessment -----------------------------------------------------

    def assess_row(self, row: dict[str, Any]) -> AssessmentResult:
        """LLM-assess one row (no DB write). Returns the outcome."""
        theme = row.get("theme")
        playbook = self.playbooks.get(theme or "")
        try:
            resp = self.client.complete(
                messages=[
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(role="user", content=self._user_prompt(row)),
                ],
                model=self.model,
                max_tokens=self.max_tokens,
            )
            payload = extract_json_object(resp.text)
            assessment = normalise_assessment(
                payload, row["headline"], playbook, self._fallback_tradables,
            )
            status = "ASSESSED"
        except Exception as exc:
            # Parse/validation/LLM failure → DISMISSED with a rationale;
            # the row is never retried, so the reason must be on it.
            logger.warning(
                "impact agent dismissed event id=%s: %s", row.get("id"), exc,
            )
            assessment = {
                "rationale": f"impact agent failure: {exc}"[:500],
            }
            status = "DISMISSED"
        return AssessmentResult(
            event_id=int(row["id"]),
            headline=str(row["headline"]),
            theme=theme,
            status=status,
            assessment=assessment,
        )

    def assess_new_events(self, limit: int = 20) -> list[AssessmentResult]:
        """Process up to ``limit`` NEW rows (newest first — freshest
        events are the only ones with any edge left) and persist each
        outcome. Returns per-event results for summary logging."""
        with self.engine.connect() as conn:
            rows = [
                dict(r._mapping)
                for r in conn.execute(text(
                    "SELECT id, seen_at, headline, url, theme "
                    "FROM geo_events WHERE status = 'NEW' "
                    "ORDER BY seen_at DESC LIMIT :lim",
                ), {"lim": limit})
            ]
        results: list[AssessmentResult] = []
        for row in rows:
            result = self.assess_row(row)
            self._persist(result)
            results.append(result)
        return results

    def _persist(self, result: AssessmentResult) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(
                "UPDATE geo_events "
                "SET assessment = :assessment, status = :status, "
                "    status_updated_at = :now "
                "WHERE id = :id AND status = 'NEW'",
            ), {
                "assessment": json.dumps(result.assessment),
                "status": result.status,
                "now": datetime.now(UTC),
                "id": result.event_id,
            })
