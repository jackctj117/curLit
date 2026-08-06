"""Reflective self-tuning loop (CL-g8jl) — the system reviews its own track
record and proposes concrete tuning, which the operator approves or ignores.

Reads the finalised outcomes (:mod:`src.events.outcome_tracker`), aggregates
performance by dimension (theme, action, hop depth, confidence bucket, niche
vs not, survived-red-team vs not), and hands that summary to an LLM that
proposes SPECIFIC, conservative config changes — name the knob, the direction,
and the data-grounded rationale.

Deliberately ADVISORY: it never auto-edits config. It produces a proposal (and
can push it to Telegram); the operator decides. And it refuses to opine on thin
data — below ``min_sample`` finalised outcomes it returns ``insufficient_data``
rather than over-fitting noise.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from src.events.impact_agent import extract_json_object
from src.research.llm import Message
from src.research.llm.client import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_MODEL = "claude-sonnet-4-6"
#: Finalised outcomes required before the reviewer will opine.
DEFAULT_MIN_SAMPLE = 20

_FINALISED = ("win", "loss", "flat")


def _bucket_confidence(c: float | None) -> str:
    if c is None:
        return "unknown"
    if c >= 0.7:
        return "high"
    if c >= 0.5:
        return "medium"
    return "low"


def _hop_bucket(hop: int | None) -> str:
    if hop is None:
        return "unknown"
    return "1-2" if hop <= 2 else "3+"


def _group_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Win/loss/flat counts, win rate and average return for a group."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    wins = sum(1 for r in rows if r["outcome"] == "win")
    losses = sum(1 for r in rows if r["outcome"] == "loss")
    flats = n - wins - losses
    rets = [float(r["return_pct"]) for r in rows if r["return_pct"] is not None]
    avg_ret = sum(rets) / len(rets) if rets else None
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate": round(wins / n, 3),
        "avg_return": round(avg_ret, 4) if avg_ret is not None else None,
    }


def _by(rows: Sequence[Mapping[str, Any]], keyfn: Any) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(keyfn(r)), []).append(r)
    return {k: _group_stats(v) for k, v in sorted(groups.items())}


def aggregate_outcomes(engine: Any) -> dict[str, Any]:
    """Aggregate FINALISED outcomes into an overall + by-dimension summary."""
    with engine.connect() as conn:
        rows = [
            dict(r._mapping)
            for r in conn.execute(
                text(
                    "SELECT outcome, return_pct, theme, action, direction, confidence, "
                    "       is_niche, hop_count, red_team_survived "
                    "FROM idea_outcomes WHERE outcome IN ('win', 'loss', 'flat')",
                )
            )
        ]
    return {
        "sample": len(rows),
        "overall": _group_stats(rows),
        "by_theme": _by(rows, lambda r: r["theme"] or "unknown"),
        "by_action": _by(rows, lambda r: r["action"] or "unknown"),
        "by_direction": _by(rows, lambda r: r["direction"] or "unknown"),
        "by_hop": _by(rows, lambda r: _hop_bucket(r["hop_count"])),
        "by_confidence": _by(
            rows,
            lambda r: _bucket_confidence(
                float(r["confidence"]) if r["confidence"] is not None else None
            ),
        ),
        "by_niche": _by(rows, lambda r: "niche" if r["is_niche"] else "core"),
        "by_red_team": _by(rows, lambda r: "survived" if r["red_team_survived"] else "n/a"),
    }


@dataclass
class ReviewResult:
    status: str  # "ok" | "insufficient_data"
    sample: int
    summary: dict[str, Any] = field(default_factory=dict)
    proposal: dict[str, Any] | None = None
    raw_text: str = ""

    def to_telegram(self) -> str:
        """Compact operator-facing message."""
        if self.status == "insufficient_data":
            return (
                f"🔬 <b>Reflective review</b>\nInsufficient finalised outcomes "
                f"to tune yet ({self.sample}). Building the track record."
            )
        p = self.proposal or {}
        lines = [f"🔬 <b>Reflective review</b> — {self.sample} finalised ideas"]
        if p.get("overall_read"):
            lines.append(str(p["overall_read"]))
        changes = p.get("proposed_changes") or []
        if changes:
            lines.append("\n<b>Proposed tuning (you approve):</b>")
            for c in changes[:8]:
                if isinstance(c, dict):
                    lines.append(
                        f"• <b>{c.get('knob', '?')}</b>: {c.get('change', '')} "
                        f"— {c.get('rationale', '')}"
                    )
        return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are a quant reviewing an event-driven desk's IDEA TRACK RECORD to propose
tuning. You get win rates and average returns broken down by dimension.

Your job: identify what's working and what isn't, then propose SPECIFIC,
CONSERVATIVE config changes. Name the knob, the direction, and the
data-grounded reason. Available knobs include: min_urgency, min_confidence
(confluence Gate A), asymmetry_threshold and the hop/torque/smallness weights
(niche scoring), NICHE_MAX_CYCLES, per-theme trust (suggest de-prioritising a
theme), confidence calibration.

Rules:
- Do NOT over-fit small samples. If a group has few ideas, say so and DON'T
  propose a change off it.
- Prefer a few high-confidence changes over many speculative ones. It's fine to
  propose nothing if nothing is clearly actionable.
- Every change must cite the number that justifies it.

Respond with ONE JSON object and nothing else:
{
  "overall_read": "<2-3 sentences: how is the desk doing>",
  "findings": ["<data-grounded observation>", ...],
  "proposed_changes": [
    {"knob": "<config knob>", "change": "<specific change>",
     "rationale": "<the number that justifies it>"}
  ]
}
Leave "proposed_changes" empty when nothing is clearly warranted.
"""


class ReflectiveReviewer:
    """Reviews finalised outcomes → an advisory tuning proposal."""

    def __init__(
        self,
        engine: Any,
        client: LLMClient | None = None,
        model: str | None = None,
        min_sample: int | None = None,
        max_tokens: int = 1600,
    ) -> None:
        self.engine = engine
        if client is not None:
            self.client = client
        else:
            from src.research.llm import get_client  # noqa: PLC0415

            self.client = get_client("claude-code")
        self.model = model or os.environ.get("NICHE_REVIEW_MODEL", DEFAULT_REVIEW_MODEL)
        self.max_tokens = max_tokens
        if min_sample is not None:
            self.min_sample = min_sample
        else:
            try:
                self.min_sample = int(os.environ.get("NICHE_REVIEW_MIN_SAMPLE", DEFAULT_MIN_SAMPLE))
            except ValueError:
                self.min_sample = DEFAULT_MIN_SAMPLE

    def review(self) -> ReviewResult:
        """Aggregate + (if enough data) get an LLM tuning proposal. Fail-soft:
        an LLM/parse failure returns the aggregation with no proposal."""
        summary = aggregate_outcomes(self.engine)
        sample = summary["sample"]
        if sample < self.min_sample:
            return ReviewResult(status="insufficient_data", sample=sample, summary=summary)
        import json  # noqa: PLC0415

        try:
            resp = self.client.complete(
                # Single-shot JSON over injected data — no toolset (CL-scup).
                no_tools=True,
                messages=[
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(
                        role="user",
                        content=(
                            "Track record by dimension (JSON):\n"
                            f"{json.dumps(summary, indent=2)}\n\n"
                            "Review it and propose tuning. JSON only."
                        ),
                    ),
                ],
                model=self.model,
                max_tokens=self.max_tokens,
            )
            proposal = extract_json_object(resp.text)
            raw = resp.text
        except Exception as exc:
            logger.warning("reflective review: LLM/parse failed: %s", str(exc)[:200])
            return ReviewResult(status="ok", sample=sample, summary=summary, proposal=None)
        return ReviewResult(
            status="ok", sample=sample, summary=summary, proposal=proposal, raw_text=raw
        )
