"""Truth Social post classifier (CL-s9as) — RESEARCH ONLY.

Labels each ingested post for the event study: relevance, topic, tone,
named entities, explicit market language. Runs on the Haiku triage tier
(same subscription client as event triage — cheap, batched). Rows are
stamped with ``classifier_version`` so a future reclassification pass
can be diffed against these labels.

DELIBERATE EXCLUSIONS (operator-approved scope): no family-holdings or
portfolio-alignment features of any kind, no trading signals, no alerts.
The output feeds measurement, not execution.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from src.events.triage import DEFAULT_TRIAGE_MODEL, extract_json_array
from src.research.llm import Message, get_client
from src.research.llm.client import LLMClient

logger = logging.getLogger(__name__)

CLASSIFIER_VERSION = "v1-haiku"

_TOPICS = {
    "tariffs",
    "china",
    "energy",
    "defense",
    "appointments",
    "broad_market",
    "self_referential",
    "trade",
    "monetary",
    "other",
}
_TONES = {"positive", "negative", "threatening", "de-escalatory", "neutral"}

_SYSTEM_PROMPT = """You label social-media posts for an academic-style \
event study of market reactions. For each numbered post return one JSON \
object; respond with a JSON array only, no prose.

Fields per post:
  id: the number given
  relevant: true ONLY if the post could reasonably move markets (policy, \
tariffs, trade, specific companies/sectors, appointments, monetary policy, \
explicit market encouragement). Personal attacks, campaign content, and \
pleasantries are false.
  topic: one of tariffs|china|energy|defense|appointments|broad_market|\
self_referential|trade|monetary|other (single best fit; self_referential = \
about Truth Social/DJT/his own ventures)
  secondary: array of additional applicable topics from the same list \
(may be empty)
  tone: positive|negative|threatening|de-escalatory|neutral \
(threatening = aggressive threat/tariff language; de-escalatory = \
walk-back, pause, conciliatory)
  entities: companies, countries, or sectors EXPLICITLY named in the text \
— no inference, no tickers unless literally written
  explicit_market: true ONLY for direct phrases like "buy", "great time \
to buy", "sell"
  confidence: 0.0-1.0"""


def _clamp01(v: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def classify_pending(
    engine: Any,
    client: LLMClient | None = None,
    model: str | None = None,
    batch_size: int = 10,
    now: datetime | None = None,
) -> int:
    """Classify up to one batch of unclassified posts. Returns rows
    written. Media-only posts (empty text) are fast-pathed to
    not-relevant without an LLM call. LLM failure → 0 (posts stay
    unclassified; the loop retries)."""
    now = now or datetime.now(UTC)
    with engine.connect() as conn:
        rows = [
            dict(r._mapping)
            for r in conn.execute(
                text("""
            SELECT p.post_id, p.text FROM truth_posts p
            LEFT JOIN truth_classifications c ON c.post_id = p.post_id
            WHERE c.post_id IS NULL
            ORDER BY p.posted_at
            LIMIT :lim
        """),
                {"lim": batch_size},
            )
        ]
    if not rows:
        return 0

    written = 0
    empties = [r for r in rows if not str(r["text"]).strip()]
    for r in empties:
        _store(
            engine,
            r["post_id"],
            {
                "relevant": False,
                "topic": None,
                "secondary": [],
                "tone": None,
                "entities": [],
                "explicit_market": False,
                "confidence": 1.0,
            },
            now,
        )
        written += 1
    to_classify = [r for r in rows if str(r["text"]).strip()]
    if not to_classify:
        return written

    llm = client if client is not None else get_client("claude-code")
    chosen_model = model or os.environ.get(
        "TRUTH_CLASSIFIER_MODEL",
        DEFAULT_TRIAGE_MODEL,
    )
    numbered = "\n\n".join(f"[{i}] {str(r['text'])[:1500]}" for i, r in enumerate(to_classify))
    try:
        resp = llm.complete(
            messages=[
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=numbered),
            ],
            model=chosen_model,
            max_tokens=1600,
            # Single-shot classification of injected text (CL-scup).
            no_tools=True,
        )
        items = extract_json_array(resp.text)
    except Exception as exc:
        logger.warning(
            "truth classifier: LLM failed — %d post(s) retry next cycle: %s",
            len(to_classify),
            str(exc)[:200],
        )
        return written

    by_idx: dict[int, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict):
            try:
                by_idx[int(item["id"])] = item
            except (KeyError, TypeError, ValueError):
                continue
    for i, r in enumerate(to_classify):
        item = by_idx.get(i)
        if item is None:
            continue  # absent from response — retried next cycle
        _store(engine, r["post_id"], item, now)
        written += 1
    logger.info("truth classifier: %d post(s) classified", written)
    return written


def _store(
    engine: Any,
    post_id: str,
    item: dict[str, Any],
    now: datetime,
) -> None:
    import json  # noqa: PLC0415

    topic = str(item.get("topic") or "").strip().lower() or None
    if topic is not None and topic not in _TOPICS:
        topic = "other"
    tone = str(item.get("tone") or "").strip().lower() or None
    if tone is not None and tone not in _TONES:
        tone = "neutral"
    secondary = [
        s for s in (item.get("secondary") or []) if isinstance(s, str) and s.lower() in _TOPICS
    ]
    entities = [
        str(e)[:80] for e in (item.get("entities") or []) if isinstance(e, str) and e.strip()
    ][:10]
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO truth_classifications
                (post_id, is_market_relevant, primary_topic,
                 secondary_topics, tone, named_entities,
                 explicit_market_language, confidence,
                 classifier_version, classified_at)
            VALUES (:i, :rel, :top, :sec, :tone, :ent, :xm, :conf, :v, :at)
            ON CONFLICT (post_id) DO NOTHING
        """),
            {
                "i": post_id,
                "rel": bool(item.get("relevant")),
                "top": topic,
                "sec": json.dumps(secondary),
                "tone": tone,
                "ent": json.dumps(entities),
                "xm": bool(item.get("explicit_market")),
                "conf": _clamp01(item.get("confidence")),
                "v": CLASSIFIER_VERSION,
                "at": now,
            },
        )
