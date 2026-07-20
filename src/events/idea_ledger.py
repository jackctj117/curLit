"""Trade-idea ledger (CL-mgcp) — persists the impact agent's advisory
``trade_ideas`` into the ``trade_ideas`` table (migration 007) so every
idea the operator sees in Telegram is TRACKED, AGED, and AUTO-EXPIRED
instead of scrolling away in chat history.

  * :func:`persist_ideas` — upsert one assessment's ideas.
    ``idea_id = sha1(geo_event_id:ticker:action)[:16]`` so re-running
    an assess cycle (or two writers racing) dedups via
    ``ON CONFLICT (idea_id) DO NOTHING``. Gaps the LLM left —
    ``preferred_instrument`` / ``stop_loss_pct`` — are filled by the
    horizon-based selector (:mod:`src.events.instrument_selector`);
    when the selector DISAGREES with the LLM's action it annotates
    ``notes`` ("selector prefers buy_puts (short horizon)") but NEVER
    overrides the action — the LLM saw the event, the selector only
    saw the horizon.
  * :func:`expire_stale` — pending ideas whose ``time_stop_days`` have
    elapsed flip to ``expired`` (guarded UPDATE: only rows still
    ``pending`` transition — an operator's future manual transition is
    never clobbered). Runs every pipeline cycle.
  * :func:`list_open` — pending ideas, newest first, for the bot's
    read-only ``ideas`` command.

Status lifecycle v1: ``pending → expired`` is the ONLY automatic
transition; ``taken`` / ``cancelled`` / ``closed`` are schema-reserved
for future operator commands (the bot listing is read-only for now).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import bindparam, text

from src.events.instrument_selector import decision_for_idea
from src.events.prices import parse_ts

logger = logging.getLogger(__name__)

_INSERT_SQL = text(
    "INSERT INTO trade_ideas "
    "(idea_id, geo_event_id, ticker, action, direction, confidence, "
    " time_horizon, holding_period_days, time_stop_days, stop_loss_pct, "
    " preferred_instrument, instrument_reason, rationale, suggested_entry, "
    " notes, price_at_signal, created_at, status, status_updated_at) "
    "VALUES "
    "(:idea_id, :geo_event_id, :ticker, :action, :direction, :confidence, "
    " :time_horizon, :holding_period_days, :time_stop_days, :stop_loss_pct, "
    " :preferred_instrument, :instrument_reason, :rationale, :suggested_entry, "
    " :notes, :price_at_signal, :created_at, 'pending', :created_at) "
    "ON CONFLICT (idea_id) DO NOTHING"
)

_LIST_OPEN_SQL = text(
    "SELECT idea_id, geo_event_id, ticker, action, direction, confidence, "
    "       time_horizon, holding_period_days, time_stop_days, stop_loss_pct, "
    "       preferred_instrument, instrument_reason, rationale, "
    "       suggested_entry, notes, price_at_signal, created_at "
    "FROM trade_ideas WHERE status = 'pending' "
    "ORDER BY created_at DESC, id DESC"
)


def make_idea_id(geo_event_id: Any, ticker: str, action: str) -> str:
    """Stable dedup key: same event + ticker + action → same row."""
    raw = f"{geo_event_id}:{ticker}:{action}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]  # noqa: S324


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_params(
    geo_event_id: int,
    idea: dict[str, Any],
    prices: Mapping[str, Mapping[str, Any]] | None,
    now: datetime,
) -> dict[str, Any] | None:
    """One idea dict → insert params, selector-gap-filled. ``None``
    for unusable entries (no ticker/action) — dropped individually,
    same posture as the impact agent's normaliser."""
    ticker = str(idea.get("ticker") or "").strip()
    action = str(idea.get("action") or "").strip().lower()
    if not ticker or not action:
        logger.debug("idea ledger: dropping idea without ticker/action: %r", idea)
        return None

    preferred = str(idea.get("preferred_instrument") or "").strip()
    stop_loss_pct = _float_or_none(idea.get("stop_loss_pct"))
    instrument_reason = ""
    notes = str(idea.get("notes") or "").strip()

    decision = decision_for_idea(idea)
    if decision is not None:
        # Fill ONLY the gaps; the LLM's own values always win when set.
        if not preferred:
            preferred = decision.preferred_instrument
            instrument_reason = decision.reason
        if stop_loss_pct is None:
            stop_loss_pct = decision.stop_loss_pct
            if not instrument_reason:
                instrument_reason = decision.reason
        if decision.action != action:
            annotation = (
                f"selector prefers {decision.action} "
                f"({decision.horizon_band} horizon)"
            )
            notes = f"{notes}; {annotation}" if notes else annotation

    price_info = (prices or {}).get(ticker) or {}
    return {
        "idea_id": make_idea_id(geo_event_id, ticker, action),
        "geo_event_id": int(geo_event_id),
        "ticker": ticker,
        "action": action,
        "direction": str(idea.get("direction") or "") or None,
        "confidence": _float_or_none(idea.get("confidence")),
        "time_horizon": str(idea.get("time_horizon") or "") or None,
        "holding_period_days": str(idea.get("holding_period_days") or "") or None,
        "time_stop_days": _int_or_none(idea.get("time_stop_days")),
        "stop_loss_pct": stop_loss_pct,
        "preferred_instrument": preferred or None,
        "instrument_reason": instrument_reason or None,
        "rationale": str(idea.get("rationale") or "") or None,
        "suggested_entry": str(idea.get("suggested_entry") or "") or None,
        "notes": notes or None,
        "price_at_signal": _float_or_none(price_info.get("price")),
        "created_at": now,
    }


def persist_ideas(
    engine: Any,
    geo_event_id: int,
    assessment: Mapping[str, Any],
    prices: Mapping[str, Mapping[str, Any]] | None = None,
    now: datetime | None = None,
) -> int:
    """Upsert one assessment's ``trade_ideas`` rows; returns how many
    rows were actually INSERTED (conflicts dedup silently to 0-cost
    no-ops). ``prices`` (from :func:`src.events.prices.get_prices`)
    supplies ``price_at_signal`` where available. DB errors propagate —
    the pipeline wraps this fail-soft, tests want the loud version."""
    ideas = assessment.get("trade_ideas")
    if not isinstance(ideas, list) or not ideas:
        return 0
    now = now or datetime.now(UTC)
    params = [
        p for idea in ideas
        if isinstance(idea, dict)
        and (p := _row_params(geo_event_id, idea, prices, now)) is not None
    ]
    if not params:
        return 0
    inserted = 0
    with engine.begin() as conn:
        for p in params:  # per-row so rowcount attributes cleanly
            result = conn.execute(_INSERT_SQL, p)
            inserted += max(0, result.rowcount or 0)
    return inserted


def expire_stale(engine: Any, now: datetime | None = None) -> int:
    """``pending`` ideas whose ``created_at + time_stop_days`` has
    elapsed → ``expired``. The UPDATE is guarded on ``status =
    'pending'`` so a concurrent (future) operator transition wins the
    race. Expiry math runs in Python — portable across Postgres and
    the sqlite test engines. Returns rows transitioned."""
    now = now or datetime.now(UTC)
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, created_at, time_stop_days "
            "FROM trade_ideas WHERE status = 'pending'",
        )).all()
    stale: list[int] = []
    for row_id, created_at, time_stop_days in rows:
        created = parse_ts(created_at)
        if created is None or time_stop_days is None:
            continue  # no time stop → the idea never auto-expires
        if created + timedelta(days=int(time_stop_days)) <= now:
            stale.append(int(row_id))
    if not stale:
        return 0
    stmt = text(
        "UPDATE trade_ideas "
        "SET status = 'expired', status_updated_at = :now "
        "WHERE id IN :ids AND status = 'pending'",
    ).bindparams(bindparam("ids", expanding=True))  # Postgres + sqlite alike
    with engine.begin() as conn:
        result = conn.execute(stmt, {"now": now, "ids": stale})
    expired = max(0, result.rowcount or 0)
    if expired:
        logger.info("idea ledger: auto-expired %d stale idea(s)", expired)
    return expired


def list_open(engine: Any) -> list[dict[str, Any]]:
    """Pending ideas, newest first, ``created_at`` parsed to an aware
    datetime. Raises on DB errors — the bot command wraps this with a
    graceful 'unavailable' reply."""
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(_LIST_OPEN_SQL)]
    for row in rows:
        row["created_at"] = parse_ts(row.get("created_at"))
    return rows
