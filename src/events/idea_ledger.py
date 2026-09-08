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
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import bindparam, text

from src.events.instrument_selector import decision_for_idea
from src.events.prices import parse_ts
from src.events.trade_card import build_trade_card
from src.events.trade_idea import TradeIdea

logger = logging.getLogger(__name__)

_INSERT_SQL = text(
    "INSERT INTO trade_ideas "
    "(idea_id, geo_event_id, ticker, action, direction, confidence, "
    " time_horizon, holding_period_days, time_stop_days, stop_loss_pct, "
    " preferred_instrument, instrument_reason, rationale, suggested_entry, "
    " notes, price_at_signal, stop_price, target_prices, risk_reward, "
    " entry_trigger, invalidation, dte_window, suggested_strike, "
    " created_at, status, status_updated_at) "
    "VALUES "
    "(:idea_id, :geo_event_id, :ticker, :action, :direction, :confidence, "
    " :time_horizon, :holding_period_days, :time_stop_days, :stop_loss_pct, "
    " :preferred_instrument, :instrument_reason, :rationale, :suggested_entry, "
    " :notes, :price_at_signal, :stop_price, :target_prices, :risk_reward, "
    " :entry_trigger, :invalidation, :dte_window, :suggested_strike, "
    " :created_at, 'pending', :created_at) "
    "ON CONFLICT (idea_id) DO NOTHING"
)

_LIST_OPEN_SQL = text(
    "SELECT idea_id, geo_event_id, ticker, action, direction, confidence, "
    "       time_horizon, holding_period_days, time_stop_days, stop_loss_pct, "
    "       preferred_instrument, instrument_reason, rationale, "
    "       suggested_entry, notes, price_at_signal, stop_price, "
    "       target_prices, risk_reward, entry_trigger, invalidation, "
    "       dte_window, suggested_strike, created_at "
    "FROM trade_ideas WHERE status = 'pending' "
    "ORDER BY created_at DESC, id DESC"
)


def make_idea_id(geo_event_id: Any, ticker: str, action: str) -> str:
    """Stable dedup key: same event + ticker + action → same row."""
    raw = f"{geo_event_id}:{ticker}:{action}"
    # Persisted dedup key, NOT authentication/integrity. Preserve existing IDs.
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
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
    same posture as the impact agent's normaliser.

    The idea is parsed ONCE into a :class:`~src.events.trade_idea.TradeIdea`
    (CL-59mk) and read by attribute from there. The raw dict is still
    handed to :func:`decision_for_idea` / :func:`build_trade_card`, which
    also serve ``trade_ideas`` table rows and legacy shapes."""
    parsed = TradeIdea.from_dict(idea)
    # from_dict is a lossless read (it does not strip), so the four fields
    # this ledger has always stripped keep stripping explicitly; the seven
    # other string columns below are stored verbatim, exactly as before.
    ticker = parsed.ticker.strip()
    action = parsed.action.strip().lower()
    if not ticker or not action:
        logger.debug("idea ledger: dropping idea without ticker/action: %r", idea)
        return None

    preferred = parsed.preferred_instrument.strip()
    stop_loss_pct = parsed.stop_loss_pct
    instrument_reason = ""
    notes = parsed.notes.strip()

    # Niche/asymmetry metadata (CL-u2ph) — the trade_ideas table has no
    # niche columns (additive-only rule), so fold the tags into `notes`
    # as a compact prefix so a persisted niche idea stays self-describing
    # in the audit trail. hop_count / asymmetry_score / liquidity_flag
    # travel; torque_reason is already the idea's `notes` body.
    if parsed.niche:
        hops = parsed.hop_count
        asym = parsed.asymmetry_score
        marker = "niche"
        if hops is not None:
            marker += f" {hops}hop"
        if asym is not None:
            marker += f" asym{asym:.2f}"
        if parsed.liquidity_flag:
            marker += " illiquid"
        notes = f"[{marker}] {notes}".strip() if notes else f"[{marker}]"

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
            annotation = f"selector prefers {decision.action} ({decision.horizon_band} horizon)"
            notes = f"{notes}; {annotation}" if notes else annotation

    price_info = (prices or {}).get(ticker) or {}

    # Ground the idea's PERCENTAGES in the real price fetched this cycle
    # (CL-jiqq). Build the card from the gap-filled stop so an idea the
    # LLM left blank still gets a dollar stop from the selector's value.
    # No price → the card returns %-only (dollar fields None); we still
    # persist the entry_trigger / invalidation / dte_window text.
    card_idea = dict(idea)
    if stop_loss_pct is not None:
        card_idea["stop_loss_pct"] = stop_loss_pct
    card = build_trade_card(
        card_idea,
        _float_or_none(price_info.get("price")),
        _float_or_none(price_info.get("change_pct")),
    )
    target_prices = card.get("target_prices") or []

    return {
        "idea_id": make_idea_id(geo_event_id, ticker, action),
        "geo_event_id": int(geo_event_id),
        "ticker": ticker,
        "action": action,
        "direction": parsed.direction or None,
        "confidence": parsed.confidence,
        "time_horizon": parsed.time_horizon or None,
        "holding_period_days": parsed.holding_period_days or None,
        "time_stop_days": parsed.time_stop_days,
        "stop_loss_pct": stop_loss_pct,
        "preferred_instrument": preferred or None,
        "instrument_reason": instrument_reason or None,
        "rationale": parsed.rationale or None,
        "suggested_entry": parsed.suggested_entry or None,
        "notes": notes or None,
        "price_at_signal": _float_or_none(price_info.get("price")),
        # Grounded trade-card levels (CL-jiqq). Dollar fields are NULL
        # when there was no live price; text fields persist regardless.
        "stop_price": card.get("stop_price"),
        "target_prices": json.dumps(target_prices) if target_prices else None,
        "risk_reward": card.get("risk_reward"),
        "entry_trigger": parsed.entry_trigger or None,
        "invalidation": parsed.invalidation or None,
        "dte_window": card.get("dte_window") or None,
        "suggested_strike": card.get("suggested_strike"),
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
        p
        for idea in ideas
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
        rows = conn.execute(
            text(
                "SELECT id, created_at, time_stop_days FROM trade_ideas WHERE status = 'pending'",
            )
        ).all()
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
    graceful 'unavailable' reply.

    This is the RAW per-event view (one row per geo_event that proposed
    an idea) — the audit trail. For the operator-facing bot listing use
    :func:`list_open_consolidated`, which collapses (ticker, action)."""
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(_LIST_OPEN_SQL)]
    for row in rows:
        row["created_at"] = parse_ts(row.get("created_at"))
        row["target_prices"] = _parse_target_prices(row.get("target_prices"))
    return rows


def list_open_consolidated(engine: Any) -> list[dict[str, Any]]:
    """Open ideas CONSOLIDATED on (ticker, action) for the bot display
    (CL-5mkf). The ``trade_ideas`` table keeps one row per geo_event
    (the audit trail — :func:`list_open` returns those unchanged); this
    view shows ONE entry per (ticker, action) so a gold idea proposed by
    six events reads as one high-conviction idea, not six near-duplicates.

    The kept row is the most-recent/highest-confidence of the group (raw
    rows are newest-first; ties break on higher confidence), so its
    levels/short-id are the freshest. An ``event_count`` field carries
    how many raw rows collapsed into it (1 = a plain single-event idea);
    the bot renders ``×N events`` when ``event_count > 1``."""
    rows = list_open(engine)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (
            str(row.get("ticker") or ""),
            str(row.get("action") or ""),
        )
        existing = groups.get(key)
        if existing is None:
            kept = dict(row)
            kept["event_count"] = 1
            groups[key] = kept
            order.append(key)
            continue
        existing["event_count"] += 1
        # Rows arrive newest-first, so the incumbent already wins on
        # recency; only a strictly-higher confidence unseats it (its
        # levels are then the ones the operator sees).
        row_conf = _float_or_none(row.get("confidence"))
        cur_conf = _float_or_none(existing.get("confidence"))
        if row_conf is not None and row_conf > (cur_conf if cur_conf is not None else -1.0):
            kept = dict(row)
            kept["event_count"] = existing["event_count"]
            groups[key] = kept
    return [groups[k] for k in order]


_GET_ONE_SQL = text(
    "SELECT idea_id, geo_event_id, ticker, action, direction, confidence, "
    "       time_horizon, holding_period_days, time_stop_days, stop_loss_pct, "
    "       preferred_instrument, instrument_reason, rationale, "
    "       suggested_entry, notes, price_at_signal, stop_price, "
    "       target_prices, risk_reward, entry_trigger, invalidation, "
    "       dte_window, suggested_strike, created_at, status "
    "FROM trade_ideas WHERE idea_id LIKE :prefix "
    "ORDER BY created_at DESC, id DESC"
)


def get_idea(engine: Any, idea_id_prefix: str) -> dict[str, Any] | None:
    """One idea by ``idea_id`` prefix (the short id the bot shows), most
    recent first on ties. ``None`` when nothing matches. Fields are
    parsed like :func:`list_open` (aware ``created_at``, list
    ``target_prices``). Raises only on DB errors — the bot wraps it."""
    prefix = str(idea_id_prefix or "").strip()
    if not prefix:
        return None
    with engine.connect() as conn:
        row = conn.execute(_GET_ONE_SQL, {"prefix": f"{prefix}%"}).first()
    if row is None:
        return None
    out = dict(row._mapping)
    out["created_at"] = parse_ts(out.get("created_at"))
    out["target_prices"] = _parse_target_prices(out.get("target_prices"))
    return out


def _parse_target_prices(value: Any) -> list[float]:
    """``target_prices`` normalised to a list of floats. Postgres JSONB
    hands back a Python list; sqlite (tests) stores the JSON string —
    parse either. Garbage → ``[]`` (never a lie, never a raise)."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out
