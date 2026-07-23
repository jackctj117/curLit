"""Closed-loop outcome tracking (CL-6axf) — the system's own track record.

Every surfaced trade idea (``trade_ideas``) already captured an entry price
(``price_at_signal``) and time. This scores what happened NEXT: fetch the
current price, compute the signed, direction-adjusted return, track the best
and worst excursion (MFE/MAE), and — once the idea's horizon elapses — stamp a
final win / loss / flat verdict into ``idea_outcomes``.

Idea attributes are denormalised at score time (theme, action, direction,
confidence, niche/hop, red-team-survived) so the reflective loop (CL-#2) can
aggregate performance by dimension with a plain GROUP BY.

Reuses ``src.events.prices.get_prices`` for the current price — the SAME source
that set the entry — so entry and exit are measured on one ruler. Fail-soft: a
ticker that can't be priced stays ``no_data`` and is retried next cycle; the
job never raises.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from src.events.prices import parse_ts

logger = logging.getLogger(__name__)

_BULLISH_ACTIONS = frozenset({"long", "buy_calls"})
_BEARISH_ACTIONS = frozenset({"short", "buy_puts"})


def _idea_is_bullish(action: str, direction: str) -> bool | None:
    """Canonical long/short sign for scoring (CL-67m9, P1).

    ``action`` is the concrete instruction (long/short/buy_calls/buy_puts) and
    WINS; ``direction`` (bullish/bearish) is descriptive and only used when the
    action is absent/unknown. The old code OR-ed the two, so a contradictory
    idea like ``action=buy_puts`` (bearish) with ``direction=bullish`` scored
    as a LONG — inverting the outcome and poisoning the reflective loop.
    Returns ``None`` when neither field yields a sign (→ no_data, don't guess).
    """
    a = action.lower()
    if a in _BULLISH_ACTIONS:
        return True
    if a in _BEARISH_ACTIONS:
        return False
    d = direction.lower()
    if d == "bullish":
        return True
    if d == "bearish":
        return False
    return None

#: Matches the niche marker the ledger folds into notes, e.g. "[niche 3hop …]".
_NICHE_RE = re.compile(r"\[niche(?:\s+(\d+)hop)?[^\]]*\]", re.I)

#: Price lookup: tickers → {ticker: last_price}. Injectable for tests.
PriceFn = Callable[[list[str]], dict[str, float]]


@dataclass(frozen=True)
class OutcomeConfig:
    #: |signed return| at/above this at horizon = win (below −this = loss).
    win_threshold: float = 0.05
    #: Fallback evaluation horizon when an idea has no time_stop_days.
    default_horizon_days: int = 10


def parse_niche_marker(notes: str | None) -> tuple[bool, int | None]:
    """``(is_niche, hop_count)`` from the ledger's notes prefix."""
    if not notes:
        return (False, None)
    m = _NICHE_RE.search(notes)
    if not m:
        return (False, None)
    return (True, int(m.group(1)) if m.group(1) else None)


def _default_price_fn(engine: Any) -> PriceFn:
    def _fetch(tickers: list[str]) -> dict[str, float]:
        from src.events.prices import get_prices  # noqa: PLC0415
        out = get_prices(tickers, engine=engine)
        return {
            t: float(v["price"]) for t, v in out.items()
            if v.get("price") is not None
        }
    return _fetch


_SELECT_OPEN = text("""
    SELECT ti.idea_id, ti.ticker, ti.action, ti.direction, ti.confidence,
           ti.time_horizon, ti.time_stop_days, ti.price_at_signal,
           ti.created_at, ti.notes, ge.theme AS theme,
           io.max_favorable_pct AS prev_mfe, io.max_adverse_pct AS prev_mae
    FROM trade_ideas ti
    LEFT JOIN geo_events ge ON ge.id = ti.geo_event_id
    LEFT JOIN idea_outcomes io ON io.idea_id = ti.idea_id
    WHERE ti.price_at_signal IS NOT NULL
      AND (io.outcome IS NULL OR io.outcome IN ('open', 'no_data'))
""")

_UPSERT = text("""
    INSERT INTO idea_outcomes (
        idea_id, ticker, action, direction, theme, time_horizon, confidence,
        is_niche, hop_count, red_team_survived, entry_price, entry_at,
        last_price, last_at, return_pct, max_favorable_pct, max_adverse_pct,
        horizon_days, outcome, scored_count, created_at, updated_at
    ) VALUES (
        :idea_id, :ticker, :action, :direction, :theme, :time_horizon,
        :confidence, :is_niche, :hop_count, :red_team_survived, :entry_price,
        :entry_at, :last_price, :last_at, :return_pct, :mfe, :mae,
        :horizon_days, :outcome, 1, :now, :now
    )
    ON CONFLICT (idea_id) DO UPDATE SET
        last_price = excluded.last_price,
        last_at = excluded.last_at,
        return_pct = excluded.return_pct,
        max_favorable_pct = excluded.max_favorable_pct,
        max_adverse_pct = excluded.max_adverse_pct,
        outcome = excluded.outcome,
        scored_count = idea_outcomes.scored_count + 1,
        updated_at = excluded.updated_at
""")


def score_open_ideas(
    engine: Any,
    price_fn: PriceFn | None = None,
    now: datetime | None = None,
    config: OutcomeConfig | None = None,
) -> dict[str, int]:
    """Score every open idea once and upsert its outcome. Returns a count by
    resulting outcome (``open`` / ``win`` / ``loss`` / ``flat`` / ``no_data``).
    """
    config = config or OutcomeConfig()
    now = now or datetime.now(UTC)
    with engine.connect() as conn:
        rows = conn.execute(_SELECT_OPEN).mappings().all()
    counts = {"open": 0, "win": 0, "loss": 0, "flat": 0, "no_data": 0}
    if not rows:
        return counts

    tickers = sorted({str(r["ticker"]) for r in rows})
    fetch = price_fn or _default_price_fn(engine)
    try:
        current = fetch(tickers)
    except Exception:
        logger.warning("outcome scoring: price fetch failed", exc_info=True)
        current = {}

    with engine.begin() as conn:
        for r in rows:
            entry = float(r["price_at_signal"])
            entry_at = parse_ts(r["created_at"])
            horizon = (
                int(r["time_stop_days"]) if r["time_stop_days"] is not None
                else config.default_horizon_days
            )
            due = entry_at is not None and now >= entry_at + timedelta(days=horizon)
            bullish = _idea_is_bullish(
                str(r["action"] or ""), str(r["direction"] or ""),
            )
            last = current.get(str(r["ticker"]))
            prev_mfe = r["prev_mfe"]
            prev_mae = r["prev_mae"]

            # bullish is None → the idea's sign is indeterminate; score no_data
            # rather than guess a direction (which would mislabel the outcome).
            if last is None or entry <= 0 or bullish is None:
                outcome = "no_data"
                signed = None
                mfe = float(prev_mfe) if prev_mfe is not None else None
                mae = float(prev_mae) if prev_mae is not None else None
            else:
                raw = (last - entry) / entry
                signed = raw if bullish else -raw
                mfe = max(signed, float(prev_mfe)) if prev_mfe is not None else signed
                mae = min(signed, float(prev_mae)) if prev_mae is not None else signed
                if due:
                    if signed >= config.win_threshold:
                        outcome = "win"
                    elif signed <= -config.win_threshold:
                        outcome = "loss"
                    else:
                        outcome = "flat"
                else:
                    outcome = "open"

            is_niche, hop = parse_niche_marker(r["notes"])
            red_team = bool(r["notes"] and "red-team" in str(r["notes"]).lower())
            conn.execute(_UPSERT, {
                "idea_id": r["idea_id"], "ticker": r["ticker"],
                "action": r["action"], "direction": r["direction"],
                "theme": r["theme"], "time_horizon": r["time_horizon"],
                "confidence": r["confidence"], "is_niche": is_niche,
                "hop_count": hop, "red_team_survived": red_team,
                "entry_price": entry, "entry_at": entry_at,
                "last_price": last, "last_at": now if last is not None else None,
                "return_pct": signed, "mfe": mfe, "mae": mae,
                "horizon_days": horizon, "outcome": outcome, "now": now,
            })
            counts[outcome] = counts.get(outcome, 0) + 1

    logger.info("outcome scoring: %s", counts)
    return counts
