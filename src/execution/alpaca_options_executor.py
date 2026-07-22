"""Alpaca paper options executor (CL-ldd2).

Reads the desk's surfaced advisory options ideas and places PAPER option orders
on Alpaca, per the operator's chosen policy:

  * only NICHE ideas that SURVIVED the red-team critic (notes carry the
    "[niche …]" marker and a "red-team" note), confidence >= min_confidence;
  * 1 contract per idea, but SKIP if the estimated premium (ask*100*qty)
    exceeds ``max_premium_usd``;
  * a hard daily cap on new option orders.

Terminal decisions (submitted / skipped_premium) are recorded in
``alpaca_option_orders`` keyed by idea_id, so an idea is executed at most once
and the daily cap can be counted. Transient misses (no price/contract/quote)
are NOT recorded, so they retry next cycle. The Telegram advisory feed is a
separate path and is untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from src.execution.alpaca_options import (
    AlpacaOptionsClient,
    ContractSelectionConfig,
    resolve_contract,
)

logger = logging.getLogger(__name__)

#: ticker -> last price. Injectable; the live default reuses get_prices.
PriceFn = Callable[[str], float | None]


@dataclass(frozen=True)
class OptionsExecConfig:
    min_confidence: float = 0.55
    max_premium_usd: float = 500.0
    qty: int = 1
    max_per_day: int = 5
    require_niche: bool = True
    require_red_team: bool = True
    selection: ContractSelectionConfig = ContractSelectionConfig()
    # Technical-alignment gate (CL-3xoj): skip an idea whose computed price
    # structure is strongly AGAINST the thesis (alignment_score in [-1,1];
    # e.g. buying calls into a confirmed downtrend at the lows = -1.0).
    # Fail-open: no history/context -> no gate (consistent with the repo's
    # missing-data posture). -1.01 disables the gate entirely.
    min_alignment: float = -0.4


def _default_price_fn(engine: Any) -> PriceFn:
    def _fetch(ticker: str) -> float | None:
        from src.events.prices import get_prices  # noqa: PLC0415
        out = get_prices([ticker], engine=engine)
        v = out.get(ticker)
        return float(v["price"]) if v and v.get("price") is not None else None
    return _fetch


def fetch_executable_ideas(
    engine: Any, cfg: OptionsExecConfig,
) -> list[dict[str, Any]]:
    """Pending buy_calls/buy_puts ideas matching the policy that haven't been
    acted on yet (highest confidence first)."""
    where = [
        "action IN ('buy_calls','buy_puts')",
        "confidence >= :min_conf",
        "status = 'pending'",
        "NOT EXISTS (SELECT 1 FROM alpaca_option_orders a "
        "WHERE a.idea_id = ti.idea_id)",
    ]
    if cfg.require_niche:
        where.append("lower(notes) LIKE '%niche%'")
    if cfg.require_red_team:
        where.append("lower(notes) LIKE '%red-team%'")
    sql = (
        "SELECT idea_id, ticker, action, confidence, preferred_instrument, notes "
        "FROM trade_ideas ti WHERE " + " AND ".join(where) +
        " ORDER BY confidence DESC, created_at DESC"
    )
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(
            text(sql), {"min_conf": cfg.min_confidence})]


def _submitted_today(engine: Any, now: datetime) -> int:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    with engine.connect() as conn:
        return int(conn.execute(text(
            "SELECT COUNT(*) FROM alpaca_option_orders "
            "WHERE status = 'submitted' AND submitted_at >= :start",
        ), {"start": start}).scalar() or 0)


def _record(engine: Any, row: dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO alpaca_option_orders
                (idea_id, ticker, occ_symbol, opt_type, qty, premium_est,
                 alpaca_order_id, status, detail, submitted_at)
            VALUES (:idea_id,:ticker,:occ_symbol,:opt_type,:qty,:premium_est,
                    :alpaca_order_id,:status,:detail,:submitted_at)
            ON CONFLICT (idea_id) DO NOTHING
        """), row)


def execute_pending_options(
    engine: Any,
    client: AlpacaOptionsClient,
    price_fn: PriceFn | None = None,
    cfg: OptionsExecConfig | None = None,
    now: datetime | None = None,
    technicals_fn: Any = None,
) -> dict[str, int]:
    """Execute eligible options ideas within the daily cap. Returns a count by
    outcome. Never raises — a per-idea failure is recorded/logged and the loop
    continues."""
    cfg = cfg or OptionsExecConfig()
    now = now or datetime.now(UTC)
    fetch = price_fn or _default_price_fn(engine)
    if technicals_fn is None:
        from src.events.technical_context import compute_for_ticker  # noqa: PLC0415
        technicals_fn = compute_for_ticker
    counts = {"submitted": 0, "skipped_premium": 0, "no_contract": 0,
              "no_quote": 0, "no_price": 0, "error": 0, "market_closed": 0,
              "misaligned": 0}

    # Options MARKET orders are 422-rejected outside regular hours — don't even
    # try; just wait for the next open (CL-ldd2).
    if not client.is_market_open():
        logger.info("alpaca options: market closed — no orders this cycle")
        counts["market_closed"] = 1
        return counts

    budget = max(0, cfg.max_per_day - _submitted_today(engine, now))
    if budget <= 0:
        logger.info("alpaca options: daily cap reached — no new orders")
        return counts

    for idea in fetch_executable_ideas(engine, cfg):
        if counts["submitted"] >= budget:
            break
        ticker = str(idea.get("ticker") or "")
        right = "call" if str(idea.get("action")) == "buy_calls" else "put"
        try:
            price = fetch(ticker)
            if not price:
                counts["no_price"] += 1
                continue  # transient — retry next cycle (not recorded)
            # Technical-alignment gate (CL-3xoj): don't buy calls into a
            # confirmed downtrend (or puts into an uptrend). Transient (not
            # recorded) — structure changes; the idea retries next cycle.
            # Fail-open when no context is computable.
            try:
                ctx = technicals_fn(ticker)
            except Exception:
                ctx = None
            if ctx is not None:
                from src.events.technical_context import alignment_score  # noqa: PLC0415
                direction = "bullish" if right == "call" else "bearish"
                score = alignment_score(ctx, direction)
                if score < cfg.min_alignment:
                    counts["misaligned"] += 1
                    logger.info(
                        "alpaca options: skipped %s %s — technical alignment "
                        "%.2f < %.2f (trend=%s, %s)", ticker, idea.get("action"),
                        score, cfg.min_alignment, ctx.trend, ctx.breakout_state,
                    )
                    continue
            contract = resolve_contract(client, idea, price, now.date(),
                                        cfg.selection)
            if not contract:
                counts["no_contract"] += 1
                continue
            occ = str(contract.get("symbol") or "")
            ask = client.get_option_ask(occ)
            if ask is None:
                counts["no_quote"] += 1
                continue
            premium = ask * 100.0 * cfg.qty
            base = {
                "idea_id": idea["idea_id"], "ticker": ticker,
                "occ_symbol": occ, "opt_type": right, "qty": cfg.qty,
                "premium_est": premium, "submitted_at": now,
            }
            if premium > cfg.max_premium_usd:
                _record(engine, {**base, "alpaca_order_id": None,
                                 "status": "skipped_premium",
                                 "detail": f"premium ${premium:.0f} > "
                                           f"${cfg.max_premium_usd:.0f}"})
                counts["skipped_premium"] += 1
                logger.info("alpaca options: skipped %s (%s) premium $%.0f > cap",
                            ticker, occ, premium)
                continue
            order = client.submit_option_order(occ, cfg.qty, "buy")
            _record(engine, {**base,
                             "alpaca_order_id": str(order.get("id") or ""),
                             "status": "submitted", "detail": None})
            counts["submitted"] += 1
            logger.info("alpaca options: BOUGHT %d %s (%s) ~$%.0f premium "
                        "[idea %s]", cfg.qty, occ, ticker, premium,
                        idea["idea_id"])
        except Exception as exc:
            counts["error"] += 1
            logger.warning("alpaca options: error executing %s: %s",
                           ticker, str(exc)[:200])
    logger.info("alpaca options: %s", counts)
    return counts
