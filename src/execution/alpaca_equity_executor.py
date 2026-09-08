"""Alpaca paper EQUITY (shares) executor (CL-ncbq).

Expresses the SAME advisory ideas the options executor (CL-ldd2) trades as
contracts — ``buy_calls`` / ``buy_puts`` on niche, red-team-survived events —
as plain shares on Alpaca PAPER, so the two expressions of one signal set can
be compared head-to-head.

WHY (CL-4c7o): the review measured the desk's equity ideas hitting DIRECTION
on the underlying 80% of the time (41/51), while the short-dated OTM options
expressing those same ideas won only 7%. The theses were producing real 1-2%
moves; spread + theta were eating them before they could be realized. Shares
carry the identical directional call with a penny spread and no time decay.

Consequences of that, visible as things this module deliberately does NOT do:
  * no spread filter (the options path's ``max_entry_spread_pct``) — a liquid
    US equity's spread is basis points, not 35% of the ask;
  * no entry-mid baseline (mig 018) — an equity fill IS the honest basis, so
    there is no spread to cancel on both sides;
  * no contract selection at all — the ticker is the instrument.

What IS mirrored, because those lessons are about the IDEA and not the
instrument: the eligibility query, the daily/hourly caps, the market-hours
guard, the technical-alignment gate, the open-spread entry delay, the
idea-life floor, and idea-level dedup keyed on ``idea_id`` in
``alpaca_equity_orders`` (mig 019).

Direction mapping: ``buy_calls`` → long shares (``buy``); ``buy_puts`` → short
shares (``sell_short``, submitted to Alpaca as a plain ``sell``).

Terminal decisions (submitted / skipped_expiring / skipped_short_disabled /
skipped_price_too_high) are recorded so an idea is acted on at most once;
transient misses (no price, misaligned, entry-delayed, concentration) are NOT
recorded and retry next cycle.

The ``trade_ideas`` row is deliberately left UNTOUCHED on entry: the options
executor may independently express the same idea, and marking it would break
that A/B (and the other book's eligibility query).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from src.events.prices import parse_ts
from src.execution.alpaca_equity import AlpacaEquityClient
from src.execution.alpaca_exposure import (
    ExposureUnavailableError,
    position_quantity,
    require_positive_int,
    validated_positions,
)

# Shared with the options book on purpose — these encode lessons about the
# IDEA (opening spreads are widest),
# not about options, so forking them would let the two books drift.
from src.execution.alpaca_options_executor import (
    _default_price_fn,
    _entry_delay_active,
    _is_duplicate_client_order_id,
)
from src.monitoring.logging_setup import LogContext

logger = logging.getLogger(__name__)

#: ticker -> last price. Injectable; the live default reuses get_prices.
PriceFn = Callable[[str], float | None]

#: Bookkeeping values for ``alpaca_equity_orders.side``. Alpaca itself only
#: knows buy/sell — a sell with no position OPENS a short — but the book has
#: to remember which one we meant so P&L can be signed correctly at exit.
SIDE_LONG = "buy"
SIDE_SHORT = "sell_short"


@dataclass(frozen=True)
class EquityExecConfig:
    min_confidence: float = 0.55
    #: Target dollar exposure per idea. qty = floor(notional / price), so a
    #: $1,000 sleeve buys 10 shares of a $95 name and 1 share of a $900 one.
    #: Fixed notional (not fixed qty like the options book) is what makes the
    #: per-idea outcomes comparable across very different share prices.
    notional_usd: float = 1000.0
    max_per_day: int = 10
    #: Intraday pacing, same rationale as the options book (CL-h02l): the
    #: daily budget must not be dumped in one burst at the open, so fresh
    #: afternoon events still find room.
    max_per_hour: int = 2
    #: Concentration cap read from the LIVE Alpaca book (CL-3nfm): idea_id
    #: dedup cannot see that a DIFFERENT idea already put us in this name —
    #: the same theme re-confirmed by a later event, or the impact agent and
    #: the niche agent independently surfacing it. One position per ticker.
    max_positions_per_ticker: int = 1
    require_niche: bool = True
    require_red_team: bool = True
    #: Technical-alignment gate (CL-3xoj): skip an idea whose computed price
    #: structure is strongly AGAINST the thesis. Fail-open when no context is
    #: computable. -1.01 disables.
    min_alignment: float = -0.4
    #: No entries in the first N minutes of the regular session. Equity
    #: spreads are pennies, so this is NOT about spread cost here — it is
    #: about the opening auction's price discovery: a gap that fully reverses
    #: by 9:45 is not the move the thesis called. 0 disables.
    entry_delay_min: int = 15
    #: During the delay window, ideas at/above this confidence enter anyway.
    #: 1.01 disables the override.
    entry_delay_override_conf: float = 0.80
    #: Minimum remaining idea life (created_at + time_stop_days - now) to open
    #: a position (CL-v2m9): with less life than this, the time-stop exit rule
    #: would force-close the position almost immediately — churn for a thesis
    #: that was never given room to play out. 0 disables.
    min_idea_life_days: float = 3.0
    #: Master switch for the short side. Shorting carries unbounded loss and
    #: borrow/locate risk that the long side does not, so it is separately
    #: killable without disabling the whole book. When off, ``buy_puts`` ideas
    #: are recorded TERMINALLY (the policy will not change mid-idea).
    allow_short: bool = True

    def __post_init__(self) -> None:
        for name in ("max_positions_per_ticker", "max_per_day", "max_per_hour"):
            require_positive_int(name, getattr(self, name), allow_zero=True)
        if (
            isinstance(self.notional_usd, bool)
            or not isinstance(self.notional_usd, (int, float))
            or not math.isfinite(self.notional_usd)
            or self.notional_usd <= 0
        ):
            raise ValueError("notional_usd must be finite and positive")


def _held_ticker_qty(client: Any, ticker: str) -> Decimal:
    """Shares (long or short) already held in ``ticker`` on the live book.

    CL-0deu.1.1: raise on unknown exposure and preserve fractional holdings.
    """
    try:
        logger.info("alpaca equity: reading exposure before entry")
        positions = validated_positions(client.list_equity_positions(), asset_class="us_equity")
    except ExposureUnavailableError:
        raise
    except Exception as exc:
        raise ExposureUnavailableError("position_lookup_failed") from exc
    root = str(ticker).upper()
    total = Decimal(0)
    for p in positions:
        if str(p.get("symbol") or "").upper() != root:
            continue
        total += abs(position_quantity(p["qty"]))
    assert total >= 0
    return total


def fetch_executable_ideas(
    engine: Any,
    cfg: EquityExecConfig,
) -> list[dict[str, Any]]:
    """Pending buy_calls/buy_puts ideas matching the policy that this book has
    not acted on yet (highest confidence first).

    Same shape as the options executor's query — the two books MUST see the
    same candidate set for the A/B to mean anything — except that the NOT
    EXISTS guard reads ``alpaca_equity_orders``, so an idea already bought as
    contracts is still eligible for shares.
    """
    where = [
        "action IN ('buy_calls','buy_puts')",
        "confidence >= :min_conf",
        "status = 'pending'",
        "NOT EXISTS (SELECT 1 FROM alpaca_equity_orders a WHERE a.idea_id = ti.idea_id)",
    ]
    if cfg.require_niche:
        where.append("lower(notes) LIKE '%niche%'")
    if cfg.require_red_team:
        where.append("lower(notes) LIKE '%red-team%'")
    # CL-u59z: where contains only literals above; min_conf is a bind parameter.
    sql = (
        "SELECT idea_id, ticker, action, confidence, preferred_instrument, notes, "  # nosec B608
        "created_at, time_stop_days "
        "FROM trade_ideas ti WHERE "
        + " AND ".join(where)
        + " ORDER BY confidence DESC, created_at DESC"
    )
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql), {"min_conf": cfg.min_confidence})]


def _submitted_since(engine: Any, since: datetime) -> int:
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM alpaca_equity_orders "
                    "WHERE status = 'submitted' AND submitted_at >= :since",
                ),
                {"since": since},
            ).scalar()
            or 0
        )


def _submitted_today(engine: Any, now: datetime) -> int:
    return _submitted_since(engine, now.replace(hour=0, minute=0, second=0, microsecond=0))


def _submitted_last_hour(engine: Any, now: datetime) -> int:
    return _submitted_since(engine, now - timedelta(hours=1))


def _record(engine: Any, row: dict[str, Any]) -> None:
    """Write one terminal decision. ON CONFLICT DO NOTHING makes the idea_id
    dedup authoritative even against a concurrent cycle."""
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO alpaca_equity_orders
                (idea_id, ticker, side, qty, notional_est, entry_price,
                 alpaca_order_id, status, detail, submitted_at)
            VALUES (:idea_id,:ticker,:side,:qty,:notional_est,:entry_price,
                    :alpaca_order_id,:status,:detail,:submitted_at)
            ON CONFLICT (idea_id) DO NOTHING
        """),
            row,
        )


def _fill_price(client: Any, order: dict[str, Any], fallback: float) -> float:
    """The order's REAL average fill, or ``fallback`` (the submit-time price).

    Market orders usually return before the fill, so the POST response carries
    ``filled_avg_price: null``; one read-back catches the common case where it
    filled in between. This matters more here than on the options book: the
    share fill IS the P&L basis, with no mid to reconstruct it from later.
    Fail-soft — a missing fill is not a reason to skip a trade we just made.
    """
    raw: Any = order.get("filled_avg_price")
    order_id = str(order.get("id") or "")
    if raw in (None, "") and order_id:
        get_order = getattr(client, "get_order", None)
        if callable(get_order):
            try:
                fetched = get_order(order_id) or {}
                raw = fetched.get("filled_avg_price")
            except Exception:
                logger.debug("alpaca equity: fill read-back failed for %s", order_id, exc_info=True)
    try:
        price = float(raw) if raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        price = 0.0
    return price if price > 0 else fallback


def execute_pending_equities(
    engine: Any,
    client: AlpacaEquityClient,
    price_fn: PriceFn | None = None,
    cfg: EquityExecConfig | None = None,
    now: datetime | None = None,
    technicals_fn: Any = None,
) -> dict[str, int]:
    """Execute eligible ideas as shares within the caps. Returns a count by
    outcome. Never raises — a per-idea failure is recorded/logged and the loop
    continues (same posture as the options executor)."""
    cfg = cfg or EquityExecConfig()
    now = now or datetime.now(UTC)
    fetch = price_fn or _default_price_fn(engine)
    if technicals_fn is None:
        from src.events.technical_context import compute_for_ticker  # noqa: PLC0415

        technicals_fn = compute_for_ticker
    counts = {
        "submitted": 0,
        "skipped_expiring": 0,
        "skipped_short_disabled": 0,
        "skipped_price_too_high": 0,
        "skipped_concentration": 0,
        "blocked_exposure": 0,
        "no_price": 0,
        "misaligned": 0,
        "entry_delayed": 0,
        "error": 0,
        "market_closed": 0,
    }

    # MARKET orders outside regular hours are rejected — don't even try.
    if not client.is_market_open():
        logger.info("alpaca equity: market closed — no orders this cycle")
        counts["market_closed"] = 1
        return counts

    day_left = max(0, cfg.max_per_day - _submitted_today(engine, now))
    hour_left = max(0, cfg.max_per_hour - _submitted_last_hour(engine, now))
    budget = min(day_left, hour_left)
    if budget <= 0:
        if day_left <= 0:
            logger.info("alpaca equity: daily cap reached — no new orders")
        else:
            logger.info(
                "alpaca equity: hourly pace reached (%d/hr) — %d left today, resuming next cycle",
                cfg.max_per_hour,
                day_left,
            )
        return counts

    delay_active = _entry_delay_active(now, cfg.entry_delay_min)

    for idea in fetch_executable_ideas(engine, cfg):
        if counts["submitted"] >= budget:
            break
        ticker = str(idea.get("ticker") or "")
        side = SIDE_LONG if str(idea.get("action")) == "buy_calls" else SIDE_SHORT
        base_skip = {
            "idea_id": idea["idea_id"],
            "ticker": ticker,
            "side": side,
            "qty": None,
            "notional_est": None,
            "entry_price": None,
            "alpaca_order_id": None,
            "submitted_at": now,
        }
        # Idea about to auto-expire (CL-v2m9). TERMINAL: remaining life only
        # shrinks, so the skip can never un-trigger.
        if cfg.min_idea_life_days > 0 and idea.get("time_stop_days") is not None:
            created = parse_ts(idea.get("created_at"))
            if created is not None:
                remaining = created + timedelta(days=int(idea["time_stop_days"])) - now
                if remaining < timedelta(days=cfg.min_idea_life_days):
                    left_days = remaining.total_seconds() / 86400.0
                    _record(
                        engine,
                        {
                            **base_skip,
                            "status": "skipped_expiring",
                            "detail": (
                                f"idea life left {left_days:.1f}d "
                                f"< {cfg.min_idea_life_days:.1f}d floor"
                            ),
                        },
                    )
                    counts["skipped_expiring"] += 1
                    logger.info(
                        "alpaca equity: skipped %s (%s) — idea expires in %.1fd (< %.1fd floor)",
                        ticker,
                        idea["idea_id"],
                        left_days,
                        cfg.min_idea_life_days,
                    )
                    continue
        # Short side disabled by policy. TERMINAL: the idea's direction never
        # changes, so re-evaluating it every cycle would only burn queries —
        # and if the operator re-enables shorts, that is a NEW policy for NEW
        # ideas, not a licence to open a stale one.
        if side == SIDE_SHORT and not cfg.allow_short:
            _record(
                engine,
                {
                    **base_skip,
                    "status": "skipped_short_disabled",
                    "detail": "buy_puts idea but allow_short is off",
                },
            )
            counts["skipped_short_disabled"] += 1
            logger.info(
                "alpaca equity: skipped %s (%s) — short side disabled", ticker, idea["idea_id"]
            )
            continue
        # Opening-auction delay. Transient (not recorded) — the 5-min loop
        # re-evaluates, so a 9:30 signal simply enters at ~9:45.
        if delay_active and (float(idea.get("confidence") or 0.0) < cfg.entry_delay_override_conf):
            counts["entry_delayed"] += 1
            continue
        try:
            price = fetch(ticker)
            if not price or price <= 0:
                counts["no_price"] += 1
                continue  # transient — retry next cycle (not recorded)
            # Technical-alignment gate (CL-3xoj): don't go long into a
            # confirmed downtrend (or short into an uptrend). Transient —
            # structure changes. Fail-open when no context is computable.
            try:
                ctx = technicals_fn(ticker)
            except Exception:
                ctx = None
            if ctx is not None:
                from src.events.technical_context import alignment_score  # noqa: PLC0415

                direction = "bullish" if side == SIDE_LONG else "bearish"
                score = alignment_score(ctx, direction)
                if score < cfg.min_alignment:
                    counts["misaligned"] += 1
                    logger.info(
                        "alpaca equity: skipped %s %s — technical alignment %.2f < %.2f "
                        "(trend=%s, %s)",
                        ticker,
                        idea.get("action"),
                        score,
                        cfg.min_alignment,
                        ctx.trend,
                        ctx.breakout_state,
                    )
                    continue
            qty = int(math.floor(cfg.notional_usd / price))
            if qty < 1:
                # A single share costs more than the whole sleeve. TERMINAL:
                # rounding UP to one share would silently take a position
                # several times the intended size, which is exactly the
                # accidental-concentration failure the caps exist to prevent.
                _record(
                    engine,
                    {
                        **base_skip,
                        "status": "skipped_price_too_high",
                        "detail": (
                            f"1 share @ ${price:.2f} > ${cfg.notional_usd:.0f} notional sleeve"
                        ),
                    },
                )
                counts["skipped_price_too_high"] += 1
                logger.info(
                    "alpaca equity: skipped %s — share price $%.2f exceeds the $%.0f sleeve",
                    ticker,
                    price,
                    cfg.notional_usd,
                )
                continue

            # CONCENTRATION cap (CL-3nfm) against the LIVE book. Transient:
            # once the earlier position closes, this idea may legitimately
            # enter — recording it terminally would silently drop it.
            with LogContext(idea_id=str(idea["idea_id"]), symbol=ticker, venue="alpaca_equity"):
                try:
                    held = _held_ticker_qty(client, ticker)
                except ExposureUnavailableError as exc:
                    counts["blocked_exposure"] += 1
                    logger.warning(
                        "alpaca equity: entry blocked — exposure unavailable (%s)",
                        exc,
                        extra={
                            "extra_data": {
                                "reason": "blocked_exposure",
                                "detail": str(exc),
                                "idea_id": str(idea["idea_id"]),
                                "symbol": ticker,
                            }
                        },
                    )
                    continue
            # Alpaca NETS equity holdings, so a ticker is either held or not —
            # one position, whatever its size. The cap therefore answers "may
            # we open in a name we are already in?"; share counts don't stack
            # the way option contracts did.
            if (1 if held > 0 else 0) >= cfg.max_positions_per_ticker:
                counts["skipped_concentration"] += 1
                logger.info(
                    "alpaca equity: skipped %s — already hold %s shares (max %d position/ticker)",
                    ticker,
                    held,
                    cfg.max_positions_per_ticker,
                )
                continue

            notional = qty * price
            base = {
                "idea_id": idea["idea_id"],
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "notional_est": notional,
                "submitted_at": now,
            }
            # Alpaca has no "short" side: a sell with no position opens one.
            wire_side = "buy" if side == SIDE_LONG else "sell"
            # client_order_id = curlit-eq-<idea_id> (mirrors the options
            # book's crash-safety, namespaced so the two books can never
            # collide on one idea): a crash after the fill but before the DB
            # row makes the next cycle's resubmit fail Alpaca's uniqueness
            # check, which we recover below as already-executed.
            try:
                logger.info(
                    "alpaca equity: submitting %s entry %d %s [idea %s]",
                    wire_side,
                    qty,
                    ticker,
                    idea["idea_id"],
                )
                order = client.submit_equity_order(
                    ticker,
                    qty,
                    wire_side,
                    client_order_id=f"curlit-eq-{idea['idea_id']}",
                )
            except Exception as sub_exc:
                if _is_duplicate_client_order_id(sub_exc):
                    _record(
                        engine,
                        {
                            **base,
                            "entry_price": price,
                            "alpaca_order_id": "recovered",
                            "status": "submitted",
                            "detail": (
                                "recovered: prior cycle already traded (duplicate client_order_id)"
                            ),
                        },
                    )
                    counts["recovered"] = counts.get("recovered", 0) + 1
                    logger.warning(
                        "alpaca equity: idea %s was ALREADY traded by a prior crashed "
                        "cycle — recorded, not re-traded",
                        idea["idea_id"],
                    )
                    continue
                raise
            fill = _fill_price(client, order, price)
            _record(
                engine,
                {
                    **base,
                    "entry_price": fill,
                    "alpaca_order_id": str(order.get("id") or ""),
                    "status": "submitted",
                    "detail": None,
                },
            )
            counts["submitted"] += 1
            logger.info(
                "alpaca equity: %s %d %s @ ~$%.2f (~$%.0f notional) [idea %s]",
                "BOUGHT" if side == SIDE_LONG else "SHORTED",
                qty,
                ticker,
                fill,
                notional,
                idea["idea_id"],
            )
        except Exception as exc:
            counts["error"] += 1
            logger.warning("alpaca equity: error executing %s: %s", ticker, str(exc)[:200])
    logger.info("alpaca equity: %s", counts)
    return counts
