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

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text

from src.events.prices import parse_ts
from src.execution.alpaca_options import (
    AlpacaOptionsClient,
    ContractSelectionConfig,
    mid_and_spread,
    resolve_contract,
)

logger = logging.getLogger(__name__)

#: ticker -> last price. Injectable; the live default reuses get_prices.
PriceFn = Callable[[str], float | None]


def _is_duplicate_client_order_id(exc: BaseException) -> bool:
    """True when Alpaca rejected the order because our client_order_id was
    already used — i.e. a PRIOR (crashed) cycle already bought this idea.
    Alpaca enforces uniqueness and returns a 422 naming client_order_id."""
    text = str(exc).lower()
    body = ""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = str(getattr(resp, "text", "")).lower()
        except Exception:
            body = ""
    hay = text + " " + body
    return "client_order_id" in hay and ("unique" in hay or "duplicate" in hay or "already" in hay)


def _held_contract_counts(
    client: Any,
    occ_symbol: str,
    underlying: str,
) -> tuple[int, int]:
    """(contracts held of THIS occ_symbol, contracts held on the underlying).

    Reads the live Alpaca book for the concentration cap (CL-3nfm) — the DB
    order log is not enough, since a position can also be closed externally.

    FAIL-OPEN: any lookup problem returns (0, 0) so a broker/API blip cannot
    block entries outright. A cap is a ceiling, not a safety interlock; the
    premium and daily/hourly caps still bound the damage.
    """
    try:
        positions = client.list_option_positions() or []
    except Exception:
        logger.warning(
            "alpaca options: position lookup failed — concentration cap not applied",
            exc_info=True,
        )
        return (0, 0)
    same_symbol = 0
    same_underlying = 0
    root = str(underlying).upper()
    for p in positions:
        sym = str(p.get("symbol") or "")
        try:
            qty = abs(int(float(p.get("qty") or 0)))
        except (TypeError, ValueError):
            qty = 0
        if not sym or qty <= 0:
            continue
        if sym == occ_symbol:
            same_symbol += qty
        # OCC symbols are <ROOT><YYMMDD><C|P><strike>; the root is the leading
        # alpha run, so compare that rather than a prefix match (which would
        # make "AS" collide with "ASC"/"ASTL").
        m = re.match(r"^([A-Z]+)", sym)
        if m and m.group(1) == root:
            same_underlying += qty
    return (same_symbol, same_underlying)


@dataclass(frozen=True)
class OptionsExecConfig:
    min_confidence: float = 0.55
    max_premium_usd: float = 500.0
    qty: int = 1
    max_per_day: int = 10
    #: Intraday pacing (CL-h02l): at most this many new buys per rolling
    #: 60 min, so the daily budget is NOT dumped in one burst at the open —
    #: entries spread across the session and fresh intraday events can still
    #: get bought in the afternoon instead of finding the cap already spent.
    max_per_hour: int = 2
    #: Concentration caps (CL-3nfm). Dedup is keyed on idea_id, so two
    #: DIFFERENT ideas naming the same name — the same theme re-confirmed on a
    #: later event, or the impact agent and the niche agent independently
    #: surfacing it — both pass and both buy. Observed 2026-07-27: FRO
    #: accumulated to qty=3 on ONE contract across two cycles. The FX book has
    #: real concentration caps (per_instrument_max_pct, haven cluster); the
    #: options path had only premium + count caps, nothing per-name.
    #: Contracts already held of the SAME OCC symbol before a buy is skipped.
    max_contracts_per_symbol: int = 1
    #: Open contracts across ALL strikes/expiries of one underlying.
    max_contracts_per_underlying: int = 2
    #: Widest quoted spread (fraction of the ask) we will market-BUY into
    #: (CL-d44a). A contract quoted 0.13/0.42 is a 69% spread: buying the ask
    #: and marking the bid is an instant −69% before the underlying moves,
    #: and it needs a ~69% move just to break even. Measured 2026-07-28,
    #: this is what made 17 of 18 exits losses. 1.01 disables the filter.
    max_entry_spread_pct: float = 0.35
    require_niche: bool = True
    require_red_team: bool = True
    selection: ContractSelectionConfig = ContractSelectionConfig()
    #: No entries in the first N minutes of the regular session — option
    #: spreads are widest right after the 9:30 ET open (day-one lesson:
    #: market-buying cheap OTM contracts in the opening minute paid the
    #: worst spreads of the day and instantly registered −40%+ "losses").
    #: Ideas just wait — the 5-min loop naturally re-evaluates them once
    #: the window passes. 0 disables.
    entry_delay_min: int = 15
    #: Urgency override: during the delay window, ideas at/above this
    #: confidence may enter immediately (extremely strong signal on a
    #: breaking event beats spread costs). 1.01 disables the override.
    entry_delay_override_conf: float = 0.80
    #: Minimum remaining idea life (created_at + time_stop_days − now) to
    #: open a position (CL-v2m9). 2026-07-31: RTX/FLNG/ASC contracts were
    #: bought 1–2 days before their ideas auto-expired, and the
    #: "idea auto-expired by pipeline" exit rule force-closed them the next
    #: morning — the spread paid twice for theses that could not play out.
    #: 0 disables.
    min_idea_life_days: float = 3.0
    #: Minimum assessed urgency of the SOURCE EVENT (CL-khf7). The CL-4c7o
    #: review measured 7% option win rate against an 80% directional hit
    #: on the underlyings — options stay on only as a restricted execution-
    #: learning channel for the strongest signals (operator policy sets 8
    #: via ALPACA_OPT_MIN_URGENCY; shares are the primary expression,
    #: CL-ncbq). When > 0, ideas without a linked event or without an
    #: assessed urgency are EXCLUDED — restriction means provably urgent.
    #: 0 disables (default: no behavior change).
    min_urgency: int = 0
    # Technical-alignment gate (CL-3xoj): skip an idea whose computed price
    # structure is strongly AGAINST the thesis (alignment_score in [-1,1];
    # e.g. buying calls into a confirmed downtrend at the lows = -1.0).
    # Fail-open: no history/context -> no gate (consistent with the repo's
    # missing-data posture). -1.01 disables the gate entirely.
    min_alignment: float = -0.4


_NY = ZoneInfo("America/New_York")


def _entry_delay_active(now: datetime, delay_min: int) -> bool:
    """True while inside the first ``delay_min`` minutes of the regular
    session (measured from 9:30 ET). Negative minutes (pre-open — only
    reachable if the market-open gate was faked) also count as delayed."""
    if delay_min <= 0:
        return False
    ny = now.astimezone(_NY)
    session_open = ny.replace(hour=9, minute=30, second=0, microsecond=0)
    return (ny - session_open).total_seconds() < delay_min * 60


def _default_price_fn(engine: Any) -> PriceFn:
    def _fetch(ticker: str) -> float | None:
        from src.events.prices import get_prices  # noqa: PLC0415

        out = get_prices([ticker], engine=engine)
        v = out.get(ticker)
        return float(v["price"]) if v and v.get("price") is not None else None

    return _fetch


def fetch_executable_ideas(
    engine: Any,
    cfg: OptionsExecConfig,
) -> list[dict[str, Any]]:
    """Pending buy_calls/buy_puts ideas matching the policy that haven't been
    acted on yet (highest confidence first)."""
    where = [
        "ti.action IN ('buy_calls','buy_puts')",
        "ti.confidence >= :min_conf",
        "ti.status = 'pending'",
        "NOT EXISTS (SELECT 1 FROM alpaca_option_orders a WHERE a.idea_id = ti.idea_id)",
    ]
    if cfg.require_niche:
        where.append("lower(ti.notes) LIKE '%niche%'")
    if cfg.require_red_team:
        where.append("lower(ti.notes) LIKE '%red-team%'")
    sql = (
        "SELECT ti.idea_id, ti.ticker, ti.action, ti.confidence, ti.preferred_instrument, "
        "ti.notes, ti.created_at, ti.time_stop_days, g.assessment AS event_assessment "
        "FROM trade_ideas ti LEFT JOIN geo_events g ON g.id = ti.geo_event_id WHERE "
        + " AND ".join(where)
        + " ORDER BY ti.confidence DESC, ti.created_at DESC"
    )
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(text(sql), {"min_conf": cfg.min_confidence})]
    out: list[dict[str, Any]] = []
    below = 0
    for row in rows:
        assessment = row.pop("event_assessment", None)
        if cfg.min_urgency > 0 and _event_urgency(assessment) < cfg.min_urgency:
            below += 1
            continue
        out.append(row)
    if below:
        logger.info(
            "alpaca options: %d idea(s) below the urgency floor %d — options are the "
            "restricted channel (CL-khf7); shares express the rest",
            below,
            cfg.min_urgency,
        )
    return out


def _event_urgency(assessment: Any) -> int:
    """Assessed urgency of an idea's source event; 0 when unknowable."""
    if isinstance(assessment, str) and assessment.strip():
        try:
            assessment = json.loads(assessment)
        except json.JSONDecodeError:
            return 0
    if not isinstance(assessment, dict):
        return 0
    try:
        return int(assessment.get("urgency") or 0)
    except (TypeError, ValueError):
        return 0


def _submitted_since(engine: Any, since: datetime) -> int:
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM alpaca_option_orders "
                    "WHERE status = 'submitted' AND submitted_at >= :since",
                ),
                {"since": since},
            ).scalar()
            or 0
        )


def _submitted_today(engine: Any, now: datetime) -> int:
    return _submitted_since(
        engine,
        now.replace(hour=0, minute=0, second=0, microsecond=0),
    )


def _submitted_last_hour(engine: Any, now: datetime) -> int:
    return _submitted_since(engine, now - timedelta(hours=1))


def _record(engine: Any, row: dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO alpaca_option_orders
                (idea_id, ticker, occ_symbol, opt_type, qty, premium_est,
                 alpaca_order_id, status, detail, submitted_at,
                 entry_mid, entry_spread_pct)
            VALUES (:idea_id,:ticker,:occ_symbol,:opt_type,:qty,:premium_est,
                    :alpaca_order_id,:status,:detail,:submitted_at,
                    :entry_mid,:entry_spread_pct)
            ON CONFLICT (idea_id) DO NOTHING
        """),
            row,
        )


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
    counts = {
        "submitted": 0,
        "skipped_premium": 0,
        "skipped_concentration": 0,
        "skipped_spread": 0,
        "skipped_expiring": 0,
        "no_contract": 0,
        "no_quote": 0,
        "no_price": 0,
        "error": 0,
        "market_closed": 0,
        "misaligned": 0,
        "entry_delayed": 0,
    }

    # Options MARKET orders are 422-rejected outside regular hours — don't even
    # try; just wait for the next open (CL-ldd2).
    if not client.is_market_open():
        logger.info("alpaca options: market closed — no orders this cycle")
        counts["market_closed"] = 1
        return counts

    day_left = max(0, cfg.max_per_day - _submitted_today(engine, now))
    hour_left = max(0, cfg.max_per_hour - _submitted_last_hour(engine, now))
    budget = min(day_left, hour_left)
    if budget <= 0:
        if day_left <= 0:
            logger.info("alpaca options: daily cap reached — no new orders")
        else:
            logger.info(
                "alpaca options: hourly pace reached (%d/hr) — %d left today, resuming next cycle",
                cfg.max_per_hour,
                day_left,
            )
        return counts

    delay_active = _entry_delay_active(now, cfg.entry_delay_min)

    for idea in fetch_executable_ideas(engine, cfg):
        if counts["submitted"] >= budget:
            break
        ticker = str(idea.get("ticker") or "")
        # Idea about to auto-expire (CL-v2m9): with less life left than the
        # floor, the "idea auto-expired by pipeline" exit rule would
        # force-close the position almost immediately — churn that pays the
        # spread twice. TERMINAL (recorded): remaining life only shrinks,
        # so the skip can never un-trigger.
        if cfg.min_idea_life_days > 0 and idea.get("time_stop_days") is not None:
            created = parse_ts(idea.get("created_at"))
            if created is not None:
                remaining = created + timedelta(days=int(idea["time_stop_days"])) - now
                if remaining < timedelta(days=cfg.min_idea_life_days):
                    left_days = remaining.total_seconds() / 86400.0
                    _record(
                        engine,
                        {
                            "idea_id": idea["idea_id"],
                            "ticker": ticker,
                            "occ_symbol": None,
                            "opt_type": (
                                "call" if str(idea.get("action")) == "buy_calls" else "put"
                            ),
                            "qty": cfg.qty,
                            "premium_est": None,
                            "alpaca_order_id": None,
                            "status": "skipped_expiring",
                            "detail": (
                                f"idea life left {left_days:.1f}d "
                                f"< {cfg.min_idea_life_days:.1f}d floor"
                            ),
                            "submitted_at": now,
                            "entry_mid": None,
                            "entry_spread_pct": None,
                        },
                    )
                    counts["skipped_expiring"] += 1
                    logger.info(
                        "alpaca options: skipped %s (%s) — idea expires in %.1fd "
                        "(< %.1fd floor)",
                        ticker,
                        idea["idea_id"],
                        left_days,
                        cfg.min_idea_life_days,
                    )
                    continue
        # Open-spread protection: no entries in the first N minutes of the
        # session unless the signal is extremely strong. Transient (not
        # recorded) — the idea re-evaluates on the next 5-min cycle, so a
        # 9:30 signal simply enters at ~9:45 instead.
        if delay_active and (float(idea.get("confidence") or 0.0) < cfg.entry_delay_override_conf):
            counts["entry_delayed"] += 1
            continue
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
                        "%.2f < %.2f (trend=%s, %s)",
                        ticker,
                        idea.get("action"),
                        score,
                        cfg.min_alignment,
                        ctx.trend,
                        ctx.breakout_state,
                    )
                    continue
            contract = resolve_contract(client, idea, price, now.date(), cfg.selection)
            if not contract:
                counts["no_contract"] += 1
                continue
            occ = str(contract.get("symbol") or "")
            # One quote for the ask, the spread filter, and the entry-mid
            # baseline (CL-d44a). Falls back to the ask-only lookup for
            # clients that predate get_option_quote.
            bid: float | None = None
            ask: float | None = None
            get_quote = getattr(client, "get_option_quote", None)
            if callable(get_quote):
                try:
                    bid, ask = get_quote(occ)
                except Exception:
                    logger.debug("alpaca options: quote probe failed for %s", occ, exc_info=True)
            if ask is None:
                ask = client.get_option_ask(occ)
            if ask is None:
                counts["no_quote"] += 1
                continue
            entry_mid, spread = mid_and_spread(bid, ask)
            # SPREAD filter: a market buy into a very wide book pays the whole
            # spread up front and then needs a move that size just to break
            # even — and the bid-marked stop machinery liquidates it long
            # before that. Skipped as a TRANSIENT miss (not recorded), so the
            # idea retries when the market tightens.
            if spread is not None and spread > cfg.max_entry_spread_pct:
                counts["skipped_spread"] += 1
                logger.info(
                    "alpaca options: skipped %s (%s) — spread %.0f%% of ask "
                    "(bid %.2f / ask %.2f) > %.0f%% cap",
                    ticker,
                    occ,
                    spread * 100,
                    bid or 0.0,
                    ask,
                    cfg.max_entry_spread_pct * 100,
                )
                continue
            premium = ask * 100.0 * cfg.qty
            base = {
                "idea_id": idea["idea_id"],
                "ticker": ticker,
                "occ_symbol": occ,
                "opt_type": right,
                "qty": cfg.qty,
                "premium_est": premium,
                "submitted_at": now,
                # Baseline for honest MID-to-MID exit P&L (CL-d44a, mig 018).
                "entry_mid": entry_mid,
                "entry_spread_pct": spread,
            }
            if premium > cfg.max_premium_usd:
                _record(
                    engine,
                    {
                        **base,
                        "alpaca_order_id": None,
                        "status": "skipped_premium",
                        "detail": f"premium ${premium:.0f} > ${cfg.max_premium_usd:.0f}",
                    },
                )
                counts["skipped_premium"] += 1
                logger.info(
                    "alpaca options: skipped %s (%s) premium $%.0f > cap", ticker, occ, premium
                )
                continue

            # CONCENTRATION cap (CL-3nfm): consult what we ALREADY hold before
            # adding. idea_id dedup can't see this — two different ideas naming
            # the same contract both pass it — so repeated confirmations on one
            # theme silently stacked size in a single name (FRO reached qty=3).
            held_symbol, held_underlying = _held_contract_counts(client, occ, ticker)
            if held_symbol >= cfg.max_contracts_per_symbol:
                counts["skipped_concentration"] += 1
                logger.info(
                    "alpaca options: skipped %s (%s) — already hold %d of this "
                    "contract (max %d/symbol)",
                    ticker,
                    occ,
                    held_symbol,
                    cfg.max_contracts_per_symbol,
                )
                continue
            if held_underlying >= cfg.max_contracts_per_underlying:
                counts["skipped_concentration"] += 1
                logger.info(
                    "alpaca options: skipped %s (%s) — already hold %d contracts on "
                    "%s (max %d/underlying)",
                    ticker,
                    occ,
                    held_underlying,
                    ticker,
                    cfg.max_contracts_per_underlying,
                )
                continue
            # client_order_id = idea_id (review P1): the order-then-record
            # sequence could double-buy if we crashed after the fill but
            # before the DB row — the next cycle re-fetched the idea and
            # bought again. Alpaca's client_order_id uniqueness makes the
            # resubmit fail, which we recover below as already-executed.
            try:
                order = client.submit_option_order(
                    occ,
                    cfg.qty,
                    "buy",
                    client_order_id=f"curlit-{idea['idea_id']}",
                )
            except Exception as sub_exc:
                if _is_duplicate_client_order_id(sub_exc):
                    _record(
                        engine,
                        {
                            **base,
                            "alpaca_order_id": "recovered",
                            "status": "submitted",
                            "detail": "recovered: prior cycle already "
                            "bought (duplicate "
                            "client_order_id)",
                        },
                    )
                    counts["recovered"] = counts.get("recovered", 0) + 1
                    logger.warning(
                        "alpaca options: idea %s was ALREADY bought by a "
                        "prior crashed cycle — recorded, not re-bought",
                        idea["idea_id"],
                    )
                    continue
                raise
            _record(
                engine,
                {
                    **base,
                    "alpaca_order_id": str(order.get("id") or ""),
                    "status": "submitted",
                    "detail": None,
                },
            )
            counts["submitted"] += 1
            logger.info(
                "alpaca options: BOUGHT %d %s (%s) ~$%.0f premium [idea %s]",
                cfg.qty,
                occ,
                ticker,
                premium,
                idea["idea_id"],
            )
        except Exception as exc:
            counts["error"] += 1
            logger.warning("alpaca options: error executing %s: %s", ticker, str(exc)[:200])
    logger.info("alpaca options: %s", counts)
    return counts
