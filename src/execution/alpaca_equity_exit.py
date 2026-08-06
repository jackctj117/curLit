"""Alpaca paper EQUITY (shares) EXIT manager (CL-ncbq).

Completes the share book — idea → entry (``alpaca_equity_executor``) →
monitoring → EXIT — as the twin of the options exit manager (CL-3rho). Each
cycle every open paper share position is matched back to its originating idea
(the ``alpaca_equity_orders`` row keyed by idea_id) and evaluated against
prioritized rules; the FIRST matching rule closes the FULL position with a
market order and records the reason on the same row.

Priority order (first hit wins):
  1. thesis_invalidated — the originating geo_event was DISMISSED or the idea
     cancelled: the desk ACTIVELY no longer believes the story. (geo_event
     EXPIRED does NOT count — that is the ~2h intraday FX-confluence window
     lapsing, not a verdict on a multi-day equity thesis.)
  2. time_stop         — held >= the idea's ``time_stop_days`` (config default
     when the idea carries none), or the pipeline already auto-expired it.
  3. stop_loss         — the UNDERLYING moved against us by the idea's OWN
     advisory ``stop_loss_pct`` when it has one, else the config default.
  4. profit_target     — likewise, the idea's own advisory target if it has
     one, else the config default.
  5. stale             — safety net at time stop + grace; only reachable when
     rule 2 could not evaluate earlier (e.g. an unparseable entry timestamp
     that later heals).

WHY the thresholds are so much tighter than the options book's ±40/80%
(CL-4c7o): those were premium percentages on a levered instrument. Here a
5% stop and a 10% target are moves in the SHARE, which is the size of move
the desk's ideas were actually producing — direction was right 80% of the
time (41/51) while the options expressing it won 7%.

Things this manager deliberately does NOT have, all downstream of trading
shares instead of contracts:
  * no expiry_protect / expired_worthless — shares do not expire, so the
    idea's own deadline is the ONLY clock;
  * no wide-spread stop suppression and no unsellable/no-bid guard — a liquid
    US equity always has a two-sided penny market, which is the whole reason
    this book exists;
  * no ``entry_mid`` reconstruction — the share fill IS the basis.

Safety posture (mirrored):
  * Alpaca positions with NO matching row are flagged and never auto-managed.
  * A vanished position is finalized honestly: a pending exit is confirmed
    closed, anything else becomes ``closed_external``.
  * Exits reuse crash-safe client_order_id dedup (``curlit-eq-exit-<idea_id>``).
  * Market-hours gated (MARKET orders are rejected outside RTH).
  * Missing price data skips only the P&L rules this cycle; the date-based
    rules still run, so bad data cannot hold a position forever.
  * ``trade_ideas`` is NOT marked closed — the options book may hold the same
    idea, and the two books are a deliberate A/B (CL-ncbq).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import text

from src.execution.alpaca_equity import AlpacaEquityClient
from src.execution.alpaca_equity_executor import SIDE_SHORT
from src.execution.alpaca_options_executor import _is_duplicate_client_order_id

logger = logging.getLogger(__name__)


class ExitReason(StrEnum):
    THESIS_INVALIDATED = "thesis_invalidated"
    TIME_STOP = "time_stop"
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"
    STALE = "stale"
    # Finalization-only reason (no order was needed/possible):
    CLOSED_EXTERNAL = "closed_external"


@dataclass(frozen=True)
class EquityExitConfig:
    #: Close when the SHARE is down this fraction from entry (signed for the
    #: side). Used only when the idea carries no advisory stop of its own.
    stop_loss_pct: float = 0.05
    #: Close when the SHARE is up this fraction from entry. Same fallback rule.
    profit_target_pct: float = 0.10
    #: Time stop when the idea itself carries no time_stop_days.
    default_time_stop_days: int = 10
    #: Safety-net margin past the time stop (rule 5). Wider than the options
    #: book's 2 days because nothing here decays: an extra few days on a share
    #: position costs nothing but opportunity, so the net can be loose.
    stale_grace_days: int = 5


# --------------------------------------------------------------------------- #
# parsing helpers (DB rows arrive as Decimal from Postgres NUMERIC, str from
# the sqlite test engine, and None whenever the pipeline had no number)
# --------------------------------------------------------------------------- #


def _as_dt(value: Any) -> datetime | None:
    """Parse a DB timestamp (datetime from Postgres, str from sqlite tests)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


def _as_float(value: Any) -> float | None:
    """Numeric DB value → float, or None.

    Postgres NUMERIC hands back ``Decimal``, which cannot be mixed with the
    float prices coming off the quote API (``Decimal - float`` raises). Every
    number that crosses that boundary goes through here — a bug the sqlite
    tests could never catch, since the fixtures shim NUMERIC to FLOAT.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_or_none(value: Any) -> float | None:
    """An advisory percentage the LLM produced, sanity-checked.

    Only a strictly positive fraction BELOW 1.0 is usable: 0 would exit
    instantly and >= 1 would mean a 100% move in a share, which is not a stop
    the pipeline meant. Anything else → None (caller uses the config default)
    rather than a silently absurd threshold.
    """
    pct = _as_float(value)
    if pct is None or pct <= 0.0 or pct >= 1.0:
        return None
    return pct


def idea_stop_pct(row: dict[str, Any], cfg: EquityExitConfig) -> float:
    """The idea's OWN advisory stop fraction, else the config default.

    The impact agent already reasoned about how much adverse move kills this
    specific thesis (``trade_ideas.stop_loss_pct``, mig 007); a flat 5% would
    throw that away and stop a volatile name out on noise.
    """
    return _pct_or_none(row.get("stop_loss_pct")) or cfg.stop_loss_pct


def idea_target_pct(row: dict[str, Any], cfg: EquityExitConfig) -> float:
    """The idea's OWN advisory profit target fraction, else the default.

    ``trade_ideas`` stores the agent's targets as DOLLAR levels
    (``target_prices``, mig 008) computed from its percentages against
    ``price_at_signal``; the raw ``target_pct`` list never got its own column.
    So: use an explicit ``target_pct`` if a caller supplies one, else recover
    the fraction from the first dollar target relative to the signal price.
    ``abs()`` is correct for both sides — a short idea's target sits BELOW the
    signal price, and P&L is signed for the side separately.
    """
    explicit = row.get("target_pct")
    if isinstance(explicit, list | tuple):
        explicit = explicit[0] if explicit else None
    pct = _pct_or_none(explicit)
    if pct is not None:
        return pct
    signal = _as_float(row.get("price_at_signal"))
    targets = _parse_target_prices(row.get("target_prices"))
    if signal and signal > 0 and targets:
        return _pct_or_none(abs(targets[0] - signal) / signal) or cfg.profit_target_pct
    return cfg.profit_target_pct


def _parse_target_prices(value: Any) -> list[float]:
    """``target_prices`` → list of floats. Postgres JSONB hands back a list;
    sqlite (tests) stores the JSON string. Garbage → [] (never a raise)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if not isinstance(value, list | tuple):
        return []
    out: list[float] = []
    for item in value:
        num = _as_float(item)
        if num is not None:
            out.append(num)
    return out


def _quote_mid(quote: tuple[float | None, float | None] | None) -> float | None:
    """Mid of a two-sided quote, or None. A one-sided or crossed book yields
    no usable mid and callers must NOT invent one."""
    if not quote:
        return None
    bid, ask = quote
    if bid is None or ask is None or ask <= 0 or bid <= 0 or bid > ask:
        return None
    return (bid + ask) / 2.0


def current_price(
    pos: dict[str, Any], quote: tuple[float | None, float | None] | None
) -> float | None:
    """What the share is worth right now: the quote MID when a two-sided quote
    is available, else Alpaca's own position mark. None when neither exists —
    the P&L rules then skip this cycle rather than fabricate a number."""
    mid = _quote_mid(quote)
    if mid is not None:
        return mid
    return _as_float(pos.get("current_price"))


def signed_pnl_pct(side: str, entry: float | None, price: float | None) -> float | None:
    """Return on the position, SIGNED for the side: a short that fell is a
    WIN. Everything downstream (stop, target, the recorded pnl_pct) reads this
    one number, so the short book can never be judged by long-side signs.

    Coerces defensively — a Postgres ``Decimal`` reaching here directly
    (bypassing :func:`entry_basis`/:func:`current_price`) would raise on the
    float mix, and this seam is exactly one refactor away from that."""
    entry = _as_float(entry)
    price = _as_float(price)
    if not entry or entry <= 0 or price is None or price <= 0:
        return None
    raw = (price - entry) / entry
    return -raw if side == SIDE_SHORT else raw


def entry_basis(row: dict[str, Any], pos: dict[str, Any]) -> float | None:
    """Entry price for P&L: the recorded fill, falling back to Alpaca's
    ``avg_entry_price``. The fallback covers the row written before the fill
    was reported, where the submit-time last price was stored instead."""
    return _as_float(row.get("entry_price")) or _as_float(pos.get("avg_entry_price"))


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #


def evaluate_equity_exit(
    row: dict[str, Any],
    pos: dict[str, Any],
    cfg: EquityExitConfig,
    now: datetime,
    quote: tuple[float | None, float | None] | None = None,
) -> tuple[ExitReason, str] | None:
    """First matching exit rule for one open share position, or None (hold).

    Pure decision function — all I/O stays in :func:`manage_equity_exits`.
    """
    side = str(row.get("side") or "")
    entry = entry_basis(row, pos)
    price = current_price(pos, quote)
    pnl = signed_pnl_pct(side, entry, price)
    entered = _as_dt(row.get("submitted_at"))
    days_held = (now - entered).days if entered else None
    time_stop = int(row.get("time_stop_days") or cfg.default_time_stop_days)

    # 1. Thesis invalidated — the desk ACTIVELY no longer believes the story.
    #    Deliberately NOT geo_event EXPIRED: that is the intraday FX gate
    #    lapsing (median EXPIRED lifetime ~2h), a gate lifecycle rather than a
    #    verdict; the equity deadline is the idea's own time stop (rule 2).
    if str(row.get("event_status") or "").upper() == "DISMISSED":
        return (ExitReason.THESIS_INVALIDATED, "geo_event dismissed")
    if str(row.get("idea_status") or "") == "cancelled":
        return (ExitReason.THESIS_INVALIDATED, "idea cancelled")

    # 2. Hard time stop — the idea's own deadline (pipeline auto-expiry is the
    #    same clock observed from the other side). Shares don't decay, but an
    #    event thesis still has a shelf life: past it we are holding beta.
    if days_held is not None and days_held >= time_stop:
        return (ExitReason.TIME_STOP, f"held {days_held}d >= time stop {time_stop}d")
    if str(row.get("idea_status") or "") == "expired":
        return (ExitReason.TIME_STOP, "idea auto-expired by pipeline")

    # 3/4. Price rules. No entry-day grace and no spread guard here (the
    #      options book needs both because a cheap OTM contract's own spread
    #      reads as a -40% "loss" on day one); a penny-wide share quote means
    #      an adverse move is a real adverse move from the first minute.
    if pnl is not None:
        stop_pct = idea_stop_pct(row, cfg)
        if pnl <= -stop_pct:
            return (
                ExitReason.STOP_LOSS,
                f"{'short' if side == SIDE_SHORT else 'long'} {pnl:+.1%} <= -{stop_pct:.1%}",
            )
        target_pct = idea_target_pct(row, cfg)
        if pnl >= target_pct:
            return (
                ExitReason.PROFIT_TARGET,
                f"{'short' if side == SIDE_SHORT else 'long'} {pnl:+.1%} >= +{target_pct:.1%}",
            )

    # 5. Stale safety net (normally unreachable past rule 2).
    if days_held is not None and days_held >= time_stop + cfg.stale_grace_days:
        return (ExitReason.STALE, f"held {days_held}d > time stop + {cfg.stale_grace_days}d")

    return None


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def _fetch_open_rows(engine: Any) -> list[dict[str, Any]]:
    """Real positions (status='submitted') not yet finalized, joined to their
    idea + originating event for the thesis check and the ADVISORY levels."""
    sql = """
        SELECT a.idea_id, a.ticker, a.side, a.qty, a.entry_price,
               a.submitted_at, a.exit_status, a.exit_reason, a.exit_order_id,
               -- The idea's OWN levels drive rules 3/4 (config is only the
               -- fallback), so they must be SELECTed: a missing column here
               -- would silently degrade every exit to the flat defaults.
               ti.time_stop_days, ti.stop_loss_pct, ti.target_prices,
               ti.price_at_signal, ti.status AS idea_status,
               ge.status AS event_status
        FROM alpaca_equity_orders a
        LEFT JOIN trade_ideas ti ON ti.idea_id = a.idea_id
        LEFT JOIN geo_events ge ON ge.id = ti.geo_event_id
        WHERE a.status = 'submitted'
          AND (a.exit_status IS NULL OR a.exit_status = 'submitted')
    """
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(sql))]


def _mark_exit(
    engine: Any,
    idea_id: str,
    now: datetime,
    *,
    exit_status: str,
    exit_reason: str | None = None,
    exit_order_id: str | None = None,
    exit_price: float | None = None,
    pnl_pct: float | None = None,
    overwrite_fill: bool = False,
) -> None:
    """Record the exit on the SAME order row.

    COALESCE keeps first-write values (the reason recorded at submit survives
    the later fill confirmation). ``overwrite_fill`` is the one exception: on
    confirmation we may know the REAL average fill, which is strictly more
    honest than the submit-time estimate — and unlike the options book there
    is no mid to reconstruct it from later, so it must be written now.

    Deliberately does NOT touch ``trade_ideas`` (the options exit does): the
    options book may hold the same idea, and closing it here would both
    corrupt that book's view and end the A/B early (CL-ncbq).
    """
    fill_sql = (
        "exit_price = :ep, pnl_pct = :pp"
        if overwrite_fill
        else "exit_price = COALESCE(exit_price, :ep), pnl_pct = COALESCE(pnl_pct, :pp)"
    )
    with engine.begin() as conn:
        conn.execute(
            text(f"""
            UPDATE alpaca_equity_orders SET
                exit_status   = :es,
                exit_reason   = COALESCE(exit_reason, :er),
                exit_order_id = COALESCE(exit_order_id, :eo),
                {fill_sql},
                exited_at     = COALESCE(exited_at, :ea)
            WHERE idea_id = :i
        """),
            {
                "es": exit_status,
                "er": exit_reason,
                "eo": exit_order_id,
                "ep": exit_price,
                "pp": pnl_pct,
                "ea": now,
                "i": idea_id,
            },
        )


def _confirm_fill(
    engine: Any,
    client: Any,
    row: dict[str, Any],
    now: datetime,
) -> bool:
    """True when a submitted exit order has FILLED, recording the real fill.

    The position list is the primary signal (a gone position = a closed one),
    but Alpaca can report the fill before the book refreshes; asking the order
    directly closes that window instead of leaving the row 'submitted' for a
    cycle. Fail-soft: an unreadable order is simply "not confirmed yet".
    """
    order_id = str(row.get("exit_order_id") or "")
    if not order_id or order_id == "recovered":
        return False
    get_order = getattr(client, "get_order", None)
    if not callable(get_order):
        return False
    try:
        order = get_order(order_id) or {}
    except Exception:
        logger.debug("equity exit: order probe failed for %s", order_id, exc_info=True)
        return False
    if str(order.get("status") or "").lower() != "filled":
        return False
    fill = _as_float(order.get("filled_avg_price"))
    entry = _as_float(row.get("entry_price"))
    pnl = signed_pnl_pct(str(row.get("side") or ""), entry, fill)
    _mark_exit(
        engine,
        str(row["idea_id"]),
        now,
        exit_status="closed",
        exit_price=fill,
        pnl_pct=pnl,
        overwrite_fill=fill is not None,
    )
    return True


def _finalize_vanished(
    engine: Any,
    row: dict[str, Any],
    now: datetime,
    counts: dict[str, int],
) -> None:
    """Row says open, Alpaca shows no position — record the truth. Shares
    cannot expire worthless, so there are only two cases."""
    idea_id = str(row["idea_id"])
    if row.get("exit_status") == "submitted":
        # Our close filled — confirm it (reason/price kept from submit time).
        _mark_exit(engine, idea_id, now, exit_status="closed")
        counts["closed_confirmed"] += 1
        logger.info(
            "equity exit: %s close CONFIRMED [%s]", row.get("ticker"), row.get("exit_reason")
        )
        return
    _mark_exit(engine, idea_id, now, exit_status="closed", exit_reason=ExitReason.CLOSED_EXTERNAL)
    counts["closed_external"] += 1
    logger.warning(
        "equity exit: %s closed OUTSIDE the manager — marked closed_external", row.get("ticker")
    )


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #


def manage_equity_exits(
    engine: Any,
    client: AlpacaEquityClient,
    cfg: EquityExitConfig | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Evaluate every open share position and close what the rules say.

    Run BEFORE entries each cycle (manage what we hold, then buy). Returns a
    count by outcome. Never raises — a per-position failure is logged and the
    loop continues."""
    cfg = cfg or EquityExitConfig()
    now = now or datetime.now(UTC)
    counts: dict[str, int] = {
        "held": 0,
        "exit_submitted": 0,
        "exit_pending": 0,
        "closed_confirmed": 0,
        "closed_external": 0,
        "unmatched": 0,
        "recovered": 0,
        "error": 0,
        "market_closed": 0,
    }

    # Closing orders are market orders too — rejected outside regular hours.
    if not client.is_market_open():
        logger.info("equity exit: market closed — no evaluation this cycle")
        counts["market_closed"] = 1
        return counts

    try:
        positions = {str(p.get("symbol") or "").upper(): p for p in client.list_equity_positions()}
    except Exception:
        logger.warning("equity exit: positions fetch failed — retry next cycle", exc_info=True)
        counts["error"] += 1
        return counts

    for row in _fetch_open_rows(engine):
        idea_id = str(row["idea_id"])
        ticker = str(row.get("ticker") or "").upper()
        pos = positions.pop(ticker, None)
        try:
            if pos is None:
                _finalize_vanished(engine, row, now, counts)
                continue
            if row.get("exit_status") == "submitted":
                # Close already placed — never resubmit (the client_order_id
                # dedup would reject it anyway). Ask the order whether it
                # filled; if it did, record the real fill now.
                if _confirm_fill(engine, client, row, now):
                    counts["closed_confirmed"] += 1
                    logger.info(
                        "equity exit: %s close CONFIRMED via order status [%s]",
                        ticker,
                        row.get("exit_reason"),
                    )
                else:
                    counts["exit_pending"] += 1
                continue
            # ONE quote per position per cycle, reused for the mark and the
            # recorded P&L. FAIL-SOFT: (None, None) degrades to Alpaca's own
            # position mark rather than skipping the position.
            quote: tuple[float | None, float | None] = (None, None)
            get_quote = getattr(client, "get_stock_quote", None)
            if callable(get_quote):
                try:
                    quote = get_quote(ticker)
                except Exception:
                    logger.debug("equity exit: quote probe failed for %s", ticker, exc_info=True)

            decision = evaluate_equity_exit(row, pos, cfg, now, quote=quote)
            if decision is None:
                counts["held"] += 1
                continue
            reason, detail = decision
            qty = abs(int(float(pos.get("qty") or row.get("qty") or 0)))
            if qty <= 0:
                counts["error"] += 1
                logger.warning(
                    "equity exit: %s has non-positive qty %r — not closing", ticker, pos.get("qty")
                )
                continue
            side = str(row.get("side") or "")
            price = current_price(pos, quote)
            pnl = signed_pnl_pct(side, entry_basis(row, pos), price)
            # Sell to close a long, buy to cover a short.
            wire_side = "buy" if side == SIDE_SHORT else "sell"
            try:
                order = client.submit_equity_order(
                    ticker,
                    qty,
                    wire_side,
                    client_order_id=f"curlit-eq-exit-{idea_id}",
                )
            except Exception as sub_exc:
                if _is_duplicate_client_order_id(sub_exc):
                    # Prior crashed cycle already closed — record, don't retry.
                    _mark_exit(
                        engine,
                        idea_id,
                        now,
                        exit_status="submitted",
                        exit_reason=reason,
                        exit_order_id="recovered",
                        exit_price=price,
                        pnl_pct=pnl,
                    )
                    counts["recovered"] += 1
                    logger.warning(
                        "equity exit: %s was ALREADY being closed by a prior crashed "
                        "cycle — recorded",
                        ticker,
                    )
                    continue
                raise
            _mark_exit(
                engine,
                idea_id,
                now,
                exit_status="submitted",
                exit_reason=reason,
                exit_order_id=str(order.get("id") or ""),
                exit_price=price,
                pnl_pct=pnl,
            )
            counts["exit_submitted"] += 1
            logger.info(
                "equity exit: %s %d %s — %s (%s); pnl=%s [idea %s]",
                "COVERING" if side == SIDE_SHORT else "SELLING",
                qty,
                ticker,
                reason,
                detail,
                f"{pnl:+.1%}" if pnl is not None else "n/a",
                idea_id,
            )
        except Exception:
            counts["error"] += 1
            logger.warning("equity exit: error managing %s", ticker, exc_info=True)

    # Anything Alpaca holds that we can't explain: flag, never auto-manage.
    # The account is shared with the options book and with any manual trade,
    # so "not ours" is the common case — we don't sell what we can't explain.
    for symbol, _pos in positions.items():
        counts["unmatched"] += 1
        logger.warning(
            "equity exit: UNMATCHED Alpaca position %s — no alpaca_equity_orders "
            "row; NOT auto-managed",
            symbol,
        )

    logger.info("equity exit: %s", counts)
    return counts
