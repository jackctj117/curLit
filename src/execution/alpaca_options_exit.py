"""Alpaca paper options EXIT manager (CL-3rho).

Completes the options loop — idea → entry (``alpaca_options_executor``) →
monitoring → EXIT. Each cycle, every open paper option position is matched
back to its originating idea (the ``alpaca_option_orders`` row keyed by
idea_id) and evaluated against prioritized exit rules; the FIRST matching
rule closes the FULL position with a market sell-to-close and records the
exact reason on the same row.

Priority order (first hit wins):
  1. thesis_invalidated — the originating geo_event was DISMISSED or the
     trade idea cancelled: the desk ACTIVELY no longer believes the story.
     (geo_event EXPIRED does NOT trigger this — that's the ~2h intraday
     FX-confluence window lapsing, not an options-thesis verdict.)
  2. time_stop        — held ≥ the idea's ``time_stop_days`` (config default
     when the idea carries none), or the pipeline already auto-expired the
     idea. Event options are theses with deadlines; the deadline is the exit.
  3. stop_loss        — premium down ≥ ``stop_loss_pct`` from entry.
     DISABLED on the entry day (NY calendar date): opening spreads on
     cheap OTM contracts register as instant −40% "losses" without any
     real move. The ``entry_day_extreme_stop_pct`` safety valve (−60%)
     stays active even then.
  4. profit_target    — premium up ≥ ``profit_target_pct`` from entry.
  5. expiry_protect   — ≤ ``expiry_protect_days`` to expiration and not
     meaningfully profitable (< ``expiry_protect_min_profit``); a
     near-expiry winner is left to run. Backstop: at ≤ ``final_day_dte``
     the position closes REGARDLESS of P&L — an event option is never
     ridden into expiry without a decision, and closing IS the decision.
     Date-based (parsed from the OCC symbol), so it fires even when
     quotes are missing.
  6. stale            — safety net at time_stop + grace; only reachable if
     rule 2 could not evaluate on earlier cycles (e.g. unparseable entry
     timestamp that later heals).

Safety posture:
  * Alpaca positions with NO matching row are flagged (WARN + counted) and
    never auto-managed — we don't sell what we can't explain.
  * A vanished position (row open, Alpaca shows nothing) is finalized
    honestly: a pending sell is confirmed closed; a past-expiry row becomes
    ``expired_worthless`` (−100%); anything else is ``closed_external``.
  * Sells reuse the crash-safe client_order_id dedup
    (``curlit-exit-<idea_id>``): a crash between fill and record can never
    double-sell — the resubmit is 422-rejected and recovered as closed.
  * Market-hours gated (options MARKET orders are 422-rejected outside RTH).
  * Missing price data skips only the P&L rules for that cycle (retry next
    time); the date-based rules still run, so bad data can't hold a
    position forever.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text

from src.execution.alpaca_options import AlpacaOptionsClient, mid_and_spread
from src.execution.alpaca_options_executor import _is_duplicate_client_order_id
from src.research.notifications import notify_operator

logger = logging.getLogger(__name__)

# OCC option symbol: ROOT (1-6 chars) + YYMMDD + C/P + strike*1000 (8 digits).
_OCC_RE = re.compile(r"^[A-Z.]{1,6}(\d{2})(\d{2})(\d{2})[CP]\d{8}$")

_NY = ZoneInfo("America/New_York")


class ExitReason(StrEnum):
    THESIS_INVALIDATED = "thesis_invalidated"
    TIME_STOP = "time_stop"
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"
    EXPIRY_PROTECT = "expiry_protect"
    STALE = "stale"
    # Finalization-only reasons (no sell was needed/possible):
    EXPIRED_WORTHLESS = "expired_worthless"
    CLOSED_EXTERNAL = "closed_external"


@dataclass(frozen=True)
class OptionsExitConfig:
    #: Close when premium is down this fraction from entry (0.40 = −40%).
    #: DISABLED on the entry day (NY calendar date) — opening spreads on
    #: cheap OTM contracts register as instant −40% "losses" without any
    #: real move; from day 2 the mark is meaningful.
    stop_loss_pct: float = 0.40
    #: Settle window (CL-h02l): for the first N minutes after entry, NO
    #: premium-based stop fires — not even the extreme valve. A cheap OTM
    #: contract's bid/ask spread alone can exceed 60% right after the buy,
    #: so the indicative mark shows a huge phantom loss that isn't a real
    #: move (2026-07-23: EE/FLNG sold at −60/−73% six minutes after entry
    #: on spread noise). Date/thesis/profit rules still apply. 0 disables.
    entry_settle_min: int = 15
    #: Entry-day safety valve: an extreme adverse move still closes even
    #: during the grace period (0.60 = −60%).
    entry_day_extreme_stop_pct: float = 0.60
    #: Widest quoted spread (as a fraction of the ask) at which a
    #: premium-based STOP may still fire (CL-d44a). Above this the market is
    #: too wide to tell "moved against me" from "wide market": the mid is
    #: barely informative and a market sell would realize the spread. Date
    #: and thesis rules are UNAFFECTED — a dismissed event or a time stop
    #: still closes, wide market or not. 1.01 disables the guard.
    max_stop_spread_pct: float = 0.50
    #: Close when premium is up this fraction from entry (0.80 = +80%).
    profit_target_pct: float = 0.80
    #: Close when ≤ this many days to expiration UNLESS meaningfully
    #: profitable (see expiry_protect_min_profit).
    expiry_protect_days: int = 4
    #: A near-expiry position at/above this premium gain is left to run —
    #: it exits via profit target, time stop, or the final-day backstop.
    expiry_protect_min_profit: float = 0.25
    #: Absolute backstop: at ≤ this many DTE, close REGARDLESS of P&L.
    #: An event option is never ridden into expiration without a decision.
    final_day_dte: int = 1
    #: Time stop when the idea itself carries no time_stop_days.
    default_time_stop_days: int = 10
    #: Safety-net margin past the time stop (rule 6).
    stale_grace_days: int = 2


def occ_expiry(occ_symbol: str | None) -> date | None:
    """Expiration date parsed from an OCC symbol, or None."""
    m = _OCC_RE.match(str(occ_symbol or "").strip().upper())
    if not m:
        return None
    yy, mm, dd = (int(g) for g in m.groups())
    try:
        return date(2000 + yy, mm, dd)
    except ValueError:
        return None


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


def _mid_pnl_pct(entry_mid: float | None, mid: float | None) -> float | None:
    """Premium return measured MID-to-MID (CL-d44a) — the only honest
    "did the thesis move" number, because it cancels the bid/ask spread on
    BOTH sides. ``entry_mid`` is the mid recorded at buy time (migration
    018); None for rows written before it, which fall back to the legacy
    mark plus the spread guard."""
    if entry_mid is None or mid is None or entry_mid <= 0:
        return None
    return (mid - entry_mid) / entry_mid


def _pnl_pct(pos: dict[str, Any]) -> float | None:
    """Premium return of a live position, or None when unpriceable.

    LEGACY MARK (bid-based) — kept only for rows with no ``entry_mid``
    baseline. Prefers Alpaca's own ``unrealized_plpc``; falls back to
    current/avg-entry. Never fabricates — no data → None (P&L rules skip
    this cycle, date rules still apply).

    Both inputs are BID-marked while entries fill at the ASK, so on a wide
    contract this reads as a large loss with no real move (CL-d44a). The
    ``max_stop_spread_pct`` guard exists to stop that firing a sell."""
    plpc = pos.get("unrealized_plpc")
    if plpc is not None and plpc != "":
        try:
            return float(plpc)
        except (TypeError, ValueError):
            pass
    try:
        avg = float(pos.get("avg_entry_price") or 0)
        cur = float(pos.get("current_price") or 0)
    except (TypeError, ValueError):
        return None
    if avg > 0 and cur > 0:
        return (cur - avg) / avg
    return None


def evaluate_exit(
    row: dict[str, Any],
    pos: dict[str, Any],
    cfg: OptionsExitConfig,
    now: datetime,
    quote: tuple[float | None, float | None] | None = None,
) -> tuple[ExitReason, str] | None:
    """First matching exit rule for one open position, or None (hold).

    Pure decision function — all I/O stays in :func:`manage_option_exits`.

    ``quote`` is the live (bid, ask) (CL-d44a). Given it, P&L is measured
    MID-to-MID against the entry mid, which cancels the spread on both
    sides; and a market too WIDE to judge (spread > max_stop_spread_pct)
    suppresses the premium STOPS entirely. Omitted → legacy bid-marked
    behavior, which is what pre-018 rows without an entry_mid still use.
    """
    mid, spread = mid_and_spread(*(quote or (None, None)))
    # MID-to-MID is the honest measure; fall back to the legacy bid mark
    # only when there is no entry baseline (rows predating migration 018).
    entry_mid = row.get("entry_mid")
    try:
        entry_mid = float(entry_mid) if entry_mid is not None else None
    except (TypeError, ValueError):
        entry_mid = None
    pnl = _mid_pnl_pct(entry_mid, mid)
    pnl_is_mid = pnl is not None
    if pnl is None:
        pnl = _pnl_pct(pos)
    # A market wider than the stop threshold cannot distinguish "moved
    # against me" from "wide market" — and selling into it REALIZES the
    # spread. Suppress premium stops only; date/thesis rules still fire.
    # A mid-measured P&L is already spread-neutral, so the guard applies
    # only to the legacy bid-marked path.
    stops_suppressed = not pnl_is_mid and spread is not None and spread > cfg.max_stop_spread_pct
    entered = _as_dt(row.get("submitted_at"))
    days_held = (now - entered).days if entered else None
    time_stop = int(row.get("time_stop_days") or cfg.default_time_stop_days)
    expiry = occ_expiry(row.get("occ_symbol") or pos.get("symbol"))
    dte = (expiry - now.date()).days if expiry else None
    # Grace period runs on the NY calendar date — "entry day" means the
    # trading session the position was opened in, not a UTC window.
    is_entry_day = (
        entered is not None and entered.astimezone(_NY).date() == now.astimezone(_NY).date()
    )
    # Minutes since the buy — used to suppress premium stops until the
    # indicative quote settles (CL-h02l). Unknown entry time → treat as
    # settled (don't hold a position hostage to a missing timestamp).
    mins_held = (now - entered).total_seconds() / 60.0 if entered is not None else None
    settled = mins_held is None or mins_held >= cfg.entry_settle_min

    # 1. Thesis invalidated — the desk ACTIVELY no longer believes the
    #    story: event DISMISSED or idea cancelled. Deliberately NOT
    #    geo_event EXPIRED — expiry is the ~2h intraday FX-confluence
    #    window lapsing (median EXPIRED lifetime is 2h03m), a gate
    #    lifecycle, not a verdict on a multi-week options thesis; the
    #    options deadline is the idea's own time stop (rule 2).
    event_status = str(row.get("event_status") or "").upper()
    if event_status == "DISMISSED":
        return (ExitReason.THESIS_INVALIDATED, "geo_event dismissed")
    if str(row.get("idea_status") or "") == "cancelled":
        return (ExitReason.THESIS_INVALIDATED, "idea cancelled")

    # 2. Hard time stop — the idea's own deadline (pipeline auto-expiry of
    #    the idea is the same clock, observed from the other side).
    if days_held is not None and days_held >= time_stop:
        return (ExitReason.TIME_STOP, f"held {days_held}d >= time stop {time_stop}d")
    if str(row.get("idea_status") or "") == "expired":
        return (ExitReason.TIME_STOP, "idea auto-expired by pipeline")

    # 3. Stop loss on premium — DISABLED on entry day (grace period:
    #    opening spreads masquerade as losses), except the extreme-move
    #    safety valve.
    if pnl is not None and not stops_suppressed:
        if is_entry_day:
            # Extreme valve fires only AFTER the settle window (CL-h02l) —
            # inside it, a cheap contract's spread masquerades as a −60%+
            # loss and would sell on noise, not a real move.
            if settled and pnl <= -cfg.entry_day_extreme_stop_pct:
                return (
                    ExitReason.STOP_LOSS,
                    f"extreme adverse move {pnl:+.0%} on entry day "
                    f"(<= -{cfg.entry_day_extreme_stop_pct:.0%} valve, "
                    f"{mins_held:.0f}min held)",
                )
        elif pnl <= -cfg.stop_loss_pct:
            return (ExitReason.STOP_LOSS, f"premium {pnl:+.0%} <= -{cfg.stop_loss_pct:.0%}")

    # 4. Profit target on premium.
    if pnl is not None and pnl >= cfg.profit_target_pct:
        return (ExitReason.PROFIT_TARGET, f"premium {pnl:+.0%} >= +{cfg.profit_target_pct:.0%}")

    # 5. Expiration protection. Final-day backstop closes REGARDLESS of
    #    P&L (never ride into expiration without a decision); inside the
    #    protect window, a meaningful winner is left to run. Date-based,
    #    so missing quotes (pnl None) still close — bad data can't hold.
    if dte is not None:
        if dte <= cfg.final_day_dte:
            return (ExitReason.EXPIRY_PROTECT, f"{dte}d to expiry — final-day close")
        if dte <= cfg.expiry_protect_days and (pnl is None or pnl < cfg.expiry_protect_min_profit):
            return (
                ExitReason.EXPIRY_PROTECT,
                f"{dte}d to expiry <= {cfg.expiry_protect_days}d, not meaningfully profitable",
            )

    # 6. Stale safety net (normally unreachable past rule 2).
    if days_held is not None and days_held >= time_stop + cfg.stale_grace_days:
        return (ExitReason.STALE, f"held {days_held}d > time stop + {cfg.stale_grace_days}d")

    return None


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #


def _fetch_open_rows(engine: Any) -> list[dict[str, Any]]:
    """Real positions (status='submitted') not yet finalized, joined to their
    idea + originating event for the thesis check."""
    sql = """
        SELECT a.idea_id, a.ticker, a.occ_symbol, a.opt_type, a.qty,
               a.premium_est, a.submitted_at, a.exit_status, a.exit_reason,
               -- entry_mid (CL-d44a, mig 018): WITHOUT this column the
               -- mid-to-mid P&L in evaluate_exit silently never engages
               -- (row.get returns None) and every exit falls back to the
               -- bid mark — found by the CL-0hr9 e2e agent.
               a.entry_mid,
               ti.time_stop_days, ti.status AS idea_status,
               ge.status AS event_status
        FROM alpaca_option_orders a
        LEFT JOIN trade_ideas ti ON ti.idea_id = a.idea_id
        LEFT JOIN geo_events ge ON ge.id = ti.geo_event_id
        WHERE a.status = 'submitted'
          AND (a.exit_status IS NULL
               OR a.exit_status = 'submitted'
               -- 'unsellable' (CL-hptt) MUST stay in the working set: it means
               -- the contract had no bid at the last attempt, not that we are
               -- done with it. Excluding it would silently ABANDON an open
               -- position instead of retrying once a bid returns. The
               -- per-cycle alert is deduped on the stored status, not here.
               OR a.exit_status = 'unsellable')
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
    exit_premium: float | None = None,
    pnl_pct: float | None = None,
    close_idea: bool = True,
) -> None:
    """Record the exit on the order row; mirror 'closed' onto the idea.

    COALESCE keeps first-write values (e.g. the reason recorded at sell
    submit survives the later fill confirmation)."""
    with engine.begin() as conn:
        conn.execute(
            text("""
            UPDATE alpaca_option_orders SET
                exit_status   = :es,
                exit_reason   = COALESCE(exit_reason, :er),
                exit_order_id = COALESCE(exit_order_id, :eo),
                exit_premium  = COALESCE(exit_premium, :ep),
                pnl_pct       = COALESCE(pnl_pct, :pp),
                exited_at     = COALESCE(exited_at, :ea)
            WHERE idea_id = :i
        """),
            {
                "es": exit_status,
                "er": exit_reason,
                "eo": exit_order_id,
                "ep": exit_premium,
                "pp": pnl_pct,
                "ea": now,
                "i": idea_id,
            },
        )
        if close_idea:
            conn.execute(
                text("""
                UPDATE trade_ideas SET status = 'closed',
                                       status_updated_at = :now
                WHERE idea_id = :i
                  AND status IN ('pending', 'taken', 'expired')
            """),
                {"now": now, "i": idea_id},
            )


def _finalize_vanished(
    engine: Any,
    row: dict[str, Any],
    now: datetime,
    counts: dict[str, int],
) -> None:
    """Row says open, Alpaca shows no position — record the truth."""
    idea_id = str(row["idea_id"])
    if row.get("exit_status") == "submitted":
        # Our sell filled — confirm the close (reason/premium kept from
        # submit time via COALESCE).
        _mark_exit(engine, idea_id, now, exit_status="closed")
        counts["closed_confirmed"] += 1
        logger.info(
            "options exit: %s (%s) close CONFIRMED [%s]",
            row.get("occ_symbol"),
            row.get("ticker"),
            row.get("exit_reason"),
        )
        return
    expiry = occ_expiry(row.get("occ_symbol"))
    if expiry is not None and expiry < now.date():
        _mark_exit(
            engine,
            idea_id,
            now,
            exit_status="closed",
            exit_reason=ExitReason.EXPIRED_WORTHLESS,
            exit_premium=0.0,
            pnl_pct=-1.0,
        )
        counts["expired_worthless"] += 1
        logger.warning(
            "options exit: %s (%s) EXPIRED WORTHLESS (-100%%)",
            row.get("occ_symbol"),
            row.get("ticker"),
        )
        return
    _mark_exit(engine, idea_id, now, exit_status="closed", exit_reason=ExitReason.CLOSED_EXTERNAL)
    counts["closed_external"] += 1
    logger.warning(
        "options exit: %s (%s) closed OUTSIDE the manager — marked closed_external",
        row.get("occ_symbol"),
        row.get("ticker"),
    )


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #


def manage_option_exits(
    engine: Any,
    client: AlpacaOptionsClient,
    cfg: OptionsExitConfig | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Evaluate every open option position and close what the rules say.

    Returns a count by outcome. Never raises — a per-position failure is
    logged and the loop continues (same posture as the entry executor)."""
    cfg = cfg or OptionsExitConfig()
    now = now or datetime.now(UTC)
    counts: dict[str, int] = {
        "held": 0,
        "exit_submitted": 0,
        "exit_pending": 0,
        "closed_confirmed": 0,
        "expired_worthless": 0,
        "closed_external": 0,
        "unmatched": 0,
        "recovered": 0,
        "unsellable": 0,
        "error": 0,
        "market_closed": 0,
    }

    # Sells are market orders too — 422 outside regular hours.
    if not client.is_market_open():
        logger.info("options exit: market closed — no evaluation this cycle")
        counts["market_closed"] = 1
        return counts

    try:
        positions = {str(p.get("symbol")): p for p in client.list_option_positions()}
    except Exception:
        logger.warning("options exit: positions fetch failed — retry next cycle", exc_info=True)
        counts["error"] += 1
        return counts

    for row in _fetch_open_rows(engine):
        idea_id = str(row["idea_id"])
        occ = str(row.get("occ_symbol") or "")
        pos = positions.pop(occ, None)
        try:
            if pos is None:
                _finalize_vanished(engine, row, now, counts)
                continue
            if row.get("exit_status") == "submitted":
                # Sell already placed, not yet filled — never resubmit
                # (client_order_id dedup would reject it anyway).
                counts["exit_pending"] += 1
                continue
            # ONE quote per position per cycle (CL-d44a), reused for both the
            # mid-based P&L / spread guard below and the no-bid check further
            # down. FAIL-SOFT: (None, None) degrades to the legacy bid mark
            # rather than skipping the position.
            quote: tuple[float | None, float | None] = (None, None)
            try:
                get_quote = getattr(client, "get_option_quote", None)
                if callable(get_quote):
                    quote = get_quote(occ)
            except Exception:
                logger.debug("options exit: quote probe failed for %s", occ, exc_info=True)

            decision = evaluate_exit(row, pos, cfg, now, quote=quote)
            if decision is None:
                counts["held"] += 1
                continue
            reason, detail = decision
            qty = int(float(pos.get("qty") or row.get("qty") or 0))
            if qty <= 0:
                counts["error"] += 1
                logger.warning(
                    "options exit: %s has non-positive qty %r — not selling", occ, pos.get("qty")
                )
                continue
            pnl = _pnl_pct(pos)
            try:
                cur = float(pos.get("current_price") or 0)
            except (TypeError, ValueError):
                cur = 0.0
            exit_premium = cur * 100.0 * qty if cur > 0 else None

            # UNSELLABLE guard (CL-hptt): a deep-OTM contract can carry a live
            # ask and NO bid (bp=0, bs=0). A MARKET sell into an empty book is
            # rejected by Alpaca with a 403, and before this the generic
            # handler below just logged "error managing" and retried EVERY
            # cycle forever — never backing off, never escalating, and never
            # recording an exit_status, so the book believed an exit was in
            # flight while the position quietly stayed open. Record it once,
            # alert once, and stop re-submitting until a bid returns.
            # FAIL-OPEN by construction: a None bid (quote unavailable), a
            # probe that raises, or a client without the method at all falls
            # through to the submit attempt exactly as before. An exit must
            # never be blocked because a QUOTE lookup failed — only a venue
            # that explicitly quotes no bid stops it.
            bid = quote[0]
            if bid is None:
                # No quote from the batch probe above — fall back to the
                # dedicated bid lookup (older/stub clients expose only that).
                try:
                    get_bid = getattr(client, "get_option_bid", None)
                    if callable(get_bid):
                        bid = get_bid(occ)
                except Exception:
                    logger.debug("options exit: bid probe failed for %s", occ, exc_info=True)
            if bid is not None and bid <= 0:
                if str(row.get("exit_status") or "") != "unsellable":
                    _mark_exit(
                        engine,
                        idea_id,
                        now,
                        exit_status="unsellable",
                        exit_reason=f"{reason} (no bid — market sell would be rejected)",
                        pnl_pct=pnl,
                        close_idea=False,
                    )
                    notify_operator(
                        "⚠️ Option exit blocked — no bid",
                        f"{occ} ({row.get('ticker')}) wants to exit ({reason}) but the "
                        f"contract has NO BID, so a market sell is rejected. Position "
                        f"stays OPEN and will retry when a bid returns. Close it "
                        f"manually if you want out now.",
                    )
                    logger.warning(
                        "options exit: %s is UNSELLABLE (no bid) — wanted %s; "
                        "recorded + alerted, not retrying until a bid returns",
                        occ,
                        reason,
                    )
                counts["unsellable"] += 1
                continue

            try:
                order = client.submit_option_order(
                    occ,
                    qty,
                    "sell",
                    client_order_id=f"curlit-exit-{idea_id}",
                )
            except Exception as sub_exc:
                if _is_duplicate_client_order_id(sub_exc):
                    # Prior crashed cycle already sold — record, don't retry.
                    _mark_exit(
                        engine,
                        idea_id,
                        now,
                        exit_status="submitted",
                        exit_reason=reason,
                        exit_order_id="recovered",
                        exit_premium=exit_premium,
                        pnl_pct=pnl,
                    )
                    counts["recovered"] += 1
                    logger.warning(
                        "options exit: %s was ALREADY being sold "
                        "by a prior crashed cycle — recorded",
                        occ,
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
                exit_premium=exit_premium,
                pnl_pct=pnl,
            )
            counts["exit_submitted"] += 1
            logger.info(
                "options exit: SELLING %d %s (%s) — %s (%s); pnl=%s [idea %s]",
                qty,
                occ,
                row.get("ticker"),
                reason,
                detail,
                f"{pnl:+.0%}" if pnl is not None else "n/a",
                idea_id,
            )
        except Exception:
            counts["error"] += 1
            logger.warning("options exit: error managing %s", occ, exc_info=True)

    # Anything Alpaca holds that we can't explain: flag, never auto-manage.
    for occ, _pos in positions.items():
        counts["unmatched"] += 1
        logger.warning(
            "options exit: UNMATCHED Alpaca position %s — no "
            "alpaca_option_orders row; NOT auto-managed",
            occ,
        )

    logger.info("options exit: %s", counts)
    return counts
