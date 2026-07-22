"""Morning Telegram digests (CL-ydp8, supersedes long-only CL-lpai).

Two messages at the start of each trading day:

1. **POSITIONS** — everything actually HELD, long AND short, both venues,
   each FX pair with its economic reading (short USD_CHF = long CHF),
   plus what closed in the last 24h with realized P&L, plus balances.
2. **LONG IDEAS** — what the event pipeline currently wants to be LONG
   (pending bullish trade ideas, deduped by ticker, top-N by
   confidence). This is the desk's shopping list, independent of what
   the executors have bought; the auto-buy takes the top 5 each open.

Pure builders + clock gate live here; the daemon script owns venue I/O.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.execution.broker import currency_pair
from src.research.notifications import html_escape

logger = logging.getLogger(__name__)

_NY = ZoneInfo("America/New_York")

# OCC: ROOT + YYMMDD + C/P + strike*1000 (8 digits).
_OCC_RE = re.compile(r"^([A-Z.]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


def _describe_occ(occ: str) -> str:
    """``DHT260821C00020000`` → ``DHT $20 call exp 08-21`` (raw symbol
    when unparseable — never hide a position over formatting)."""
    m = _OCC_RE.match(str(occ or "").strip().upper())
    if not m:
        return str(occ)
    root, _yy, mm, dd, right, strike_raw = m.groups()
    strike = int(strike_raw) / 1000.0
    strike_txt = f"${strike:.2f}".rstrip("0").rstrip(".")
    kind = "call" if right == "C" else "put"
    return f"{root} {strike_txt} {kind} exp {mm}-{dd}"


def _fx_reading(symbol: str, qty: float) -> str:
    """Economic reading of an FX position: short USD_CHF = long CHF."""
    pair = currency_pair(symbol)
    if pair is None:
        return "long" if qty > 0 else "short"
    base, quote = pair
    if qty > 0:
        return f"long {base} / short {quote}"
    return f"short {base} / long {quote}"


def _fmt_qty(q: float) -> str:
    return f"{q:+,.0f}" if abs(q) >= 1 else f"{q:+.2f}"


def build_position_digest(
    oanda_positions: list[Any] | None,
    alpaca_positions: list[Mapping[str, Any]] | None,
    closed_24h: list[Mapping[str, Any]] | None = None,
    balances: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> str:
    """Telegram-HTML body for the FULL position digest (longs AND shorts).

    ``None`` for a venue means "couldn't read it" and renders an explicit
    unavailable line — an unreadable venue must never look flat.
    ``closed_24h`` entries: {"venue", "desc", "pl"} strings pre-computed
    by the daemon, so this builder stays pure and testable.
    """
    now = now or datetime.now(UTC)
    lines: list[str] = []

    lines.append("<b>OANDA FX</b>")
    if oanda_positions is None:
        lines.append("  (unavailable — could not read positions)")
    elif not oanda_positions:
        lines.append("  flat")
    else:
        for p in oanda_positions:
            qty = float(p.quantity)
            upl = float(getattr(p, "unrealized_pnl", 0.0) or 0.0)
            lines.append(
                f"  {html_escape(str(p.symbol))} {_fmt_qty(qty)}"
                f" @ {float(p.avg_price):.5g} (uPL {upl:+.2f})"
                f" — {html_escape(_fx_reading(str(p.symbol), qty))}"
            )

    lines.append("")
    lines.append("<b>Alpaca options</b>")
    if alpaca_positions is None:
        lines.append("  (unavailable — could not read positions)")
    elif not alpaca_positions:
        lines.append("  flat")
    else:
        for pos in alpaca_positions:
            occ = str(pos.get("symbol") or "")
            qty = int(float(pos.get("qty") or 0))
            desc = _describe_occ(occ)
            try:
                entry = float(pos.get("avg_entry_price") or 0) * 100.0 * qty
                cur = float(pos.get("current_price") or 0) * 100.0 * qty
            except (TypeError, ValueError):
                entry = cur = 0.0
            pnl_txt = ""
            plpc = pos.get("unrealized_plpc")
            if plpc is not None and plpc != "":
                try:
                    pnl_txt = f" ({float(plpc):+.0%})"
                except (TypeError, ValueError):
                    pnl_txt = ""
            val_txt = (
                f" — entry ${entry:,.0f}, now ${cur:,.0f}" if entry > 0 else ""
            )
            lines.append(
                f"  {html_escape(desc)} ×{qty}{html_escape(val_txt)}"
                f"{html_escape(pnl_txt)}"
            )

    if closed_24h:
        lines.append("")
        lines.append("<b>Closed last 24h</b>")
        for c in closed_24h:
            lines.append(
                f"  {html_escape(str(c.get('venue', '')))} "
                f"{html_escape(str(c.get('desc', '')))} → "
                f"{html_escape(str(c.get('pl', '')))}"
            )

    if balances:
        lines.append("")
        lines.append("<i>" + " | ".join(
            f"{html_escape(str(k))}: {html_escape(str(v))}"
            for k, v in balances.items()
        ) + "</i>")

    lines.append("")
    lines.append(f"<i>{now.astimezone(_NY).strftime('%Y-%m-%d %H:%M ET')}</i>")
    return "\n".join(lines)


def build_long_ideas_digest(
    ideas: list[Mapping[str, Any]],
    now: datetime | None = None,
    limit: int = 12,
) -> str:
    """Telegram-HTML body for "what the desk wants to be LONG".

    ``ideas``: pending BULLISH trade-idea rows (ticker, action,
    confidence, preferred_instrument, rationale) — the pipeline's view of
    what deserves long exposure, independent of execution. Deduped by
    ticker keeping the highest-confidence row, sorted by confidence,
    capped at ``limit`` with an honest "of N" so a cap never reads as the
    whole pool.
    """
    now = now or datetime.now(UTC)
    best: dict[str, Mapping[str, Any]] = {}
    for idea in ideas:
        t = str(idea.get("ticker") or "").upper()
        if not t:
            continue
        conf = float(idea.get("confidence") or 0.0)
        if t not in best or conf > float(best[t].get("confidence") or 0.0):
            best[t] = idea
    ranked = sorted(
        best.values(),
        key=lambda i: float(i.get("confidence") or 0.0),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("<b>What the desk wants to be LONG (event-driven)</b>")
    if not ranked:
        lines.append("  no pending bullish ideas")
    for idea in ranked[:limit]:
        conf = float(idea.get("confidence") or 0.0)
        pref = str(idea.get("preferred_instrument") or "").strip()
        rat = str(idea.get("rationale") or "").strip()
        if len(rat) > 90:
            rat = rat[:87] + "…"
        via = f" via {pref}" if pref else ""
        tail = f" — {rat}" if rat else ""
        lines.append(
            f"  <b>{html_escape(str(idea.get('ticker')))}</b> "
            f"{conf:.2f}{html_escape(via)}{html_escape(tail)}"
        )
    lines.append("")
    lines.append(
        f"<i>Top {min(limit, len(ranked))} of {len(ranked)} tickers — "
        f"the auto-buy takes the top 5 at ~9:45 ET</i>"
    )
    lines.append(f"<i>{now.astimezone(_NY).strftime('%Y-%m-%d %H:%M ET')}</i>")
    return "\n".join(lines)


def should_send(
    now: datetime,
    last_sent_ny_date: str | None,
    send_time_et: str = "09:15",
) -> bool:
    """True when it's at/after the send time in New York and today's
    digest hasn't gone out yet (dedup by NY calendar date — one send per
    trading morning, resilient to daemon restarts)."""
    ny = now.astimezone(_NY)
    if str(last_sent_ny_date or "") == ny.date().isoformat():
        return False
    try:
        hh, mm = (int(x) for x in send_time_et.split(":", 1))
    except ValueError:
        hh, mm = 9, 15
        logger.warning(
            "morning digest: bad MORNING_DIGEST_TIME_ET %r — using 09:15",
            send_time_et,
        )
    return (ny.hour, ny.minute) >= (hh, mm)
