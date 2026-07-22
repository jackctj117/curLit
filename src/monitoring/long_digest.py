"""Morning Telegram digest of all LONG positions (CL-lpai).

One message at the start of each trading day answering "what am I long
going into today?" across both venues:

  * OANDA — trades with POSITIVE units (long the instrument);
  * Alpaca — every open option position (bought calls/puts are long
    premium by construction; the desk never writes options).

Short positions are COUNTED in a footer, never listed — that keeps the
message honest on days like today (OANDA all shorts) without burying the
long book. Send-time and once-per-day dedup live here as pure functions;
the daemon script owns the clock loop and venue I/O.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.research.notifications import html_escape

logger = logging.getLogger(__name__)

_NY = ZoneInfo("America/New_York")

# OCC: ROOT + YYMMDD + C/P + strike*1000 (8 digits).
_OCC_RE = re.compile(r"^([A-Z.]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")


def _describe_occ(occ: str) -> str:
    """``DHT260821C00020000`` → ``DHT $20 call exp 08-21`` (or the raw
    symbol when unparseable — never hide a position over formatting)."""
    m = _OCC_RE.match(str(occ or "").strip().upper())
    if not m:
        return str(occ)
    root, _yy, mm, dd, right, strike_raw = m.groups()
    strike = int(strike_raw) / 1000.0
    strike_txt = f"${strike:.2f}".rstrip("0").rstrip(".")
    kind = "call" if right == "C" else "put"
    return f"{root} {strike_txt} {kind} exp {mm}-{dd}"


def _fmt_qty(q: float) -> str:
    return f"{q:+,.0f}" if abs(q) >= 1 else f"{q:+.2f}"


def build_long_digest(
    oanda_positions: list[Any] | None,
    alpaca_positions: list[Mapping[str, Any]] | None,
    now: datetime | None = None,
) -> str:
    """Telegram-HTML body listing LONG holdings; interpolations escaped.

    ``None`` for a venue means "couldn't read it" and renders an explicit
    unavailable line — an unreadable venue must never look flat.
    """
    now = now or datetime.now(UTC)
    lines: list[str] = []

    # ---- OANDA longs (positive units) ---------------------------------
    lines.append("<b>OANDA FX — long</b>")
    oanda_shorts = 0
    if oanda_positions is None:
        lines.append("  (unavailable — could not read positions)")
    else:
        longs = [p for p in oanda_positions if float(p.quantity) > 0]
        oanda_shorts = sum(1 for p in oanda_positions if float(p.quantity) < 0)
        if not longs:
            lines.append("  none")
        for p in longs:
            upl = float(getattr(p, "unrealized_pnl", 0.0) or 0.0)
            lines.append(
                f"  {html_escape(str(p.symbol))} {_fmt_qty(float(p.quantity))}"
                f" @ {float(p.avg_price):.5g} (uPL {upl:+.2f})"
            )

    # ---- Alpaca options (long premium) --------------------------------
    lines.append("")
    lines.append("<b>Alpaca options — long premium</b>")
    if alpaca_positions is None:
        lines.append("  (unavailable — could not read positions)")
    elif not alpaca_positions:
        lines.append("  none")
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

    # ---- footer -------------------------------------------------------
    if oanda_shorts:
        lines.append("")
        lines.append(
            f"<i>Shorts not listed: OANDA {oanda_shorts} "
            f"(this digest covers longs only)</i>"
        )
    lines.append("")
    lines.append(
        f"<i>{now.astimezone(_NY).strftime('%Y-%m-%d %H:%M ET')}</i>"
    )
    return "\n".join(lines)


def should_send(
    now: datetime,
    last_sent_ny_date: str | None,
    send_time_et: str = "09:15",
) -> bool:
    """True when it's at/after the send time in New York and today's
    digest hasn't gone out yet (dedup by NY calendar date — one per
    trading morning, resilient to daemon restarts)."""
    ny = now.astimezone(_NY)
    if str(last_sent_ny_date or "") == ny.date().isoformat():
        return False
    try:
        hh, mm = (int(x) for x in send_time_et.split(":", 1))
    except ValueError:
        hh, mm = 9, 15
        logger.warning(
            "long digest: bad LONG_DIGEST_TIME_ET %r — using 09:15",
            send_time_et,
        )
    return (ny.hour, ny.minute) >= (hh, mm)
