"""Morning Telegram digests (CL-ydp8, supersedes long-only CL-lpai).

Two messages at the start of each trading day:

1. **POSITIONS** — everything actually HELD, long AND short, both venues,
   each FX pair with its economic reading (short USD_CHF = long CHF),
   plus what closed in the last 24h with realized P&L, plus balances.
2. **LONG IDEAS** — dated research and pending backlog (CL-ayd8).
   Fresh verified niche ideas precede other fresh ideas and older backlog.
   A report timestamp is never substituted for a discovery timestamp.

Pure builders + clock gate live here; the daemon script owns venue I/O.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
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
    unsellable: list[Mapping[str, Any]] | None = None,
) -> str:
    """Telegram-HTML body for the FULL position digest (longs AND shorts).

    ``None`` for a venue means "couldn't read it" and renders an explicit
    unavailable line — an unreadable venue must never look flat.
    ``closed_24h`` entries: {"venue", "desc", "pl"} strings pre-computed
    by the daemon, so this builder stays pure and testable.

    ``unsellable`` (CL-p0pe): option rows whose exit is BLOCKED because the
    contract has no bid (CL-hptt). Those page the operator once and then go
    quiet, so a position stuck for days would otherwise lose visibility
    entirely — this resurfaces it every morning until it clears. Omitted
    from the message when empty, so a normal morning gains no noise.
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
            val_txt = f" — entry ${entry:,.0f}, now ${cur:,.0f}" if entry > 0 else ""
            lines.append(
                f"  {html_escape(desc)} ×{qty}{html_escape(val_txt)}{html_escape(pnl_txt)}"
            )

    if unsellable:
        lines.append("")
        lines.append("<b>⚠️ Exit BLOCKED — no bid</b>")
        for u in unsellable:
            occ = str(u.get("occ_symbol") or "")
            reason = str(u.get("exit_reason") or "").split(" (no bid")[0] or "exit"
            lines.append(
                f"  {html_escape(_describe_occ(occ))} — wants "
                f"{html_escape(reason)}; no bid, sells when one returns"
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
        lines.append(
            "<i>"
            + " | ".join(
                f"{html_escape(str(k))}: {html_escape(str(v))}" for k, v in balances.items()
            )
            + "</i>"
        )

    lines.append("")
    lines.append(f"<i>{now.astimezone(_NY).strftime('%Y-%m-%d %H:%M ET')}</i>")
    return "\n".join(lines)


#: Max company-name length before truncation in the LONG-ideas digest
#: (CL-ikz2) — keeps "<b>VG</b> (Venture Global, Inc.)" phone-readable.
_MAX_NAME_LEN = 28


def _timestamp(value: Any) -> datetime | None:
    """Unknown/naive dates stay unknown; do not silently invent a timezone."""
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def _verified_niche(idea: Mapping[str, Any]) -> bool:
    """Only joined write-once status, never LLM notes, identifies verified niche."""
    return (
        idea.get("research_eligible") in (True, 1)
        and bool(idea.get("research_invocation_id"))
        and idea.get("discovery_status") == "completed"
        and idea.get("evidence_status") == "source_backed"
        and idea.get("review_status") == "supported"
        and idea.get("liquidity_status") == "sufficient"
    )


def build_long_ideas_digest(
    ideas: list[Mapping[str, Any]] | None,
    now: datetime | None = None,
    limit: int = 12,
    names: dict[str, str] | None = None,
    research: list[Mapping[str, Any]] | None = None,
) -> str:
    """CL-ayd8: last-24h is a DISPLAY window, not trading eligibility.

    None means a failed read, [] a successful empty read. Keep historical rows,
    expose references and ages, and never infer research success from a backlog.
    Within a category/ticker show the newest record and references to duplicates.
    """
    now = now or datetime.now(UTC)
    if now.tzinfo is None or limit < 1:
        raise ValueError("aware now and positive limit required")
    cutoff = now - timedelta(hours=24)  # One daily digest window; not an expiry policy.
    labels = (
        "New verified niche (last 24h)",
        "Other new pending ideas (last 24h)",
        "Older pending backlog",
        "Unknown/invalid idea date",
    )
    groups: list[list[Mapping[str, Any]]] = [[], [], [], []]
    for idea in ideas or []:
        created = _timestamp(idea.get("created_at"))
        category = (
            3
            if created is None or created > now
            else 2
            if created < cutoff
            else 0
            if _verified_niche(idea)
            else 1
        )
        groups[category].append(idea)
    best: dict[tuple[int, str], Mapping[str, Any]] = {}
    duplicates: dict[tuple[int, str], list[str]] = {}
    for category, group in enumerate(groups):
        for idea in group:
            t = str(idea.get("ticker") or "").upper()
            if not t:
                continue
            key = category, t
            duplicates.setdefault(key, []).append(str(idea.get("idea_id") or "unknown"))

            def rank(row: Mapping[str, Any]) -> tuple[datetime, float]:
                return (
                    _timestamp(row.get("created_at")) or datetime.min.replace(tzinfo=UTC),
                    float(row.get("confidence") or 0.0),
                )

            if key not in best or rank(idea) > rank(best[key]):
                best[key] = idea
    ranked = sorted(
        best.items(),
        key=lambda item: (item[0][0], -float(item[1].get("confidence") or 0.0), item[0][1]),
    )

    lines = ["<b>Research ideas — freshness and pending backlog</b>"]
    if research is None:
        lines.append("  research status unavailable — freshness cannot be confirmed")
    elif not research:
        lines.append("  No niche research attempts recorded in the last 24h")
    else:
        lines.append(f"  Niche attempts last 24h: {len(research)}")
        # Latest three diagnoses keep the daily Telegram payload phone-readable.
        for run in research[:3]:
            lines.append(
                html_escape(
                    f"  {run.get('recorded_at')} — {run.get('discovery_status', 'unknown')}"
                    f": {run.get('reason') or 'no failure reason'}"
                    f" [run {run.get('invocation_id')}; eligible {run.get('eligible_count', 'unknown')}]"
                    f" published: {run.get('published_count', 'unknown')};"
                    f" job: {run.get('job_state') or 'legacy/prior attempt'};"
                    f" reviews: {run.get('reviews') or 'none recorded'}"
                )
            )
    lines.extend(f"{label}: {len(group)}" for label, group in zip(labels, groups, strict=True))
    if not groups[0]:
        lines.append("  No new verified niche ideas in this pending feed")
    if ideas is None:
        lines.append("  pending ideas unavailable — not an empty book")
    elif not ranked:
        lines.append("  no pending bullish ideas")
    for key, idea in ranked[:limit]:
        conf = float(idea.get("confidence") or 0.0)
        pref = str(idea.get("preferred_instrument") or "").strip()
        rat = str(idea.get("rationale") or "").strip()
        if len(rat) > 90:
            rat = rat[:87] + "…"
        via = f" via {pref}" if pref else ""
        tail = f" — {rat}" if rat else ""
        ticker = str(idea.get("ticker") or "")
        name = (names or {}).get(ticker.upper())
        if name and len(name) > _MAX_NAME_LEN:
            name = name[: _MAX_NAME_LEN - 1].rstrip() + "…"
        name_part = f" ({html_escape(name)})" if name else ""
        lines.append(
            f"  <b>{html_escape(ticker)}</b>{name_part} "
            f"{conf:.2f}{html_escape(via)}{html_escape(tail)}"
        )
        created = _timestamp(idea.get("created_at"))
        stamp = created.isoformat() if created is not None and created <= now else "unknown/invalid"
        provenance = (
            f"verified niche; run {idea.get('research_invocation_id')}"
            if _verified_niche(idea)
            else "provenance unverified (not a verified niche idea)"
        )
        refs = duplicates[key]
        lines.append(
            html_escape(
                f"    {labels[key[0]]}; idea {stamp}; event {idea.get('geo_event_id', 'unknown')}"
                f" seen {idea.get('event_seen_at') or 'unknown'}; {provenance}"
                f"; IDs {', '.join(refs[:5])}"
                + (f" (+{len(refs) - 5} more records)" if len(refs) > 5 else "")
            )
        )
    lines.append("")
    lines.append(
        f"<i>Top {min(limit, len(ranked))} of {len(ranked)} tickers — "
        f"grouped by freshness; Pending records: {len(ideas or [])}. "
        "Research only; this report does not authorize orders.</i>"
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
