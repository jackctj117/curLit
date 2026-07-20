"""Tickers-of-interest digest for the event pipeline (CL-b92t).

After each ``--assess`` cycle the pipeline can send the operator ONE
compact Telegram-HTML digest of the events assessed in that cycle
whose urgency clears a threshold (default 5). This doubles as the
"the bots surfaced these tickers" feed: alongside the per-event lines
it unions every affected instrument into a deduped ``Tradable:`` line
(OANDA-reachable symbols with a long/short call, direction arrows
↑/↓) and a ``Watch:`` line (equity/polymarket alerts plus tradables
demoted to watch).

Never spams: an empty cycle, or a cycle where nothing clears the
threshold, sends nothing at all. Long cycles are capped at
``MAX_EVENTS`` event lines with a ``+N more`` note; the instrument
union still covers every qualifying event, so the ticker feed stays
complete even when event lines are elided.

Volume marks (CL-i4sr): ``Watch:`` tickers with an unusual
relative-volume spike in the last 24h (per the ``volume_spikes`` table
written by :class:`src.scanners.relative_volume.RelativeVolumeScanner`)
are annotated ``FRO×3.2`` style. :func:`fetch_volume_marks` is
deliberately fail-soft — a missing table, empty DB, or connection blip
degrades to no annotations, never to a lost digest.

Enrichment (CL-mgcp): when the caller passes ``prices`` (from
:func:`src.events.prices.get_prices`) every Tradable/Watch token gains
last close + daily change (``BCO_USD 78.4 (+2.1%)↑`` /
``FRO $24.10 (+3.2%)×3.2``); ``seen_ats`` adds per-event age
(``(2h ago)``); the assessments' advisory ``trade_ideas`` /
``fade_candidates`` render under ``Ideas:`` / ``Fade:`` — the same
ideas the pipeline just persisted to the ``trade_ideas`` ledger.
Everything degrades by omission: no price → bare ticker, no seen_at →
no age, no ideas → no section.

All interpolated content (headlines, themes, instruments, idea fields)
is escaped via :func:`src.research.notifications.html_escape` — GDELT
headlines and LLM output are hostile input.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from src.events.impact_agent import AssessmentResult
from src.events.prices import format_age, format_price
from src.research.notifications import (
    DispatchResult,
    html_escape,
    notify_operator,
)

logger = logging.getLogger(__name__)

#: Default minimum urgency (1-10) for an event to make the digest.
#: Overridable per-call and via EVENT_DIGEST_MIN_URGENCY (CLI wiring
#: in scripts/event_pipeline.py).
DEFAULT_MIN_URGENCY = 5

#: Cap on per-event lines in one digest; the rest collapse into a
#: "+N more" note. Telegram messages max out at 4096 chars and a
#: phone screen much sooner.
MAX_EVENTS = 15

#: Headline truncation length for the one-line-per-event format.
_HEADLINE_MAX = 80

#: ``affected`` kinds that are directly tradable through the broker.
_TRADABLE_KINDS = frozenset({"oanda", "fx"})

_ARROWS = {"long": "↑", "short": "↓"}  # ↑ / ↓

#: Lookback for "recent" unusual volume spikes on the Watch line.
VOLUME_MARK_LOOKBACK_HOURS = 24

#: Max tickers per Tradable/Watch line before wrapping — price+change
#: annotations make tokens ~4x wider than bare tickers (CL-mgcp).
TOKENS_PER_LINE = 6

#: Caps for the advisory sections; anything beyond collapses into a
#: "+N more" note. Ideas are the operator's action feed — most urgent
#: events' ideas render first.
MAX_IDEAS = 8
MAX_FADES = 5

#: Truncation for idea rationales / fade reasons — one phone line.
_RATIONALE_MAX = 60


def fetch_volume_marks(
    engine: Any,
    lookback_hours: int = VOLUME_MARK_LOOKBACK_HOURS,
) -> dict[str, float]:
    """Latest unusual RVOL per ticker within the lookback window —
    ``{"FRO": 3.2, ...}`` for Watch-line annotation.

    Fail-soft BY DESIGN: the ``volume_spikes`` table not existing yet
    (migration not run), an empty table, or any DB error all return
    ``{}`` so the digest renders unannotated instead of dying — the
    scanner is an optional enhancement, not a digest dependency.
    """
    try:
        from sqlalchemy import text  # noqa: PLC0415

        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT ticker, rvol FROM volume_spikes "
                    "WHERE is_unusual AND scanned_at >= :cutoff "
                    "ORDER BY scanned_at ASC"
                ),
                {"cutoff": cutoff},
            )
            # ASC + dict overwrite → the LATEST scan wins per ticker.
            return {str(t): float(r) for t, r in result}
    except Exception:
        logger.debug("volume marks unavailable; digest renders unannotated", exc_info=True)
        return {}


def _urgency(result: AssessmentResult) -> int:
    """Defensive urgency read — assessments come from an LLM via the
    DB; a missing/garbage value sorts the event out rather than
    crashing the digest."""
    try:
        return int(result.assessment.get("urgency", 0))
    except (TypeError, ValueError):
        return 0


def _truncate(text: str, limit: int = _HEADLINE_MAX) -> str:
    text = " ".join(str(text).split())  # collapse newlines/runs
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _price_part(
    instrument: str, prices: Mapping[str, Mapping[str, Any]] | None,
) -> str:
    """`` $24.10 (+3.2%)`` (leading space) or ``""`` — format_price
    output is HTML-safe by construction (digits and ``$%().+-``)."""
    rendered = format_price(instrument, (prices or {}).get(instrument))
    return f" {rendered}" if rendered else ""


def _instrument_token(
    instrument: str,
    directions: set[str],
    prices: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """``BCO_USD`` + {long} → ``BCO_USD↑`` (``BCO_USD 78.4 (+2.1%)↑``
    when priced); conflicting directions across events drop the arrow
    rather than pick a side."""
    arrow = ""
    if directions == {"long"}:
        arrow = _ARROWS["long"]
    elif directions == {"short"}:
        arrow = _ARROWS["short"]
    return f"{html_escape(instrument)}{_price_part(instrument, prices)}{arrow}"


def _watch_token(
    instrument: str,
    volume_marks: Mapping[str, float] | None,
    prices: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Watch-line token, price-annotated when available and
    RVOL-annotated when the ticker had a recent unusual spike:
    ``FRO $24.10 (+3.2%)×3.2``. The price/rvol suffixes are
    machine-generated, but the ticker is escaped like every other
    interpolated value."""
    token = html_escape(instrument) + _price_part(instrument, prices)
    rvol = (volume_marks or {}).get(instrument)
    if rvol is not None and rvol > 0:
        token += f"×{rvol:.1f}"
    return token


def _token_lines(label_html: str, tokens: list[str]) -> list[str]:
    """``<b>Tradable:</b> tok tok …`` wrapped every TOKENS_PER_LINE
    tokens so priced tokens stay phone-readable."""
    lines = [f"{label_html} " + " ".join(tokens[:TOKENS_PER_LINE])]
    for i in range(TOKENS_PER_LINE, len(tokens), TOKENS_PER_LINE):
        lines.append(" ".join(tokens[i : i + TOKENS_PER_LINE]))
    return lines


def _action_label(action: str) -> str:
    """Human, imperative action for the compact line: ``buy_puts`` →
    ``BUY PUTS``, ``short`` → ``SHORT``. Keeps the operator's own words
    (puts/shorts) front and center (CL-jiqq)."""
    return str(action or "?").replace("_", " ").strip().upper() or "?"


def _idea_line(
    idea: Mapping[str, Any],
    prices: Mapping[str, Mapping[str, Any]] | None,
) -> str:
    """One compact Ideas: line, now carrying the GROUNDED numbers
    (CL-jiqq): ``TSM $172.40 (-1.8%) — BUY PUTS 1-3wk | entry <trigger>
    | stop $181 | tgt $150 | R:R 2.1 | 5d stop``. Dollar levels come
    from :func:`src.events.trade_card.build_trade_card` off the real
    price; absent a price they simply don't render. Every LLM-sourced
    field is escaped; machine numbers are HTML-safe by construction."""
    from src.events.trade_card import build_trade_card  # noqa: PLC0415

    ticker = str(idea.get("ticker") or "")
    info = (prices or {}).get(ticker) or {}
    card = build_trade_card(
        dict(idea), info.get("price"), info.get("change_pct"),
    )

    # Lead: ticker + price + imperative action + DTE window (options).
    head = f"{html_escape(ticker)}{_price_part(ticker, prices)} — "
    call = _action_label(str(idea.get("action") or "?"))
    dte = str(card.get("dte_window") or "").replace(" weeks", "wk").replace(
        " months", "mo",
    )
    if dte:
        call += f" {dte}"
    line = head + html_escape(call)

    # Grounded segments, pipe-separated — only what actually resolved.
    segs: list[str] = []
    trigger = str(idea.get("entry_trigger") or "").strip()
    if trigger:
        segs.append(f"entry {html_escape(_truncate(trigger, 28))}")
    stop_price = card.get("stop_price")
    if stop_price is not None:
        segs.append(f"stop ${stop_price:,.0f}")
    targets = card.get("target_prices") or []
    if targets:
        segs.append("tgt " + "/".join(f"${t:,.0f}" for t in targets))
    rr = card.get("risk_reward")
    if rr is not None:
        segs.append(f"R:R {rr}")
    time_stop = idea.get("time_stop_days")
    if time_stop is not None:
        segs.append(f"{time_stop}d stop")
    if segs:
        line += " | " + " | ".join(segs)
    return line


def _fade_line(fade: Mapping[str, Any]) -> str:
    """One Fade: line — ``NVDA — fade the spike — reason…``."""
    line = html_escape(str(fade.get("ticker") or ""))
    action = str(fade.get("action") or "").strip() or "fade"
    line += f" — {html_escape(_truncate(action, _RATIONALE_MAX))}"
    reason = str(fade.get("reason") or "").strip()
    if reason:
        line += f" — {html_escape(_truncate(reason, _RATIONALE_MAX))}"
    return line


def _advisory_entries(
    qualifying: Sequence[AssessmentResult], key: str,
) -> list[dict[str, Any]]:
    """Union the assessments' advisory lists (``trade_ideas`` /
    ``fade_candidates``) across qualifying events, urgency-desc order,
    deduped on (ticker, action) — first (most urgent) occurrence wins."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in qualifying:
        entries = r.assessment.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            dedup = (
                str(entry.get("ticker") or ""), str(entry.get("action") or ""),
            )
            if not dedup[0] or dedup in seen:
                continue
            seen.add(dedup)
            out.append(entry)
    return out


def build_digest(
    results: Sequence[AssessmentResult],
    min_urgency: int = DEFAULT_MIN_URGENCY,
    max_events: int = MAX_EVENTS,
    volume_marks: Mapping[str, float] | None = None,
    prices: Mapping[str, Mapping[str, Any]] | None = None,
    seen_ats: Mapping[int, Any] | None = None,
    now: datetime | None = None,
) -> tuple[str, str] | None:
    """Build ``(title, html_message)`` for one assess cycle, or
    ``None`` when nothing qualifies (never send an empty digest).

    Only ASSESSED results with ``urgency >= min_urgency`` are
    included, grouped by theme, most urgent first within each group.

    Enrichment inputs (all optional, all degrade by omission —
    CL-mgcp): ``prices`` from :func:`src.events.prices.get_prices`
    annotates every ticker token and Ideas line; ``seen_ats``
    (event_id → seen_at) adds per-event age; ``now`` pins the age
    clock for tests.
    """
    qualifying = sorted(
        (r for r in results if r.status == "ASSESSED" and _urgency(r) >= min_urgency),
        key=_urgency,
        reverse=True,
    )
    if not qualifying:
        return None

    shown = qualifying[:max_events]
    elided = len(qualifying) - len(shown)

    # Group shown events by theme, preserving urgency-desc order both
    # of the groups (by their most urgent member) and within a group.
    by_theme: dict[str, list[AssessmentResult]] = {}
    for r in shown:
        by_theme.setdefault(r.theme or "other", []).append(r)

    # Instrument union across ALL qualifying events (not just shown) —
    # the ticker feed stays complete even when event lines are elided.
    tradable: dict[str, set[str]] = {}
    watch: dict[str, set[str]] = {}
    for r in qualifying:
        affected = r.assessment.get("affected")
        if not isinstance(affected, list):
            continue
        for entry in affected:
            if not isinstance(entry, dict):
                continue
            instrument = str(entry.get("instrument", "")).strip()
            if not instrument:
                continue
            kind = str(entry.get("kind", "")).strip().lower()
            direction = str(entry.get("direction", "")).strip().lower()
            if kind in _TRADABLE_KINDS and direction in ("long", "short"):
                tradable.setdefault(instrument, set()).add(direction)
            else:
                watch.setdefault(instrument, set()).add(direction)
    # An instrument that is tradable in ANY event doesn't also need a
    # watch entry — the stronger call wins.
    for instrument in tradable:
        watch.pop(instrument, None)

    lines: list[str] = []
    for theme, events in by_theme.items():
        lines.append(f"<i>{html_escape(theme)}</i>")
        for r in events:
            line = f"<b>{_urgency(r)}/10</b> {html_escape(_truncate(r.headline))}"
            age = format_age((seen_ats or {}).get(r.event_id), now=now)
            if age:
                line += f" ({age} ago)"
            lines.append(line)
        lines.append("")
    if elided:
        lines.append(f"+{elided} more above urgency {min_urgency}")
        lines.append("")

    if tradable:
        lines.extend(_token_lines(
            "<b>Tradable:</b>",
            [_instrument_token(i, d, prices) for i, d in tradable.items()],
        ))
    if watch:
        lines.extend(_token_lines(
            "<b>Watch:</b>",
            [_watch_token(i, volume_marks, prices) for i in watch],
        ))

    ideas = _advisory_entries(qualifying, "trade_ideas")
    if ideas:
        lines.append("")
        lines.append("<b>Ideas:</b>")
        lines.extend(_idea_line(idea, prices) for idea in ideas[:MAX_IDEAS])
        if len(ideas) > MAX_IDEAS:
            lines.append(f"+{len(ideas) - MAX_IDEAS} more ideas")
    fades = _advisory_entries(qualifying, "fade_candidates")
    if fades:
        lines.append("")
        lines.append("<b>Fade:</b>")
        lines.extend(_fade_line(fade) for fade in fades[:MAX_FADES])

    while lines and not lines[-1]:
        lines.pop()

    title = (
        f"Event scan — {len(qualifying)} "
        f"event{'s' if len(qualifying) != 1 else ''}, "
        f"{len(tradable)} tradable instrument{'s' if len(tradable) != 1 else ''}"
    )
    return title, "\n".join(lines)


def send_digest(
    results: Sequence[AssessmentResult],
    min_urgency: int = DEFAULT_MIN_URGENCY,
    max_events: int = MAX_EVENTS,
    volume_marks: Mapping[str, float] | None = None,
    prices: Mapping[str, Mapping[str, Any]] | None = None,
    seen_ats: Mapping[int, Any] | None = None,
) -> DispatchResult | None:
    """Build and dispatch the cycle digest via Telegram.

    Returns the ``DispatchResult``, or ``None`` when nothing cleared
    the threshold (in which case NO message is sent — the digest never
    spams quiet cycles). Telegram being unconfigured is the
    dispatcher's no-op, same as every other notification.

    ``volume_marks`` (ticker → recent unusual RVOL, see
    :func:`fetch_volume_marks`) annotates the Watch line;
    ``prices`` / ``seen_ats`` enrich tokens, Ideas lines, and event
    ages (CL-mgcp). ``None`` / ``{}`` for any of them renders that
    annotation off.
    """
    built = build_digest(
        results, min_urgency=min_urgency, max_events=max_events,
        volume_marks=volume_marks, prices=prices, seen_ats=seen_ats,
    )
    if built is None:
        logger.info(
            "digest: no ASSESSED events at urgency >= %d; nothing sent",
            min_urgency,
        )
        return None
    title, message = built
    return notify_operator(title, message, html=True)
