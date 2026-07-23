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
from src.events.retail_proxy import compact_label
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

#: How many corroborating themes to name inline before a "+N more" tail
#: on the corroboration note (CL-5mkf) — keeps the line phone-readable.
_MAX_CORROB_THEMES = 3

#: Pure safe-haven instruments (and their retail proxies) — the
#: haven-CLUSTER concentration reminder fires when these dominate the
#: advisory ideas as a group (CL-5mkf). Advisory display only; the
#: machine cap lives in the EventDrivenStrategy.
_HAVEN_TICKERS = frozenset({"XAU_USD", "XAG_USD", "GLD", "SLV", "IAU"})

#: Max per-instrument concentration reminder lines before a "+N more"
#: tail (CL-wbmw) — keeps the Ideas section phone-readable.
_MAX_CONCENTRATION_NOTES = 3


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
    instrument: str,
    prices: Mapping[str, Mapping[str, Any]] | None,
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
        dict(idea),
        info.get("price"),
        info.get("change_pct"),
    )

    # A multi-line block (CL-jiqq follow-up: single truncated lines hid
    # the entry conditions the operator needs to act). Line 1: ticker +
    # price + action + DTE. Line 2: the FULL entry trigger. Line 3: the
    # grounded number segments. Blank lines between blocks are added by
    # the assembler.
    head = f"<b>{html_escape(ticker)}</b>{_price_part(ticker, prices)} — "
    call = _action_label(str(idea.get("action") or "?"))
    dte = (
        str(card.get("dte_window") or "")
        .replace(" weeks", "wk")
        .replace(
            " months",
            "mo",
        )
    )
    if dte:
        call += f" {dte}"
    block: list[str] = [head + html_escape(call)]

    trigger = str(idea.get("entry_trigger") or "").strip()
    if trigger:
        # Full trigger (generous cap only to bound pathological output).
        block.append(f"  entry: {html_escape(_truncate(trigger, 240))}")

    # Grounded number segments — only what actually resolved.
    segs: list[str] = []
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
        block.append("  " + " | ".join(segs))

    # Corroboration note (CL-5mkf): when >1 qualifying event proposed the
    # SAME (ticker, action), this idea is CONVICTION, not a duplicate to
    # drop. Name the corroborating themes (capped, +N more). All themes
    # are LLM/config-sourced → escaped like every interpolated value.
    corrob = idea.get("corroboration")
    if isinstance(corrob, Mapping):
        count = int(corrob.get("count") or 0)
        themes = [str(t) for t in (corrob.get("themes") or []) if str(t)]
        if count > 1:
            shown = themes[:_MAX_CORROB_THEMES]
            extra = len(themes) - len(shown)
            theme_str = ", ".join(html_escape(t) for t in shown)
            if extra > 0:
                theme_str += f" +{extra} more"
            note = f"  ✓ corroborated by {count} events"
            if theme_str:
                note += f" ({theme_str})"
            block.append(note)

    # Niche / asymmetry tag (CL-u2ph): a multi-hop under-followed name the
    # obvious trade missed. Show the hop count + asymmetry score + the
    # torque mechanism, and — when it cleared verification but tripped the
    # liquidity floor — a size-small/check-spread caveat. torque_reason is
    # LLM-sourced → escaped; the numbers are HTML-safe by construction.
    if idea.get("niche"):
        try:
            hops = int(idea.get("hop_count") or 0)
        except (TypeError, ValueError):
            hops = 0
        asym = idea.get("asymmetry_score")
        asym_str = f"{float(asym):.2f}" if isinstance(asym, (int, float)) else "?"
        torque = _truncate(str(idea.get("torque_reason") or ""), 80)
        tag = f"  🎯 niche ({hops} hop{'s' if hops != 1 else ''}, asym {asym_str})"
        if torque:
            tag += f" — {html_escape(torque)}"
        block.append(tag)
        if idea.get("liquidity_flag"):
            block.append("  ⚠ small/illiquid — size small, check spread")

    # Robinhood execution proxy (CL-vowz): the operator trades Robinhood,
    # which has no FX/CFDs/futures — so a raw 'XAU_USD LONG' idea is not
    # placeable. Show the tradable version on its own indented line.
    # ``direction`` (falling back to ``action``) picks the short side.
    proxy = compact_label(
        ticker,
        str(idea.get("direction") or idea.get("action") or ""),
    )
    block.append(f"  RH: {html_escape(proxy)}")
    return "\n".join(block)


def _idea_event_count(idea: Mapping[str, Any]) -> int:
    """How many events back this idea: its corroboration count (>= 1),
    or 1 when it carries no corroboration block."""
    corrob = idea.get("corroboration")
    if isinstance(corrob, Mapping):
        return max(1, int(corrob.get("count") or 1))
    return 1


def _concentration_notes(ideas: Sequence[Mapping[str, Any]]) -> list[str]:
    """Concentration reminders for the Ideas section (CL-wbmw, generalizes
    CL-5mkf's gold-only note), or ``[]``. Rendered ONCE, before the idea
    blocks, not per-idea.

    These are DISPLAYED reminders for hand-executed Robinhood trades — we
    can't ENFORCE concentration on trades the operator places by hand, so
    this is honestly just a nudge. The real machine cap
    (per_instrument_max_pct + haven_max_pct) lives in the
    EventDrivenStrategy and applies only to the OANDA paper legs.

    Two flavours:

      * Per-instrument line, for ANY over-weight instrument — one that is
        (a) corroborated by multiple events (corroboration.count > 1) OR
        (b) appears in >= 2 distinct ideas in this digest:
        ``⚠️ already exposed to <INSTRUMENT> via N ideas — watch
        concentration``. N = distinct ideas + extra corroborating events
        beyond the first on each (how many ways the operator is nudged
        into that name). Capped at _MAX_CONCENTRATION_NOTES lines with a
        ``+N more`` tail, most-exposed first.
      * Haven-CLUSTER line, when combined gold/silver ideas are over-weight
        AS A GROUP: ``⚠️ watch gold/silver (haven) concentration``.

    Don't double-warn: a haven instrument already flagged by its own
    per-instrument line is excluded from the cluster tally — we prefer the
    specific line. The cluster line fires only for a group over-weight the
    individual lines don't already cover.

    Instrument names are HTML-escaped (LLM/config-sourced); the fixed
    prose is HTML-safe by construction.
    """
    # Tally exposure per instrument (distinct-idea count + extra
    # corroborating events), preserving first-seen (urgency-desc) order.
    order: list[str] = []
    distinct: dict[str, int] = {}
    events: dict[str, int] = {}
    corroborated: dict[str, bool] = {}
    for idea in ideas:
        ticker = str(idea.get("ticker") or "")
        if not ticker:
            continue
        count = _idea_event_count(idea)
        if ticker not in distinct:
            order.append(ticker)
            distinct[ticker] = 0
            events[ticker] = 0
            corroborated[ticker] = False
        distinct[ticker] += 1
        events[ticker] += count
        if count > 1:
            corroborated[ticker] = True

    # Over-weight = corroborated by >1 event OR appears in >= 2 ideas.
    overweight = [t for t in order if corroborated[t] or distinct[t] >= 2]

    notes: list[str] = []
    for ticker in overweight[:_MAX_CONCENTRATION_NOTES]:
        notes.append(
            f"⚠️ already exposed to {html_escape(ticker)} via "
            f"{events[ticker]} ideas — watch concentration"
        )
    extra = len(overweight) - _MAX_CONCENTRATION_NOTES
    if extra > 0:
        notes.append(f"+{extra} more over-weight instruments")

    # Haven-cluster line: the metals as a GROUP, but ONLY counting havens
    # NOT already covered by a specific per-instrument line above (no
    # double-warning — the specific line wins).
    overweight_havens = {t for t in overweight if t.upper() in _HAVEN_TICKERS}
    cluster_ideas = [
        i
        for i in ideas
        if str(i.get("ticker") or "").upper() in _HAVEN_TICKERS
        and str(i.get("ticker") or "") not in overweight_havens
    ]
    cluster_distinct = len({str(i.get("ticker") or "") for i in cluster_ideas})
    cluster_corroborated = any(_idea_event_count(i) > 1 for i in cluster_ideas)
    if cluster_distinct >= 2 or cluster_corroborated:
        notes.append("⚠️ watch gold/silver (haven) concentration")

    return notes


def _fade_line(fade: Mapping[str, Any]) -> str:
    """One Fade: block — ticker + action on line 1, full reason on
    line 2 (the reason is what tells the operator WHY to fade)."""
    ticker = html_escape(str(fade.get("ticker") or ""))
    action = str(fade.get("action") or "").strip() or "fade"
    block = [f"<b>{ticker}</b> — {html_escape(_truncate(action, 160))}"]
    reason = str(fade.get("reason") or "").strip()
    if reason:
        block.append(f"  {html_escape(_truncate(reason, 240))}")
    return "\n".join(block)


def _advisory_entries(
    qualifying: Sequence[AssessmentResult],
    key: str,
) -> list[dict[str, Any]]:
    """Union the assessments' advisory lists (``trade_ideas`` /
    ``fade_candidates``) across qualifying events, urgency-desc order,
    deduped on (ticker, action) — first (most urgent) occurrence wins.

    Redundancy is CONVICTION, not noise (CL-5mkf): instead of silently
    dropping duplicate (ticker, action) entries, we COUNT how many
    qualifying events proposed each and collect their (distinct) themes,
    attaching ``corroboration = {"count": N, "themes": [...]}`` to the
    kept entry. ``count`` is the number of events that proposed it
    (>= 1); ``themes`` preserves first-seen (urgency-desc) order. The
    kept entry is a shallow copy so the source assessment is untouched.
    """
    out: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in qualifying:
        entries = r.assessment.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            dedup = (
                str(entry.get("ticker") or ""),
                str(entry.get("action") or ""),
            )
            if not dedup[0]:
                continue
            theme = str(r.theme or "").strip()
            kept = by_key.get(dedup)
            if kept is None:
                kept = dict(entry)
                kept["corroboration"] = {
                    "count": 1,
                    "themes": [theme] if theme else [],
                }
                by_key[dedup] = kept
                out.append(kept)
                continue
            # A later (less-urgent) event proposing the same idea →
            # corroboration, not a drop. Bump the count; add its theme.
            corrob = kept["corroboration"]
            corrob["count"] += 1
            if theme and theme not in corrob["themes"]:
                corrob["themes"].append(theme)
    return out


def build_cross_asset_line(result: Any) -> str | None:
    """Render one compact cross-asset corroboration line for a confirmed
    event (CL-6mzn), or ``None`` when there is nothing to say.

    ``result`` is a :class:`src.events.cross_asset.CrossAssetResult`.
    Formats as::

        Cross-asset: BCO_USD +1.8% ✓ · USD_CAD -0.2% ✗ · confirms (2/3)

    or, when the corroboration failed, the operator's fade warning::

        Cross-asset: related assets NOT confirming — fade risk (0/3)

    Returns ``None`` (line omitted) when the read is UNKNOWN — no
    instrument had usable price data — so we never render a hollow line.
    Instrument names are HTML-escaped (they come from config, but the
    line is embedded in a Telegram-HTML body); the numeric/✓✗ suffixes
    are HTML-safe by construction.
    """
    if result is None:
        return None
    confirmed = getattr(result, "confirmed", None)
    details = list(getattr(result, "details", []) or [])
    # Unknown (no data) → omit entirely; a line with nothing measured is
    # noise and could be mistaken for "no corroboration".
    if confirmed is None:
        return None

    voting = [d for d in details if getattr(d, "agrees", None) is not None]
    n_voting = len(voting)
    n_agree = sum(1 for d in voting if d.agrees)
    if n_voting == 0:  # defensive — confirmed is non-None but nothing voted
        return None

    parts: list[str] = []
    for d in voting:
        move = d.actual_move_pct or 0.0
        mark = "✓" if d.agrees else "✗"  # ✓ / ✗
        parts.append(f"{html_escape(str(d.instrument))} {move:+.1f}% {mark}")

    body = " · ".join(parts)  # " · " separator
    if confirmed:
        return f"<b>Cross-asset:</b> {body} · confirms ({n_agree}/{n_voting})"
    return (
        f"<b>Cross-asset:</b> related assets NOT confirming — fade risk "
        f"({n_agree}/{n_voting}) · {body}"
    )


def _poly_context_line(poly_signal: Any, theme: str) -> str | None:
    """One corroboration line for a theme's top prediction market, or
    None (CL-r1ep). ``Prediction mkt: Hormuz-closure 18% ↑`` — display
    only, NOT a gate. Fail-soft: any error yields no line. The slug and
    prob are machine-shaped, but the slug is escaped defensively."""
    if poly_signal is None:
        return None
    try:
        probs = poly_signal.latest_prob_for_theme(theme)
    except Exception:
        logger.debug("poly corroboration lookup failed for %s", theme, exc_info=True)
        return None
    if not probs:
        return None
    # Cite the highest-probability market in the theme — the most
    # market-relevant read of the situation.
    slug, info = max(probs.items(), key=lambda kv: kv[1].get("yes_prob") or 0.0)
    pct = round(float(info.get("yes_prob") or 0.0) * 100)
    rising = info.get("rising")
    arrow = " ↑" if rising is True else (" ↓" if rising is False else "")
    return f"<i>Prediction mkt:</i> {html_escape(slug)} {pct}%{arrow}"


def build_digest(
    results: Sequence[AssessmentResult],
    min_urgency: int = DEFAULT_MIN_URGENCY,
    max_events: int = MAX_EVENTS,
    volume_marks: Mapping[str, float] | None = None,
    prices: Mapping[str, Mapping[str, Any]] | None = None,
    seen_ats: Mapping[int, Any] | None = None,
    now: datetime | None = None,
    poly_signal: Any = None,
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
        # Prediction-market corroboration for this theme (CL-r1ep) —
        # display only, one line, degrades to nothing when absent.
        poly_line = _poly_context_line(poly_signal, theme)
        if poly_line:
            lines.append(poly_line)
        lines.append("")
    if elided:
        lines.append(f"+{elided} more above urgency {min_urgency}")
        lines.append("")

    if tradable:
        lines.extend(
            _token_lines(
                "<b>Tradable:</b>",
                [_instrument_token(i, d, prices) for i, d in tradable.items()],
            )
        )
    if watch:
        lines.extend(
            _token_lines(
                "<b>Watch:</b>",
                [_watch_token(i, volume_marks, prices) for i in watch],
            )
        )

    ideas = _advisory_entries(qualifying, "trade_ideas")
    if ideas:
        lines.append("")
        lines.append("<b>Ideas:</b>")
        # Concentration reminders (CL-wbmw, generalizes CL-5mkf) — once,
        # before the ideas: a line per over-weight instrument (corroborated
        # or repeated across ideas) plus a gold/silver cluster line when the
        # havens are over-weight as a group. Displayed nudges only — can't
        # enforce hand-executed Robinhood trades.
        lines.extend(_concentration_notes(ideas))
        for idea in ideas[:MAX_IDEAS]:
            lines.append(_idea_line(idea, prices))
            lines.append("")  # blank line between ideas for readability
        if len(ideas) > MAX_IDEAS:
            lines.append(f"+{len(ideas) - MAX_IDEAS} more ideas")
    fades = _advisory_entries(qualifying, "fade_candidates")
    if fades:
        lines.append("")
        lines.append("<b>Fade:</b>")
        for fade in fades[:MAX_FADES]:
            lines.append(_fade_line(fade))
            lines.append("")  # blank line between fades

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
    poly_signal: Any = None,
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
    annotation off. ``poly_signal`` (a
    :class:`src.events.polymarket_signal.PolymarketSignal`) adds a
    per-theme prediction-market corroboration line (CL-r1ep) — display
    only, never a gate.
    """
    built = build_digest(
        results,
        min_urgency=min_urgency,
        max_events=max_events,
        volume_marks=volume_marks,
        prices=prices,
        seen_ats=seen_ats,
        poly_signal=poly_signal,
    )
    if built is None:
        logger.info(
            "digest: no ASSESSED events at urgency >= %d; nothing sent",
            min_urgency,
        )
        return None
    title, message = built
    return notify_operator(title, message, html=True)
