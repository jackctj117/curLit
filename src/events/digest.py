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

All interpolated content (headlines, themes, instruments) is escaped
via :func:`src.research.notifications.html_escape` — GDELT headlines
are hostile input.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from src.events.impact_agent import AssessmentResult
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


def _instrument_token(instrument: str, directions: set[str]) -> str:
    """``BCO_USD`` + {long} → ``BCO_USD↑``; conflicting directions
    across events drop the arrow rather than pick a side."""
    arrow = ""
    if directions == {"long"}:
        arrow = _ARROWS["long"]
    elif directions == {"short"}:
        arrow = _ARROWS["short"]
    return f"{html_escape(instrument)}{arrow}"


def _watch_token(
    instrument: str, volume_marks: Mapping[str, float] | None,
) -> str:
    """Watch-line token, RVOL-annotated when the ticker had a recent
    unusual spike: ``FRO`` + 3.2 → ``FRO×3.2``. The rvol suffix is
    machine-generated (float), but the ticker is escaped like every
    other interpolated value."""
    token = html_escape(instrument)
    rvol = (volume_marks or {}).get(instrument)
    if rvol is not None and rvol > 0:
        token += f"×{rvol:.1f}"
    return token


def build_digest(
    results: Sequence[AssessmentResult],
    min_urgency: int = DEFAULT_MIN_URGENCY,
    max_events: int = MAX_EVENTS,
    volume_marks: Mapping[str, float] | None = None,
) -> tuple[str, str] | None:
    """Build ``(title, html_message)`` for one assess cycle, or
    ``None`` when nothing qualifies (never send an empty digest).

    Only ASSESSED results with ``urgency >= min_urgency`` are
    included, grouped by theme, most urgent first within each group.
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
            lines.append(
                f"<b>{_urgency(r)}/10</b> {html_escape(_truncate(r.headline))}"
            )
        lines.append("")
    if elided:
        lines.append(f"+{elided} more above urgency {min_urgency}")
        lines.append("")

    if tradable:
        tokens = " ".join(
            _instrument_token(i, d) for i, d in tradable.items()
        )
        lines.append(f"<b>Tradable:</b> {tokens}")
    if watch:
        lines.append(
            "<b>Watch:</b> "
            + " ".join(_watch_token(i, volume_marks) for i in watch)
        )

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
) -> DispatchResult | None:
    """Build and dispatch the cycle digest via Telegram.

    Returns the ``DispatchResult``, or ``None`` when nothing cleared
    the threshold (in which case NO message is sent — the digest never
    spams quiet cycles). Telegram being unconfigured is the
    dispatcher's no-op, same as every other notification.

    ``volume_marks`` (ticker → recent unusual RVOL, see
    :func:`fetch_volume_marks`) annotates the Watch line; ``None`` /
    ``{}`` renders it unannotated.
    """
    built = build_digest(
        results, min_urgency=min_urgency, max_events=max_events,
        volume_marks=volume_marks,
    )
    if built is None:
        logger.info(
            "digest: no ASSESSED events at urgency >= %d; nothing sent",
            min_urgency,
        )
        return None
    title, message = built
    return notify_operator(title, message, html=True)
