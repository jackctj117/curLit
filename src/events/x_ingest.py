"""X → geo_events bridge (CL-esyo) — the X watchlist feeds the analysis
pipeline instead of merely relaying tweets to Telegram.

The X monitor (``src.data.x_monitor``) polls a curated watchlist of
news / flow / OSINT / mining accounts. This module turns each relevant
post into a NEW ``geo_events`` row — exactly the shape the GDELT
ingester writes — so the EXISTING Event Impact Agent assesses it into
tickers / trade ideas, no second assessment path.

The QUOTA-CRITICAL relevance gate (LLM assessment is metered):

  * A post is only ingested if its text keyword-matches a playbook
    ``watch_term`` — and THAT match assigns the row's ``theme``. No
    match → not ingested, never assessed (see :func:`match_theme`).
  * The whole ``small_traders`` category is excluded — unverified
    entertainment, never a geopolitical event.
  * A per-call ``cap`` (default 10) stops one chatty account flooding
    the queue in a single cycle.

Provenance for the impact agent: the row's ``source`` is
``x:<handle>`` (e.g. ``x:DeItaone``). The prompt builder passes it
through :func:`source_credibility_note` so the model can calibrate
confidence on a fast-but-unconfirmed headline account vs an OSINT
account that needs cross-checking. No schema change: source alone
carries the handle, and a small handle/category mapping does the rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from src.data.x_monitor import Post, WatchAccount
    from src.events.playbooks import Playbook

logger = logging.getLogger(__name__)

#: Per-call ingest cap — one chatty account can't flood the NEW queue.
DEFAULT_INGEST_CAP = 10

#: Category excluded from ingestion entirely (unverified entertainment).
EXCLUDED_CATEGORY = "small_traders"

#: Max headline length stored — geo_events.headline is TEXT, but a
#: pathological tweet (or thread-quote blob) shouldn't bloat the prompt.
_MAX_HEADLINE_LEN = 500


@dataclass(frozen=True)
class IngestResult:
    """Counts from one :func:`ingest_posts` call — the monitor logs these."""

    ingested: int = 0
    skipped_no_theme: int = 0
    skipped_category: int = 0
    deduped: int = 0

    def summary_line(self, handle: str) -> str:
        return (
            f"x-ingest @{handle}: {self.ingested} ingested, "
            f"{self.skipped_no_theme} no-theme, "
            f"{self.skipped_category} category-excluded, "
            f"{self.deduped} deduped"
        )


# --------------------------------------------------------------------- #
# Theme matching (pure — unit-tested without a DB)
# --------------------------------------------------------------------- #


def _term_matches(text_lower: str, term_lower: str) -> bool:
    """True if ``term_lower`` appears in ``text_lower``.

    A single-word term must match on a word boundary (so 'coup' does not
    fire on 'couple'); a multi-word / phrase term is a plain substring
    match (spacing/punctuation between words is the operator's concern,
    and phrase terms are already specific enough).
    """
    if not term_lower:
        return False
    if " " in term_lower or "-" in term_lower:
        return term_lower in text_lower
    # Single word: require non-alphanumeric boundaries around the hit so
    # 'war' doesn't fire inside 'warehouse'. Scan every occurrence.
    start = 0
    n = len(term_lower)
    while True:
        idx = text_lower.find(term_lower, start)
        if idx < 0:
            return False
        before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
        after = idx + n
        after_ok = after >= len(text_lower) or not text_lower[after].isalnum()
        if before_ok and after_ok:
            return True
        start = idx + 1


def match_theme(
    post_text: str,
    playbooks: dict[str, Playbook],
) -> str | None:
    """Match ``post_text`` against every playbook's ``watch_terms``.

    Case-insensitive. Returns the theme with the STRONGEST match, or
    ``None`` when nothing matches (which is the relevance gate: a
    no-match post is never ingested). "Strongest" = most terms matched;
    ties broken by the longest single matched term (a specific phrase
    like 'strait of hormuz' beats a lone generic word). Deterministic on
    ties: playbook insertion order is the final tiebreak.
    """
    if not post_text or not post_text.strip():
        return None
    text_lower = post_text.lower()
    best_theme: str | None = None
    best_count = 0
    best_longest = 0
    for theme, pb in playbooks.items():
        count = 0
        longest = 0
        for term in pb.watch_terms:
            term_lower = term.strip().lower()
            if _term_matches(text_lower, term_lower):
                count += 1
                longest = max(longest, len(term_lower))
        if count == 0:
            continue
        if (count, longest) > (best_count, best_longest):
            best_theme = theme
            best_count = count
            best_longest = longest
    return best_theme


# --------------------------------------------------------------------- #
# Source-credibility provenance for the impact prompt
# --------------------------------------------------------------------- #

#: Per-category credibility framing. The impact agent sees the row's
#: ``source`` (``x:<handle>``) and uses this to calibrate confidence.
_CATEGORY_CREDIBILITY: dict[str, str] = {
    "financial_flow": (
        "fast market-moving headline/flow account — usually reliable but "
        "unconfirmed; treat as an early wire, not a filed report"
    ),
    "conflict_osint": (
        "conflict OSINT — frequently first but frequently wrong; "
        "cross-check before raising confidence, may be unverified"
    ),
    "africa_mining": (
        "Africa mining/commodities specialist — good sector intel, but a "
        "single-source claim; treat asset/permit specifics as unconfirmed"
    ),
}

#: Handle-specific overrides win over the category framing (lower-cased).
_HANDLE_CREDIBILITY: dict[str, str] = {
    "deitaone": (
        "Walter Bloomberg / @DeItaone — fast headline mirror of newswire "
        "flashes; reliable wording but unconfirmed and often already priced"
    ),
    "sentdefender": (
        "@sentdefender — high-volume OSINT aggregator; fast but noisy and "
        "frequently unverified — cross-check before acting"
    ),
}


def source_credibility_note(source: str, category: str | None = None) -> str | None:
    """One line of provenance for an ``x:<handle>`` ``geo_events.source``.

    Returns ``None`` for non-X sources (e.g. plain ``gdelt``) so the
    prompt builder adds nothing. ``category`` is optional context: it is
    unavailable when only the persisted row is in hand, so the handle
    mapping carries the load, falling back to the category framing.
    """
    if not source or not source.startswith("x:"):
        return None
    handle = source[len("x:"):].strip()
    handle_lower = handle.lower()
    detail = _HANDLE_CREDIBILITY.get(handle_lower)
    if detail is None and category:
        detail = _CATEGORY_CREDIBILITY.get(category)
    if detail is None:
        detail = (
            "X/social post — a single unverified source; weigh accordingly"
        )
    return f"SOURCE: X/@{handle} ({detail})."


# --------------------------------------------------------------------- #
# seen_at parsing
# --------------------------------------------------------------------- #


def _parse_created_at(created_at: str) -> datetime:
    """Parse a post's ``created_at`` into a tz-aware UTC datetime.

    Defensive on purpose: the API v2 backend gives ISO-8601
    ('2026-07-20T21:00:54.000Z'); a bird-style CLI gives the classic
    Twitter format ('Mon Jul 20 21:00:54 +0000 2026'). Anything
    unparseable (or empty) falls back to now() — a fresh X post with a
    slightly-wrong timestamp is still worth assessing, and a bad string
    must never drop the row.
    """
    raw = (created_at or "").strip()
    if not raw:
        return datetime.now(UTC)
    # ISO-8601 first (API v2). Python's fromisoformat handles the 'Z'
    # suffix from 3.11 on; be defensive anyway.
    iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(iso)
        return _as_utc(dt)
    except ValueError:
        pass
    # Classic Twitter / RFC-2822-ish format ('Mon Jul 20 ... +0000 2026').
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            return _as_utc(dt)
    except (TypeError, ValueError):
        pass
    # bird's 'Mon Jul 20 21:00:54 +0000 2026' isn't RFC-2822 order —
    # try the classic Twitter strptime pattern before giving up.
    try:
        dt = datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
        return _as_utc(dt)
    except ValueError:
        logger.debug("unparseable created_at %r — using now()", raw)
        return datetime.now(UTC)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _normalise_headline(text_value: str) -> str:
    """Collapse whitespace and cap length for the stored headline."""
    collapsed = " ".join((text_value or "").split())
    if len(collapsed) <= _MAX_HEADLINE_LEN:
        return collapsed
    return collapsed[: _MAX_HEADLINE_LEN - 1].rstrip() + "…"


# --------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------- #

#: ON CONFLICT DO NOTHING on the external_id unique key — the same post
#: seen on two polls collapses to one row (mirrors the GDELT dedup).
_INSERT_SQL = text(
    "INSERT INTO geo_events "
    "(seen_at, source, external_id, headline, url, theme, status, "
    " status_updated_at) "
    "VALUES (:seen_at, :source, :external_id, :headline, :url, :theme, "
    "        'NEW', :now) "
    "ON CONFLICT (external_id) DO NOTHING",
)


def ingest_posts(
    engine: Engine,
    account: WatchAccount,
    posts: list[Post],
    playbooks: dict[str, Playbook],
    cap: int = DEFAULT_INGEST_CAP,
) -> IngestResult:
    """Turn relevant posts from one account into NEW ``geo_events`` rows.

    Relevance gate (see the module docstring): the whole
    ``small_traders`` category is skipped; a post with no matched theme
    is skipped; everything else is inserted with ``theme`` = the match,
    ``source`` = ``x:<handle>``, and ``external_id`` = ``x:<post id>``
    (ON CONFLICT DO NOTHING dedups re-seen posts). ``cap`` bounds the
    inserts per call so one burst can't flood the assessment queue.

    Returns per-category counts; never raises for a DB problem (the
    monitor loop must survive) — a failure is logged and counted as
    zero-ingested for that call.
    """
    if account.category == EXCLUDED_CATEGORY:
        # Whole category excluded — count every post as category-skipped
        # so the monitor's log is honest about what was dropped and why.
        return IngestResult(skipped_category=len(posts))

    ingested = 0
    skipped_no_theme = 0
    deduped = 0
    rows: list[dict[str, object]] = []
    now = datetime.now(UTC)
    for post in posts:
        if ingested >= cap:
            # Cap reached: remaining posts are neither ingested nor
            # counted as skipped-for-theme — they simply don't fit this
            # call. The next poll's since_id has already advanced past
            # them in the monitor, so this is a deliberate flood guard.
            break
        theme = match_theme(post.text, playbooks)
        if theme is None:
            skipped_no_theme += 1
            continue
        rows.append({
            "seen_at": _parse_created_at(post.created_at),
            "source": f"x:{account.handle}",
            "external_id": f"x:{post.id}",
            "headline": _normalise_headline(post.text),
            "url": f"https://x.com/{account.handle}/status/{post.id}",
            "theme": theme,
            "now": now,
        })
        ingested += 1

    if not rows:
        return IngestResult(
            ingested=0,
            skipped_no_theme=skipped_no_theme,
            skipped_category=0,
            deduped=0,
        )

    try:
        with engine.begin() as conn:
            inserted_ids: set[str] = set()
            for row in rows:
                result = conn.execute(_INSERT_SQL, row)
                # rowcount is 1 on insert, 0 when ON CONFLICT skipped it.
                if getattr(result, "rowcount", 0) and result.rowcount > 0:
                    inserted_ids.add(str(row["external_id"]))
        inserted = len(inserted_ids)
        deduped = len(rows) - inserted
        ingested = inserted
    except Exception as exc:
        # A DB failure must never break the monitor loop — log and
        # report zero ingested for this call.
        logger.warning(
            "x-ingest DB write failed for @%s (%d candidate rows): %s",
            account.handle, len(rows), str(exc)[:200],
        )
        return IngestResult(
            ingested=0,
            skipped_no_theme=skipped_no_theme,
            skipped_category=0,
            deduped=0,
        )

    return IngestResult(
        ingested=ingested,
        skipped_no_theme=skipped_no_theme,
        skipped_category=0,
        deduped=deduped,
    )
