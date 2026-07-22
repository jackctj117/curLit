"""Truth Social public-post ingester (CL-s9as) — RESEARCH ONLY.

Polls the public trumpstruth.org RSS archive of @realDonaldTrump posts
(no auth, no scraping of Truth Social itself, no early-access feeds —
the operator-approved scope is PUBLIC data only) and upserts into
``truth_posts`` (migration 017). Downstream: classification
(``src/events/truth_classifier.py``) and event-study reaction
measurement (``src/data/truth_reactions.py``). This module never
generates signals, alerts, or orders.
"""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

FEED_URL_DEFAULT = "https://trumpstruth.org/feed"

#: Injectable transport: url -> response body text.
HttpGet = Callable[[str], str]

_TRUTH_NS = "https://truthsocial.com/ns"
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class TruthPost:
    post_id: str
    posted_at: datetime
    text: str
    url: str | None


def _strip_html(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", fragment or "")).strip()


def parse_feed(xml_text: str) -> list[TruthPost]:
    """RSS 2.0 → posts. Media-only posts come through with text='' (kept —
    the archive should be complete; the classifier fast-paths them to
    not-relevant). Unparseable items are skipped with a warning, never
    silently dropped mid-item."""
    out: list[TruthPost] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("truth feed: unparseable XML", exc_info=True)
        return out
    for item in root.iter("item"):
        try:
            original_id = item.findtext(f"{{{_TRUTH_NS}}}originalId")
            guid = item.findtext("guid") or ""
            post_id = str(original_id or guid).strip()
            if not post_id:
                continue
            pub = item.findtext("pubDate") or ""
            posted_at = parsedate_to_datetime(pub)
            if posted_at.tzinfo is None:
                posted_at = posted_at.replace(tzinfo=UTC)
            title = (item.findtext("title") or "").strip()
            desc = _strip_html(item.findtext("description") or "")
            # The archive puts full text in BOTH title and description;
            # title is truncation-free for short posts, description wins
            # for long ones. "[No Title]" placeholders mean media-only.
            body = desc if len(desc) > len(title) else title
            if body.startswith("[No Title]"):
                body = ""
            url = (
                item.findtext(f"{{{_TRUTH_NS}}}originalUrl")
                or item.findtext("link")
            )
            out.append(TruthPost(
                post_id=post_id,
                posted_at=posted_at.astimezone(UTC),
                text=body,
                url=url,
            ))
        except Exception:
            logger.warning("truth feed: skipping unparseable item",
                           exc_info=True)
    return out


def _default_http_get(url: str) -> str:
    import httpx  # noqa: PLC0415 — lazy

    resp = httpx.get(url, timeout=20.0, follow_redirects=True, headers={
        "User-Agent": "curlit-research/1.0 (event-study; public archive)",
    })
    resp.raise_for_status()
    return resp.text


def ingest_posts(
    engine: Any,
    http_get: HttpGet | None = None,
    feed_url: str = FEED_URL_DEFAULT,
    now: datetime | None = None,
) -> int:
    """Fetch the public feed and upsert new posts. Returns the number of
    NEW rows. Fail-soft: a fetch/parse failure logs and returns 0 (the
    5-min loop retries)."""
    fetch = http_get or _default_http_get
    now = now or datetime.now(UTC)
    try:
        posts = parse_feed(fetch(feed_url))
    except Exception:
        logger.warning("truth feed: fetch failed — retry next cycle",
                       exc_info=True)
        return 0
    if not posts:
        logger.warning("truth feed: feed returned ZERO items — possible "
                       "feed outage; posts published now may be missed")
        return 0
    with engine.connect() as conn:
        had_rows = conn.execute(
            text("SELECT 1 FROM truth_posts LIMIT 1"),
        ).scalar() is not None
    new = 0
    with engine.begin() as conn:
        for p in posts:
            result = conn.execute(text("""
                INSERT INTO truth_posts
                    (post_id, posted_at, text, url, ingested_at)
                VALUES (:i, :t, :x, :u, :n)
                ON CONFLICT (post_id) DO NOTHING
            """), {"i": p.post_id, "t": p.posted_at, "x": p.text,
                   "u": p.url, "n": now})
            new += int(result.rowcount or 0)
    # Gap detection: the feed is a sliding window (~100 posts). If NOTHING
    # in the current feed was already stored, the daemon was likely down
    # longer than the window and posts in between are silently missing —
    # say so loudly (a silent cluster miss is worse than a noisy warning).
    if had_rows and posts and new == len(posts):
        logger.warning(
            "truth feed: ZERO overlap between feed (%d items) and stored "
            "posts — possible ingestion GAP; posts between the last stored "
            "item and the oldest feed item may be missing", len(posts),
        )
    if new:
        logger.info("truth feed: %d new post(s) ingested", new)
    return new
