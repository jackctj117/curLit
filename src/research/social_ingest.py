"""Social / forum fetchers for the paper-streams pipeline.

Adds five new feed adapters to the existing src.research.ingest registry:

  reddit       — PRAW-backed; pulls top recent posts from configured subreddits
  hackernews   — Algolia API (free, no auth); pulls recent front-page items
  fourchan     — public 4chan API; pulls /biz/ + configured boards
  lainchan     — JSON API via /<board>/catalog.json
  twitter      — official v2 search/recent (requires TWITTER_BEARER_TOKEN)

Design choices:
- All sources are RECENT-ONLY. Old social posts have nearly zero
  trading edge; the value is in fresh signal vs decayed.
- Each adapter falls through to a logged warning when missing
  credentials or rate-limited rather than crashing the ingest run.
- All adapters write into the same Paper dataclass + the same
  research_papers table → MetaLearner (CL-fsj) downweights the
  signal-poor sources automatically over time.
- 4chan / lainchan signal-to-noise is intentionally flagged
  (source_label includes "(low-signal)") so the RelevanceScorer
  weighs them appropriately and the operator can filter them out
  of the triage dashboard with a single click.

Operator note: Twitter requires a paid API tier (was free, now
$100/mo basic). Without TWITTER_BEARER_TOKEN, the Twitter fetcher
logs a warning and returns [].

Reference: CL-poly-1 follow-up (social-source aggressive feed).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from src.research.ingest import FeedConfig, HttpGet, Paper, _default_http_get

logger = logging.getLogger(__name__)


# Default freshness windows per source. Older posts are dropped because
# social-data tradeable edge decays in hours, not days. Operator can
# override per-feed in paper_streams.yaml via the ``max_age_hours``
# extra field.
_DEFAULT_REDDIT_MAX_AGE_HOURS: int = 24
_DEFAULT_HN_MAX_AGE_HOURS: int = 12
_DEFAULT_FOURCHAN_MAX_AGE_HOURS: int = 6
_DEFAULT_LAINCHAN_MAX_AGE_HOURS: int = 24
_DEFAULT_TWITTER_MAX_AGE_HOURS: int = 4

# Per-call result caps. Goal: pull enough for the LLM agents to triage,
# not so much that we blow the extractor budget. Reddit's 100 is a hot
# default; 4chan / lainchan threads are larger so we cap tighter.
_DEFAULT_REDDIT_LIMIT: int = 100
_DEFAULT_HN_LIMIT: int = 50
_DEFAULT_FOURCHAN_LIMIT: int = 30
_DEFAULT_LAINCHAN_LIMIT: int = 30
_DEFAULT_TWITTER_LIMIT: int = 50


# --------------------------------------------------------------------- #
# Reddit (PRAW)
# --------------------------------------------------------------------- #


@dataclass
class RedditFetcher:
    """Pull top + new posts from configured subreddits. Uses PRAW via
    OAuth (requires ``REDDIT_CLIENT_ID``, ``REDDIT_CLIENT_SECRET``, and
    a ``REDDIT_USER_AGENT`` string identifying this script).

    Feed config schema (``configs/paper_streams.yaml``):
      adapter: reddit
      query_url: <subreddit list as comma-separated; or 'r/all'>
      source_label: <human label>
      max_age_hours: <int, optional>
      sort: hot | new | top (default 'hot')

    Recent-only filter: posts older than ``max_age_hours`` are dropped.
    """

    http_get: HttpGet = field(default=_default_http_get)
    max_age_hours: int = _DEFAULT_REDDIT_MAX_AGE_HOURS

    def fetch(self, feed: FeedConfig) -> list[Paper]:
        try:
            import praw
        except ImportError:
            logger.warning(
                "Reddit fetcher: praw not installed. "
                "Install with `pip install praw`.",
            )
            return []

        client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        user_agent = os.environ.get(
            "REDDIT_USER_AGENT",
            "curLit-research/1.0 by /u/anon",
        )
        if not client_id or not client_secret:
            logger.warning(
                "Reddit fetcher: REDDIT_CLIENT_ID/SECRET not set — "
                "skipping %s", feed.name,
            )
            return []

        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            reddit.read_only = True
        except Exception:
            logger.exception(
                "Reddit fetcher: client init failed for %s", feed.name,
            )
            return []

        cutoff = datetime.now(UTC) - timedelta(hours=self.max_age_hours)
        subreddits = self._parse_subs(feed.query_url)

        out: list[Paper] = []
        limit = _DEFAULT_REDDIT_LIMIT
        for sub in subreddits:
            try:
                for post in reddit.subreddit(sub).hot(limit=limit):
                    created = datetime.fromtimestamp(post.created_utc, UTC)
                    if created < cutoff:
                        continue
                    body = (
                        getattr(post, "selftext", "") or ""
                    )[:4000]
                    out.append(Paper(
                        title=str(post.title or "")[:300],
                        authors=(f"u/{post.author}",) if post.author else (),
                        year=created.year,
                        url=f"https://reddit.com{post.permalink}",
                        doi="",
                        abstract=body,
                        source_label=feed.source_label,
                    ))
            except Exception:
                logger.exception(
                    "Reddit fetcher: subreddit %s failed", sub,
                )
        return out

    @staticmethod
    def _parse_subs(query_url: str) -> list[str]:
        """``query_url`` can be 'wallstreetbets,forex' or
        'r/wallstreetbets,r/forex'. Returns clean sub names."""
        raw = query_url.strip()
        parts = re.split(r"[,\s]+", raw)
        return [p.removeprefix("r/").strip("/").strip()
                for p in parts if p.strip()]


# --------------------------------------------------------------------- #
# Hacker News (Algolia API, free)
# --------------------------------------------------------------------- #


@dataclass
class HackerNewsFetcher:
    """Pull recent HN front-page stories via the Algolia search API.

    Feed config schema:
      adapter: hackernews
      query_url: <Algolia search params or 'front'> — 'front' = front page
      source_label: <human label>
      max_age_hours: <int, optional>
    """

    http_get: HttpGet = field(default=_default_http_get)
    max_age_hours: int = _DEFAULT_HN_MAX_AGE_HOURS

    def fetch(self, feed: FeedConfig) -> list[Paper]:
        params_str = feed.query_url.strip()
        if params_str == "front":
            url = "https://hn.algolia.com/api/v1/search"
            params_str = (
                f"tags=front_page&hitsPerPage={_DEFAULT_HN_LIMIT}"
            )
        else:
            url = "https://hn.algolia.com/api/v1/search"

        full_url = f"{url}?{params_str}"
        try:
            body = self.http_get(full_url)
        except Exception:
            logger.exception(
                "HackerNews fetcher: fetch failed for %s", feed.name,
            )
            return []

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("HackerNews fetcher: invalid JSON for %s", feed.name)
            return []

        hits = data.get("hits", [])
        cutoff = datetime.now(UTC) - timedelta(hours=self.max_age_hours)
        out: list[Paper] = []
        for h in hits:
            ts = h.get("created_at_i")
            if ts is None:
                continue
            created = datetime.fromtimestamp(int(ts), UTC)
            if created < cutoff:
                continue
            title = h.get("title") or h.get("story_title") or ""
            url_field = h.get("url") or (
                f"https://news.ycombinator.com/item?id={h.get('objectID')}"
            )
            body_text = (h.get("story_text") or "")[:4000]
            author = h.get("author") or ""
            out.append(Paper(
                title=str(title)[:300],
                authors=(author,) if author else (),
                year=created.year,
                url=str(url_field),
                doi="",
                abstract=body_text,
                source_label=feed.source_label,
            ))
        return out


# --------------------------------------------------------------------- #
# 4chan (public read API)
# --------------------------------------------------------------------- #
# 4chan has a public read-only API: https://a.4cdn.org/<board>/catalog.json
# Returns the full board catalog (threads + first ~5 replies). No auth.
# Signal-to-noise on /biz/ and /pol/ is intentionally low; we ingest
# anyway because the operator asked. MetaLearner (CL-fsj) downweights
# the source if it doesn't produce winners.


@dataclass
class FourchanFetcher:
    """Pull threads from configured 4chan boards.

    Feed config schema:
      adapter: fourchan
      query_url: <board list — e.g. 'biz' or 'biz,pol'>
      source_label: <human label — convention: include '(low-signal)'>
      max_age_hours: <int, optional>
    """

    http_get: HttpGet = field(default=_default_http_get)
    max_age_hours: int = _DEFAULT_FOURCHAN_MAX_AGE_HOURS

    def fetch(self, feed: FeedConfig) -> list[Paper]:
        cutoff = datetime.now(UTC) - timedelta(hours=self.max_age_hours)
        boards = [
            b.strip().strip("/") for b in feed.query_url.split(",")
            if b.strip()
        ]
        out: list[Paper] = []
        for board in boards:
            url = f"https://a.4cdn.org/{board}/catalog.json"
            try:
                body = self.http_get(url)
            except Exception:
                logger.warning(
                    "4chan fetcher: catalog fetch failed for %s", board,
                    exc_info=True,
                )
                continue
            try:
                pages = json.loads(body)
            except json.JSONDecodeError:
                logger.warning("4chan fetcher: invalid JSON for %s", board)
                continue

            for page in pages:
                for thread in page.get("threads", [])[:_DEFAULT_FOURCHAN_LIMIT]:
                    ts = thread.get("time")
                    if ts is None:
                        continue
                    created = datetime.fromtimestamp(int(ts), UTC)
                    if created < cutoff:
                        continue
                    title = (
                        thread.get("sub")
                        or _strip_html(thread.get("com", ""))[:120]
                    )
                    body_text = _strip_html(thread.get("com", ""))[:4000]
                    thread_no = thread.get("no")
                    out.append(Paper(
                        title=str(title or f"thread {thread_no}")[:300],
                        authors=(thread.get("name", "Anonymous"),),
                        year=created.year,
                        url=(
                            f"https://boards.4chan.org/{board}/thread/"
                            f"{thread_no}"
                        ),
                        doi="",
                        abstract=body_text,
                        source_label=feed.source_label,
                    ))
        return out


# --------------------------------------------------------------------- #
# lainchan
# --------------------------------------------------------------------- #
# lainchan exposes a /catalog.json per board similar to 4chan.


@dataclass
class LainchanFetcher:
    """Pull threads from configured lainchan boards.

    Feed config schema:
      adapter: lainchan
      query_url: <board list — e.g. 'tech' or 'tech,sec'>
      source_label: <human label — convention: include '(low-signal)'>
      max_age_hours: <int, optional>
    """

    http_get: HttpGet = field(default=_default_http_get)
    max_age_hours: int = _DEFAULT_LAINCHAN_MAX_AGE_HOURS

    def fetch(self, feed: FeedConfig) -> list[Paper]:
        cutoff = datetime.now(UTC) - timedelta(hours=self.max_age_hours)
        boards = [
            b.strip().strip("/") for b in feed.query_url.split(",")
            if b.strip()
        ]
        out: list[Paper] = []
        for board in boards:
            url = f"https://lainchan.org/{board}/catalog.json"
            try:
                body = self.http_get(url)
            except Exception:
                logger.warning(
                    "lainchan fetcher: catalog fetch failed for %s", board,
                    exc_info=True,
                )
                continue
            try:
                pages = json.loads(body)
            except json.JSONDecodeError:
                logger.warning(
                    "lainchan fetcher: invalid JSON for %s", board,
                )
                continue
            for page in pages:
                for thread in page.get("threads", [])[:_DEFAULT_LAINCHAN_LIMIT]:
                    ts = thread.get("time") or thread.get("last_modified")
                    if ts is None:
                        continue
                    created = datetime.fromtimestamp(int(ts), UTC)
                    if created < cutoff:
                        continue
                    title = (
                        thread.get("sub")
                        or _strip_html(thread.get("com", ""))[:120]
                    )
                    body_text = _strip_html(thread.get("com", ""))[:4000]
                    thread_no = thread.get("no")
                    out.append(Paper(
                        title=str(title or f"thread {thread_no}")[:300],
                        authors=(thread.get("name", "Anonymous"),),
                        year=created.year,
                        url=(
                            f"https://lainchan.org/{board}/res/"
                            f"{thread_no}.html"
                        ),
                        doi="",
                        abstract=body_text,
                        source_label=feed.source_label,
                    ))
        return out


# --------------------------------------------------------------------- #
# Twitter / X (official v2 search/recent)
# --------------------------------------------------------------------- #
# Requires a paid API tier (basic = $200/mo as of 2026). Without
# TWITTER_BEARER_TOKEN we log a warning and return []. The buzzword
# search uses the v2 ``GET /2/tweets/search/recent`` endpoint — pulls
# tweets from the last 7 days matching a query.


@dataclass
class TwitterFetcher:
    """Pull buzzword-matched tweets from the last few hours.

    Feed config schema:
      adapter: twitter
      query_url: <Twitter v2 query — e.g. 'EURUSD OR "fed cut" lang:en'>
      source_label: <human label>
      max_age_hours: <int, optional>

    Honest scope: this hits the official API. ``TWITTER_BEARER_TOKEN``
    must be set; otherwise the fetcher logs WARNING and returns [].
    Scraping alternatives (nitter, snscrape) are ToS-hostile and break
    every couple weeks; we don't ship them.
    """

    http_get: HttpGet = field(default=_default_http_get)
    max_age_hours: int = _DEFAULT_TWITTER_MAX_AGE_HOURS

    def fetch(self, feed: FeedConfig) -> list[Paper]:
        token = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()
        if not token:
            logger.warning(
                "Twitter fetcher: TWITTER_BEARER_TOKEN not set — "
                "skipping %s. Provision via developer.twitter.com.",
                feed.name,
            )
            return []

        # v2 search/recent has a 'start_time' RFC3339 param. Cap to
        # max_age_hours so we don't pull stale buzzword hits.
        start_time = (
            datetime.now(UTC) - timedelta(hours=self.max_age_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            resp = httpx.get(
                "https://api.twitter.com/2/tweets/search/recent",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "query": feed.query_url,
                    "start_time": start_time,
                    "max_results": _DEFAULT_TWITTER_LIMIT,
                    "tweet.fields": "created_at,author_id,public_metrics",
                    "expansions": "author_id",
                    "user.fields": "username",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
        except Exception:
            logger.exception(
                "Twitter fetcher: API call failed for %s", feed.name,
            )
            return []

        data = resp.json()
        users = {
            u["id"]: u["username"]
            for u in data.get("includes", {}).get("users", [])
        }
        tweets = data.get("data", [])

        out: list[Paper] = []
        for t in tweets:
            tid = str(t.get("id", ""))
            text = str(t.get("text", ""))[:4000]
            created_at = t.get("created_at", "")
            try:
                created = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00"),
                )
            except Exception:
                created = datetime.now(UTC)
            author_id = t.get("author_id", "")
            handle = users.get(author_id, "")
            out.append(Paper(
                title=text[:200],
                authors=(f"@{handle}",) if handle else (),
                year=created.year,
                url=(
                    f"https://twitter.com/{handle or 'i'}/status/{tid}"
                ),
                doi="",
                abstract=text,
                source_label=feed.source_label,
            ))
        return out


# --------------------------------------------------------------------- #
# Registry — extend the existing ingest._FETCHER_REGISTRY
# --------------------------------------------------------------------- #


def register_social_fetchers() -> None:
    """Side-effect: register the five new adapters in the existing
    ingest fetcher registry. Idempotent — re-running is a no-op.
    Called once at module import via the ``src.research`` __init__.
    """
    from src.research.ingest import _FETCHER_REGISTRY

    additions: dict[str, Callable[[HttpGet], Any]] = {
        "reddit":     lambda http_get: RedditFetcher(http_get=http_get),
        "hackernews": lambda http_get: HackerNewsFetcher(http_get=http_get),
        "fourchan":   lambda http_get: FourchanFetcher(http_get=http_get),
        "lainchan":   lambda http_get: LainchanFetcher(http_get=http_get),
        "twitter":    lambda http_get: TwitterFetcher(http_get=http_get),
    }
    for name, factory in additions.items():
        _FETCHER_REGISTRY.setdefault(name, factory)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


_HTML_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE: re.Pattern[str] = re.compile(r"&\w+;")


def _strip_html(s: str) -> str:
    """Strip HTML tags + collapse whitespace. 4chan posts use
    <br> + <a class="quotelink"> liberally."""
    if not s:
        return ""
    cleaned = _HTML_TAG_RE.sub(" ", s)
    cleaned = _HTML_ENTITY_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


# Register at module import so callers don't have to remember.
register_social_fetchers()
