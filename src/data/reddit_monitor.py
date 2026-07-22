"""Reddit → geo_events bridge (CL-okww) — tiered subreddit monitoring that
feeds the analysis pipeline, mirroring the X ingest path (CL-esyo).

Polls a tiered watchlist of subreddits (configs/reddit_watchlist.yaml) via
Reddit's public JSON listings — keyless; a descriptive User-Agent and gentle,
tier-gated cadence keep us well inside the unauthenticated limits. Each
relevant post becomes a NEW ``geo_events`` row (source ``reddit:<subreddit>``)
so the EXISTING triage → Event Impact Agent path assesses it. No second
assessment path, no raw forwarding to Telegram.

The QUOTA-CRITICAL relevance gates (LLM assessment is metered), in order of
cheapness:
  1. tier scheduling  — tier 1 polls every cycle, tier 2 every 2nd, tier 3
     every 4th (and slower tiers read ``hot``, already community-filtered);
  2. min_score        — a per-subreddit score floor (0 on ``new`` where fresh
     posts start at 1; meaningful on ``hot``);
  3. theme match      — :func:`src.events.x_ingest.match_theme` against the
     playbooks' watch_terms; no match → never ingested;
  4. per-call cap     — one busy subreddit can't flood the NEW queue.

Transport is injectable (no live HTTP in unit tests); OAuth/PRAW can be
swapped in later behind the same shim if volume ever needs it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from src.events.x_ingest import match_theme

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from src.events.playbooks import Playbook

logger = logging.getLogger(__name__)

#: Public JSON listing endpoint — Reddit 403-blocks this for most clients in
#: 2026; kept only as a last-resort fallback when no OAuth creds are set.
_LISTING_URL = "https://www.reddit.com/r/{name}/{listing}.json"
#: OAuth listing endpoint (the supported path — free tier, ~100 QPM).
_OAUTH_LISTING_URL = "https://oauth.reddit.com/r/{name}/{listing}"
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

#: A descriptive User-Agent is REQUIRED — Reddit throttles/blocks generic ones.
DEFAULT_USER_AGENT = "curlit-event-monitor/1.0 (geopolitical research)"

#: Per-call ingest cap — one busy subreddit can't flood the NEW queue.
DEFAULT_INGEST_CAP = 5

#: Posts fetched per listing call (Reddit caps at 100; 25 is plenty).
DEFAULT_FETCH_LIMIT = 25

VALID_LISTINGS = ("new", "hot", "rising")

#: Injectable transport: (url, params, headers) → parsed JSON dict.
HttpGetJson = Callable[[str, dict[str, str], dict[str, str]], dict[str, Any]]


def _default_http_get_json(
    url: str, params: dict[str, str], headers: dict[str, str],
) -> dict[str, Any]:
    import httpx  # noqa: PLC0415 — lazy
    resp = httpx.get(url, params=params, headers=headers, timeout=15.0,
                     follow_redirects=True)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


#: Injectable token transport: (url, auth(user,pass), data, headers) → JSON.
TokenPostFn = Callable[
    [str, tuple[str, str], dict[str, str], dict[str, str]], dict[str, Any],
]


def _default_token_post(
    url: str, auth: tuple[str, str], data: dict[str, str],
    headers: dict[str, str],
) -> dict[str, Any]:
    import httpx  # noqa: PLC0415 — lazy
    resp = httpx.post(url, auth=auth, data=data, headers=headers, timeout=15.0)
    resp.raise_for_status()
    out: dict[str, Any] = resp.json()
    return out


class RedditOAuth:
    """Application-only OAuth (client_credentials) — the supported read path.

    Reddit 403-blocks unauthenticated JSON scraping (verified live, 2026), so
    listings go through ``oauth.reddit.com`` with a bearer token from the
    operator's Reddit app (https://www.reddit.com/prefs/apps, "script" type).
    Tokens are cached until shortly before expiry. Fail-soft: a token failure
    logs and returns None (the caller skips the cycle rather than crashing)."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str = DEFAULT_USER_AGENT,
        token_post: TokenPostFn | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self._token_post = token_post or _default_token_post
        self._clock = clock or __import__("time").time
        self._token: str | None = None
        self._expires_at: float = 0.0

    def token(self) -> str | None:
        now = self._clock()
        if self._token and now < self._expires_at - 60:
            return self._token
        try:
            payload = self._token_post(
                _TOKEN_URL,
                (self.client_id, self.client_secret),
                {"grant_type": "client_credentials"},
                {"User-Agent": self.user_agent},
            )
            self._token = str(payload["access_token"])
            self._expires_at = now + float(payload.get("expires_in", 3600))
            return self._token
        except Exception as exc:
            logger.warning("reddit oauth: token fetch failed: %s", str(exc)[:160])
            self._token = None
            return None


def oauth_from_env() -> RedditOAuth | None:
    """Build a :class:`RedditOAuth` from REDDIT_CLIENT_ID/SECRET, or None."""
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if not cid or not secret:
        return None
    return RedditOAuth(
        cid, secret,
        user_agent=os.environ.get("REDDIT_USER_AGENT", DEFAULT_USER_AGENT),
    )


@dataclass(frozen=True)
class WatchSubreddit:
    """One watchlist entry (see configs/reddit_watchlist.yaml)."""

    name: str
    tier: int = 1
    listing: str = "new"
    min_score: int = 0
    note: str = ""


@dataclass(frozen=True)
class RedditPost:
    """The fields we keep from a Reddit listing child."""

    id: str
    subreddit: str
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float
    permalink: str

    @property
    def text(self) -> str:
        """Title + a head of the selftext — what theme matching sees."""
        body = " ".join((self.selftext or "").split())
        return f"{self.title} {body[:400]}".strip()

    @property
    def url(self) -> str:
        return f"https://www.reddit.com{self.permalink}"


def load_reddit_watchlist(path: str | Any) -> list[WatchSubreddit]:
    """Parse the watchlist YAML → validated entries. Raises on a corrupt
    file (fail loud); skips individual malformed entries with a warning."""
    import yaml  # noqa: PLC0415

    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or not isinstance(raw.get("subreddits"), list):
        msg = f"reddit watchlist {path}: expected top-level 'subreddits' list"
        raise ValueError(msg)
    out: list[WatchSubreddit] = []
    for entry in raw["subreddits"]:
        if not isinstance(entry, dict) or not str(entry.get("name") or "").strip():
            logger.warning("reddit watchlist: skipping malformed entry %r", entry)
            continue
        listing = str(entry.get("listing") or "new").strip().lower()
        if listing not in VALID_LISTINGS:
            logger.warning(
                "reddit watchlist: %s has invalid listing %r — using 'new'",
                entry.get("name"), listing,
            )
            listing = "new"
        try:
            tier = max(1, min(3, int(entry.get("tier", 1))))
        except (TypeError, ValueError):
            tier = 1
        try:
            min_score = max(0, int(entry.get("min_score", 0)))
        except (TypeError, ValueError):
            min_score = 0
        out.append(WatchSubreddit(
            name=str(entry["name"]).strip().lstrip("r/"),
            tier=tier,
            listing=listing,
            min_score=min_score,
            note=str(entry.get("note") or ""),
        ))
    return out


def due_this_cycle(sub: WatchSubreddit, cycle: int) -> bool:
    """Tier cadence: tier 1 every cycle, tier 2 every 2nd, tier 3 every 4th."""
    interval = {1: 1, 2: 2, 3: 4}.get(sub.tier, 1)
    return cycle % interval == 0


def parse_listing(payload: dict[str, Any], subreddit: str) -> list[RedditPost]:
    """Reddit listing JSON → posts. Malformed children are skipped; never
    raises (the monitor loop must survive a weird payload)."""
    out: list[RedditPost] = []
    children = ((payload.get("data") or {}).get("children")) or []
    if not isinstance(children, list):
        return out
    for child in children:
        data = child.get("data") if isinstance(child, dict) else None
        if not isinstance(data, dict):
            continue
        post_id = str(data.get("id") or "").strip()
        title = str(data.get("title") or "").strip()
        if not post_id or not title:
            continue
        if data.get("stickied") or data.get("pinned"):
            continue  # mod announcements, daily threads
        try:
            score = int(data.get("score") or 0)
            num_comments = int(data.get("num_comments") or 0)
            created = float(data.get("created_utc") or 0.0)
        except (TypeError, ValueError):
            score, num_comments, created = 0, 0, 0.0
        out.append(RedditPost(
            id=post_id,
            subreddit=subreddit,
            title=title,
            selftext=str(data.get("selftext") or ""),
            score=score,
            num_comments=num_comments,
            created_utc=created,
            permalink=str(data.get("permalink") or f"/r/{subreddit}/comments/{post_id}/"),
        ))
    return out


def fetch_posts(
    sub: WatchSubreddit,
    http_get: HttpGetJson | None = None,
    limit: int = DEFAULT_FETCH_LIMIT,
    user_agent: str | None = None,
    oauth: RedditOAuth | None = None,
) -> list[RedditPost]:
    """One listing call for a subreddit → parsed posts.

    With ``oauth`` (the supported path) the call goes to ``oauth.reddit.com``
    with a bearer token; without it, the legacy public JSON URL is tried
    (Reddit 403-blocks that for most clients — expect failures). Fail-soft:
    any fetch/parse error logs and returns [] (the loop moves on)."""
    getter = http_get or _default_http_get_json
    ua = user_agent or os.environ.get("REDDIT_USER_AGENT", DEFAULT_USER_AGENT)
    headers = {"User-Agent": ua}
    if oauth is not None:
        token = oauth.token()
        if token is None:
            return []  # token failure already logged — skip this cycle
        url = _OAUTH_LISTING_URL.format(name=sub.name, listing=sub.listing)
        headers["Authorization"] = f"bearer {token}"
    else:
        url = _LISTING_URL.format(name=sub.name, listing=sub.listing)
    try:
        payload = getter(url, {"limit": str(limit)}, headers)
    except Exception as exc:
        logger.warning("reddit fetch failed for r/%s: %s", sub.name,
                       str(exc)[:160])
        return []
    return parse_listing(payload, sub.name)


# --------------------------------------------------------------------- #
# Ingestion (mirrors x_ingest.ingest_posts)
# --------------------------------------------------------------------- #

_INSERT_SQL = text(
    "INSERT INTO geo_events "
    "(seen_at, source, external_id, headline, url, theme, status, "
    " status_updated_at) "
    "VALUES (:seen_at, :source, :external_id, :headline, :url, :theme, "
    "        'NEW', :now) "
    "ON CONFLICT (external_id) DO NOTHING",
)

_MAX_HEADLINE_LEN = 500


@dataclass(frozen=True)
class RedditIngestResult:
    ingested: int = 0
    skipped_low_score: int = 0
    skipped_no_theme: int = 0
    deduped: int = 0

    def summary_line(self, name: str) -> str:
        return (
            f"reddit-ingest r/{name}: {self.ingested} ingested, "
            f"{self.skipped_low_score} low-score, "
            f"{self.skipped_no_theme} no-theme, {self.deduped} deduped"
        )


def _seen_at(created_utc: float) -> datetime:
    if created_utc and created_utc > 0:
        try:
            return datetime.fromtimestamp(created_utc, tz=UTC)
        except (OverflowError, OSError, ValueError):
            pass
    return datetime.now(UTC)


def ingest_reddit_posts(
    engine: Engine,
    sub: WatchSubreddit,
    posts: list[RedditPost],
    playbooks: dict[str, Playbook],
    cap: int = DEFAULT_INGEST_CAP,
) -> RedditIngestResult:
    """Relevant posts from one subreddit → NEW ``geo_events`` rows.

    Gates in order: min_score, theme match, per-call cap. ``external_id`` is
    ``reddit:<post id>`` (ON CONFLICT DO NOTHING dedups re-seen posts —
    essential here because unlike the X monitor there is no since_id cursor;
    every poll re-reads the listing). Never raises for a DB problem."""
    ingested = 0
    low_score = 0
    no_theme = 0
    rows: list[dict[str, object]] = []
    now = datetime.now(UTC)
    for post in posts:
        if ingested >= cap:
            break
        if post.score < sub.min_score:
            low_score += 1
            continue
        theme = match_theme(post.text, playbooks)
        if theme is None:
            no_theme += 1
            continue
        headline = " ".join(post.title.split())
        if len(headline) > _MAX_HEADLINE_LEN:
            headline = headline[: _MAX_HEADLINE_LEN - 1].rstrip() + "…"
        rows.append({
            "seen_at": _seen_at(post.created_utc),
            "source": f"reddit:{sub.name}",
            "external_id": f"reddit:{post.id}",
            "headline": headline,
            "url": post.url,
            "theme": theme,
            "now": now,
        })
        ingested += 1

    if not rows:
        return RedditIngestResult(
            skipped_low_score=low_score, skipped_no_theme=no_theme,
        )

    try:
        with engine.begin() as conn:
            inserted = 0
            for row in rows:
                result = conn.execute(_INSERT_SQL, row)
                if getattr(result, "rowcount", 0) and result.rowcount > 0:
                    inserted += 1
        return RedditIngestResult(
            ingested=inserted,
            skipped_low_score=low_score,
            skipped_no_theme=no_theme,
            deduped=len(rows) - inserted,
        )
    except Exception as exc:
        logger.warning(
            "reddit-ingest DB write failed for r/%s (%d rows): %s",
            sub.name, len(rows), str(exc)[:200],
        )
        return RedditIngestResult(
            skipped_low_score=low_score, skipped_no_theme=no_theme,
        )
