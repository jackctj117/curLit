"""GDELT Doc 2.0 ingester (CL-6iu7) — free, keyless news firehose for
the current-events pipeline.

Polls https://api.gdeltproject.org/api/v2/doc/doc (mode=ArtList,
format=json) with one query per playbook theme (OR-joined watch terms
from ``configs/event_playbooks.yaml``), dedups on a sha256 hash of the
article URL, and inserts NEW rows into ``geo_events``.

HONEST LATENCY NOTE: GDELT's front-file updates roughly every 15
minutes, and article ingestion adds its own lag — this source is NOT a
low-latency wire. The system's edge is interpretation (the Event Impact
Agent mapping a headline onto pre-researched instrument playbooks), not
speed. Anyone with a Bloomberg terminal saw the headline first; the bet
is that a disciplined, always-on reaction to the *second* 15 minutes is
still worth having.

Rate-limit posture: one request per theme per run (6 themes → 6
requests), an ADAPTIVE inter-request pause targeting a fixed cadence
(GDELT 429s aggressively — observed even at ~12s spacing under load),
one cool-off retry per theme on 429, 30s timeout, and per-theme failure
tolerance (one flaky query never kills the run).

Cadence, not blind sleep (CL-7vn9): the politeness constraint is the
SPACING between request *starts*, not the sleep itself. We aim for one
request every ``pause_sec`` seconds; the wall-clock a fetch already
consumed counts toward that interval, so a 4s fetch under a 6s cadence
sleeps only ~2s, and a fetch slower than the cadence sleeps nothing.
Request timing to GDELT is byte-for-byte unchanged from the old fixed
pause when fetches are instantaneous (unit tests); against the live,
often-slow endpoint it strips the redundant sleep that used to stack
on top of already-slow responses — ~90s/cycle of pure sleep at 16
themes collapses toward the fetch time itself, with no tighter spacing.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.data.base import BaseIngester
from src.events.playbooks import (
    DEFAULT_PLAYBOOKS_PATH,
    Playbook,
    load_playbooks,
)

logger = logging.getLogger(__name__)

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

#: GDELT's datetime parameter format (UTC, no separators).
_GDELT_DT_FMT = "%Y%m%d%H%M%S"
#: GDELT's `seendate` field format in ArtList JSON.
_SEENDATE_FMT = "%Y%m%dT%H%M%SZ"


def build_theme_query(playbook: Playbook) -> str:
    """OR-join a playbook's watch terms into one GDELT query string.

    Multi-word terms are phrase-quoted; the group is parenthesized (GDELT
    requires parens around OR lists) and restricted to English sources so
    the impact agent isn't fed headlines it can't read.
    """
    parts = []
    for term in playbook.watch_terms:
        parts.append(f'"{term}"' if " " in term else term)
    return f"({' OR '.join(parts)}) sourcelang:english"


def url_external_id(url: str) -> str:
    """Stable dedup key: sha256 of the stripped URL."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


class GdeltDocProvider:
    """Thin client for the GDELT Doc 2.0 ArtList endpoint."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_records: int = 50,
        rate_limit_cooloff_sec: float = 20.0,
    ) -> None:
        self.timeout = timeout
        self.max_records = max_records
        self.rate_limit_cooloff_sec = rate_limit_cooloff_sec

    def fetch_articles(
        self, query: str, start: datetime, end: datetime,
    ) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(self.max_records),
            "sort": "DateDesc",
            "startdatetime": start.astimezone(UTC).strftime(_GDELT_DT_FMT),
            "enddatetime": end.astimezone(UTC).strftime(_GDELT_DT_FMT),
        }
        resp = httpx.get(GDELT_DOC_URL, params=params, timeout=self.timeout)
        if resp.status_code == 429 and self.rate_limit_cooloff_sec > 0:
            # GDELT's limiter trips readily; one polite cool-off retry
            # rescues most themes. Still 429 after that → raise, and the
            # ingester skips this theme for the run.
            logger.info(
                "GDELT 429 — cooling off %.0fs before one retry",
                self.rate_limit_cooloff_sec,
            )
            time.sleep(self.rate_limit_cooloff_sec)
            resp = httpx.get(GDELT_DOC_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:
            # GDELT returns plain-text error strings (rate limits, query
            # syntax complaints) with HTTP 200 — treat as an empty batch.
            logger.warning(
                "GDELT non-JSON response (first 200 chars): %r",
                resp.text[:200],
            )
            return []
        articles = payload.get("articles") or []
        if not isinstance(articles, list):
            return []
        return articles


class GdeltIngester(BaseIngester):
    """Themed GDELT poll → NEW rows in ``geo_events``."""

    def __init__(
        self,
        db_url: str,
        playbooks_path: Path | str = DEFAULT_PLAYBOOKS_PATH,
        max_records_per_theme: int = 50,
        pause_sec: float = 6.0,
    ) -> None:
        super().__init__(db_url, "gdelt")
        self.playbooks = load_playbooks(playbooks_path)
        self.provider = GdeltDocProvider(max_records=max_records_per_theme)
        self.pause_sec = pause_sec

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        # Adaptive cadence (CL-7vn9): sleep only enough to keep request
        # STARTS ``pause_sec`` apart, crediting the time the previous
        # fetch already spent. ``last_start`` is the monotonic clock at
        # the previous request's launch; ``None`` before the first.
        last_start: float | None = None
        for theme, playbook in self.playbooks.items():
            if last_start is not None and self.pause_sec > 0:
                # Time already elapsed since the last request began counts
                # toward the cadence — a slow fetch shortens (or zeroes)
                # the wait, a fast one still pauses to stay polite.
                elapsed = time.monotonic() - last_start
                remaining = self.pause_sec - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            query = build_theme_query(playbook)
            last_start = time.monotonic()
            try:
                articles = self.provider.fetch_articles(query, start, end)
            except Exception:
                # One flaky theme query must not kill the whole poll.
                logger.warning(
                    "GDELT fetch failed for theme %s", theme, exc_info=True,
                )
                continue
            for art in articles:
                rows.append({
                    "theme": theme,
                    "url": art.get("url"),
                    "title": art.get("title"),
                    "seendate": art.get("seendate"),
                })
            logger.debug("GDELT theme %s: %d articles", theme, len(articles))
        return pd.DataFrame(rows)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["url"] = df["url"].astype("string").str.strip()
        df["headline"] = df["title"].astype("string").str.strip()
        df = df[df["url"].notna() & (df["url"] != "")]
        df = df[df["headline"].notna() & (df["headline"] != "")]
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "seen_at", "source", "external_id", "headline",
                    "url", "theme", "status", "status_updated_at",
                ],
            )
        now = datetime.now(UTC)
        seen = pd.to_datetime(
            df["seendate"], format=_SEENDATE_FMT, utc=True, errors="coerce",
        )
        df["seen_at"] = seen.fillna(pd.Timestamp(now))
        df["external_id"] = df["url"].map(url_external_id)
        df["source"] = self.source
        df["status"] = "NEW"
        df["status_updated_at"] = pd.Timestamp(now)
        keep = [
            "seen_at", "source", "external_id", "headline",
            "url", "theme", "status", "status_updated_at",
        ]
        return df[keep]

    def _key_columns(self) -> list[str]:
        # URL-hash dedup: the same article matched by two theme queries
        # (or seen on consecutive polls) collapses to one row; the base
        # validate() drops in-batch dupes, _upsert_dataframe anti-joins
        # against rows already in the table.
        return ["external_id"]

    def upsert(self, df: pd.DataFrame) -> int:
        return self._upsert_dataframe(
            df, "geo_events", self.engine, self._key_columns(),
        )
