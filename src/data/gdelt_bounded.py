"""Bounded, resumable GDELT slices for the single owned event pipeline.

No background thread/second assessment writer. Network calls run outside DB
transactions. A cursor advances only AFTER article persistence succeeds. The
source-wide lease also prevents independently invoked slices overlapping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd
from sqlalchemy import text

from src.data.gdelt import GDELT_DOC_URL, build_theme_query

if TYPE_CHECKING:
    from src.data.gdelt import GdeltIngester

logger = logging.getLogger(__name__)
# Existing provider timeout/cool-off and cadence are retained, not tightened.
REQUEST_SECONDS = 30.0
MIN_BACKOFF_SECONDS = 20.0
MAX_BACKOFF_SECONDS = 3600.0
# Allow final short DB writes/unlock after the network scheduling budget ends.
LEASE_GRACE_SECONDS = 30.0


@dataclass(frozen=True)
class GdeltRead:
    status: str
    articles: list[dict[str, Any]] = field(default_factory=list)
    retry_after: float = 0.0


async def fetch_once(
    query: str, start: datetime, end: datetime, timeout: float, max_records: int
) -> GdeltRead:
    """One HTTP request with an outer wall-clock deadline; no hidden sleeps."""
    try:
        async with asyncio.timeout(timeout), httpx.AsyncClient(follow_redirects=False) as client:
            response = await client.get(
                GDELT_DOC_URL,
                params={
                    "query": query,
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": str(max_records),
                    "sort": "DateDesc",
                    "startdatetime": start.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
                    "enddatetime": end.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
                },
                timeout=timeout,
            )
        if response.status_code == 429:
            raw = response.headers.get("Retry-After", "")
            retry = MIN_BACKOFF_SECONDS
            try:
                retry = float(raw)
            except ValueError:
                with suppress(ValueError, TypeError, OverflowError):
                    retry = parsedate_to_datetime(raw).timestamp() - time.time()
            return GdeltRead(
                "rate_limited",
                retry_after=max(
                    MIN_BACKOFF_SECONDS, retry if math.isfinite(retry) else MIN_BACKOFF_SECONDS
                ),
            )
        if response.status_code != 200:
            return GdeltRead(f"http_{response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            return GdeltRead("invalid_json")
        articles = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(articles, list) or any(
            not isinstance(a, dict)
            or any(not isinstance(a.get(k), str) or not a[k] for k in ("url", "title", "seendate"))
            for a in articles
        ):
            return GdeltRead("invalid_articles")
        try:
            for article in articles:
                datetime.strptime(article["seendate"], "%Y%m%dT%H%M%SZ")
        except ValueError:
            return GdeltRead("invalid_article_date")
        return GdeltRead("success", articles)
    except (TimeoutError, httpx.HTTPError) as exc:
        return GdeltRead(type(exc).__name__)


def run_slice(
    ingester: GdeltIngester,
    start: datetime,
    end: datetime,
    *,
    budget_sec: float = 60.0,
    read: Callable[[str, datetime, datetime, float, int], GdeltRead] | None = None,
    wall_clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Yield to assessment on 429, budget exhaustion, or checkpoint failure.

    The default minute is a scheduling budget inside the existing 15-minute
    loop, not a higher GDELT request rate. DB availability is still required for
    assessment itself; no network call starts without a durable pending window.
    """
    if not math.isfinite(budget_sec) or budget_sec <= 0 or start >= end:
        raise ValueError("invalid ingestion bounds")
    deadline = monotonic() + budget_sec
    result = {"rows": 0, "attempted": 0, "completed": 0, "deferred": 0}
    engine = ingester.engine
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO gdelt_ingest_cursors(theme,payload) VALUES ('_source','{}') ON CONFLICT(theme) DO NOTHING"
            )
        )
        states = {
            r[0]: json.loads(r[1])
            for r in conn.execute(text("SELECT theme,payload FROM gdelt_ingest_cursors"))
        }
    source = states.pop("_source")
    if source.get("lease_until", 0) > wall_clock() or source.get("next_eligible", 0) > wall_clock():
        result["deferred"] = len(ingester.playbooks)
        logger.info("GDELT slice deferred: source lease or retry deadline active")
        return result
    # Compare exact stored JSON text, including the initial '{}' representation.
    with engine.connect() as conn:
        original = conn.execute(
            text("SELECT payload FROM gdelt_ingest_cursors WHERE theme='_source'")
        ).scalar_one()
    source = json.loads(original)
    if source.get("lease_until", 0) > wall_clock() or source.get("next_eligible", 0) > wall_clock():
        result["deferred"] = len(ingester.playbooks)
        return result
    owner = uuid.uuid4().hex
    source.update(lease_owner=owner, lease_until=wall_clock() + budget_sec + LEASE_GRACE_SECONDS)
    with engine.begin() as conn:
        acquired = (
            conn.execute(
                text(
                    "UPDATE gdelt_ingest_cursors SET payload=:new WHERE theme='_source' AND payload=:old"
                ),
                {"new": json.dumps(source, sort_keys=True), "old": original},
            ).rowcount
            == 1
        )
    if not acquired:
        result["deferred"] = len(ingester.playbooks)
        return result
    with engine.connect() as conn:
        states = {
            r[0]: json.loads(r[1])
            for r in conn.execute(
                text("SELECT theme,payload FROM gdelt_ingest_cursors WHERE theme != '_source'")
            )
        }

    def save(theme: str, state: dict[str, Any]) -> None:
        with engine.begin() as conn:
            current = conn.execute(
                text("SELECT payload FROM gdelt_ingest_cursors WHERE theme='_source'")
            ).scalar_one()
            lease = json.loads(current)
            if lease.get("lease_owner") != owner or lease.get("lease_until", 0) <= wall_clock():
                raise RuntimeError("ingestion_lease_lost")
            # Lock/compare source ownership in the same short transaction as
            # the cursor write. A stale worker cannot overwrite its successor.
            if (
                conn.execute(
                    text(
                        "UPDATE gdelt_ingest_cursors SET payload=payload WHERE theme='_source' AND payload=:old"
                    ),
                    {"old": current},
                ).rowcount
                != 1
            ):
                raise RuntimeError("ingestion_lease_changed")
            conn.execute(
                text(
                    "INSERT INTO gdelt_ingest_cursors(theme,payload) VALUES (:theme,:payload) ON CONFLICT(theme) DO UPDATE SET payload=excluded.payload"
                ),
                {"theme": theme, "payload": json.dumps(state, sort_keys=True, allow_nan=False)},
            )

    def live_read(
        query: str, lower: datetime, upper: datetime, timeout: float, limit: int
    ) -> GdeltRead:
        return asyncio.run(fetch_once(query, lower, upper, timeout, limit))

    request = read or live_read
    try:
        themes = sorted(
            ingester.playbooks, key=lambda t: (states.get(t, {}).get("last_attempt", 0), t)
        )
        for theme in themes:
            state = states.get(theme, {})
            if state.get("next_eligible", 0) > wall_clock():
                continue
            wait = max(0.0, source.get("next_eligible", 0) - wall_clock())
            if wait >= deadline - monotonic():
                break
            if wait:
                sleep(wait)
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            lower = datetime.fromisoformat(
                state.get("window_start") or state.get("covered_until") or start.isoformat()
            )
            upper = datetime.fromisoformat(state.get("window_end") or end.isoformat())
            if lower >= upper:
                continue
            state.update(
                schema_version=1,
                window_start=lower.isoformat(),
                window_end=upper.isoformat(),
                status="pending",
                last_attempt=wall_clock(),
            )
            save(theme, state)
            source["next_eligible"] = wall_clock() + max(ingester.pause_sec, 0.0)
            save("_source", source)
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            logger.info(
                "GDELT bounded request: theme=%s window=%s..%s remaining=%.1fs",
                theme,
                lower,
                upper,
                remaining,
            )
            response = request(
                build_theme_query(ingester.playbooks[theme]),
                lower,
                upper,
                min(REQUEST_SECONDS, remaining),
                ingester.provider.max_records,
            )
            result["attempted"] += 1
            if response.status == "success":
                if response.articles:
                    raw = pd.DataFrame([{**a, "theme": theme} for a in response.articles])
                    result["rows"] += ingester.upsert(ingester.validate(ingester.transform(raw)))
                if len(response.articles) < ingester.provider.max_records:
                    # Upsert has committed (or verified a genuinely empty batch).
                    state.update(
                        covered_until=upper.isoformat(),
                        window_start=None,
                        window_end=None,
                        status="complete",
                        failures=0,
                        next_eligible=0,
                    )
                    result["completed"] += 1
                else:
                    # Retry a smaller window rather than declaring capped
                    # results complete. After that prefix commits, covered_until
                    # resumes the remaining tail up to the next cycle's cutoff.
                    seconds = int((upper - lower).total_seconds())
                    if seconds >= 2:  # GDELT request timestamps have second precision.
                        state.update(
                            window_end=(lower + timedelta(seconds=seconds // 2)).isoformat(),
                            next_eligible=source["next_eligible"],
                        )
                    else:
                        state["next_eligible"] = wall_clock() + MAX_BACKOFF_SECONDS
                    state["status"] = "result_cap_reached"
            else:
                failures = int(state.get("failures", 0)) + 1
                # Bound exponent before evaluation, not only its result.
                backoff = min(MAX_BACKOFF_SECONDS, MIN_BACKOFF_SECONDS * 2 ** min(failures - 1, 8))
                retry = max(backoff, response.retry_after)
                state.update(
                    status=response.status, failures=failures, next_eligible=wall_clock() + retry
                )
                if response.status == "rate_limited":
                    source["next_eligible"] = state["next_eligible"]
                    save("_source", source)
            save(theme, state)
            logger.info("GDELT checkpoint: theme=%s status=%s", theme, state["status"])
            if response.status == "rate_limited":
                break  # Do not sleep out a source-wide throttle ahead of assessment.
    finally:
        source.update(lease_until=0, lease_owner=None)
        # Expired owners must never overwrite a successor's source state.
        with engine.begin() as conn:
            current = conn.execute(
                text("SELECT payload FROM gdelt_ingest_cursors WHERE theme='_source'")
            ).scalar_one()
            if json.loads(current).get("lease_owner") == owner:
                conn.execute(
                    text(
                        "UPDATE gdelt_ingest_cursors SET payload=:new WHERE theme='_source' AND payload=:old"
                    ),
                    {"new": json.dumps(source, sort_keys=True), "old": current},
                )
    result["deferred"] = len(ingester.playbooks) - result["completed"]
    logger.info("GDELT bounded slice: %s", result)
    return result
