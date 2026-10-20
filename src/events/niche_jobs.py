"""Durable bounded niche research jobs, not order jobs (CL-ut1d).

Only recent, still-ASSESSED inputs can publish. Database compare-and-swap leases
fence old workers; no DB transaction is held during model/retrieval calls.
Audit persistence precedes atomic event + ledger + completion. A retry never
resets the event or promotes an old assessment's model-written status fields.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import text

from src.events.idea_ledger import persist_ideas
from src.events.prices import parse_ts
from src.events.research_status import ResearchStatus, research_key

logger = logging.getLogger(__name__)
# Version pins provider, prompt, retrieval and review policy in event_pipeline.
POLICY_VERSION = "claude-subscription-refinement-v2"
# Research on yesterday's event is not automatically a fresh trading thesis.
MAX_EVENT_AGE = timedelta(hours=24)
# Bounded per-event quota: initial attempt, 15-minute retry, then one-hour retry.
MAX_ATTEMPTS = 3
RETRY_DELAYS = (15 * 60, 60 * 60)
# Three bounded CLI calls plus source retrieval; stale completion is fenced.
LEASE_SECONDS = 60 * 60


def _object(value: Any) -> dict[str, Any]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def snapshot(event: Any) -> dict[str, Any]:
    """Only decision input; this worker's prior audit annotation is not input."""
    assessment = dict(_object(event["assessment"]))
    assessment.pop("niche_research", None)
    return {
        "id": int(event["id"]),
        "headline": event["headline"],
        "theme": event["theme"],
        "assessment": assessment,
    }


def _current_input_matches(event: Any, expected: dict[str, Any], minimum: int) -> bool:
    try:
        return (
            event is not None
            and event["status"] == "ASSESSED"
            and snapshot(event) == expected
            and int(_object(event["assessment"]).get("urgency", 0)) >= minimum
        )
    except (ValueError, TypeError, KeyError):
        logger.warning(
            "niche queue: changed/malformed current event; refusing research publication"
        )
        return False


def retryable(report: dict[str, Any]) -> bool:
    discovery = report.get("discovery") or {}
    # Completed empty/unsupported research and malformed output are terminal.
    # Transport/auth/quota failures (including partial refinement) may recover.
    if discovery.get("status") in {"unavailable", "partial", "budget_exhausted"}:
        return True
    return any(
        (candidate.get("research") or {}).get("review_status") == "review_unavailable"
        for candidate in report.get("candidates", [])
        if isinstance(candidate, dict)
    )


@dataclass(frozen=True)
class NicheJob:
    job_id: str
    event_id: int
    input_snapshot: dict[str, Any]
    attempts: int
    invocation_id: str


class NicheJobs:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.min_urgency = 7  # Same baseline as niche_agent.DEFAULT_MIN_URGENCY.

    def seed(self, min_urgency: int, now: datetime) -> int:
        """Recover the assessment→enqueue crash gap from the durable event queue.

        No historical bulk replay: only the explicit 24h research window. Existing
        jobs survive a restart; a new input version gets a different identity.
        """
        cutoff = now - MAX_EVENT_AGE
        self.min_urgency = min_urgency
        added = 0
        with self.engine.begin() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT id, headline, theme, assessment, seen_at FROM geo_events "
                        "WHERE status = 'ASSESSED' AND seen_at >= :cutoff AND seen_at <= :now "
                        "ORDER BY seen_at, id"
                    ),
                    {"cutoff": cutoff, "now": now},
                )
                .mappings()
                .all()
            )
            for row in rows:
                try:
                    assessment = _object(row["assessment"])
                    if int(assessment.get("urgency", 0)) < min_urgency:
                        continue
                    prior = assessment.get("niche_research")
                    if isinstance(prior, dict) and not retryable(prior):
                        continue
                    seen = parse_ts(row["seen_at"])
                    if seen is None or seen > now or seen < cutoff:
                        continue
                    data = snapshot(row)
                    serialized = json.dumps(data, sort_keys=True, allow_nan=False)
                except (ValueError, TypeError):
                    logger.warning("niche queue: invalid event input id=%s", row["id"])
                    continue
                job_id = hashlib.sha256(f"{POLICY_VERSION}:{serialized}".encode()).hexdigest()
                logger.info("niche queue: enqueue event=%s job=%s", row["id"], job_id)
                result = conn.execute(
                    text(
                        "INSERT INTO niche_jobs "
                        "(job_id,event_id,policy_version,input_snapshot,state,next_attempt,"
                        "expires_at,updated_at) VALUES "
                        "(:job,:event,:policy,:snapshot,'pending',:now,:expires,:now) "
                        "ON CONFLICT (job_id) DO NOTHING"
                    ),
                    {
                        "job": job_id,
                        "event": row["id"],
                        "policy": POLICY_VERSION,
                        "snapshot": serialized,
                        "now": now.timestamp(),
                        "expires": (seen + MAX_EVENT_AGE).timestamp(),
                    },
                )
                added += max(0, result.rowcount)
        return added

    def claim(self, now: datetime, limit: int) -> list[NicheJob]:
        if limit < 1:
            raise ValueError("positive claim limit required")
        tick = now.timestamp()
        jobs = []
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE niche_jobs SET state='expired', reason='event_age_limit', updated_at=:now "
                    "WHERE state IN ('pending','retry','running') AND expires_at <= :now"
                ),
                {"now": tick},
            )
            conn.execute(
                text(
                    "UPDATE niche_jobs SET state='failed', reason='attempt_limit', updated_at=:now "
                    "WHERE state IN ('retry','running') AND attempts >= :max "
                    "AND (lease_until IS NULL OR lease_until <= :now)"
                ),
                {"now": tick, "max": MAX_ATTEMPTS},
            )
            rows = (
                conn.execute(
                    text(
                        "SELECT * FROM niche_jobs WHERE policy_version=:policy AND attempts < :max "
                        "AND expires_at > :now AND "
                        "((state IN ('pending','retry') AND next_attempt <= :now) "
                        "OR (state='running' AND lease_until <= :now)) "
                        "ORDER BY next_attempt, job_id LIMIT :limit"
                    ),
                    {"policy": POLICY_VERSION, "now": tick, "max": MAX_ATTEMPTS, "limit": limit},
                )
                .mappings()
                .all()
            )
            for row in rows:
                event = (
                    conn.execute(
                        text(
                            "SELECT id,headline,theme,assessment,status FROM geo_events WHERE id=:id"
                        ),
                        {"id": row["event_id"]},
                    )
                    .mappings()
                    .first()
                )
                if not _current_input_matches(
                    event,
                    _object(row["input_snapshot"]),
                    self.min_urgency,
                ):
                    conn.execute(
                        text(
                            "UPDATE niche_jobs SET state='superseded', reason='event_or_input_changed',"
                            "updated_at=:now WHERE job_id=:job AND attempts=:attempts "
                            "AND (lease_until IS NULL OR lease_until <= :now)"
                        ),
                        {"now": tick, "job": row["job_id"], "attempts": row["attempts"]},
                    )
                    continue
                invocation = str(uuid4())
                won = conn.execute(
                    text(
                        "UPDATE niche_jobs SET state='running', attempts=attempts+1,"
                        "invocation_id=:invocation,lease_until=:lease,updated_at=:now "
                        "WHERE job_id=:job AND attempts=:attempts AND state=:state "
                        "AND (lease_until IS NULL OR lease_until <= :now)"
                    ),
                    {
                        "invocation": invocation,
                        "lease": tick + LEASE_SECONDS,
                        "now": tick,
                        "job": row["job_id"],
                        "attempts": row["attempts"],
                        "state": row["state"],
                    },
                )
                if won.rowcount == 1:
                    jobs.append(
                        NicheJob(
                            row["job_id"],
                            row["event_id"],
                            _object(row["input_snapshot"]),
                            row["attempts"] + 1,
                            invocation,
                        )
                    )
                    logger.info(
                        "niche queue: claimed event=%s attempt=%s invocation=%s",
                        row["event_id"],
                        row["attempts"] + 1,
                        invocation,
                    )
        return jobs

    def finish(
        self,
        job: NicheJob,
        enriched: dict[str, Any],
        report: dict[str, Any],
        statuses: dict[tuple[str, str], ResearchStatus],
        now: datetime,
    ) -> bool:
        """Return True only if current input, audit and ledger committed together.

        Caller committed audit independently first. Conditional job UPDATE locks
        this attempt, and event CAS fences assessment/status changes. Exceptions
        roll everything here back, retaining the lease for bounded crash recovery.
        """
        if retryable(report) and statuses:
            raise ValueError("incomplete research cannot publish a partial subset")
        tick = now.timestamp()
        with self.engine.begin() as conn:
            locked = conn.execute(
                text(
                    "UPDATE niche_jobs SET updated_at=:now WHERE job_id=:job AND state='running' "
                    "AND invocation_id=:invocation AND lease_until > :now AND expires_at > :now"
                ),
                {"now": tick, "job": job.job_id, "invocation": job.invocation_id},
            )
            if locked.rowcount != 1:
                logger.warning("niche queue: stale completion fenced job=%s", job.job_id)
                return False
            audit = (
                conn.execute(
                    text(
                        "SELECT event_id,input_snapshot,report FROM niche_research_audit "
                        "WHERE invocation_id=:invocation"
                    ),
                    {"invocation": job.invocation_id},
                )
                .mappings()
                .one()
            )
            if (
                audit["event_id"] != job.event_id
                or _object(audit["input_snapshot"]) != job.input_snapshot
                or _object(audit["report"]) != report
            ):
                raise ValueError("niche completion does not match committed audit")
            row = (
                conn.execute(
                    text("SELECT id,headline,theme,assessment,status FROM geo_events WHERE id=:id"),
                    {"id": job.event_id},
                )
                .mappings()
                .first()
            )
            valid = _current_input_matches(row, job.input_snapshot, self.min_urgency)
            if not valid:
                self._state(conn, job, "superseded", "event_or_input_changed", tick, tick)
                return False
            assert row is not None
            # Production uses JSONB; SQLite fixtures compare canonical stored text.
            original = row["assessment"]
            original_json = json.dumps(original) if isinstance(original, dict) else original
            predicate = (
                "CAST(:original AS jsonb)" if conn.dialect.name == "postgresql" else ":original"
            )
            written = conn.execute(
                text(
                    "UPDATE geo_events SET assessment=:assessment WHERE id=:id "
                    "AND status='ASSESSED' AND assessment=" + predicate  # nosec B608
                ),
                {"assessment": json.dumps(enriched), "id": job.event_id, "original": original_json},
            )
            if written.rowcount != 1:
                self._state(conn, job, "superseded", "event_changed_during_merge", tick, tick)
                return False
            # Only fresh niche additions belong to this transaction. Ordinary
            # impact ideas still pass through the normal price enrichment path.
            niche_only = {
                "trade_ideas": [
                    idea
                    for idea in enriched.get("trade_ideas", [])
                    if research_key(idea.get("ticker"), idea.get("action")) in statuses
                ]
            }
            persist_ideas(
                self.engine,
                job.event_id,
                niche_only,
                now=now,
                research_status=statuses,
                connection=conn,
            )
            retry = retryable(report)
            state = (
                "retry"
                if retry and job.attempts < MAX_ATTEMPTS
                else "completed"
                if not retry and report["discovery"]["status"] in {"completed", "abstained"}
                else "failed"
            )
            reason = str(report["discovery"].get("reason") or report["discovery"]["status"])
            delay = RETRY_DELAYS[min(job.attempts - 1, len(RETRY_DELAYS) - 1)]
            self._state(conn, job, state, reason, tick, tick + delay)
        logger.info(
            "niche queue: finished event=%s state=%s reason=%s", job.event_id, state, reason
        )
        return True

    @staticmethod
    def _state(
        conn: Any, job: NicheJob, state: str, reason: str, now: float, next_attempt: float
    ) -> None:
        conn.execute(
            text(
                "UPDATE niche_jobs SET state=:state, reason=:reason, next_attempt=:next,"
                "updated_at=:now,lease_until=NULL WHERE job_id=:job AND invocation_id=:invocation"
            ),
            {
                "state": state,
                "reason": reason,
                "next": next_attempt,
                "now": now,
                "job": job.job_id,
                "invocation": job.invocation_id,
            },
        )
