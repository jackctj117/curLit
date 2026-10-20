"""CL-ut1d: PostgreSQL JSONB, concurrent leases and atomic publication.

Uses the opt-in disposable curlit_test_* fixture, never the operational DB.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from migrations.run import _strip_sql_comments
from sqlalchemy import text

from src.events.niche_audit import persist_niche_audit
from src.events.niche_jobs import NicheJobs
from tests.integration import test_niche_audit_postgres as audit_fixtures
from tests.unit._research_status_fixture import eligible_status

audit_engine = audit_fixtures.engine


@pytest.fixture
def engine(audit_engine: Any) -> Any:
    with audit_engine.begin() as conn:
        for column in ("headline TEXT", "theme TEXT", "seen_at TIMESTAMPTZ"):
            conn.execute(text("ALTER TABLE geo_events ADD COLUMN " + column))
        for name in (
            "007_trade_ideas.sql",
            "008_trade_idea_levels.sql",
            "025_idea_research_status.sql",
            "029_niche_jobs.sql",
        ):
            for _ in range(2):  # Deployment migrations are idempotent.
                for statement in _strip_sql_comments(Path("migrations", name).read_text()).split(
                    ";"
                ):
                    if statement.strip():
                        conn.execute(text(statement))
        conn.execute(
            text(
                "INSERT INTO geo_events VALUES (1,'ASSESSED',:assessment,'captured','energy',:seen)"
            ),
            {
                "assessment": json.dumps({"urgency": 8, "trade_ideas": []}),
                "seen": datetime.now(UTC) - timedelta(hours=1),
            },
        )
    return audit_engine


def test_postgres_concurrency_and_publication(engine: Any) -> None:
    from scripts.morning_digest import _fetch_bullish_ideas, _fetch_niche_activity

    now = datetime.now(UTC)
    queue = NicheJobs(engine)
    assert queue.seed(7, now) == 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: NicheJobs(engine).claim(now, 1), range(2)))
    assert sum(map(len, claims)) == 1
    job = next(batch[0] for batch in claims if batch)
    report = {"discovery": {"status": "completed"}, "candidates": [], "eligible_count": 1}
    persist_niche_audit(engine, job.invocation_id, 1, job.input_snapshot, report)
    enriched = {
        "urgency": 8,
        "trade_ideas": [
            {"ticker": "FRO", "action": "long", "niche": True, "confidence": 0.7},
        ],
        "niche_research": report,
    }
    status = eligible_status(research_invocation_id=job.invocation_id)
    assert queue.finish(job, enriched, report, {("FRO", "long"): status}, now)
    assert not queue.finish(job, enriched, report, {("FRO", "long"): status}, now)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT state FROM niche_jobs")).scalar_one() == "completed"
        assert conn.execute(text("SELECT COUNT(*) FROM trade_ideas")).scalar_one() == 1
        assert (
            conn.execute(
                text("SELECT research_invocation_id FROM idea_research_status")
            ).scalar_one()
            == job.invocation_id
        )
    ideas = _fetch_bullish_ideas(engine)
    assert ideas and ideas[0]["research_invocation_id"] == job.invocation_id
    activity = _fetch_niche_activity(engine, datetime.now(UTC))
    assert activity and activity[0]["job_state"] == "completed"
    assert activity[0]["published_count"] == 1


def test_postgres_failed_commit_retries_without_reassessment(engine: Any) -> None:
    now = datetime.now(UTC)
    q = NicheJobs(engine)
    q.seed(7, now)
    first = q.claim(now, 1)[0]
    report = {"discovery": {"status": "unavailable", "reason": "auth_failure"}, "candidates": []}
    persist_niche_audit(engine, first.invocation_id, 1, first.input_snapshot, report)
    enriched = {**first.input_snapshot["assessment"], "niche_research": report}
    assert q.finish(first, enriched, report, {}, now)
    restarted = NicheJobs(engine)
    assert restarted.seed(7, now + timedelta(minutes=15)) == 0
    retry = restarted.claim(now + timedelta(minutes=15), 1)[0]
    assert retry.attempts == 2
    with engine.begin() as conn:
        conn.execute(text("UPDATE geo_events SET status='CONFIRMED'"))
    persist_niche_audit(engine, retry.invocation_id, 1, retry.input_snapshot, report)
    assert not restarted.finish(retry, enriched, report, {}, now + timedelta(minutes=15))
    with engine.connect() as conn:
        assert conn.execute(text("SELECT state FROM niche_jobs")).scalar_one() == "superseded"
        assert conn.execute(text("SELECT COUNT(*) FROM trade_ideas")).scalar_one() == 0
