"""CL-ut1d: durable recovery oracle on real disposable SQL transactions."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from migrations.run import _strip_sql_comments
from sqlalchemy import create_engine, text

from src.events.niche_audit import persist_niche_audit
from src.events.niche_jobs import LEASE_SECONDS, NicheJobs, retryable
from tests.unit._research_status_fixture import eligible_status, install_research_status
from tests.unit.test_idea_ledger import _shim_pg_types_for_sqlite, _sqlite_statements

NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)
INPUT = {
    "id": 1,
    "headline": "Captured event",
    "theme": "energy",
    "assessment": {"urgency": 8, "trade_ideas": []},
}


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    eng = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE geo_events (id INTEGER PRIMARY KEY, headline TEXT, theme TEXT,"
                "assessment TEXT, seen_at TEXT, status TEXT)"
            )
        )
        for migration in (
            "007_trade_ideas.sql",
            "008_trade_idea_levels.sql",
            "020_niche_research_audit.sql",
            "029_niche_jobs.sql",
        ):
            sql = _shim_pg_types_for_sqlite(
                _strip_sql_comments(Path("migrations", migration).read_text())
            ).replace("NOW()", "CURRENT_TIMESTAMP")
            for stmt in _sqlite_statements(sql):
                conn.execute(text(stmt))
        conn.execute(
            text("INSERT INTO geo_events VALUES (1,:headline,:theme,:assessment,:seen,'ASSESSED')"),
            {
                **INPUT,
                "assessment": json.dumps(INPUT["assessment"]),
                "seen": NOW - timedelta(hours=1),
            },
        )
    install_research_status(eng)
    yield eng
    eng.dispose()


def _report(status: str, reason: str = "") -> dict[str, Any]:
    return {
        "discovery": {"status": status, "reason": reason},
        "candidates": [],
        "eligible_count": 0,
    }


def _finish(queue: NicheJobs, job: Any, report: dict, now: datetime) -> bool:
    persist_niche_audit(queue.engine, job.invocation_id, 1, job.input_snapshot, report)
    enriched = copy.deepcopy(job.input_snapshot["assessment"])
    enriched["niche_research"] = {**report, "invocation_id": job.invocation_id}
    return queue.finish(job, enriched, report, {}, now)


def test_failed_discovery_retries_without_reassessment_after_restart(engine: Any) -> None:
    q = NicheJobs(engine)
    assert q.seed(7, NOW) == 1
    job = q.claim(NOW, 1)[0]
    assert _finish(q, job, _report("unavailable", "subscription_auth_unverified"), NOW)
    restarted = NicheJobs(engine)
    assert restarted.seed(7, NOW + timedelta(minutes=15)) == 0
    assert restarted.claim(NOW + timedelta(minutes=14), 1) == []
    retry = restarted.claim(NOW + timedelta(minutes=15), 1)[0]
    assert retry.attempts == 2 and retry.invocation_id != job.invocation_id
    assert _finish(restarted, retry, _report("abstained"), NOW + timedelta(minutes=15))
    assert restarted.claim(NOW + timedelta(hours=3), 1) == []
    with engine.connect() as conn:
        assert conn.execute(text("SELECT status FROM geo_events")).scalar_one() == "ASSESSED"
        assert conn.execute(text("SELECT COUNT(*) FROM niche_research_audit")).scalar_one() == 2
        assert conn.execute(text("SELECT COUNT(*) FROM trade_ideas")).scalar_one() == 0


@pytest.mark.parametrize("status", ["abstained", "completed", "invalid_output"])
def test_terminal_outcomes_do_not_hot_loop(engine: Any, status: str) -> None:
    q = NicheJobs(engine)
    q.seed(7, NOW)
    assert _finish(q, q.claim(NOW, 1)[0], _report(status), NOW)
    assert q.seed(7, NOW + timedelta(hours=2)) == 0
    assert q.claim(NOW + timedelta(hours=2), 1) == []


def test_two_workers_only_one_claim_and_stale_worker_cannot_publish(engine: Any) -> None:
    q = NicheJobs(engine)
    q.seed(7, NOW)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: NicheJobs(engine).claim(NOW, 1), range(2)))
    assert sum(map(len, claims)) == 1
    old = next(jobs[0] for jobs in claims if jobs)
    later = NOW + timedelta(seconds=LEASE_SECONDS + 1)
    new = q.claim(later, 1)[0]
    assert _finish(q, old, _report("completed"), later) is False
    assert _finish(q, new, _report("abstained"), later)


@pytest.mark.parametrize("change", ["status", "assessment"])
def test_concurrent_event_change_fences_publication(engine: Any, change: str) -> None:
    q = NicheJobs(engine)
    q.seed(7, NOW)
    job = q.claim(NOW, 1)[0]
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE geo_events SET status='CONFIRMED'"
                if change == "status"
                else "UPDATE geo_events SET assessment=:assessment"
            ),
            {"assessment": json.dumps({"urgency": 9, "trade_ideas": []})},
        )
    assert not _finish(q, job, _report("completed"), NOW)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT state FROM niche_jobs")).scalar_one() == "superseded"
        assert conn.execute(text("SELECT COUNT(*) FROM niche_research_audit")).scalar_one() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM trade_ideas")).scalar_one() == 0


def test_expiry_and_attempt_limits(engine: Any) -> None:
    q = NicheJobs(engine)
    q.seed(7, NOW)
    for offset in (0, 15, 75):
        instant = NOW + timedelta(minutes=offset)
        job = q.claim(instant, 1)[0]
        assert _finish(q, job, _report("unavailable"), instant)
    assert q.claim(NOW + timedelta(hours=3), 1) == []
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE geo_events SET assessment=:assessment"),
            {"assessment": json.dumps({"urgency": 9, "trade_ideas": []})},
        )
    assert q.seed(7, NOW) == 1
    assert q.claim(NOW + timedelta(days=1), 1) == []
    with engine.connect() as conn:
        assert set(conn.execute(text("SELECT state FROM niche_jobs")).scalars()) == {
            "failed",
            "expired",
        }


def test_ledger_failure_rolls_back_event_and_job_but_keeps_audit(
    engine: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = NicheJobs(engine)
    q.seed(7, NOW)
    job = q.claim(NOW, 1)[0]

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr("src.events.niche_jobs.persist_ideas", fail)
    with pytest.raises(RuntimeError, match="ledger unavailable"):
        _finish(q, job, _report("completed"), NOW)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT state FROM niche_jobs")).scalar_one() == "running"
        assert (
            json.loads(conn.execute(text("SELECT assessment FROM geo_events")).scalar_one())
            == (INPUT["assessment"])
        )
        assert conn.execute(text("SELECT COUNT(*) FROM niche_research_audit")).scalar_one() == 1


def test_eligible_idea_and_immutable_status_commit_once(engine: Any) -> None:
    q = NicheJobs(engine)
    q.seed(7, NOW)
    job = q.claim(NOW, 1)[0]
    report = _report("completed")
    persist_niche_audit(engine, job.invocation_id, 1, job.input_snapshot, report)
    enriched = {
        "urgency": 8,
        "trade_ideas": [
            {"ticker": "FRO", "action": "long", "niche": True, "confidence": 0.7},
        ],
        "niche_research": report,
    }
    status = eligible_status(research_invocation_id=job.invocation_id)
    assert q.finish(job, enriched, report, {("FRO", "long"): status}, NOW)
    assert not q.finish(job, enriched, report, {("FRO", "long"): status}, NOW)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM trade_ideas")).scalar_one() == 1
        assert (
            conn.execute(
                text("SELECT research_invocation_id FROM idea_research_status")
            ).scalar_one()
            == job.invocation_id
        )


def test_review_failure_retries_but_unsupported_evidence_does_not() -> None:
    report = _report("completed")
    report["candidates"] = [{"research": {"review_status": "review_unavailable"}}]
    assert retryable(report)
    report["candidates"][0]["research"]["review_status"] = "insufficient_evidence"
    assert not retryable(report)


def test_completed_discovery_with_failed_review_records_review_retry_reason(engine: Any) -> None:
    queue = NicheJobs(engine)
    queue.seed(7, NOW)
    report = _report("completed")
    report["candidates"] = [{"research": {"review_status": "review_unavailable"}}]
    assert _finish(queue, queue.claim(NOW, 1)[0], report, NOW)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT state, reason FROM niche_jobs")).one()
        assert tuple(row) == ("retry", "review_unavailable")


def test_pipeline_retries_with_no_new_assessments(
    engine: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import event_pipeline

    from src.events.niche_agent import NicheReport
    from src.events.research_evidence import DiscoveryOutcome

    clock = [NOW]
    captured = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return clock[0]

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["subscription_only"] is True
            assert kwargs["tool_agent_enabled"] is False

        def run_report(self, row: dict, playbook: Any = None) -> NicheReport:
            captured.append(copy.deepcopy(row))
            return NicheReport(
                DiscoveryOutcome(
                    "unavailable" if len(captured) == 1 else "abstained",
                    "subscription_quota_exhausted" if len(captured) == 1 else "",
                ),
                [],
            )

        @staticmethod
        def merge_into_assessment(*args: Any, **kwargs: Any) -> int:
            return 0

    monkeypatch.setattr(event_pipeline, "datetime", Clock)
    monkeypatch.setattr("src.events.niche_agent.NicheAgent", Agent)
    monkeypatch.setattr("src.data.symbols.SymbolUniverse", lambda engine: object())
    assert event_pipeline._niche_step(engine, [], 7, durable=True) == 0
    clock[0] += timedelta(minutes=15)
    results: list = []
    assert event_pipeline._niche_step(engine, results, 7, durable=True) == 0
    assert len(captured) == 2 and captured[0] == captured[1] == INPUT
    assert results[0].assessment["niche_research"]["discovery"]["status"] == "abstained"
    clock[0] += timedelta(hours=2)
    assert event_pipeline._niche_step(engine, [], 7, durable=True) == 0
    assert len(captured) == 2  # A completed abstention is never a reason to hot-loop.


def test_morning_queries_preserve_real_status_and_failure_visibility(engine: Any) -> None:
    from scripts.morning_digest import _fetch_bullish_ideas, _fetch_niche_activity

    from src.events.idea_ledger import persist_ideas

    status = eligible_status(research_invocation_id="verified-run")
    persist_ideas(
        engine,
        1,
        {
            "trade_ideas": [
                {
                    "ticker": "FRO",
                    "action": "long",
                    "direction": "bullish",
                    "niche": True,
                }
            ]
        },
        now=NOW,
        research_status={("FRO", "long"): status},
    )
    ideas = _fetch_bullish_ideas(engine)
    assert ideas and ideas[0]["research_invocation_id"] == "verified-run"
    assert ideas[0]["event_seen_at"] and ideas[0]["created_at"]
    persist_niche_audit(engine, "failed-run", 1, INPUT, _report("unavailable", "auth_failure"))
    with engine.begin() as conn:
        conn.execute(text("UPDATE niche_research_audit SET recorded_at=:now"), {"now": NOW})
    activity = _fetch_niche_activity(engine, NOW)
    assert activity and activity[0]["discovery_status"] == "unavailable"
    assert activity[0]["reason"] == "auth_failure"

    class Unavailable:
        def connect(self) -> Any:
            raise RuntimeError("test read failure")

    assert _fetch_bullish_ideas(Unavailable()) is None
    assert _fetch_niche_activity(Unavailable(), NOW) is None


def test_client_initialization_failure_is_audited_and_deferred(
    engine: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import event_pipeline

    class Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return NOW

    def unavailable(**kwargs: Any) -> Any:
        raise RuntimeError("test missing CLI")

    monkeypatch.setattr(event_pipeline, "datetime", Clock)
    monkeypatch.setattr("src.events.niche_agent.NicheAgent", unavailable)
    monkeypatch.setattr("src.data.symbols.SymbolUniverse", lambda engine: object())
    assert event_pipeline._niche_step(engine, [], 7, durable=True) == 0
    with engine.connect() as conn:
        assert conn.execute(text("SELECT state FROM niche_jobs")).scalar_one() == "retry"
        report = json.loads(
            conn.execute(text("SELECT report FROM niche_research_audit")).scalar_one()
        )
        assert report["discovery"]["reason"] == "agent_initialization_failed"


def test_weeks_old_event_is_not_reissued_as_new_research(engine: Any) -> None:
    with engine.begin() as conn:
        conn.execute(text("UPDATE geo_events SET seen_at=:old"), {"old": NOW - timedelta(days=30)})
    q = NicheJobs(engine)
    assert q.seed(7, NOW) == 0
    assert q.claim(NOW, 1) == []


@pytest.mark.parametrize("when", ["before_claim", "before_finish"])
def test_corrupt_changed_input_is_superseded_not_published(engine: Any, when: str) -> None:
    q = NicheJobs(engine)
    q.seed(7, NOW)
    job = q.claim(NOW, 1)[0] if when == "before_finish" else None
    with engine.begin() as conn:
        conn.execute(text("UPDATE geo_events SET assessment='null'"))
    if job:
        assert not _finish(q, job, _report("completed"), NOW)
    else:
        assert q.claim(NOW, 1) == []
    with engine.connect() as conn:
        assert conn.execute(text("SELECT state FROM niche_jobs")).scalar_one() == "superseded"
