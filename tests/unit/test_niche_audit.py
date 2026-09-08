"""CL-27s0: real transactions, replay and collision checks on disposable SQLite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.events.niche_audit import persist_niche_audit


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    from migrations.run import _strip_sql_comments

    engine = create_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    sql = _strip_sql_comments(Path("migrations/020_niche_research_audit.sql").read_text())
    sql = (
        sql.replace("JSONB", "TEXT")
        .replace("TIMESTAMPTZ", "TEXT")
        .replace("NOW()", "CURRENT_TIMESTAMP")
    )
    with engine.begin() as conn:
        # Apply twice: the deployment runner replays additive migrations.
        for _ in range(2):
            for statement in sql.split(";"):
                if statement.strip():
                    conn.execute(text(statement))
    yield engine
    engine.dispose()


def test_audit_replay_does_not_overwrite_or_duplicate(engine: Any) -> None:
    snapshot = {"id": 1, "assessment": {"urgency": 8}}
    report = {"discovery": {"status": "partial"}, "eligible_count": 0}
    persist_niche_audit(engine, "invocation-1", 1, snapshot, report)
    persist_niche_audit(engine, "invocation-1", 1, snapshot, report)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT event_id, input_snapshot, report FROM niche_research_audit")
        ).all()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert json.loads(rows[0][1]) == snapshot
    assert json.loads(rows[0][2]) == report
    with pytest.raises(ValueError, match="collision"):
        persist_niche_audit(
            engine, "invocation-1", 1, snapshot, {"discovery": {"status": "completed"}}
        )


def test_audit_survives_independent_failed_merge_and_changed_status(engine: Any) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE geo_events (id INTEGER PRIMARY KEY, status TEXT, assessment TEXT)")
        )
        conn.execute(text("INSERT INTO geo_events VALUES (1, 'CONFIRMED', '{}')"))
    persist_niche_audit(engine, "invocation-1", 1, {"id": 1}, {"discovery": {"status": "partial"}})
    with pytest.raises(RuntimeError), engine.begin() as conn:
        result = conn.execute(
            text("UPDATE geo_events SET assessment = 'unsafe' WHERE id = 1 AND status = 'ASSESSED'")
        )
        assert result.rowcount == 0
        raise RuntimeError("merge rolled back")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM niche_research_audit")).scalar_one() == 1
        assert conn.execute(text("SELECT assessment FROM geo_events")).scalar_one() == "{}"


def test_failed_audit_commit_is_not_success_and_can_be_replayed(engine: Any) -> None:
    class FailedCommit:
        def begin(self) -> Any:
            from contextlib import contextmanager

            @contextmanager
            def transaction() -> Any:
                with engine.begin() as conn:
                    yield conn
                    raise RuntimeError("commit failed")

            return transaction()

    with pytest.raises(RuntimeError, match="commit failed"):
        persist_niche_audit(FailedCommit(), "invocation-1", 1, {}, {})
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM niche_research_audit")).scalar_one() == 0
    persist_niche_audit(engine, "invocation-1", 1, {}, {})
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM niche_research_audit")).scalar_one() == 1
