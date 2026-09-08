"""CL-27s0: opt-in disposable PostgreSQL proof; no operational DB defaults."""

from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from migrations.run import _strip_sql_comments
from scripts import event_pipeline
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.events.niche_audit import persist_niche_audit


@pytest.fixture
def engine() -> Any:
    raw_url = os.environ.get("CURLIT_AUDIT_TEST_DB_URL")
    if not raw_url:
        pytest.skip("Set CURLIT_AUDIT_TEST_DB_URL to an isolated curlit_test_* PostgreSQL database")
    url = make_url(raw_url)
    if url.host != "127.0.0.1" or not (url.database or "").startswith("curlit_test_"):
        pytest.fail("Audit integration tests require a loopback disposable curlit_test_* database")
    # A unique schema isolates each test without deleting pre-existing objects.
    schema = "audit_test_" + uuid4().hex
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    sql = _strip_sql_comments(Path("migrations/020_niche_research_audit.sql").read_text())
    with engine.begin() as conn:
        for _ in range(2):
            for statement in sql.split(";"):
                if statement.strip():
                    conn.execute(text(statement))
        conn.execute(
            text("CREATE TABLE geo_events (id BIGINT PRIMARY KEY, status TEXT, assessment JSONB)")
        )
    try:
        yield engine
    finally:
        engine.dispose()
        # Only this fixture's randomly named schema, in the opt-in test DB.
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.mark.parametrize("change", ["status", "assessment", "none"])
def test_actual_pipeline_preserves_audit_without_overwriting_concurrent_decisions(
    engine: Any,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    initial = {"urgency": 8, "trade_ideas": []}
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO geo_events VALUES (1, 'ASSESSED', :a)"), {"a": json.dumps(initial)}
        )

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run_report(self, row: dict[str, Any], playbook: Any = None) -> Any:
            # Independent committed connection simulates the confluence worker
            # advancing a decision while the researcher is still running.
            with engine.begin() as conn:
                if change == "status":
                    conn.execute(text("UPDATE geo_events SET status = 'CONFIRMED' WHERE id = 1"))
                elif change == "assessment":
                    conn.execute(
                        text("UPDATE geo_events SET assessment = :a WHERE id = 1"),
                        {"a": json.dumps({**initial, "concurrent_decision": True})},
                    )
            return SimpleNamespace(
                eligible=["fixture-candidate"],
                to_dict=lambda: {"discovery": {"status": "completed"}},
            )

        def merge_into_assessment(self, assessment: dict[str, Any], ideas: list[str]) -> int:
            assessment["trade_ideas"].extend(ideas)
            return len(ideas)

    monkeypatch.setattr("src.events.niche_agent.NicheAgent", Agent)
    monkeypatch.setattr(
        "src.data.symbols.SymbolUniverse", lambda engine: SimpleNamespace(exists=lambda t: True)
    )
    result = SimpleNamespace(
        event_id=1,
        headline="fixture",
        theme=None,
        status="ASSESSED",
        assessment=copy.deepcopy(initial),
    )
    count = event_pipeline._niche_step(engine, [result], 7)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM niche_research_audit")).mappings().all()
        assessment = conn.execute(
            text("SELECT assessment FROM geo_events WHERE id = 1")
        ).scalar_one()
    assert len(rows) == 1
    assert rows[0]["input_snapshot"]["assessment"] == initial
    assert rows[0]["report"]["discovery"]["status"] == "completed"
    if change == "none":
        assert count == 1
        assert assessment == result.assessment
        assert assessment["trade_ideas"] == ["fixture-candidate"]
    else:
        assert count == 0
        assert result.assessment == initial
        assert assessment["trade_ideas"] == []
        if change == "assessment":
            assert assessment["concurrent_decision"] is True


def test_concurrent_replay_commits_exactly_once(engine: Any) -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: persist_niche_audit(engine, "same-run", 1, {}, {"status": "partial"}),
                range(8),
            )
        )
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM niche_research_audit")).scalar_one() == 1
    with pytest.raises(ValueError, match="collision"):
        persist_niche_audit(engine, "same-run", 1, {}, {"status": "completed"})
