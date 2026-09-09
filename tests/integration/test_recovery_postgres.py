"""Opt-in disposable PostgreSQL: recovery reads and GDELT checkpoint lifecycle."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

import pytest
from migrations.run import _strip_sql_comments
from scripts.alpaca_recovery_report import internal_snapshot
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from src.data.gdelt import GdeltIngester
from src.data.gdelt_bounded import GdeltRead, run_slice

START = datetime(2026, 9, 9, 13, tzinfo=UTC)
END = datetime(2026, 9, 9, 14, tzinfo=UTC)


@pytest.fixture
def recovery_engine() -> Any:
    raw = os.environ.get("CURLIT_AUDIT_TEST_DB_URL")
    if not raw:
        pytest.skip("Requires explicit disposable CURLIT_AUDIT_TEST_DB_URL")
    url = make_url(raw)
    if url.host != "127.0.0.1" or not (url.database or "").startswith("curlit_test_"):
        pytest.fail("Refusing non-disposable database")
    schema = "recovery_test_" + uuid4().hex
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    with engine.begin() as conn:
        for _ in range(2):
            for name in (
                "005_geo_events.sql",
                "014_alpaca_option_orders.sql",
                "016_alpaca_option_exits.sql",
                "018_option_entry_mid.sql",
                "019_alpaca_equity_orders.sql",
                "021_gdelt_ingest_cursors.sql",
            ):
                sql = _strip_sql_comments((Path("migrations") / name).read_text())
                for statement in sql.split(";"):
                    if statement.strip():
                        conn.execute(text(statement))
        conn.execute(text("CREATE TABLE trade_ideas (idea_id TEXT PRIMARY KEY)"))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def ingester_for(engine: Any) -> GdeltIngester:
    ingester = GdeltIngester("sqlite://")
    ingester.engine.dispose()
    ingester.engine = engine
    first = next(iter(ingester.playbooks))
    ingester.playbooks = {first: ingester.playbooks[first]}
    return ingester


def test_recovery_capture_is_read_only_in_actual_postgres(recovery_engine: Any) -> None:
    observed = []

    def inspect_readonly(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT * FROM alpaca_"):
            observed.append(conn.exec_driver_sql("SHOW transaction_read_only").scalar_one())

    event.listen(recovery_engine, "before_cursor_execute", inspect_readonly)
    try:
        captured = internal_snapshot(recovery_engine)
    finally:
        event.remove(recovery_engine, "before_cursor_execute", inspect_readonly)
    assert captured == {"options": [], "equities": [], "idea_ids": []}
    assert observed == ["on", "on"]


def test_two_workers_share_one_persisted_gdelt_lease(recovery_engine: Any) -> None:
    first_entered, release = Event(), Event()
    ingester = ingester_for(recovery_engine)

    def read(*args):
        first_entered.set()
        assert release.wait(5), "fixture worker was not released"
        return GdeltRead("rate_limited", retry_after=120)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run_slice, ingester, START, END, read=read)
        try:
            assert first_entered.wait(5)
            second = pool.submit(
                run_slice, ingester, START, END, read=lambda *args: pytest.fail("duplicate request")
            )
            assert second.result(timeout=5)["attempted"] == 0
        finally:
            release.set()
        assert first.result(timeout=5)["attempted"] == 1
    assert (
        run_slice(ingester, START, END, read=lambda *args: pytest.fail("ignored Retry-After"))[
            "attempted"
        ]
        == 0
    )


def test_crash_after_article_commit_replays_without_losing_coverage(
    recovery_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingester = ingester_for(recovery_engine)
    original = ingester.upsert
    article = {
        "url": "https://example.org/event",
        "title": "Fixture",
        "seendate": "20260909T133000Z",
    }

    def interrupted(df):
        original(df)  # Real committed article; then simulate failure before cursor advancement.
        raise RuntimeError("crash after article commit")

    monkeypatch.setattr(ingester, "upsert", interrupted)
    with pytest.raises(RuntimeError, match="crash after article commit"):
        run_slice(ingester, START, END, read=lambda *args: GdeltRead("success", [article]))
    with recovery_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM geo_events")).scalar_one() == 1
        state = json.loads(
            conn.execute(
                text("SELECT payload FROM gdelt_ingest_cursors WHERE theme != '_source'")
            ).scalar_one()
        )
    assert state["status"] == "pending" and "covered_until" not in state
    monkeypatch.setattr(ingester, "upsert", original)
    # The source spacing is honored with a deterministic future scheduling clock.
    import time

    now = time.time() + 60
    result = run_slice(
        ingester,
        START,
        END,
        read=lambda *args: GdeltRead("success", [article]),
        wall_clock=lambda: now,
    )
    assert result["rows"] == 0 and result["completed"] == 1
    with recovery_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM geo_events")).scalar_one() == 1
        state = json.loads(
            conn.execute(
                text("SELECT payload FROM gdelt_ingest_cursors WHERE theme != '_source'")
            ).scalar_one()
        )
    assert state["covered_until"] == END.isoformat()
