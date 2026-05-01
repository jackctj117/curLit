"""Smoke test that the new migration files apply cleanly on sqlite (CL-43l, CL-1fm).

Sqlite doesn't grok TIMESTAMPTZ / JSONB / NUMERIC, so we shim the types
before running. The point is that the SQL parses and table creation
succeeds — Postgres compatibility is verified by the production runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text


def _shim_pg_types_for_sqlite(sql: str) -> str:
    """Map Postgres-only types to sqlite equivalents for a parse smoke."""
    return (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("JSONB", "TEXT")
        .replace("NUMERIC", "REAL")
        .replace(" DEFAULT NOW()", "")
    )


@pytest.fixture
def sqlite_engine(tmp_path):  # type: ignore[no-untyped-def]
    return create_engine(f"sqlite:///{tmp_path / 'mig_test.db'}")


def _apply(engine, sql_path: Path) -> None:  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    sql = _shim_pg_types_for_sqlite(_strip_sql_comments(sql_path.read_text()))
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))


class TestFxVolatilityTable:
    def test_creates_table_and_index(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/003_fx_volatility.sql"))
        insp = inspect(sqlite_engine)
        assert "fx_volatility" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("fx_volatility")}
        assert {"date", "index_name", "value"} <= cols

    def test_insert_and_query(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/003_fx_volatility.sql"))
        with sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO fx_volatility VALUES ('2026-04-01', 'CVIX', 8.5)",
            ))
            v = conn.execute(text(
                "SELECT value FROM fx_volatility WHERE index_name='CVIX'",
            )).scalar()
            assert v == 8.5


class TestResearchPapersTable:
    def test_creates_table_and_indexes(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/004_research_papers.sql"))
        insp = inspect(sqlite_engine)
        assert "research_papers" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("research_papers")}
        assert {
            "paper_id", "title", "abstract", "url", "relevance_score",
            "read_status", "implementation_priority",
        } <= cols
        idx_names = {i["name"] for i in insp.get_indexes("research_papers")}
        assert "idx_research_papers_triage" in idx_names

    def test_default_unread(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/004_research_papers.sql"))
        with sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO research_papers (paper_id, title) "
                "VALUES ('arxiv:2024.001', 'Test paper')",
            ))
            status = conn.execute(text(
                "SELECT read_status FROM research_papers "
                "WHERE paper_id='arxiv:2024.001'",
            )).scalar()
            assert status == "unread"
