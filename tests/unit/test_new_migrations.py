"""Smoke test that the new migration files apply cleanly on sqlite
(CL-43l, CL-1fm, CL-6iu7).

Sqlite doesn't grok TIMESTAMPTZ / JSONB / NUMERIC / BIGSERIAL, so we
shim the types before running. The point is that the SQL parses and
table creation succeeds — Postgres compatibility is verified by the
production runner.
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
        .replace("BIGSERIAL", "INTEGER")
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


class TestGeoEventsTable:
    def test_creates_table_and_index(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/005_geo_events.sql"))
        insp = inspect(sqlite_engine)
        assert "geo_events" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("geo_events")}
        assert {
            "id", "seen_at", "source", "external_id", "headline",
            "url", "theme", "assessment", "status", "status_updated_at",
        } <= cols
        idx_names = {i["name"] for i in insp.get_indexes("geo_events")}
        assert "idx_geo_events_status_seen" in idx_names

    def test_default_status_new_and_unique_external_id(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/005_geo_events.sql"))
        with sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO geo_events "
                "(seen_at, source, external_id, headline, status_updated_at) "
                "VALUES ('2026-07-14T00:00:00Z', 'gdelt', 'abc', 'h1', "
                "'2026-07-14T00:00:00Z')",
            ))
            status = conn.execute(text(
                "SELECT status FROM geo_events WHERE external_id='abc'",
            )).scalar()
            assert status == "NEW"
        with pytest.raises(Exception, match="(?i)unique"), sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO geo_events "
                "(seen_at, source, external_id, headline, status_updated_at) "
                "VALUES ('2026-07-14T00:00:00Z', 'gdelt', 'abc', 'dup', "
                "'2026-07-14T00:00:00Z')",
            ))

    def test_status_check_constraint(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/005_geo_events.sql"))
        with pytest.raises(Exception, match="(?i)check"), sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO geo_events "
                "(seen_at, source, external_id, headline, status, "
                "status_updated_at) "
                "VALUES ('2026-07-14T00:00:00Z', 'gdelt', 'xyz', 'h', "
                "'BOGUS', '2026-07-14T00:00:00Z')",
            ))


class TestVolumeSpikesTable:
    def test_creates_table_and_indexes(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/006_volume_spikes.sql"))
        insp = inspect(sqlite_engine)
        assert "volume_spikes" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("volume_spikes")}
        assert {
            "id", "ticker", "scanned_at", "rvol", "volume",
            "avg_volume_20d", "price_change_pct", "is_unusual", "source",
        } <= cols
        idx_names = {i["name"] for i in insp.get_indexes("volume_spikes")}
        assert "idx_volume_spikes_ticker_scanned" in idx_names
        assert "idx_volume_spikes_unusual_scanned" in idx_names

    def test_defaults_not_unusual_yfinance_source(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/006_volume_spikes.sql"))
        with sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO volume_spikes (ticker, scanned_at, rvol) "
                "VALUES ('FRO', '2026-07-20T00:00:00Z', 3.2)",
            ))
            row = conn.execute(text(
                "SELECT is_unusual, source FROM volume_spikes "
                "WHERE ticker='FRO'",
            )).fetchone()
            assert not row[0]
            assert row[1] == "yfinance"


class TestTradeIdeasTable:
    def test_creates_table_and_indexes(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/007_trade_ideas.sql"))
        insp = inspect(sqlite_engine)
        assert "trade_ideas" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("trade_ideas")}
        assert {
            "id", "idea_id", "geo_event_id", "ticker", "action",
            "direction", "confidence", "time_horizon",
            "holding_period_days", "time_stop_days", "stop_loss_pct",
            "preferred_instrument", "instrument_reason", "rationale",
            "suggested_entry", "notes", "price_at_signal", "created_at",
            "status", "status_updated_at",
        } <= cols
        idx_names = {i["name"] for i in insp.get_indexes("trade_ideas")}
        assert "idx_trade_ideas_status_created" in idx_names
        assert "idx_trade_ideas_ticker" in idx_names

    def test_default_pending_and_unique_idea_id(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/007_trade_ideas.sql"))
        insert = text(
            "INSERT INTO trade_ideas "
            "(idea_id, geo_event_id, ticker, action, created_at, "
            "status_updated_at) "
            "VALUES ('abc123', 1, 'TSM', 'buy_puts', "
            "'2026-07-20T00:00:00Z', '2026-07-20T00:00:00Z') "
            "ON CONFLICT (idea_id) DO NOTHING",
        )
        with sqlite_engine.begin() as conn:
            conn.execute(insert)
            conn.execute(insert)  # dedups silently
            status, count = conn.execute(text(
                "SELECT status, (SELECT count(*) FROM trade_ideas) "
                "FROM trade_ideas",
            )).fetchone()
            assert status == "pending"
            assert count == 1

    def test_status_check_constraint(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        _apply(sqlite_engine, Path("migrations/007_trade_ideas.sql"))
        with pytest.raises(Exception, match="(?i)check"), sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO trade_ideas "
                "(idea_id, geo_event_id, ticker, action, created_at, "
                "status, status_updated_at) "
                "VALUES ('x', 1, 'TSM', 'long', '2026-07-20T00:00:00Z', "
                "'BOGUS', '2026-07-20T00:00:00Z')",
            ))


class TestTradeIdeaLevelsMigration:
    """CL-jiqq — migration 008 adds the concrete trade-card level columns
    to the migration-007 trade_ideas table."""

    def _apply_007_and_008(self, engine) -> None:  # type: ignore[no-untyped-def]
        from migrations.run import _strip_sql_comments

        from tests.unit.test_idea_ledger import (
            _shim_pg_types_for_sqlite,
            _sqlite_statements,
        )

        for mig in (
            "migrations/007_trade_ideas.sql",
            "migrations/008_trade_idea_levels.sql",
        ):
            sql = _shim_pg_types_for_sqlite(
                _strip_sql_comments(Path(mig).read_text()),
            )
            with engine.begin() as conn:
                for stmt in _sqlite_statements(sql):
                    conn.execute(text(stmt))

    def test_adds_all_level_columns(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        self._apply_007_and_008(sqlite_engine)
        cols = {c["name"] for c in inspect(sqlite_engine).get_columns("trade_ideas")}
        assert {
            "stop_price", "target_prices", "risk_reward", "entry_trigger",
            "invalidation", "dte_window", "suggested_strike",
        } <= cols

    def test_new_columns_default_null(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        self._apply_007_and_008(sqlite_engine)
        with sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO trade_ideas "
                "(idea_id, geo_event_id, ticker, action, created_at, "
                "status_updated_at) "
                "VALUES ('lv1', 1, 'TSM', 'buy_puts', "
                "'2026-07-20T00:00:00Z', '2026-07-20T00:00:00Z')",
            ))
            row = conn.execute(text(
                "SELECT stop_price, target_prices, risk_reward, "
                "entry_trigger, invalidation, dte_window, suggested_strike "
                "FROM trade_ideas WHERE idea_id='lv1'",
            )).fetchone()
            assert all(v is None for v in row)

    def test_level_columns_round_trip(self, sqlite_engine) -> None:  # type: ignore[no-untyped-def]
        self._apply_007_and_008(sqlite_engine)
        with sqlite_engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO trade_ideas "
                "(idea_id, geo_event_id, ticker, action, created_at, "
                "status_updated_at, stop_price, target_prices, risk_reward, "
                "entry_trigger, invalidation, dte_window, suggested_strike) "
                "VALUES ('lv2', 1, 'TSM', 'buy_puts', "
                "'2026-07-20T00:00:00Z', '2026-07-20T00:00:00Z', "
                "186.0, '[155.0, 141.0]', 1.3, 'on blockade', "
                "'denial', '1-3 weeks', 164.0)",
            ))
            row = conn.execute(text(
                "SELECT stop_price, target_prices, dte_window, suggested_strike "
                "FROM trade_ideas WHERE idea_id='lv2'",
            )).fetchone()
            assert row[0] == 186.0
            assert row[1] == "[155.0, 141.0]"
            assert row[2] == "1-3 weeks"
            assert row[3] == 164.0
