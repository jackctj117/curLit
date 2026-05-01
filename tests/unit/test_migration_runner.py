"""Tests for migrations/run.py SQL comment stripping (CL-e5ta).

Reproduces the bug where line/block comments containing ';' caused
sql.split(';') to emit empty fragments → psycopg2 'empty query' error.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text

from migrations.run import _strip_sql_comments, run_migrations


class TestStripComments:
    def test_line_comment_with_semicolon(self) -> None:
        sql = "-- explain; this is fine\nCREATE TABLE x (id INT);"
        out = _strip_sql_comments(sql)
        assert ";" in out  # the real CREATE statement keeps its terminator
        # the comment fragment is gone — splitting now gives ONE statement
        statements = [s.strip() for s in out.split(";") if s.strip()]
        assert len(statements) == 1
        assert statements[0].startswith("CREATE TABLE")

    def test_block_comment_with_semicolon(self) -> None:
        sql = "/* foo; bar; baz */\nSELECT 1;"
        out = _strip_sql_comments(sql)
        statements = [s.strip() for s in out.split(";") if s.strip()]
        assert statements == ["SELECT 1"]

    def test_multi_statement_no_comments(self) -> None:
        sql = "CREATE TABLE a (id INT); CREATE TABLE b (id INT);"
        out = _strip_sql_comments(sql)
        statements = [s.strip() for s in out.split(";") if s.strip()]
        assert len(statements) == 2

    def test_real_world_pattern(self) -> None:
        # The exact failure mode from CL-e5ta: comment with English
        # punctuation (semicolon) followed by a real statement.
        sql = (
            "-- store embeddings; later we may swap models\n"
            "CREATE TABLE embeddings (id INT, vec FLOAT[]);"
        )
        out = _strip_sql_comments(sql)
        statements = [s.strip() for s in out.split(";") if s.strip()]
        assert len(statements) == 1
        assert "CREATE TABLE embeddings" in statements[0]


class TestEndToEndOnSqlite:
    def test_runs_migration_with_comment_semicolons(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Full path: read .sql with comment-internal ';', apply to sqlite."""
        # sqlite doesn't accept TIMESTAMPTZ etc — write a generic migration.
        mig_dir = tmp_path / "migrations"
        mig_dir.mkdir()
        (mig_dir / "001_test.sql").write_text(
            "-- create the demo table; nothing special\n"
            "CREATE TABLE demo (id INTEGER PRIMARY KEY, name TEXT);\n"
            "-- now insert a row; this used to crash\n"
            "INSERT INTO demo (id, name) VALUES (1, 'a');\n",
        )

        engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
        # Monkeypatch MIGRATIONS_DIR for the test
        from migrations import run as runner

        original_dir = runner.MIGRATIONS_DIR
        runner.MIGRATIONS_DIR = mig_dir
        try:
            run_migrations(engine)
        finally:
            runner.MIGRATIONS_DIR = original_dir

        with engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM demo")).scalar()
            assert n == 1
