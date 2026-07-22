"""
Database migration runner.
Run: python -m migrations.run
"""

import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from src.data.db_env import build_db_url

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent

# Strip `-- line comments` and `/* block comments */` before splitting on ';'.
# Bare-string split would treat semicolons inside a comment as statement
# boundaries — natural English in a comment like "embeddings; if we swap..."
# would crash psycopg2 with "can't execute an empty query" (CL-e5ta).
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments before statement splitting."""
    sql = _BLOCK_COMMENT_RE.sub("", sql)
    sql = _LINE_COMMENT_RE.sub("", sql)
    return sql


def get_engine() -> Any:
    """Engine from the shared env-var convention (CL-8cw1: delegated to
    src.data.db_env.build_db_url — DATABASE_URL override, changeme warning,
    percent-safe password)."""
    return create_engine(build_db_url())


def run_migrations(engine: Any = None) -> None:
    """Apply all migrations in order."""
    if engine is None:
        engine = get_engine()

    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))

    with engine.connect() as conn:
        for migration in migrations:
            logger.info("Applying migration: %s", migration.name)
            sql = _strip_sql_comments(migration.read_text())
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                conn.execute(text(stmt))
            conn.commit()
            logger.info("  ✓ %s", migration.name)

    logger.info("All migrations applied successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migrations()
