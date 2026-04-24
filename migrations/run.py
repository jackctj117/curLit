"""
Database migration runner.
Run: python -m migrations.run
"""

import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent


def get_engine():
    db_url = (
        f"postgresql+psycopg2://"
        f"{os.environ.get('POSTGRES_USER', 'fx')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'fx')}"
    )
    return create_engine(db_url)


def run_migrations(engine=None):
    """Apply all migrations in order."""
    if engine is None:
        engine = get_engine()

    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))

    with engine.connect() as conn:
        for migration in migrations:
            logger.info(f"Applying migration: {migration.name}")
            sql = migration.read_text()
            # Split by statement (handle multi-statement files)
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                conn.execute(text(stmt))
            conn.commit()
            logger.info(f"  ✓ {migration.name}")

    logger.info("All migrations applied successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migrations()
