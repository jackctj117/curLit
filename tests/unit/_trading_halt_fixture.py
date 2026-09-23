"""Test helper: install the real migration-024 halt schema (CL-0deu.2).

Applies the production SQL (not a hand-copied schema) so fixtures exercise the
real fail-closed seed. ``active=True`` then performs an explicit, attributed
resume — exactly what an operator must do after deploying the migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.risk.trading_halt import TradingHaltStore


def install_trading_halt(engine: Any, *, active: bool = True) -> TradingHaltStore:
    from migrations.run import _strip_sql_comments

    sql = _strip_sql_comments(Path("migrations", "024_trading_halt.sql").read_text())
    sql = sql.replace("TIMESTAMPTZ", "TEXT")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    store = TradingHaltStore(engine)
    if active:
        store.resume(reason="test fixture: trading enabled", source="test", changed_by="pytest")
    return store
