"""Test helper: install the real migration-025 research-status schema (CL-7kuu).

Applies the production SQL (not a hand-copied schema) with the usual sqlite
type shims. The Postgres-only ``CREATE OR REPLACE RULE`` statement (the
no-update rule) has no sqlite equivalent and is skipped; executor tests prove
read-only behavior by capturing the SQL the executors issue instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.events.research_status import ResearchStatus, insert_research_status

MIGRATION = Path("migrations", "025_idea_research_status.sql")


def research_status_statements() -> list[str]:
    from migrations.run import _strip_sql_comments

    sql = _strip_sql_comments(MIGRATION.read_text())
    sql = sql.replace("TIMESTAMPTZ", "TEXT").replace("JSONB", "TEXT")
    return [s.strip() for s in sql.split(";") if s.strip()]


def install_research_status(engine: Any) -> None:
    with engine.begin() as conn:
        for stmt in research_status_statements():
            if stmt.upper().startswith("CREATE OR REPLACE RULE"):
                continue  # Postgres-only immutability rule.
            conn.execute(text(stmt))


def eligible_status(**overrides: Any) -> ResearchStatus:
    fields: dict[str, Any] = {
        "discovery_status": "completed",
        "evidence_status": "source_backed",
        "review_status": "supported",
        "liquidity_status": "sufficient",
        "research_eligible": True,
        "source_hashes": ("a" * 64,),
        "score_version": "evidence-coverage-v1",
        "research_invocation_id": "inv-test",
    }
    fields.update(overrides)
    return ResearchStatus(**fields)


def record_status(engine: Any, idea_id: str, status: ResearchStatus | None = None) -> bool:
    with engine.begin() as conn:
        return insert_research_status(conn, idea_id, status or eligible_status())
