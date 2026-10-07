"""Test helper: install the real migration-025 research-status schema (CL-7kuu).

Applies the production SQL (not a hand-copied schema) with the usual sqlite
type shims. The Postgres-only ``CREATE OR REPLACE RULE`` statements (the
no-update / no-delete rules) are emulated with sqlite ``RAISE(IGNORE)``
triggers; executor tests additionally capture the SQL the executors issue.
"""

from __future__ import annotations

import re
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


_RULE_RE = re.compile(
    r"CREATE OR REPLACE RULE (\w+) AS ON (UPDATE|DELETE) TO (\w+) DO INSTEAD NOTHING",
    re.IGNORECASE,
)


def install_research_status(engine: Any) -> None:
    """Apply migration 025. Each Postgres ``DO INSTEAD NOTHING`` rule is
    emulated by the sqlite equivalent, a BEFORE trigger that silently skips
    the row (``RAISE(IGNORE)``); any other rule shape fails loudly."""
    with engine.begin() as conn:
        for stmt in research_status_statements():
            if stmt.upper().startswith("CREATE OR REPLACE RULE"):
                match = _RULE_RE.fullmatch(" ".join(stmt.split()))
                assert match is not None, f"unemulated rule: {stmt}"
                name, verb, table = match.groups()
                conn.execute(
                    text(
                        f"CREATE TRIGGER {name} BEFORE {verb.upper()} ON {table} "
                        "BEGIN SELECT RAISE(IGNORE); END"
                    )
                )
                continue
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
