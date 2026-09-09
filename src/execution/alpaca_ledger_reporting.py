"""Fill-only closed performance, never legacy quote estimates (CL-0deu.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


def closed_performance(engine: Engine, cutoff: datetime) -> list[dict[str, Any]]:
    """Gross and net are separate; unavailable costs must not become zero."""
    with engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                text(
                    "SELECT p.account_scope,p.book,p.idea_id,p.symbol,p.gross_realized,p.net_realized,"
                    "p.costs_status,MAX(f.executed_at) AS closed_at FROM alpaca_ledger_allocations p "
                    "JOIN alpaca_ledger_intents i ON i.account_scope=p.account_scope AND i.book=p.book "
                    "AND i.idea_id=p.idea_id AND i.purpose='exit' "
                    "JOIN alpaca_ledger_attempts a ON a.account_scope=i.account_scope AND a.intent_id=i.intent_id "
                    "JOIN alpaca_ledger_fills f ON f.account_scope=a.account_scope AND f.broker_order_id=a.broker_order_id "
                    "WHERE p.evidence_status='fill_verified' AND p.signed_quantity=0 AND p.entry_quantity>0 "
                    "GROUP BY p.account_scope,p.book,p.idea_id,p.symbol,p.gross_realized,p.net_realized,p.costs_status "
                    "HAVING MAX(f.executed_at)>=:cut ORDER BY closed_at"
                ),
                {"cut": cutoff},
            ).mappings()
        ]
