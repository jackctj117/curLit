"""Fill-only closed performance, never legacy quote estimates (CL-0deu.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.execution.alpaca_fee_attribution import fee_summary, project_fees
from src.execution.alpaca_ledger_reconcile import project_snapshot
from src.execution.alpaca_recovery import fingerprint


def accounting_evidence(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Read-only evidence summary; never equate policy attribution with raw fills."""
    projections = project_snapshot(snapshot)
    return {
        "snapshot_hash": fingerprint(snapshot),
        "mode": "read_only",
        "history_scope": "API-visible exhausted pages; not proof of lifetime coverage",
        "fees": fee_summary(project_fees(snapshot["activities"]["records"])),
        "allocations": [
            {
                "book": p["book"],
                "idea_id": p["idea_id"],
                "symbol": p.get("symbol"),
                "evidence_status": p["evidence_status"],
                "reason": p.get("reason"),
                "allocation": p.get("allocation"),
                "historical_allocation": p.get("historical_allocation"),
            }
            for p in projections
        ],
    }


def closed_performance(engine: Engine, cutoff: datetime) -> list[dict[str, Any]]:
    """Gross and net are separate; unavailable costs must not become zero."""
    with engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                text(
                    "SELECT p.account_scope,p.book,p.idea_id,p.symbol,p.gross_realized,p.net_realized,"
                    "p.costs_status,p.closed_at,p.attribution_method FROM alpaca_ledger_allocations p "
                    "WHERE p.evidence_status='fill_verified' AND p.signed_quantity=0 AND p.entry_quantity>0 "
                    "AND p.closed_at>=:cut ORDER BY p.closed_at,p.book,p.idea_id"
                ),
                {"cut": cutoff},
            ).mappings()
        ]
