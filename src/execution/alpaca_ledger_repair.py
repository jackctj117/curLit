"""Compare-and-swap historical corrections after operator-approved cutover (CL-koeg).

No broker methods here. Caller must stop legacy writers, preserve a restored
backup, and freshly capture broker state. Every original legacy value survives
in the append-only repair audit; unknown net P&L remains NULL.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.execution.alpaca_ledger_exits import eligible_allocations, lock_key
from src.execution.alpaca_ledger_reconcile import ingest_snapshot
from src.execution.alpaca_recovery import fingerprint, quantity

logger = logging.getLogger(__name__)
Record = dict[str, Any]
TABLES = {"options": "alpaca_option_orders", "equities": "alpaca_equity_orders"}
SELECT_ROWS = {
    "options": "SELECT * FROM alpaca_option_orders WHERE idea_id=:i",
    "equities": "SELECT * FROM alpaca_equity_orders WHERE idea_id=:i",
}


def apply_repairs(
    engine: Engine,
    baseline: Record,
    current: Record,
    *,
    restore_equities: set[str],
    restore_options: set[str] | None = None,
) -> Record:
    """Approved IDs are explicit; cannot adopt a ticker or invent an entry clock."""
    if baseline["account_scope"] != current["account_scope"]:
        raise ValueError("repair account identity changed")
    account = current["account_scope"]
    approved = {"equities": restore_equities, "options": restore_options or set()}
    old = {(book, row["idea_id"]): row for book in TABLES for row in baseline["internal"][book]}
    latest = {(book, row["idea_id"]): row for book in TABLES for row in current["internal"][book]}
    projections = ingest_snapshot(engine, current)
    eligible = eligible_allocations(projections, current)
    changes = []
    for (book, idea), row in old.items():
        restoring = idea in approved[book]
        closing = book == "options" and row.get("exit_status") == "submitted"
        if not restoring and not closing:
            continue
        projection = eligible.get(f"{book}:{idea}")
        if projection is None:
            raise ValueError(f"repair target lacks current verified allocation: {book}:{idea}")
        allocation = projection["allocation"]
        after = dict(row)
        if restoring:
            if row.get("exit_reason") != "closed_external" or allocation["signed_quantity"] == 0:
                raise ValueError("restoration does not match an original closed_external holding")
            if abs(allocation["signed_quantity"]) != quantity(row["qty"]):
                raise ValueError("restoration quantity differs from original allocation")
            after.update(
                exit_status=None,
                exit_reason=None,
                exit_order_id=None,
                pnl_pct=None,
                exited_at=None,
            )
            after["exit_price" if book == "equities" else "exit_premium"] = None
        else:
            if allocation["signed_quantity"] != 0 or allocation["exit_quantity"] <= 0:
                raise ValueError("pending exit has not closed its allocation")
            exits = projection["orders"][1:]
            if any(o["status"] != "filled" for o in exits):
                raise ValueError("legacy pending exit is not fully filled")
            fills = [fill for order in exits for fill in projection["fills"][order["id"]]]
            proceeds = sum((f.qty * f.price for f in fills), Decimal(0)) * projection["multiplier"]
            after.update(
                exit_status="closed",
                exit_premium=str(proceeds),
                pnl_pct=None,
                exited_at=max(f.at for f in fills).isoformat(),
            )
        identity = fingerprint({"account": account, "book": book, "idea": idea, "before": row})
        changes.append((identity, book, idea, row, after))
    for book, ids in approved.items():
        if not ids.issubset({idea for _, b, idea, _, _ in changes if b == book}):
            raise ValueError("approved restoration target absent")
    applied = replayed = 0
    logger.info("Applying approved historical corrections with row locks and original-value audit")
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": lock_key(account)})
        for identity, book, idea, before, after in changes:
            # Table identifier selects one of two constants, never external input.
            suffix = " FOR UPDATE" if engine.dialect.name == "postgresql" else ""
            row = dict(conn.execute(text(SELECT_ROWS[book] + suffix), {"i": idea}).mappings().one())
            prior = conn.execute(
                text("SELECT after_payload FROM alpaca_ledger_repairs WHERE repair_id=:r"),
                {"r": identity},
            ).scalar_one_or_none()
            if prior is not None:
                if fingerprint(row) != fingerprint(json.loads(prior)):
                    raise ValueError("already-repaired row changed; refuse stale replay")
                replayed += 1
                continue
            if fingerprint(row) != fingerprint(before) or fingerprint(
                latest[(book, idea)]
            ) != fingerprint(before):
                raise ValueError("legacy row changed since approved baseline")
            if book == "equities":
                conn.execute(
                    text(
                        "UPDATE alpaca_equity_orders SET exit_status=NULL,exit_reason=NULL,"
                        "exit_order_id=NULL,exit_price=NULL,pnl_pct=NULL,exited_at=NULL WHERE idea_id=:i"
                    ),
                    {"i": idea},
                )
            elif idea in approved[book]:
                conn.execute(
                    text(
                        "UPDATE alpaca_option_orders SET exit_status=NULL,exit_reason=NULL,"
                        "exit_order_id=NULL,exit_premium=NULL,pnl_pct=NULL,exited_at=NULL WHERE idea_id=:i"
                    ),
                    {"i": idea},
                )
            else:
                conn.execute(
                    text(
                        "UPDATE alpaca_option_orders SET exit_status='closed',exit_premium=:p,"
                        "pnl_pct=NULL,exited_at=:at WHERE idea_id=:i"
                    ),
                    {"i": idea, "p": after["exit_premium"], "at": after["exited_at"]},
                )
            actual_after = dict(conn.execute(text(SELECT_ROWS[book]), {"i": idea}).mappings().one())
            conn.execute(
                text(
                    "INSERT INTO alpaca_ledger_repairs(repair_id,account_scope,book,idea_id,"
                    "before_payload,after_payload,snapshot_hash,applied_at) "
                    "VALUES (:r,:a,:b,:i,:before,:after,:h,:at)"
                ),
                {
                    "r": identity,
                    "a": account,
                    "b": book,
                    "i": idea,
                    "before": json.dumps(before, default=str, sort_keys=True),
                    "after": json.dumps(actual_after, default=str, sort_keys=True),
                    "h": fingerprint(current),
                    "at": datetime.now(UTC),
                },
            )
            applied += 1
        # Existing active allocations and explicitly approved restorations retain
        # management. Other closed/external rows are not automatically adopted.
        for projection in eligible.values():
            row = projection["legacy"]
            restoring = projection["idea_id"] in approved[projection["book"]]
            if projection["allocation"]["signed_quantity"] == 0 or not (
                restoring or row.get("exit_status") in {None, "submitted", "unsellable"}
            ):
                continue
            conn.execute(
                text(
                    "UPDATE alpaca_ledger_allocations SET management_enabled=TRUE "
                    "WHERE account_scope=:a AND book=:b AND idea_id=:i"
                ),
                {"a": account, "b": projection["book"], "i": projection["idea_id"]},
            )
        conn.execute(
            text(
                "UPDATE alpaca_ledger_accounts SET entries_paused=TRUE,version=version+1 "
                "WHERE account_scope=:a"
            ),
            {"a": account},
        )
    return {
        "applied": applied,
        "replayed": replayed,
        "entries_paused": True,
        "net_costs": "unknown",
        "snapshot_hash": fingerprint(current),
    }
