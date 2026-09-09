"""Replay broker evidence into per-idea allocations, never quotes (CL-0deu.3).

All raw activities survive, including non-trade events not yet allocatable.
An unexplained net position mismatch prevents automatic management for that
asset; it does not justify a synthetic expiration fill or external adoption.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.execution.alpaca_fill_ledger import (
    Ledger,
    ingest_activity,
    parse_fill,
    project_allocation,
    timestamp,
)
from src.execution.alpaca_recovery import fill_evidence, fingerprint, quantity

logger = logging.getLogger(__name__)
Record = dict[str, Any]


def original_client_id(book: str, idea_id: str, *, exit_order: bool = False) -> str:
    prefix = "curlit-eq" if book == "equities" else "curlit"
    return f"{prefix}{'-exit' if exit_order else ''}-{idea_id}"


def project_snapshot(snapshot: Record) -> list[Record]:
    """Offline immutable projection with explicit unresolved reasons per legacy row."""
    if not all(snapshot[k].get("exhausted") for k in ("orders", "activities")):
        raise ValueError("incomplete history cannot produce a repair projection")
    orders = snapshot["orders"]["records"]
    activities = snapshot["activities"]["records"]
    by_client = {}
    for order in orders:
        cid = order.get("client_order_id")
        if cid in by_client:
            raise ValueError("duplicate client-order identity")
        by_client[cid] = order
    result = []
    for book in ("options", "equities"):
        for row in snapshot["internal"][book]:
            idea = row["idea_id"]
            projected: Record = {
                "book": book,
                "idea_id": idea,
                "legacy_hash": fingerprint(row),
                "legacy": row,
                "evidence_status": "unresolved",
                "reason": None,
            }
            result.append(projected)
            try:
                if idea not in snapshot["internal"].get("idea_ids", []):
                    raise ValueError("originating idea absent; no automatic ownership")
                cid = original_client_id(book, idea)
                opening = by_client.get(cid)
                if opening is None:
                    raise ValueError("original entry order absent from covered history")
                if row.get("alpaca_order_id") not in {None, "", "recovered", opening["id"]}:
                    raise ValueError("stored entry identity conflicts with original client ID")
                symbol = row.get("occ_symbol") if book == "options" else row.get("ticker")
                if opening.get("symbol") != symbol:
                    raise ValueError("stored entry instrument conflicts with broker")
                if book == "options" and opening.get("asset_class") != "us_option":
                    raise ValueError("option instrument class unavailable")
                if book == "equities" and opening.get("asset_class") != "us_equity":
                    raise ValueError("equity instrument class unavailable")
                if book == "options" and opening.get("side") != "buy":
                    raise ValueError("only proven long option allocations supported")
                exit_id = original_client_id(book, idea, exit_order=True)
                exits = [
                    o
                    for client, o in by_client.items()
                    if client == exit_id
                    or isinstance(client, str)
                    and client.startswith(exit_id + "-r")
                ]
                if row.get("exit_order_id") not in {None, "", "recovered"} and not any(
                    o["id"] == row["exit_order_id"] for o in exits
                ):
                    raise ValueError("stored exit identity missing from original attempts")
                owned_orders = [opening, *exits]
                order_fills = {}
                for order in owned_orders:
                    if order.get("asset_id") != opening.get("asset_id"):
                        raise ValueError("exit asset identity differs from entry")
                    evidence = fill_evidence(order, activities)
                    if evidence.get("status") != "matched":
                        raise ValueError("individual executions do not match order cumulative fill")
                    order_fills[order["id"]] = [
                        parse_fill(a, order)
                        for a in activities
                        if a.get("activity_type") == "FILL" and a.get("order_id") == order["id"]
                    ]
                multiplier = Decimal(1)
                if book == "options":
                    contract = snapshot.get("contracts", {}).get(symbol, {})
                    if (
                        contract.get("id") != opening.get("asset_id")
                        or contract.get("symbol") != symbol
                    ):
                        raise ValueError("broker contract multiplier identity unavailable")
                    multiplier = quantity(contract.get("size"))
                    if multiplier <= 0:
                        raise ValueError("invalid broker contract multiplier")
                projection = project_allocation(
                    order_fills[opening["id"]],
                    [f for o in exits for f in order_fills[o["id"]]],
                    multiplier=multiplier,
                    entry_side=opening["side"],
                )
                projected.update(
                    {
                        "asset_id": opening["asset_id"],
                        "symbol": symbol,
                        "orders": owned_orders,
                        "fills": order_fills,
                        "multiplier": multiplier,
                        "allocation": asdict(projection),
                        "evidence_status": "fill_verified",
                    }
                )
            except (ValueError, KeyError, TypeError) as exc:
                projected["reason"] = str(exc)
    return result


def ingest_snapshot(engine: Engine, snapshot: Record) -> list[Record]:
    """Persist complete snapshot projections; legacy rows are never modified here."""
    projections = project_snapshot(snapshot)
    account = snapshot["account_scope"]
    ledger = Ledger(engine, account)
    logger.info("Persisting immutable account evidence and allocation projections")
    # Intent persistence precedes order observation and can safely be replayed if
    # later evidence conflicts. No network submission is possible in this module.
    for projection in projections:
        if projection["evidence_status"] != "fill_verified":
            continue
        for index, order in enumerate(projection["orders"]):
            ledger.create_intent(
                intent_id=order["client_order_id"],
                client_id=order["client_order_id"],
                book=projection["book"],
                idea_id=projection["idea_id"],
                asset_id=order["asset_id"],
                symbol=order["symbol"],
                purpose="entry" if index == 0 else "exit",
                wire_side=order["side"],
                qty=quantity(order["qty"]),
                multiplier=projection["multiplier"],
                created_at=timestamp(order["submitted_at"]),
                detail={"origin": "broker_backfill"},
            )
            ledger.observe_order(order["client_order_id"], order)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO alpaca_ledger_accounts(account_scope) VALUES (:a) "
                "ON CONFLICT (account_scope) DO NOTHING"
            ),
            {"a": account},
        )
        for activity in snapshot["activities"]["records"]:
            ingest_activity(conn, account, activity)
        for projected in projections:
            if projected["evidence_status"] != "fill_verified":
                # Do not retain yesterday's verified allocation if today's
                # evidence is incomplete or contradictory.
                conn.execute(
                    text(
                        "UPDATE alpaca_ledger_allocations SET evidence_status='unresolved' "
                        "WHERE account_scope=:a AND book=:b AND idea_id=:i"
                    ),
                    {"a": account, "b": projected["book"], "i": projected["idea_id"]},
                )
                continue
            allocation = projected["allocation"]
            for fills in projected["fills"].values():
                for fill in fills:
                    conn.execute(
                        text(
                            "INSERT INTO alpaca_ledger_fills(account_scope,activity_id,broker_order_id,"
                            "asset_id,symbol,side,quantity,price,executed_at,fees) "
                            "VALUES (:a,:i,:o,:asset,:symbol,:side,:qty,:price,:at,:fees) "
                            "ON CONFLICT (account_scope,activity_id) DO NOTHING"
                        ),
                        {
                            "a": account,
                            "i": fill.activity_id,
                            "o": fill.order_id,
                            "asset": fill.asset_id,
                            "symbol": fill.symbol,
                            "side": fill.side,
                            "qty": str(fill.qty),
                            "price": str(fill.price),
                            "at": fill.at,
                            "fees": str(fill.fees) if fill.fees is not None else None,
                        },
                    )
            payload = {k: str(v) if isinstance(v, Decimal) else v for k, v in allocation.items()}
            conn.execute(
                text(
                    "INSERT INTO alpaca_ledger_allocations(account_scope,book,idea_id,asset_id,symbol,"
                    "signed_quantity,entry_quantity,exit_quantity,entry_average,gross_realized,net_realized,"
                    "costs_status,evidence_status,original_entered_at,projection) "
                    "VALUES (:a,:b,:i,:asset,:symbol,:signed_quantity,:entry_quantity,:exit_quantity,"
                    ":entry_average,:gross_realized,:net_realized,:costs_status,'fill_verified',"
                    ":original_entered_at,:projection) ON CONFLICT (account_scope,book,idea_id) DO UPDATE SET "
                    "signed_quantity=excluded.signed_quantity,entry_quantity=excluded.entry_quantity,"
                    "exit_quantity=excluded.exit_quantity,entry_average=excluded.entry_average,"
                    "gross_realized=excluded.gross_realized,net_realized=excluded.net_realized,"
                    "costs_status=excluded.costs_status,evidence_status=excluded.evidence_status,"
                    "projection=excluded.projection"
                ),
                {
                    **payload,
                    "a": account,
                    "b": projected["book"],
                    "i": projected["idea_id"],
                    "asset": projected["asset_id"],
                    "symbol": projected["symbol"],
                    "projection": json.dumps(payload, default=str, sort_keys=True),
                },
            )
        conn.execute(
            text(
                "UPDATE alpaca_ledger_accounts SET snapshot_hash=:h,version=version+1 "
                "WHERE account_scope=:a"
            ),
            {"a": account, "h": fingerprint(snapshot)},
        )
    return projections
