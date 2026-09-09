"""Execution-linked fees and explicitly unallocated account costs (CL-z97c).

The Trading API preserves a timestamp::UUID activity identity; legacy OCC
fees carry an execution_id (Alpaca account-activities and SSE documentation).
Our captured evidence matches those exact UUIDs. Never infer links from price,
date, ticker or description. Daily aggregate fees stay at account level.
Linked costs are KNOWN costs, not proof that every applicable fee has arrived.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.execution.alpaca_fill_ledger import timestamp
from src.execution.alpaca_recovery import fingerprint, quantity

Record = dict[str, Any]  # Broker/SQL serialization boundary.
logger = logging.getLogger(__name__)
ZERO = Decimal(0)


def execution_reference(activity: Record) -> str | None:
    """Exact canonical UUID suffix only; no permissive substring matching."""
    identity = activity.get("id")
    if not isinstance(identity, str) or identity.count("::") != 1:
        return None
    prefix, suffix = identity.split("::")
    if not prefix.isdigit():
        return None
    try:
        return str(UUID(suffix)) if str(UUID(suffix)) == suffix else None
    except ValueError:
        return None


def project_fees(activities: list[Record]) -> list[Record]:
    """Replay-safe fee dispositions; rebates retain their negative cost sign."""
    unique: dict[str, Record] = {}
    for activity in activities:
        identity = activity.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("activity identity missing")
        if identity in unique and fingerprint(unique[identity]) != fingerprint(activity):
            raise ValueError("conflicting activity identity")
        unique[identity] = activity
    executions: dict[str, list[Record]] = defaultdict(list)
    for activity in unique.values():
        if activity.get("activity_type") == "FILL" and (ref := execution_reference(activity)):
            executions[ref].append(activity)
    result: list[Record] = []
    for activity in unique.values():
        if activity.get("activity_type") != "FEE":
            continue
        row: Record = {
            "fee_activity_id": activity["id"],
            "fill_activity_id": None,
            "currency": activity.get("currency"),
            "known_cost": None,
            "status": "invalid",
            "reason": None,
            "fee_date": activity.get("date"),
            "subtype": activity.get("activity_sub_type"),
            "method": "no_allocation",
            "source_hash": fingerprint(activity),
        }
        result.append(row)
        try:
            amount = quantity(activity.get("net_amount"))
            fee_date = date.fromisoformat(str(activity.get("date")))
            if activity.get("currency") != "USD":
                raise ValueError("fee currency unsupported or unavailable")
            row["known_cost"] = -amount  # Debit is positive cost; a rebate is negative.
            if activity.get("status") != "executed":
                row.update(status="pending", reason="fee not confirmed executed")
                continue
            row["status"] = "unallocated"
            ref = activity.get("execution_id")
            if not ref:
                row["reason"] = "account-level fee has no execution reference"
                continue
            matches = executions.get(ref, []) if isinstance(ref, str) else []
            if len(matches) != 1:
                row["reason"] = "execution reference absent or ambiguous in captured fills"
                continue
            fill = matches[0]
            if (
                fee_date
                < timestamp(fill["transaction_time"])
                .astimezone(ZoneInfo("America/New_York"))
                .date()
            ):
                raise ValueError("fee predates referenced execution")
            if activity.get("symbol") and activity["symbol"] != fill.get("symbol"):
                raise ValueError("fee instrument contradicts referenced execution")
            row.update(
                fill_activity_id=fill["id"],
                status="linked",
                reason=None,
                method="exact_execution_uuid",
            )
        except (ValueError, KeyError, TypeError) as exc:
            row.update(status="invalid", reason=str(exc), fill_activity_id=None)
    return result


def fee_summary(rows: list[Record]) -> Record:
    """Covered confirmed costs conserve linked + unallocated, excluding invalid/pending."""
    linked = sum((r["known_cost"] for r in rows if r["status"] == "linked"), ZERO)
    unallocated = sum((r["known_cost"] for r in rows if r["status"] == "unallocated"), ZERO)
    return {
        "currency": "USD",
        "linked_cost": linked,
        "unallocated_cost": unallocated,
        "confirmed_cost": linked + unallocated,
        "linked_count": sum(r["status"] == "linked" for r in rows),
        "unallocated_count": sum(r["status"] == "unallocated" for r in rows),
        "invalid_count": sum(r["status"] == "invalid" for r in rows),
        "pending_count": sum(r["status"] == "pending" for r in rows),
        "completeness": "captured_only_not_final_billing",
    }


def persist_fees(conn: Connection, account: str, activities: list[Record]) -> Record:
    """Derived projections may gain a link on later capture; raw activity stays immutable."""
    rows = project_fees(activities)
    logger.info("Persisting fee evidence; aggregate charges remain explicitly unallocated")
    for row in rows:
        conn.execute(
            text(
                "INSERT INTO alpaca_ledger_fee_links(account_scope,fee_activity_id,fill_activity_id,"
                "currency,known_cost,status,reason,method,source_hash,payload) "
                "VALUES (:account,:fee_activity_id,:fill_activity_id,:currency,:known_cost,:status,"
                ":reason,:method,:source_hash,:payload) ON CONFLICT(account_scope,fee_activity_id) "
                "DO UPDATE SET fill_activity_id=excluded.fill_activity_id,known_cost=excluded.known_cost,"
                "status=excluded.status,reason=excluded.reason,method=excluded.method,payload=excluded.payload"
            ),
            {
                **row,
                "account": account,
                "known_cost": str(row["known_cost"]) if row["known_cost"] is not None else None,
                "payload": json.dumps(row, default=str, sort_keys=True),
            },
        )
    return fee_summary(rows)
