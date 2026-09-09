"""Broker-evidence inventory and average-cost realized accounting (CL-0deu.3).

Pure projections are independent of quotes, positions and legacy estimates.
Activities are immutable; order status never substitutes for individual fills.
Unknown costs exclude net realized results without hiding known gross proceeds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from src.execution.alpaca_recovery import TERMINAL, WORKING, fingerprint, quantity

logger = logging.getLogger(__name__)
Record = dict[str, Any]  # Serialized broker/SQL boundary only.
ZERO = Decimal(0)
ONE = Decimal(1)


def timestamp(value: object) -> datetime:
    if not isinstance(value, (str, datetime)):
        raise ValueError("missing execution timestamp")
    result = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if result.tzinfo is None:
        raise ValueError("naive execution timestamp")
    return result.astimezone(UTC)


def side(value: object) -> str:
    if value in {"sell", "sell_short"}:
        return "sell"
    if value == "buy":
        return "buy"
    raise ValueError("unknown execution side")


@dataclass(frozen=True)
class Fill:
    activity_id: str
    order_id: str
    asset_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    at: datetime
    fees: Decimal | None


def parse_fill(activity: Record, order: Record) -> Fill:
    """Asset identity comes from the matched original order, not ticker inference."""
    if activity.get("activity_type") != "FILL" or activity.get("order_id") != order.get("id"):
        raise ValueError("activity/order mismatch")
    if not all(
        isinstance(v, str) and v
        for v in (activity.get("id"), order.get("id"), order.get("asset_id"), order.get("symbol"))
    ):
        raise ValueError("missing fill identity")
    if activity.get("symbol") != order["symbol"] or side(activity.get("side")) != side(
        order.get("side")
    ):
        raise ValueError("fill instrument/side mismatch")
    qty, price = quantity(activity.get("qty")), quantity(activity.get("price"))
    if qty <= 0 or price < 0:
        raise ValueError("invalid fill quantity/price")
    # Fees are intentionally unknown unless explicitly captured on this fill.
    # Later account-level fees require an independently justified allocation.
    fees = quantity(activity["fees"]) if activity.get("fees") is not None else None
    return Fill(
        activity["id"],
        order["id"],
        order["asset_id"],
        order["symbol"],
        side(activity["side"]),
        qty,
        price,
        timestamp(activity.get("transaction_time")),
        fees,
    )


@dataclass(frozen=True)
class Allocation:
    signed_quantity: Decimal
    entry_quantity: Decimal
    exit_quantity: Decimal
    entry_average: Decimal | None
    gross_realized: Decimal
    net_realized: Decimal | None
    costs_status: str
    original_entered_at: datetime | None


def project_allocation(
    entries: list[Fill], exits: list[Fill], *, multiplier: Decimal, entry_side: str
) -> Allocation:
    """Average-cost allocation; covers partial opens/closes and signed shorts.

    Duplicate IDs are ignored only if byte-equivalent parsed fills. A close
    before ownership, overclose, identity conflict or position flip is invalid.
    Equal-time entry/exit ambiguity is rejected rather than inventing ordering.
    """
    if not multiplier.is_finite() or multiplier <= 0 or entry_side not in {"buy", "sell"}:
        raise ValueError("invalid allocation contract")
    unique: dict[str, tuple[Fill, bool]] = {}
    for opening, fills in ((True, entries), (False, exits)):
        for fill in fills:
            if (
                fill.qty <= 0
                or fill.price < 0
                or not fill.qty.is_finite()
                or not fill.price.is_finite()
            ):
                raise ValueError("invalid allocation fill")
            value = (fill, opening)
            if fill.activity_id in unique and unique[fill.activity_id] != value:
                raise ValueError("conflicting fill identity")
            unique[fill.activity_id] = value
    identities = {(f.asset_id, f.symbol) for f, _ in unique.values()}
    if len(identities) > 1:
        raise ValueError("mixed instrument allocation")
    by_time: dict[datetime, set[bool]] = {}
    for fill, opening in unique.values():
        by_time.setdefault(fill.at, set()).add(opening)
    if any(len(kinds) > 1 for kinds in by_time.values()):
        raise ValueError("ambiguous equal-time entry/exit")
    held = entry_qty = exit_qty = cost = gross = fees = ZERO
    unknown_fees = False
    first = None
    sign = ONE if entry_side == "buy" else -ONE
    for fill, opening in sorted(
        unique.values(), key=lambda item: (item[0].at, item[0].activity_id)
    ):
        if (fill.side == entry_side) != opening:
            raise ValueError("allocation side would flip exposure")
        if opening:
            first = first or fill.at
            held += fill.qty
            entry_qty += fill.qty
            cost += fill.qty * fill.price
        else:
            if fill.qty > held or held == 0:
                raise ValueError("exit exceeds owned allocation")
            basis = cost / held
            gross += sign * fill.qty * (fill.price - basis) * multiplier
            cost -= fill.qty * basis
            held -= fill.qty
            exit_qty += fill.qty
        if fill.fees is None:
            unknown_fees = True
        else:
            fees += fill.fees
    # Net realized for a partially closed lot also needs fee allocation. Keep
    # it unknown until completely closed; no total fee assigned twice.
    costs_status = "unknown" if unknown_fees else ("complete" if held == 0 else "unallocated")
    return Allocation(
        sign * held,
        entry_qty,
        exit_qty,
        cost / held if held else None,
        gross,
        gross - fees if costs_status == "complete" else None,
        costs_status,
        first,
    )


def ingest_activity(conn: Connection, account: str, activity: Record) -> None:
    """Replay-safe immutable source record. Conflicts fail the whole transaction."""
    identity = activity.get("id")
    if not isinstance(identity, str) or not identity:
        raise ValueError("missing activity ID")
    digest = fingerprint(activity)
    conn.execute(
        text(
            "INSERT INTO alpaca_ledger_activities(account_scope,activity_id,payload,payload_hash) "
            "VALUES (:a,:i,:p,:h) ON CONFLICT (account_scope,activity_id) DO NOTHING"
        ),
        {"a": account, "i": identity, "p": json.dumps(activity, sort_keys=True), "h": digest},
    )
    saved = conn.execute(
        text(
            "SELECT payload_hash FROM alpaca_ledger_activities WHERE account_scope=:a AND activity_id=:i"
        ),
        {"a": account, "i": identity},
    ).scalar_one()
    if saved != digest:
        raise ValueError("conflicting broker activity revision requires reconciliation")


def record_order(conn: Connection, account: str, client_id: str, order: Record) -> bool:
    """Monotonic broker observation; stale messages ignored, contradictions rejected."""
    row = (
        conn.execute(
            text(
                "SELECT a.*,i.quantity,i.asset_id,i.side FROM alpaca_ledger_attempts a "
                "JOIN alpaca_ledger_intents i ON i.account_scope=a.account_scope AND i.intent_id=a.intent_id "
                "WHERE a.account_scope=:a AND a.client_order_id=:c"
            ),
            {"a": account, "c": client_id},
        )
        .mappings()
        .one()
    )
    if order.get("client_order_id") != client_id or order.get("asset_id") != row["asset_id"]:
        raise ValueError("broker order identity mismatch")
    if side(order.get("side")) != row["side"] or quantity(order.get("qty")) != quantity(
        row["quantity"]
    ):
        raise ValueError("broker order side/quantity mismatch")
    status = str(order.get("status"))
    if status not in TERMINAL | WORKING or not order.get("id"):
        raise ValueError("unrecognized broker order status/identity")
    if row["broker_order_id"] and order["id"] != row["broker_order_id"]:
        raise ValueError("client ID changed broker order identity")
    filled = quantity(order.get("filled_qty"))
    if (
        filled < 0
        or filled > quantity(row["quantity"])
        or (status == "filled" and filled != quantity(row["quantity"]))
    ):
        raise ValueError("invalid cumulative fill quantity")
    updated = timestamp(order.get("updated_at"))
    if row["broker_updated_at"] and updated < timestamp(row["broker_updated_at"]):
        return False
    if filled < quantity(row["filled_quantity"]):
        raise ValueError("cumulative fill regression")
    if row["state"] in TERMINAL and status != row["state"]:
        raise ValueError("terminal order status changed; reconcile broker correction")
    conn.execute(
        text(
            "UPDATE alpaca_ledger_attempts SET broker_order_id=:o,state=:s,filled_quantity=:f,"
            "broker_updated_at=:t,payload=:p WHERE account_scope=:a AND client_order_id=:c"
        ),
        {
            "o": order["id"],
            "s": status,
            "f": str(filled),
            "t": updated,
            "p": json.dumps(order, sort_keys=True),
            "a": account,
            "c": client_id,
        },
    )
    return True


class Ledger:
    """Persist intent BEFORE broker I/O. A timeout never releases its reservation."""

    def __init__(self, engine: Engine, account: str) -> None:
        if not account.startswith("alpaca-paper:"):
            raise ValueError("recovery ledger supports explicitly identified paper accounts only")
        self.engine, self.account = engine, account

    def create_intent(
        self,
        *,
        intent_id: str,
        client_id: str,
        book: str,
        idea_id: str,
        asset_id: str,
        symbol: str,
        purpose: str,
        wire_side: str,
        qty: Decimal,
        multiplier: Decimal,
        created_at: datetime,
        detail: Record,
    ) -> bool:
        if (
            book not in {"options", "equities"}
            or purpose not in {"entry", "exit"}
            or wire_side not in {"buy", "sell"}
            or qty <= 0
            or not qty.is_finite()
            or multiplier <= 0
            or not multiplier.is_finite()
            or not all((intent_id, client_id, idea_id, asset_id, symbol))
        ):
            raise ValueError("invalid intent")
        timestamp(created_at)
        logger.info(
            "Persisting order intent before broker submission",
            extra={"extra_data": {"intent_id": intent_id, "purpose": purpose, "symbol": symbol}},
        )
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO alpaca_ledger_accounts(account_scope) VALUES (:a) "
                    "ON CONFLICT (account_scope) DO NOTHING"
                ),
                {"a": self.account},
            )
            params = {
                "a": self.account,
                "i": intent_id,
                "b": book,
                "idea": idea_id,
                "asset": asset_id,
                "symbol": symbol,
                "purpose": purpose,
                "side": wire_side,
                "qty": str(qty),
                "mult": str(multiplier),
                "at": created_at,
                "detail": json.dumps(detail, sort_keys=True),
            }
            result = conn.execute(
                text(
                    "INSERT INTO alpaca_ledger_intents(account_scope,intent_id,book,idea_id,asset_id,"
                    "symbol,purpose,side,quantity,multiplier,currency,created_at,detail) "
                    "VALUES (:a,:i,:b,:idea,:asset,:symbol,:purpose,:side,:qty,:mult,'USD',:at,:detail) "
                    "ON CONFLICT (account_scope,intent_id) DO NOTHING"
                ),
                params,
            )
            saved = (
                conn.execute(
                    text(
                        "SELECT * FROM alpaca_ledger_intents "
                        "WHERE account_scope=:a AND intent_id=:i"
                    ),
                    params,
                )
                .mappings()
                .one()
            )
            if (
                any(
                    saved[k] != v
                    for k, v in {
                        "book": book,
                        "idea_id": idea_id,
                        "asset_id": asset_id,
                        "symbol": symbol,
                        "purpose": purpose,
                        "side": wire_side,
                    }.items()
                )
                or quantity(saved["quantity"]) != qty
                or quantity(saved["multiplier"]) != multiplier
            ):
                raise ValueError("existing intent differs; cannot reuse identity")
            conn.execute(
                text(
                    "INSERT INTO alpaca_ledger_attempts(account_scope,client_order_id,intent_id,state) "
                    "VALUES (:a,:c,:i,'created') ON CONFLICT (account_scope,client_order_id) DO NOTHING"
                ),
                {"a": self.account, "c": client_id, "i": intent_id},
            )
            matched = conn.execute(
                text(
                    "SELECT intent_id FROM alpaca_ledger_attempts "
                    "WHERE account_scope=:a AND client_order_id=:c"
                ),
                {"a": self.account, "c": client_id},
            ).scalar_one()
            if matched != intent_id:
                raise ValueError("client ID belongs to another intent")
            return result.rowcount == 1

    def begin_submission(self, client_id: str) -> bool:
        """Atomic ownership transition. Only its winner may make the FIRST POST.

        After a crash, created may be sent; submission_unknown must only be
        queried by original ID. Even confirmed 404 cannot prove a lost POST
        will never arrive, so automatic resend is deliberately disallowed.
        """
        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    "UPDATE alpaca_ledger_attempts SET state='submission_unknown' "
                    "WHERE account_scope=:a AND client_order_id=:c AND state='created'"
                ),
                {"a": self.account, "c": client_id},
            )
            return result.rowcount == 1

    def observe_order(self, client_id: str, order: Record) -> bool:
        with self.engine.begin() as conn:
            # Serialize per account; callers retain no transaction during HTTP.
            conn.execute(
                text("UPDATE alpaca_ledger_accounts SET version=version+1 WHERE account_scope=:a"),
                {"a": self.account},
            )
            return record_order(conn, self.account, client_id, order)
