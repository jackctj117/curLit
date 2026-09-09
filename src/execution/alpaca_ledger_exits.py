"""Allocation-limited close-only recovery using the existing exit rules (CL-0deu.3).

Explicit rollout mode, no entries. The account-wide PostgreSQL advisory lock
fences cooperating upgraded workers throughout one cycle without holding a SQL
transaction across broker requests. Old writers must be stopped at cutover.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from scripts.alpaca_recovery_report import capture
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.execution.alpaca_equity_exit import EquityExitConfig, evaluate_equity_exit
from src.execution.alpaca_fill_ledger import Ledger, timestamp
from src.execution.alpaca_ledger_reconcile import ingest_snapshot, original_client_id
from src.execution.alpaca_options_exit import OptionsExitConfig, evaluate_exit
from src.execution.alpaca_recovery import TERMINAL, EvidenceError, PaperEvidenceClient, quantity

logger = logging.getLogger(__name__)
Record = dict[str, Any]


def lock_key(account: str) -> int:
    """Stable signed PostgreSQL advisory key, same for both books."""
    return int.from_bytes(hashlib.sha256(account.encode()).digest()[:8], "big", signed=True)


def eligible_allocations(projections: list[Record], snapshot: Record) -> dict[str, Record]:
    """Only exact net conservation with no unexplained asset activity is manageable."""
    positions = snapshot["positions_after"]
    by_asset = {p["asset_id"]: p for p in positions}
    if len(by_asset) != len(positions):
        raise ValueError("duplicate broker position asset")
    grouped: dict[str, list[Record]] = {}
    for projection in projections:
        if projection["evidence_status"] == "fill_verified":
            grouped.setdefault(projection["asset_id"], []).append(projection)
    eligible = {}
    for asset, allocations in grouped.items():
        symbol = allocations[0]["symbol"]
        asset_orders = {
            o["id"]: o for o in snapshot["orders"]["records"] if o.get("asset_id") == asset
        }
        asset_activities = [
            a for a in snapshot["activities"]["records"] if a.get("symbol") == symbol
        ]
        # Isolate the current position lifetime using the full signed broker
        # execution stream. Older aggregate-allocation defects stay unresolved,
        # but need not strand a separately evidenced entry AFTER the book was flat.
        if any(
            a.get("activity_type") != "FILL" or a.get("order_id") not in asset_orders
            for a in asset_activities
        ):
            continue
        broker_net = Decimal(0)
        last_flat = None
        seen = set()
        for activity in sorted(
            asset_activities, key=lambda a: (timestamp(a["transaction_time"]), a["id"])
        ):
            if activity["id"] in seen:
                raise ValueError("duplicate activity in epoch projection")
            seen.add(activity["id"])
            if activity.get("side") not in {"buy", "sell", "sell_short"}:
                raise ValueError("unknown activity side")
            broker_net += quantity(activity["qty"]) * (1 if activity["side"] == "buy" else -1)
            if broker_net == 0:
                last_flat = timestamp(activity["transaction_time"])
        position = by_asset.get(asset)
        actual = quantity(position["qty"]) if position else Decimal(0)
        if broker_net != actual:
            continue
        current = [
            p
            for p in allocations
            if p["allocation"]["signed_quantity"] != 0
            and (last_flat is None or timestamp(p["orders"][0]["submitted_at"]) > last_flat)
        ]
        if any(
            (p["legacy"].get("occ_symbol") or p["legacy"].get("ticker")) == symbol
            and p["evidence_status"] != "fill_verified"
            and (last_flat is None or timestamp(p["legacy"]["submitted_at"]) > last_flat)
            for p in projections
        ):
            continue
        allocations = current + [p for p in allocations if p["allocation"]["signed_quantity"] == 0]
        owned_orders = {o["id"] for p in allocations for o in p["orders"]}
        # Non-trade activities need separate allocation evidence; do not erase
        # exercise, expiry, assignment or manual activity with a net match.
        if any(
            a.get("symbol") == symbol
            and (last_flat is None or timestamp(a["transaction_time"]) > last_flat)
            and (a.get("activity_type") != "FILL" or a.get("order_id") not in owned_orders)
            for a in snapshot["activities"]["records"]
        ):
            continue
        net = sum((p["allocation"]["signed_quantity"] for p in allocations), Decimal(0))
        if actual != net:
            continue
        # Opposing allocated longs/shorts within the same net position require
        # a netting policy; a close must not create an unintended position flip.
        if any(p["allocation"]["signed_quantity"] * actual < 0 for p in allocations):
            continue
        for projection in allocations:
            eligible[f"{projection['book']}:{projection['idea_id']}"] = projection
    return eligible


def close_only_cycle(
    engine: Engine,
    evidence: PaperEvidenceClient,
    trading: Any,
    *,
    book: str,
    cfg: OptionsExitConfig | EquityExitConfig,
) -> dict[str, int]:
    """Reconcile first, permit proven reductions only, and retain unknown attempts."""
    if book not in {"options", "equities"}:
        raise ValueError("unknown book")
    counts = {
        "submitted": 0,
        "held": 0,
        "blocked": 0,
        "pending": 0,
        "closed": 0,
        "unmanaged_allocations": 0,
        "unresolved_allocations": 0,
    }
    account = evidence.account_scope()
    ledger = Ledger(engine, account)
    # A session-scoped lock survives individual commits but is released on
    # disconnect/crash. It is not a broker fence against manual external orders.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as fence:
        if not fence.execute(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": lock_key(account)}
        ).scalar_one():
            return {**counts, "blocked": 1}
        try:
            snapshot = capture(evidence, engine, 100, include_contracts=True)
            if snapshot["account_scope"] != account:
                raise ValueError("account identity changed during reconciliation")
            before = {
                (p["asset_id"], str(quantity(p["qty"]))) for p in snapshot["positions_before"]
            }
            after = {(p["asset_id"], str(quantity(p["qty"]))) for p in snapshot["positions_after"]}
            if before != after or snapshot["internal"] != snapshot["internal_after"]:
                raise ValueError("account/internal inventory changed during snapshot")
            projections = ingest_snapshot(engine, snapshot)
            eligible = eligible_allocations(projections, snapshot)
            with engine.connect() as conn:
                managed = list(
                    conn.execute(
                        text(
                            "SELECT * FROM alpaca_ledger_allocations WHERE account_scope=:a AND book=:b "
                            "AND management_enabled=TRUE"
                        ),
                        {"a": account, "b": book},
                    ).mappings()
                )
            managed_ids = {row["idea_id"] for row in managed}
            counts["unmanaged_allocations"] = sum(
                p["book"] == book
                and p["allocation"]["signed_quantity"] != 0
                and p["idea_id"] not in managed_ids
                for p in eligible.values()
            )
            counts["unresolved_allocations"] = sum(
                p["book"] == book and p["evidence_status"] != "fill_verified" for p in projections
            )
            if counts["unmanaged_allocations"] or counts["unresolved_allocations"]:
                logger.warning(
                    "Ledger allocations require operator attention",
                    extra={"extra_data": {"book": book, **counts}},
                )
            if not trading.is_market_open():
                return {**counts, "market_closed": 1}
            for saved in managed:
                key = f"{book}:{saved['idea_id']}"
                projection = eligible.get(key)
                if projection is None:
                    counts["blocked"] += 1
                    logger.warning(
                        "Allocation reconciliation blocks exit",
                        extra={
                            "extra_data": {"idea_id": saved["idea_id"], "symbol": saved["symbol"]}
                        },
                    )
                    continue
                residual = projection["allocation"]["signed_quantity"]
                if residual == 0:
                    counts["closed"] += 1
                    continue
                asset, symbol, idea = (
                    projection["asset_id"],
                    projection["symbol"],
                    projection["idea_id"],
                )
                with engine.connect() as conn:
                    attempts = list(
                        conn.execute(
                            text(
                                "SELECT a.* FROM alpaca_ledger_attempts a JOIN alpaca_ledger_intents i "
                                "ON i.account_scope=a.account_scope AND i.intent_id=a.intent_id "
                                "WHERE a.account_scope=:a AND i.book=:b AND i.idea_id=:i AND i.purpose='exit'"
                            ),
                            {"a": account, "b": book, "i": idea},
                        ).mappings()
                    )
                unresolved = False
                for attempt in attempts:
                    if attempt["state"] not in TERMINAL:
                        try:
                            order = evidence.order(client_id=attempt["client_order_id"])
                            ledger.observe_order(attempt["client_order_id"], order)
                        except EvidenceError:
                            pass  # Still unknown, NEVER infer failure/flatness.
                        unresolved = True  # Refresh all fills/positions next cycle.
                if unresolved:
                    counts["pending"] += 1
                    continue
                # A bounded retry is a NEW attempt only after terminal evidence.
                # Three attempts per exit policy episode; further retries require
                # an operator review rather than an indefinite rejection loop.
                if len(attempts) >= 3:
                    counts["blocked"] += 1
                    logger.warning("Exit retry budget exhausted: %s", symbol)
                    continue
                with engine.connect() as conn:
                    idea_row = (
                        conn.execute(
                            text(
                                "SELECT ti.*,ge.status AS event_status FROM trade_ideas ti "
                                "LEFT JOIN geo_events ge ON ge.id=ti.geo_event_id WHERE ti.idea_id=:i"
                            ),
                            {"i": idea},
                        )
                        .mappings()
                        .one()
                    )
                row = {
                    **dict(idea_row),
                    **projection["legacy"],
                    "idea_status": idea_row["status"],
                    "entry_price": projection["allocation"]["entry_average"],
                }
                # Preserve the original recorded holding clock on restoration.
                positions = {p["asset_id"]: p for p in evidence.positions()}
                position = positions.get(asset)
                if (
                    position is None
                    or quantity(position["qty"]) * residual <= 0
                    or abs(quantity(position["qty"])) < abs(residual)
                ):
                    counts["blocked"] += 1
                    continue
                # Rule evaluation uses this allocation's actual entry basis,
                # never the aggregate broker average across multiple ideas.
                owned_position = {
                    **position,
                    "qty": str(residual),
                    "avg_entry_price": str(projection["allocation"]["entry_average"]),
                }
                owned_position.pop("unrealized_plpc", None)
                decision: tuple[str, str] | None
                if book == "options":
                    if not isinstance(cfg, OptionsExitConfig):
                        raise TypeError("options exit config required")
                    quote = trading.get_option_quote(symbol)
                    decision = evaluate_exit(row, owned_position, cfg, datetime.now(UTC), quote)
                    if quote[0] is not None and quote[0] <= 0:
                        counts["blocked"] += 1
                        continue
                else:
                    if not isinstance(cfg, EquityExitConfig):
                        raise TypeError("equity exit config required")
                    decision = evaluate_equity_exit(
                        row, owned_position, cfg, datetime.now(UTC), trading.get_stock_quote(symbol)
                    )
                if decision is None:
                    counts["held"] += 1
                    continue
                # Fresh order history: a co-owner or external working order may
                # consume the same inventory even though no fill exists yet.
                history = evidence.history("orders")
                if not history.exhausted or any(
                    o.get("asset_id") == asset and o.get("status") not in TERMINAL
                    for o in history.records
                ):
                    counts["blocked"] += 1
                    continue
                fresh = {p["asset_id"]: p for p in evidence.positions()}.get(asset)
                if fresh is None or quantity(fresh["qty"]) != quantity(position["qty"]):
                    counts["blocked"] += 1
                    continue
                cid = original_client_id(book, idea, exit_order=True)
                if attempts:
                    cid += f"-r{len(attempts)}"
                ledger.create_intent(
                    intent_id=cid,
                    client_id=cid,
                    book=book,
                    idea_id=idea,
                    asset_id=asset,
                    symbol=symbol,
                    purpose="exit",
                    wire_side="sell" if residual > 0 else "buy",
                    qty=abs(residual),
                    multiplier=projection["multiplier"],
                    created_at=datetime.now(UTC),
                    detail={"reason": str(decision[0]), "explanation": decision[1]},
                )
                if not ledger.begin_submission(cid):
                    counts["pending"] += 1
                    continue
                body = {
                    "symbol": symbol,
                    "qty": str(abs(residual)),
                    "side": "sell" if residual > 0 else "buy",
                    "type": "market",
                    "time_in_force": "day",
                    "client_order_id": cid,
                }
                if book == "options":
                    body["position_intent"] = "sell_to_close"
                logger.info(
                    "Submitting verified allocation reduction",
                    extra={
                        "extra_data": {
                            "symbol": symbol,
                            "idea_id": idea,
                            "qty": str(abs(residual)),
                            "client_order_id": cid,
                        }
                    },
                )
                try:
                    # Preserve fractional equity quantities: the old convenience
                    # submitter coerces to int. Endpoint and body are fixed here.
                    order = trading._req("POST", "/v2/orders", json_body=body)
                    ledger.observe_order(cid, order)
                except Exception as exc:
                    logger.error("Exit submission outcome unknown: %s", type(exc).__name__)
                    counts["pending"] += 1
                    return counts  # Any external effect requires a fresh full snapshot.
                counts["submitted"] += 1
                # Never mark a legacy idea/position closed at acknowledgement.
                return counts  # One submission per freshly reconciled cycle.
            logger.info("Close-only ledger cycle: %s", json.dumps(counts, sort_keys=True))
            return counts
        finally:
            fence.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": lock_key(account)})
