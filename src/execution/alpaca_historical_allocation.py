"""Conservative whole-book historical closure allocation (CL-sweu).

Only complete, noninterleaved, curLit-owned liquidation orders qualify. Raw
fills/orders are never split or rewritten. Cash proceeds are apportioned across
proven residual lots using an explicit pro-rata order-cash convention; this is
accounting attribution, not a claim of distinct per-idea broker fill prices.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_DOWN, Decimal, localcontext
from typing import Any

from src.execution.alpaca_fill_ledger import timestamp
from src.execution.alpaca_recovery import fill_evidence, fingerprint, quantity

Record = dict[str, Any]  # Broker/projection boundary.
ZERO = Decimal(0)
# Eight decimal places preserve fractional-share cash allocations below a cent.
# The final deterministic lot receives the exact residual, preserving source cash.
CASH_QUANTUM = Decimal("0.00000001")


def allocate_cash(total: Decimal, weights: dict[str, Decimal]) -> dict[str, Decimal]:
    """Deterministic pro-rata policy with exact cash conservation, not new fills."""
    if (
        not total.is_finite()
        or not weights
        or any(not q.is_finite() or q <= 0 for q in weights.values())
    ):
        raise ValueError("invalid cash allocation")
    with localcontext() as context:
        context.prec = 50  # Guard digits for division; persisted allocations use CASH_QUANTUM.
        ordered = sorted(weights)
        denominator = sum(weights.values(), ZERO)
        result = {
            i: (total * weights[i] / denominator).quantize(CASH_QUANTUM, rounding=ROUND_DOWN)
            for i in ordered[:-1]
        }
        result[ordered[-1]] = total - sum(result.values(), ZERO)
        assert sum(result.values(), ZERO) == total
        return result


def resolve_aggregate_history(snapshot: Record, projections: list[Record]) -> None:
    """Augment closed projections only when the complete inventory path is proven."""
    entries: dict[str, Record] = {}
    exits: dict[str, Record] = {}
    for p in projections:
        # These fields exist only after identity, fill coverage and multiplier checks.
        if "orders" not in p:
            continue
        entries[p["orders"][0]["id"]] = p
        for order in p["orders"][1:]:
            if order["id"] in exits and exits[order["id"]] is not p:
                raise ValueError("historical exit has conflicting original owners")
            exits[order["id"]] = p
    orders = {o["id"]: o for o in snapshot["orders"]["records"]}
    positions = {p["asset_id"]: quantity(p["qty"]) for p in snapshot["positions_after"]}
    if len(positions) != len(snapshot["positions_after"]):
        raise ValueError("duplicate position identity")
    by_asset: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for a in snapshot["activities"]["records"]:
        if a.get("activity_type") == "FILL" and a.get("order_id") in orders:
            by_asset[orders[a["order_id"]]["asset_id"]][a["order_id"]].append(a)
    nontrade_symbols = {
        a.get("symbol")
        for a in snapshot["activities"]["records"]
        if a.get("activity_type") != "FILL" and a.get("symbol")
    }
    nontrade_symbols.update(
        a.get("symbol")
        for a in snapshot["activities"]["records"]
        if a.get("activity_type") == "FILL" and a.get("order_id") not in orders
    )
    if any(
        a.get("activity_type") not in {"FILL", "FEE"}
        and a.get("qty") not in {None, "0", 0}
        and not a.get("symbol")
        for a in snapshot["activities"]["records"]
    ):
        return  # Unknown inventory-changing activity cannot be assigned to another asset.
    for asset, groups in by_asset.items():
        symbol = orders[next(iter(groups))]["symbol"]
        if symbol in nontrade_symbols:
            continue  # Expiry/exercise/splits require their own activity policy.
        signed_total = sum(
            (
                quantity(a["qty"]) * (1 if a.get("side") == "buy" else -1)
                for fills in groups.values()
                for a in fills
            ),
            ZERO,
        )
        if signed_total != positions.get(asset, ZERO):
            continue  # Unknown starting/current inventory invalidates a whole-book proof.
        spans = sorted(
            (
                min(timestamp(a["transaction_time"]) for a in fills),
                max(timestamp(a["transaction_time"]) for a in fills),
                identity,
            )
            for identity, fills in groups.items()
        )
        if any(spans[i][0] <= spans[i - 1][1] for i in range(1, len(spans))):
            continue  # Cannot reorder interleaved or tied executions by order.
        held: dict[str, Decimal] = {}
        basis: dict[str, Decimal] = {}
        gross: dict[str, Decimal] = {}
        original: dict[str, Record] = {}
        net = ZERO
        tainted = False
        for _first, last, identity in spans:
            order, fills = orders[identity], groups[identity]
            if fill_evidence(order, fills).get("status") != "matched":
                break
            qty = sum((quantity(f["qty"]) for f in fills), ZERO)
            cash = sum((quantity(f["qty"]) * quantity(f["price"]) for f in fills), ZERO)
            if order.get("side") not in {"buy", "sell"}:
                break
            sign = Decimal(1) if order["side"] == "buy" else Decimal(-1)
            before_net = net
            net += qty * sign
            opening, closing = entries.get(identity), exits.get(identity)
            if opening is not None and not tainted:
                key = f"{opening['book']}:{opening['idea_id']}"
                if held and before_net * sign < 0:
                    tainted = True  # Opposing strategy inventory needs netting policy.
                else:
                    held[key] = held.get(key, ZERO) + qty
                    basis[key] = basis.get(key, ZERO) + cash * opening["multiplier"]
                    gross.setdefault(key, ZERO)
                    original[key] = opening
            elif closing is not None and not tainted:
                key = f"{closing['book']}:{closing['idea_id']}"
                available = held.get(key, ZERO)
                if before_net * sign >= 0:
                    tainted = True
                elif qty <= available:
                    unit_cost = basis[key] / available
                    gross[key] += -sign * (cash * closing["multiplier"] - qty * unit_cost)
                    held[key] -= qty
                    basis[key] -= qty * unit_cost
                elif (
                    net == 0
                    and qty == sum(held.values(), ZERO)
                    and order.get("status") == "filled"
                    and len(held) > 1
                ):
                    weights = {i: q for i, q in held.items() if q > 0}
                    if any(
                        p["asset_id"] != asset or p["multiplier"] != closing["multiplier"]
                        for i, p in original.items()
                        if i in weights
                    ):
                        tainted = True
                    else:
                        proceeds = allocate_cash(cash * closing["multiplier"], weights)
                        for i, residual in weights.items():
                            p = original[i]
                            gross[i] += -sign * (proceeds[i] - basis[i])
                            entry_fills = p["fills"][p["orders"][0]["id"]]
                            entry_qty = sum((f.qty for f in entry_fills), ZERO)
                            detail = {
                                "method": "whole_book_close_pro_rata_cash_v1",
                                "order_id": identity,
                                "original_order_owner": key,
                                "activity_ids": sorted(f["id"] for f in fills),
                                "allocated_quantity": str(residual),
                                "allocated_cash": str(proceeds[i]),
                                "total_quantity": str(qty),
                                "total_cash": str(cash * closing["multiplier"]),
                                "closed_at": last.isoformat(),
                                "evidence_hash": fingerprint(
                                    {"order": order, "fills": sorted(fills, key=lambda f: f["id"])}
                                ),
                                "limitation": "cash attribution convention; raw executions unchanged",
                            }
                            p.update(
                                evidence_status="fill_verified",
                                reason=None,
                                historical_allocation=detail,
                                allocation={
                                    "signed_quantity": ZERO,
                                    "entry_quantity": entry_qty,
                                    "exit_quantity": entry_qty,
                                    "entry_average": None,
                                    "gross_realized": gross[i],
                                    "net_realized": None,
                                    "costs_status": "unknown",
                                    "original_entered_at": min(f.at for f in entry_fills),
                                },
                            )
                            held[i] = basis[i] = ZERO
                else:
                    tainted = True  # Partial aggregate closure is not an ownership oracle.
            else:
                tainted = True  # External fills cannot be assigned to a curLit idea.
            if net == 0:
                held.clear()
                basis.clear()
                gross.clear()
                original.clear()
                tainted = False
