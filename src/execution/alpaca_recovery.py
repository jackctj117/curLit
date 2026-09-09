"""Read-only Alpaca evidence collection and conservative recovery proposals.

This is NOT an exit manager or a repair writer (CL-cojs). Even a terminal
order does not authorize another sell. Account snapshots are not atomic with
database snapshots; consumers must revalidate before any separately approved
repair. Raw broker/SQL dictionaries are confined to this audit boundary.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import quote

import httpx

from src.monitoring.logging_setup import LogContext

logger = logging.getLogger(__name__)
Record = dict[str, Any]
# Alpaca REST limits: getallorders-1 and getaccountactivities-2 documentation.
ORDER_PAGE_SIZE = 500
ACTIVITY_PAGE_SIZE = 100
TERMINAL = frozenset({"filled", "canceled", "expired", "rejected"})
WORKING = frozenset(
    {
        "new",
        "accepted",
        "pending_new",
        "partially_filled",
        "held",
        "pending_cancel",
        "pending_replace",
        "accepted_for_bidding",
    }
)


def fingerprint(value: object) -> str:
    """Canonical artifact/row hash, not a substitute for a DB compare-and-swap."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def quantity(value: object) -> Decimal:
    """Preserve signed fractional quantities; missing/nonfinite is never zero."""
    if isinstance(value, bool) or value is None:
        raise ValueError("missing or invalid quantity")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid quantity") from exc
    if not result.is_finite():
        raise ValueError("nonfinite quantity")
    return result


@dataclass
class History:
    records: list[Record] = field(default_factory=list)
    pages: int = 0
    exhausted: bool = False
    reason: str = "not_requested"
    last_cursor: str | None = None


class EvidenceError(RuntimeError):
    """Sanitized failure: never include response bodies, headers or credentials."""


class PaperEvidenceClient:
    """Only GET operations, only the paper host, no redirects or SDK retries.

    ``transport`` is a test boundary. The operational caller supplies only keys;
    neither arbitrary endpoints nor HTTP methods are exposed.
    """

    def __init__(
        self, key: str, secret: str, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        if not key or not secret:
            raise ValueError("paper credentials required")
        self._client = httpx.Client(
            base_url="https://paper-api.alpaca.markets",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            # Bound one read, including unresponsive history endpoints.
            timeout=20.0,
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict[str, str | int] | None = None) -> Any:
        logger.info("Reading paper broker evidence", extra={"extra_data": {"path": path}})
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise EvidenceError(f"transport_{type(exc).__name__}") from None
        if response.status_code != 200:
            raise EvidenceError(f"http_{response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise EvidenceError("invalid_json") from None

    def positions(self) -> list[Record]:
        result = self._get("/v2/positions")
        if not isinstance(result, list) or not all(isinstance(r, dict) for r in result):
            raise EvidenceError("invalid_positions")
        for row in result:
            if not row.get("asset_id") or not row.get("symbol"):
                raise EvidenceError("missing_position_identity")
            quantity(row.get("qty"))
        return result

    def account_scope(self) -> str:
        account = self._get("/v2/account")
        if not isinstance(account, dict) or not account.get("id"):
            raise EvidenceError("missing_account_identity")
        return "alpaca-paper:" + fingerprint(account["id"])

    def order(self, *, broker_id: str | None = None, client_id: str) -> Record:
        if broker_id and broker_id != "recovered":
            result = self._get("/v2/orders/" + quote(broker_id, safe=""))
        else:
            result = self._get("/v2/orders:by_client_order_id", {"client_order_id": client_id})
        if not isinstance(result, dict) or not result.get("id"):
            raise EvidenceError("invalid_order")
        return result

    def contract(self, symbol: str) -> Record:
        """Captured broker contract metadata supplies multiplier, not OCC guesswork."""
        result = self._get("/v2/options/contracts/" + quote(symbol, safe=""))
        if not isinstance(result, dict) or result.get("symbol") != symbol or not result.get("id"):
            raise EvidenceError("invalid_contract_identity")
        if quantity(result.get("size")) <= 0:
            raise EvidenceError("invalid_contract_size")
        return result

    def history(self, kind: str, *, max_pages: int = 100) -> History:
        """Page to an empty response, not a short page; report boundary limits.

        Exhaustion means only the API-visible history was exhausted. It does NOT
        prove lifetime completeness (retention and delayed paper activities).
        The default bounds work to 50k orders / 10k activities per invocation.
        """
        if kind not in {"orders", "activities"} or max_pages < 1:
            raise ValueError("invalid history request")
        path = "/v2/orders" if kind == "orders" else "/v2/account/activities"
        params: dict[str, str | int] = (
            {"status": "all", "limit": ORDER_PAGE_SIZE, "direction": "desc", "nested": "false"}
            if kind == "orders"
            else {"page_size": ACTIVITY_PAGE_SIZE, "direction": "desc"}
        )
        # Paper's order-ID cursor returned recent orders again when the cursor
        # crossed archived history (observed 2026-09-09, CL-cojs). The documented
        # timestamp cursor reaches that history. A FULL page has an unverified
        # timestamp-tie boundary: retain evidence but do not claim completeness.
        cursor_key = "until" if kind == "orders" else "page_token"
        history = History(reason="page_budget_exhausted")
        seen: dict[str, Record] = {}
        cursors: set[str] = set()
        boundary_unknown = False
        for _ in range(max_pages):
            try:
                page = self._get(path, params)
                history.pages += 1
                if not isinstance(page, list) or not all(isinstance(r, dict) for r in page):
                    raise EvidenceError("invalid_page")
                if not page:
                    history.exhausted = not boundary_unknown
                    history.reason = (
                        "timestamp_boundary_unverified"
                        if boundary_unknown
                        else "api_history_exhausted"
                    )
                    break
                for row in page:
                    identity = row.get("id")
                    if not isinstance(identity, str) or not identity:
                        raise EvidenceError("missing_record_id")
                    if identity in seen and seen[identity] != row:
                        raise EvidenceError("conflicting_duplicate_id")
                    if identity not in seen:
                        history.records.append(row)
                        seen[identity] = row
                cursor = page[-1].get("submitted_at") if kind == "orders" else page[-1]["id"]
                if not isinstance(cursor, str) or not cursor:
                    raise EvidenceError("missing_cursor")
                if kind == "orders":
                    try:
                        times = [datetime.fromisoformat(r["submitted_at"]) for r in page]
                        if any(t.tzinfo is None for t in times) or times != sorted(
                            times, reverse=True
                        ):
                            raise ValueError("invalid order timestamps")
                        if history.last_cursor and any(
                            t >= datetime.fromisoformat(history.last_cursor) for t in times
                        ):
                            raise ValueError("ignored order time boundary")
                    except (ValueError, TypeError, KeyError):
                        raise EvidenceError("invalid_order_time_boundary") from None
                    boundary_unknown |= len(page) >= ORDER_PAGE_SIZE
                if cursor in cursors:
                    raise EvidenceError("nonadvancing_cursor")
                cursors.add(cursor)
                history.last_cursor = cursor
                params[cursor_key] = cursor
            except EvidenceError as exc:
                history.reason = str(exc)
                break
        logger.info(
            "History collection finished",
            extra={
                "extra_data": {
                    "kind": kind,
                    "pages": history.pages,
                    "records": len(history.records),
                    "reason": history.reason,
                }
            },
        )
        return history


def fill_evidence(order: Record, activities: list[Record]) -> Record:
    """Cross-check cumulative quantity with uniquely identified executions.

    No P&L is inferred: allocation, multiplier, and complete costs are separate
    requirements. A canceled partially filled order still has executed quantity.
    """
    try:
        filled = quantity(order.get("filled_qty"))
        requested = quantity(order.get("qty"))
        if not Decimal(0) <= filled <= requested or order.get("side") not in {"buy", "sell"}:
            raise ValueError("invalid cumulative quantity")
        unique: dict[str, Record] = {}
        for activity in activities:
            if activity.get("activity_type") != "FILL" or activity.get("order_id") != order["id"]:
                continue
            key = activity.get("id")
            if not isinstance(key, str) or not key:
                raise ValueError("missing activity identity")
            if key in unique and unique[key] != activity:
                raise ValueError("conflicting activity identity")
            unique[key] = activity
        total = Decimal(0)
        cash = Decimal(0)
        for activity in unique.values():
            qty, price = quantity(activity.get("qty")), quantity(activity.get("price"))
            activity_side = activity.get("side")
            if not isinstance(activity_side, str):
                raise ValueError("invalid fill side")
            if (
                qty <= 0
                or price < 0
                or activity.get("symbol") != order.get("symbol")
                # Alpaca FILL activities distinguish opening shorts, whereas
                # the matching order's wire side is sell (Sep 9 MGA capture).
                or {"buy": "buy", "sell": "sell", "sell_short": "sell"}.get(activity_side)
                != order.get("side")
                or not activity.get("transaction_time")
            ):
                raise ValueError("invalid fill evidence")
            total += qty
            cash += qty * price
        return {
            "status": "matched" if total == filled else "quantity_mismatch",
            "order_filled_qty": str(filled),
            "activity_qty": str(total),
            "unfilled_order_qty": str(requested - filled),
            "fill_price_quantity_sum": str(cash),
            "activity_ids": sorted(unique),
            "fees": None,
            "realized_pnl": None,
        }
    except (ValueError, KeyError) as exc:
        return {"status": "invalid_evidence", "reason": str(exc), "realized_pnl": None}


def equity_repair_candidate(
    position: Record,
    rows: list[Record],
    orders: list[Record],
    activities: list[Record],
    idea_ids: list[str],
) -> Record | None:
    """Identify a proposed repair, never authorize adoption from a symbol match.

    Require the stored broker ID AND original client ID, same asset, matching
    individual fills, observed inventory conservation, and no later/external
    activity. API retention remains an explicit limitation of the proposal.
    """
    fills = [a for a in activities if a.get("activity_type") == "FILL"]
    if not fills or len(fills) != len(activities):
        return None  # Corporate actions need their own allocation policy.
    by_id = {o["id"]: o for o in orders}
    try:
        net = Decimal(0)
        seen: set[str] = set()
        for fill in fills:
            if fill.get("id") in seen:
                return None  # Collector deduplicates; ambiguous offline input is refused.
            seen.add(fill["id"])
            order = by_id.get(fill.get("order_id"))
            if order is None or order.get("asset_id") != position["asset_id"]:
                return None
            if fill_evidence(order, activities)["status"] != "matched":
                return None
            if not str(order.get("client_order_id", "")).startswith("curlit-eq-"):
                return None
            net += quantity(fill["qty"]) * (1 if fill["side"] == "buy" else -1)
        if net != quantity(position["qty"]):
            return None
        for row in rows:
            entry = by_id.get(row.get("alpaca_order_id"))
            if (
                entry is None
                or row.get("idea_id") not in idea_ids
                or row.get("exit_reason") != "closed_external"
                or row.get("exit_order_id")
                or row.get("exit_status") != "closed"
                or entry.get("status") != "filled"
                or entry.get("client_order_id") != f"curlit-eq-{row['idea_id']}"
                or entry.get("side") != ("sell" if row.get("side") == "sell_short" else "buy")
                or quantity(entry["filled_qty"]) != abs(net)
                or quantity(row.get("qty")) != abs(net)
            ):
                continue
            if (entry["side"] == "buy") != (net > 0):
                continue
            entry_fills = [a for a in fills if a["order_id"] == entry["id"]]
            earlier_fills = [a for a in fills if a["order_id"] != entry["id"]]
            first_entry = min(datetime.fromisoformat(a["transaction_time"]) for a in entry_fills)
            if any(
                datetime.fromisoformat(a["transaction_time"]) >= first_entry for a in earlier_fills
            ):
                continue
            return {
                "idea_id": row["idea_id"],
                "row_sha256": fingerprint(row),
                "entry_order_id": entry["id"],
                "entry_fills": fill_evidence(entry, activities),
                "signed_qty": str(net),
                "original_submitted_at": row["submitted_at"],
                "proposal": "restore_original_allocation_after_approval_and_fresh_reconciliation",
                "warning": "Original holding clock is retained; overdue exits may execute. "
                "Preserve original closure/estimates in audit before correcting them.",
            }
    except (ValueError, KeyError, TypeError):
        return None
    return None


def build_report(snapshot: Record) -> Record:
    """Produce proposals only. Unknown ownership never becomes authorization."""
    positions = snapshot["positions_after"]
    orders = snapshot["orders"]["records"]
    activities = snapshot["activities"]["records"]
    option_findings: list[Record] = []
    for row in snapshot["internal"]["options"]:
        if row.get("status") != "submitted" or row.get("exit_status") != "submitted":
            continue
        with LogContext(intent_id=row["idea_id"]):
            logger.info("Classifying legacy pending exit")
            lookup = snapshot["exit_lookups"].get(row["idea_id"], {})
            order = lookup.get("order")
            finding: Record = {
                "idea_id": row["idea_id"],
                "symbol": row.get("occ_symbol"),
                "row_sha256": fingerprint(row),
                "internal": row,
                "classification": "reconciliation_required",
                "policy": "no_retry_no_accounting_finalization",
                "broker_order": order,
                "lookup_error": lookup.get("error"),
                "positions": [p for p in positions if p.get("symbol") == row.get("occ_symbol")],
            }
            if order:
                identity_ok = (
                    order.get("symbol") == row.get("occ_symbol")
                    and order.get("side") == "sell"
                    and order.get("client_order_id") == f"curlit-exit-{row['idea_id']}"
                )
                finding["identity_verified"] = identity_ok
                finding["fills"] = fill_evidence(order, activities)
                status = order.get("status")
                if not identity_ok:
                    finding["classification"] = "identity_mismatch"
                elif finding["fills"]["status"] != "matched":
                    finding["classification"] = "fill_reconciliation_required"
                elif status in WORKING:
                    finding["classification"] = "working_exit"
                    finding["policy"] = "track_original_order_do_not_resubmit"
                elif status in TERMINAL:
                    finding["classification"] = f"terminal_{status}"
                    finding["policy"] = "reconstruct_owned_residual_before_any_new_attempt"
                else:
                    finding["classification"] = "unknown_or_replaced_order"
            finding["same_contract_internal_rows"] = [
                r
                for r in snapshot["internal"]["options"]
                if r.get("occ_symbol") == row.get("occ_symbol") and r["idea_id"] != row["idea_id"]
            ]
            option_findings.append(finding)

    equity_findings: list[Record] = []
    equity_rows = snapshot["internal"]["equities"]
    # Mirror the existing manager ONLY to identify its blind spots. This is not
    # an ownership test and must not be used to adopt matching symbols.
    managed_symbols = {
        r["ticker"]
        for r in equity_rows
        if r.get("status") == "submitted" and r.get("exit_status") in {None, "submitted"}
    }
    for position in positions:
        symbol = position["symbol"]
        if position.get("asset_class") != "us_equity" or symbol in managed_symbols:
            continue
        related_orders = [o for o in orders if o.get("symbol") == symbol]
        related_activities = [a for a in activities if a.get("symbol") == symbol]
        equity_findings.append(
            {
                "symbol": symbol,
                "asset_id": position["asset_id"],
                "signed_qty": str(quantity(position["qty"])),
                "position": position,
                "classification": "ownership_unresolved",
                "policy": "include_in_risk_no_automatic_adoption_or_exit",
                "internal_rows": [r for r in equity_rows if r.get("ticker") == symbol],
                "broker_orders": related_orders,
                "activities": related_activities,
                "note": "Symbol/client-prefix matches are leads, not allocation proof. "
                "Exercise/assignment and external/manual origin require explicit policy.",
            }
        )
        if snapshot["orders"]["exhausted"] and snapshot["activities"]["exhausted"]:
            candidate = equity_repair_candidate(
                position,
                equity_findings[-1]["internal_rows"],
                related_orders,
                related_activities,
                snapshot["internal"].get("idea_ids", []),
            )
            if candidate is not None:
                equity_findings[-1]["classification"] = "curlit_entry_repair_candidate"
                equity_findings[-1]["repair_candidate"] = candidate

    # Ignore changing marks; compare identities and signed quantities, not NAV.
    def inventory(rows: list[Record]) -> list[tuple[str, str]]:
        return sorted((r["asset_id"], str(quantity(r["qty"]))) for r in rows)

    return {
        "schema_version": "alpaca-recovery-v1",
        "account_scope": snapshot["account_scope"],
        "snapshot_sha256": fingerprint(snapshot),
        "captured_at": snapshot["finished_at"],
        "inventory_stable_during_capture": inventory(snapshot["positions_before"])
        == inventory(positions),
        "internal_stable_during_capture": snapshot["internal"] == snapshot["internal_after"],
        "coverage": {
            k: {a: b for a, b in snapshot[k].items() if a != "records"}
            for k in ("orders", "activities")
        },
        "limitations": [
            "GET-only observation, not an approved or executable repair plan",
            "API exhaustion does not prove lifetime history completeness",
            "Paper non-trade activities can arrive next day",
            "Snapshots are not atomic; re-read rows, orders and positions before repair",
            "No realized P&L asserted without allocations, multipliers and costs",
        ],
        "pending_option_exits": option_findings,
        "unmatched_equity_holdings": equity_findings,
    }


def history_dict(history: History) -> Record:
    return asdict(history)
