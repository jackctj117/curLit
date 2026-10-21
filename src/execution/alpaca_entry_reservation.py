"""Atomic entry capacity in the existing Alpaca ledger (CL-0deu.1 / CL-0deu.3).

This persistence primitive is NOT an enabled entry writer. Callers must capture
account identity/version before fresh broker reads, use the shared account
writer fence, reconcile terminal fills, and pass the final halt gate before
POST. Existing close-only operation is unchanged. No automatic account resume.
The local version fence does not make broker order/position reads atomic.
Working partial fills anywhere in the account block this primitive until
terminal reconciliation; enabling it still requires the coherent snapshot path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.execution.alpaca_exposure import (
    option_underlying,
    position_quantity,
    require_positive_int,
    validated_positions,
    working_order_quantities,
)
from src.execution.alpaca_fill_ledger import Ledger, timestamp
from src.execution.alpaca_recovery import TERMINAL, WORKING, fingerprint, quantity

logger = logging.getLogger(__name__)


class ReservationBlockedError(ValueError):
    """No new capacity was committed; the caller must not POST."""


@dataclass(frozen=True)
class EntryProposal:
    book: str
    idea_id: str
    asset_id: str
    symbol: str
    underlying: str
    side: str
    qty: Decimal
    multiplier: Decimal

    @property
    def client_id(self) -> str:
        prefix = "curlit-eq-" if self.book == "equities" else "curlit-"
        return prefix + self.idea_id


@dataclass(frozen=True)
class EntryLimits:
    """Exact configured count caps; no new thresholds or risk policy."""

    max_per_day: int
    max_per_hour: int
    max_contracts_per_symbol: int = 0
    max_contracts_per_underlying: int = 0
    max_positions_per_ticker: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            require_positive_int(name, value, allow_zero=True)


@dataclass(frozen=True)
class EntrySnapshot:
    account_scope: str
    # Read BEFORE broker I/O. A different worker's reservation or observed
    # order invalidates this version rather than mixing stale/current state.
    ledger_version: int
    positions: object
    working_orders: object
    captured_at: datetime
    valid_until: datetime


def _pace_count(conn: Connection, ledger: Ledger, proposal: EntryProposal, since: datetime) -> int:
    # Explicit SQL alternatives, not untrusted interpolated identifiers. UNION
    # avoids counting a migrated legacy row and its ledger intent twice.
    sql = (
        "SELECT COUNT(*) FROM (SELECT idea_id FROM alpaca_ledger_intents "
        "WHERE account_scope=:a AND book=:b AND purpose='entry' AND created_at>=:since UNION "
        "SELECT idea_id FROM alpaca_option_orders WHERE status IN ('submitted','reserved') "
        "AND submitted_at >= :since) AS entries"
        if proposal.book == "options"
        else "SELECT COUNT(*) FROM (SELECT idea_id FROM alpaca_ledger_intents "
        "WHERE account_scope=:a AND book=:b AND purpose='entry' AND created_at>=:since UNION "
        "SELECT idea_id FROM alpaca_equity_orders WHERE status IN ('submitted','reserved') "
        "AND submitted_at >= :since) AS entries"
    )
    return int(
        conn.execute(
            text(sql), {"a": ledger.account, "b": proposal.book, "since": since}
        ).scalar_one()
    )


def _recovery_stub(conn: Connection, ledger: Ledger, proposal: EntryProposal, at: datetime) -> None:
    """Intent stub, NOT submitted/fill-verified/management-enabled.

    Legacy rows are globally keyed by idea, unlike account-scoped ledger rows.
    A collision must roll back the whole reservation, never silently adopt it.
    Account-scoped recovery captures include this row; an absent/unfilled broker
    order remains unresolved and never grants automatic exit management.
    """
    params = {
        "idea": proposal.idea_id,
        "symbol": proposal.symbol,
        "ticker": proposal.underlying,
        "qty": str(proposal.qty),
        "at": at,
        "side": "buy" if proposal.side == "buy" else "sell_short",
        "opt": "call" if proposal.symbol[-9:-8] == "C" else "put",
        "detail": json.dumps(
            {
                "account_scope": ledger.account,
                "client_order_id": proposal.client_id,
                "evidence_status": "intent_only",
            },
            sort_keys=True,
        ),
    }
    if proposal.book == "options":
        conn.execute(
            text(
                "INSERT INTO alpaca_option_orders(idea_id,ticker,occ_symbol,opt_type,qty,status,detail,submitted_at) "
                "VALUES (:idea,:ticker,:symbol,:opt,:qty,'reserved',:detail,:at)"
            ),
            params,
        )
    else:
        conn.execute(
            text(
                "INSERT INTO alpaca_equity_orders(idea_id,ticker,side,qty,status,detail,submitted_at) "
                "VALUES (:idea,:symbol,:side,:qty,'reserved',:detail,:at)"
            ),
            params,
        )


def reserve_entry(
    ledger: Ledger,
    proposal: EntryProposal,
    snapshot: EntrySnapshot,
    limits: EntryLimits,
    *,
    now: datetime,
) -> str:
    """Commit capacity + stable intent + attempt + recovery stub atomically.

    Returns the stable client ID; exceptions authorize no submission. This
    function performs zero broker/network operations. Unknown local attempts
    reserve their unfilled quantity even when absent from the broker snapshot.
    """
    now = timestamp(now)
    if not timestamp(snapshot.captured_at) <= now <= timestamp(snapshot.valid_until):
        raise ReservationBlockedError("snapshot_expired_or_future")
    if snapshot.account_scope != ledger.account:
        raise ReservationBlockedError("account_identity_mismatch")
    require_positive_int("ledger_version", snapshot.ledger_version, allow_zero=True)
    if proposal.book not in {"options", "equities"} or proposal.side not in {"buy", "sell"}:
        raise ReservationBlockedError("invalid_proposal")
    if (
        not proposal.asset_id
        or not proposal.idea_id
        or not proposal.qty.is_finite()
        or proposal.qty <= 0
        or proposal.qty != proposal.qty.to_integral_value()
        or not proposal.symbol
        or proposal.symbol != proposal.symbol.strip().upper()
    ):
        raise ReservationBlockedError("invalid_proposal")
    if not proposal.multiplier.is_finite() or proposal.multiplier <= 0:
        raise ReservationBlockedError("invalid_multiplier")
    if proposal.book == "equities" and (
        proposal.multiplier != 1 or limits.max_positions_per_ticker > 1
    ):
        raise ReservationBlockedError("unsupported_equity_allocation_policy")
    kind = "us_option" if proposal.book == "options" else "us_equity"
    if proposal.book == "options" and option_underlying(proposal.symbol) != proposal.underlying:
        raise ReservationBlockedError("underlying_identity_mismatch")
    if proposal.book == "options" and (
        # This entry path supports standard 100-share contracts only, matching
        # its premium sizing. Adjusted deliverables require separate metadata
        # and risk support, not a silently different ledger multiplier.
        proposal.side != "buy"
        or proposal.multiplier != 100
        or any(c.isdigit() for c in proposal.underlying)
    ):
        raise ReservationBlockedError("unsupported_option_entry")
    if proposal.book == "equities" and proposal.symbol != proposal.underlying:
        raise ReservationBlockedError("underlying_identity_mismatch")
    positions = validated_positions(snapshot.positions, asset_class=kind)
    working = working_order_quantities(snapshot.working_orders, asset_class=kind)
    assert isinstance(snapshot.working_orders, list)  # validated above
    by_client = {}
    for order in snapshot.working_orders:
        # An observed fill can precede its position visibility. Without a
        # broker-coherent snapshot, remaining-only arithmetic could undercount
        # even when our stored cumulative fill agrees with this order response.
        if quantity(order["filled_qty"]) > 0:
            raise ReservationBlockedError("working_partial_fill_requires_reconciliation")
        client_id = order.get("client_order_id")
        if not isinstance(client_id, str) or not client_id or client_id in by_client:
            raise ReservationBlockedError("working_client_identity_unavailable")
        by_client[client_id] = order
    exposure = dict(working)
    for position in positions:
        symbol = position["symbol"]
        if proposal.book == "options" and any(c.isdigit() for c in option_underlying(symbol)):
            raise ReservationBlockedError("adjusted_underlying_unavailable")
        exposure[symbol] = exposure.get(symbol, Decimal(0)) + abs(
            position_quantity(position["qty"])
        )
    logger.info(
        "Reserving entry capacity before broker submission",
        extra={
            "extra_data": {
                "client_order_id": proposal.client_id,
                "account_scope": ledger.account,
                "snapshot_version": snapshot.ledger_version,
            }
        },
    )
    with ledger.engine.begin() as conn:
        # Lock halt state as well as account capacity: a concurrent halt wins
        # either before this transaction or immediately after its reservation.
        # Final pre-POST halt verification remains mandatory for the caller.
        if (
            conn.execute(
                text("UPDATE trading_halt_state SET version=version WHERE id=1 AND mode='ACTIVE'")
            ).rowcount
            != 1
        ):
            raise ReservationBlockedError("entry_halted")
        if (
            conn.execute(
                text(
                    "UPDATE alpaca_ledger_accounts SET version=version+1 WHERE account_scope=:a "
                    "AND version=:v AND entries_paused=FALSE"
                ),
                {"a": ledger.account, "v": snapshot.ledger_version},
            ).rowcount
            != 1
        ):
            raise ReservationBlockedError("account_paused_missing_or_snapshot_stale")
        locals_ = conn.execute(
            text(
                "SELECT i.*,a.client_order_id,a.broker_order_id,a.state,a.filled_quantity FROM alpaca_ledger_intents i "
                "JOIN alpaca_ledger_attempts a ON a.account_scope=i.account_scope AND a.intent_id=i.intent_id "
                "WHERE i.account_scope=:a AND i.book=:b AND i.purpose='entry'"
            ),
            {"a": ledger.account, "b": proposal.book},
        ).mappings()
        for local in locals_:
            if local["state"] in TERMINAL:
                continue
            if proposal.book == "options" and any(
                c.isdigit() for c in option_underlying(local["symbol"])
            ):
                raise ReservationBlockedError("adjusted_local_underlying_unavailable")
            if local["state"] not in WORKING | {"created", "submission_unknown"}:
                raise ReservationBlockedError("local_order_state_unknown")
            if local["client_order_id"] in by_client:
                order = by_client[local["client_order_id"]]
                if (
                    order["symbol"] != local["symbol"]
                    or order.get("asset_id") != local["asset_id"]
                    or (local["broker_order_id"] and order.get("id") != local["broker_order_id"])
                    or order["side"] != local["side"]
                    or quantity(order["qty"]) != quantity(local["quantity"])
                    or quantity(order["filled_qty"]) != quantity(local["filled_quantity"])
                ):
                    raise ReservationBlockedError("working_local_identity_mismatch")
                continue
            if quantity(local["filled_quantity"]) > 0:
                raise ReservationBlockedError("partial_fill_missing_from_working_snapshot")
            remaining = quantity(local["quantity"]) - quantity(local["filled_quantity"])
            if remaining < 0:
                raise ReservationBlockedError("local_fill_exceeds_quantity")
            exposure[local["symbol"]] = exposure.get(local["symbol"], Decimal(0)) + remaining
        if (
            _pace_count(
                conn, ledger, proposal, now.replace(hour=0, minute=0, second=0, microsecond=0)
            )
            >= limits.max_per_day
        ):
            raise ReservationBlockedError("daily_capacity_exhausted")
        if _pace_count(conn, ledger, proposal, now - timedelta(hours=1)) >= limits.max_per_hour:
            raise ReservationBlockedError("hourly_capacity_exhausted")
        same = exposure.get(proposal.symbol, Decimal(0))
        if proposal.book == "options":
            underlying = sum(
                (
                    qty
                    for symbol, qty in exposure.items()
                    if option_underlying(symbol) == proposal.underlying
                ),
                Decimal(0),
            )
            if (
                same + proposal.qty > limits.max_contracts_per_symbol
                or underlying + proposal.qty > limits.max_contracts_per_underlying
            ):
                raise ReservationBlockedError("contract_capacity_exhausted")
        elif int(same > 0) >= limits.max_positions_per_ticker:
            raise ReservationBlockedError("ticker_capacity_exhausted")
        if not ledger.create_intent(
            intent_id=proposal.client_id,
            client_id=proposal.client_id,
            book=proposal.book,
            idea_id=proposal.idea_id,
            asset_id=proposal.asset_id,
            symbol=proposal.symbol,
            purpose="entry",
            wire_side=proposal.side,
            qty=proposal.qty,
            multiplier=proposal.multiplier,
            created_at=now,
            detail={"snapshot_hash": fingerprint(asdict(snapshot)), "limits": asdict(limits)},
            connection=conn,
        ):
            raise ReservationBlockedError("existing_intent_requires_reconciliation")
        _recovery_stub(conn, ledger, proposal, now)
    return proposal.client_id
