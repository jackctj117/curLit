"""Atomic ledger reservation oracles; no broker calls or operational state."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from migrations.run import _strip_sql_comments
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from src.execution.alpaca_entry_reservation import (
    EntryLimits,
    EntryProposal,
    EntrySnapshot,
    ReservationBlockedError,
    reserve_entry,
)
from src.execution.alpaca_fill_ledger import Ledger
from src.execution.alpaca_ledger_reconcile import project_snapshot
from src.execution.alpaca_recovery import fingerprint
from tests.unit._trading_halt_fixture import install_trading_halt
from tests.unit.test_alpaca_ledger_intent_logging import broker_order
from tests.unit.test_alpaca_ledger_intent_logging import snapshot as broker_snapshot

AT = datetime(2026, 10, 20, 15, tzinfo=UTC)
ACCOUNT = "alpaca-paper:" + fingerprint("fixture")
PROPOSAL = EntryProposal(
    "options", "first", "asset", "RTX261218C00105000", "RTX", "buy", Decimal(1), Decimal(100)
)
LIMITS = EntryLimits(10, 10, 2, 2, 1)


@pytest.fixture
def engine(tmp_path):
    result = create_engine(f"sqlite:///{tmp_path / 'reservations.db'}")
    with result.begin() as conn:
        for name in (
            "014_alpaca_option_orders.sql",
            "019_alpaca_equity_orders.sql",
            "022_alpaca_fill_ledger.sql",
        ):
            for statement in _strip_sql_comments((Path("migrations") / name).read_text()).split(
                ";"
            ):
                if statement.strip():
                    conn.execute(text(statement))
        conn.execute(
            text(
                "INSERT INTO alpaca_ledger_accounts(account_scope,entries_paused) VALUES (:a,FALSE)"
            ),
            {"a": ACCOUNT},
        )
    install_trading_halt(result)
    yield result
    result.dispose()


def snapshot(engine, *, working=None, positions=None, at=AT):
    with engine.connect() as conn:
        version = conn.execute(
            text("SELECT version FROM alpaca_ledger_accounts WHERE account_scope=:a"),
            {"a": ACCOUNT},
        ).scalar_one()
    # One-second fixture validity is only a test vector, not a production TTL.
    return EntrySnapshot(
        ACCOUNT, version, positions or [], working or [], at, at + timedelta(seconds=1)
    )


def saved(engine):
    with engine.connect() as conn:
        return (
            conn.execute(text("SELECT version FROM alpaca_ledger_accounts")).scalar_one(),
            conn.execute(text("SELECT COUNT(*) FROM alpaca_ledger_intents")).scalar_one(),
            conn.execute(text("SELECT COUNT(*) FROM alpaca_ledger_attempts")).scalar_one(),
        )


def working_order(**changes):
    return {
        "id": "broker-1",
        "client_order_id": PROPOSAL.client_id,
        "asset_id": "asset",
        "symbol": PROPOSAL.symbol,
        "asset_class": "us_option",
        "side": "buy",
        "qty": "1",
        "filled_qty": "0",
        "status": "accepted",
        **changes,
    }


def test_reservation_commits_intent_attempt_and_nonexecuted_stub_together(engine):
    ledger = Ledger(engine, ACCOUNT)
    assert reserve_entry(ledger, PROPOSAL, snapshot(engine), LIMITS, now=AT) == "curlit-first"
    assert saved(engine) == (1, 1, 1)
    with engine.connect() as conn:
        assert conn.execute(
            text("SELECT status,alpaca_order_id FROM alpaca_option_orders")
        ).one() == ("reserved", None)
        assert (
            conn.execute(text("SELECT state FROM alpaca_ledger_attempts")).scalar_one() == "created"
        )
    assert ledger.begin_submission(PROPOSAL.client_id)
    assert not ledger.begin_submission(PROPOSAL.client_id)


def test_stale_competing_snapshot_cannot_consume_last_slot(engine):
    ledger = Ledger(engine, ACCOUNT)
    before = snapshot(engine)
    reserve_entry(ledger, PROPOSAL, before, LIMITS, now=AT)
    with pytest.raises(ReservationBlockedError, match="stale"):
        reserve_entry(ledger, replace(PROPOSAL, idea_id="second"), before, LIMITS, now=AT)
    assert saved(engine) == (1, 1, 1)


def test_timeout_reservation_survives_day_rollover(engine):
    ledger = Ledger(engine, ACCOUNT)
    reserve_entry(ledger, PROPOSAL, snapshot(engine), LIMITS, now=AT)
    ledger.begin_submission(PROPOSAL.client_id)
    later = AT + timedelta(days=2)
    with pytest.raises(ReservationBlockedError, match="contract_capacity"):
        reserve_entry(
            ledger,
            replace(PROPOSAL, idea_id="second"),
            snapshot(engine, at=later),
            replace(LIMITS, max_contracts_per_symbol=1),
            now=later,
        )
    assert saved(engine) == (1, 1, 1)


def test_matching_broker_order_and_local_reservation_count_once(engine):
    ledger = Ledger(engine, ACCOUNT)
    reserve_entry(ledger, PROPOSAL, snapshot(engine), LIMITS, now=AT)
    reserve_entry(
        ledger,
        replace(PROPOSAL, idea_id="second"),
        snapshot(engine, working=[working_order()]),
        LIMITS,
        now=AT,
    )
    # One working contract plus one proposal exactly uses the cap of two.
    assert saved(engine) == (2, 2, 2)


@pytest.mark.parametrize("change", [{"asset_id": "wrong"}, {"side": "sell"}, {"qty": "2"}])
def test_identity_conflict_cannot_deduplicate_local_risk(engine, change):
    ledger = Ledger(engine, ACCOUNT)
    reserve_entry(ledger, PROPOSAL, snapshot(engine), LIMITS, now=AT)
    with pytest.raises(ReservationBlockedError, match="identity_mismatch"):
        reserve_entry(
            ledger,
            replace(PROPOSAL, idea_id="second"),
            snapshot(engine, working=[working_order(**change)]),
            LIMITS,
            now=AT,
        )
    assert saved(engine) == (1, 1, 1)


def test_daily_capacity_counts_reserved_attempts_across_different_symbols(engine):
    ledger = Ledger(engine, ACCOUNT)
    reserve_entry(ledger, PROPOSAL, snapshot(engine), LIMITS, now=AT)
    another = replace(
        PROPOSAL, idea_id="second", asset_id="other", symbol="FRO261218C00047000", underlying="FRO"
    )
    with pytest.raises(ReservationBlockedError, match="daily_capacity"):
        reserve_entry(ledger, another, snapshot(engine), replace(LIMITS, max_per_day=1), now=AT)
    assert saved(engine) == (1, 1, 1)


def test_legacy_collision_rolls_back_capacity_intent_and_attempt(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO alpaca_option_orders(idea_id,status,submitted_at) VALUES ('first','submitted',:at)"
            ),
            {"at": AT},
        )
    with pytest.raises(IntegrityError):
        reserve_entry(Ledger(engine, ACCOUNT), PROPOSAL, snapshot(engine), LIMITS, now=AT)
    assert saved(engine) == (0, 0, 0)


@pytest.mark.parametrize("paused", ["account", "global"])
def test_pause_blocks_without_mutation(engine, paused):
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE alpaca_ledger_accounts SET entries_paused=TRUE"
                if paused == "account"
                else "UPDATE trading_halt_state SET mode='PAUSE_ENTRIES'"
            )
        )
    with pytest.raises(ReservationBlockedError):
        reserve_entry(Ledger(engine, ACCOUNT), PROPOSAL, snapshot(engine), LIMITS, now=AT)
    assert saved(engine) == (0, 0, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"qty": Decimal("NaN")},
        {"qty": Decimal("0.5")},
        {"side": "sell"},
        {"symbol": "RTX1261218C00105000", "underlying": "RTX1"},
        {"multiplier": Decimal("NaN")},
        {"multiplier": Decimal(1)},
        {"multiplier": Decimal(150)},
    ],
)
def test_unsupported_proposal_cannot_reserve(engine, change):
    with pytest.raises(ReservationBlockedError):
        reserve_entry(
            Ledger(engine, ACCOUNT), replace(PROPOSAL, **change), snapshot(engine), LIMITS, now=AT
        )
    assert saved(engine) == (0, 0, 0)


@pytest.mark.parametrize("offset", [-2, 2])
def test_expired_or_future_snapshot_rejected(engine, offset):
    with pytest.raises(ReservationBlockedError, match="snapshot_expired_or_future"):
        reserve_entry(
            Ledger(engine, ACCOUNT),
            PROPOSAL,
            snapshot(engine, at=AT + timedelta(seconds=offset)),
            LIMITS,
            now=AT,
        )
    assert saved(engine) == (0, 0, 0)


def test_unknown_partial_fill_without_working_order_stops_new_risk(engine):
    ledger = Ledger(engine, ACCOUNT)
    proposal = replace(PROPOSAL, qty=Decimal(2))
    reserve_entry(ledger, proposal, snapshot(engine), LIMITS, now=AT)
    ledger.observe_order(
        proposal.client_id,
        working_order(
            qty="2", filled_qty="1", status="partially_filled", updated_at=AT.isoformat()
        ),
    )
    with pytest.raises(ReservationBlockedError, match="partial_fill_missing"):
        reserve_entry(ledger, replace(PROPOSAL, idea_id="second"), snapshot(engine), LIMITS, now=AT)
    assert saved(engine) == (2, 1, 1)


def test_equity_multi_allocation_policy_is_not_inferred_from_net_quantity(engine):
    proposal = EntryProposal(
        "equities", "first", "asset", "RTX", "RTX", "buy", Decimal(10), Decimal(1)
    )
    with pytest.raises(ReservationBlockedError, match="unsupported_equity_allocation_policy"):
        reserve_entry(
            Ledger(engine, ACCOUNT),
            proposal,
            snapshot(engine),
            replace(LIMITS, max_positions_per_ticker=2),
            now=AT,
        )


def test_unfilled_reserved_stub_is_not_a_fill_verified_allocation():
    order = broker_order("first", 10)
    order.update(status="accepted", filled_qty="0")
    captured = broker_snapshot([order], [("first", 10)])
    captured["activities"]["records"] = []
    captured["internal"]["equities"][0]["status"] = "reserved"
    result = project_snapshot(captured)
    assert result[0]["evidence_status"] == "unresolved"
    assert result[0]["reason"] == "reserved entry has no broker fills; intent only"


def test_missing_ledger_with_unattributed_reserved_rows_cannot_project():
    captured = broker_snapshot([], [])
    captured["internal"]["unattributed_reserved_count"] = 1
    with pytest.raises(ValueError, match="ownership unavailable"):
        project_snapshot(captured)


def test_caller_connection_must_have_transaction_for_atomic_intent(engine):
    ledger = Ledger(engine, ACCOUNT)
    with engine.connect() as conn, pytest.raises(ValueError, match="active ledger transaction"):
        ledger.create_intent(
            intent_id="fixture",
            client_id="fixture",
            book="options",
            idea_id="first",
            asset_id="asset",
            symbol=PROPOSAL.symbol,
            purpose="entry",
            wire_side="buy",
            qty=Decimal(1),
            multiplier=Decimal(100),
            created_at=AT,
            detail={},
            connection=conn,
        )
    assert saved(engine) == (0, 0, 0)


@pytest.mark.parametrize("scenario", ["advanced", "matching", "external"])
@pytest.mark.parametrize("status", ["partially_filled", "accepted", "pending_cancel"])
def test_working_partial_fills_never_rely_on_lagging_positions(engine, scenario, status):
    ledger = Ledger(engine, ACCOUNT)
    first = replace(PROPOSAL, qty=Decimal(2))
    observed = working_order(qty="2", filled_qty="1", status=status, updated_at=AT.isoformat())
    if scenario != "external":
        reserve_entry(ledger, first, snapshot(engine), LIMITS, now=AT)
        if scenario == "matching":
            ledger.observe_order(first.client_id, observed)
    before = saved(engine)
    with pytest.raises(
        ReservationBlockedError, match="working_partial_fill_requires_reconciliation"
    ):
        reserve_entry(
            ledger,
            replace(PROPOSAL, idea_id="second"),
            snapshot(engine, working=[observed]),
            LIMITS,
            now=AT,
        )
    assert saved(engine) == before


@pytest.mark.parametrize("held,allowed", [(1, True), (2, False)])
def test_terminal_partial_cancel_keeps_filled_inventory_in_cap(engine, held, allowed):
    ledger = Ledger(engine, ACCOUNT)
    first = replace(PROPOSAL, qty=Decimal(2))
    reserve_entry(ledger, first, snapshot(engine), LIMITS, now=AT)
    ledger.observe_order(
        first.client_id,
        working_order(qty="2", filled_qty="1", status="canceled", updated_at=AT.isoformat()),
    )
    # Independent arithmetic oracle assuming the fixture's fresh reconciled
    # position: held + one proposed <= two; canceled remainder does not count.
    # This is not proof that separate real broker reads are atomic.
    current = snapshot(
        engine, positions=[{"symbol": first.symbol, "asset_class": "us_option", "qty": str(held)}]
    )
    if allowed:
        reserve_entry(ledger, replace(PROPOSAL, idea_id="second"), current, LIMITS, now=AT)
        assert saved(engine) == (3, 2, 2)
    else:
        with pytest.raises(ReservationBlockedError, match="contract_capacity_exhausted"):
            reserve_entry(ledger, replace(PROPOSAL, idea_id="second"), current, LIMITS, now=AT)
        assert saved(engine) == (2, 1, 1)
