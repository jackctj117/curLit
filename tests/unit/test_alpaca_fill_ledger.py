"""Independent inventory/cashflow oracles for CL-0deu.3, no broker access."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D  # noqa: N817 - compact exact-money fixture vectors
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from src.execution.alpaca_fill_ledger import (
    Fill,
    Ledger,
    ingest_activity,
    parse_fill,
    project_allocation,
)

AT = datetime(2026, 9, 9, 14, tzinfo=UTC)
ACCOUNT = "alpaca-paper:fixture"


def fill(identity, qty, price, side="buy", minute=0, fees="0"):
    return Fill(
        identity,
        "order-" + identity,
        "asset-1",
        "ABC",
        side,
        D(qty),
        D(price),
        AT + timedelta(minutes=minute),
        D(fees) if fees is not None else None,
    )


@pytest.mark.parametrize(
    "entry_side,exit_side,exit_price,expected",
    [
        ("buy", "sell", "12", "20"),
        ("sell", "buy", "8", "20"),
        ("buy", "sell", "8", "-20"),
        ("sell", "buy", "12", "-20"),
    ],
)
def test_closed_cashflows(entry_side, exit_side, exit_price, expected):
    result = project_allocation(
        [fill("entry", "10", "10", entry_side, fees="1")],
        [fill("exit", "10", exit_price, exit_side, 1, "2")],
        multiplier=D(1),
        entry_side=entry_side,
    )
    assert result.signed_quantity == 0
    assert result.gross_realized == D(expected)
    assert result.net_realized == D(expected) - 3
    assert result.original_entered_at == AT


def test_partial_cancel_keeps_inventory_and_unknown_costs():
    result = project_allocation(
        [fill("entry", "10", "2", fees=None)],
        [fill("partial", "4", "3", "sell", 1, None)],
        multiplier=D(100),
        entry_side="buy",
    )
    assert result.signed_quantity == 6
    assert result.gross_realized == 400
    assert result.entry_average == 2
    assert result.net_realized is None
    assert result.costs_status == "unknown"


def test_average_cost_and_replayed_out_of_order_deliveries():
    entries = [fill("a", "2", "10"), fill("b", "2", "20", minute=1)]
    exits = [fill("c", "3", "25", "sell", 2)]
    result = project_allocation(
        entries[::-1] + entries, exits + exits, multiplier=D(1), entry_side="buy"
    )
    assert result.signed_quantity == 1
    assert result.entry_average == 15
    assert result.gross_realized == 30
    assert result.costs_status == "unallocated"


@pytest.mark.parametrize(
    "exits",
    [
        [fill("over", "11", "12", "sell", 1)],
        [fill("early", "1", "12", "sell", -1)],
        [fill("flip", "1", "12", "buy", 1)],
        [fill("tied", "1", "12", "sell", 0)],
    ],
)
def test_unowned_or_ambiguous_exit_rejected(exits):
    with pytest.raises(ValueError):
        project_allocation([fill("entry", "10", "10")], exits, multiplier=D(1), entry_side="buy")


@given(
    qty=st.decimals(min_value="0.001", max_value="1000", places=3),
    opening=st.decimals(min_value="0.01", max_value="1000", places=2),
    closing=st.decimals(min_value="0.01", max_value="1000", places=2),
)
def test_fractional_cashflow_conservation(qty, opening, closing):
    long = project_allocation(
        [fill("e", qty, opening)],
        [fill("x", qty, closing, "sell", 1)],
        multiplier=D(100),
        entry_side="buy",
    )
    short = project_allocation(
        [fill("e", qty, opening, "sell")],
        [fill("x", qty, closing, "buy", 1)],
        multiplier=D(100),
        entry_side="sell",
    )
    assert long.gross_realized == (qty * closing - qty * opening) * 100
    assert short.gross_realized == -long.gross_realized
    assert long.signed_quantity == short.signed_quantity == 0


@pytest.fixture
def engine():
    result = create_engine("sqlite://")
    sql = Path("migrations/022_alpaca_fill_ledger.sql").read_text()
    with result.begin() as conn:
        for statement in sql.split(";"):
            if statement.strip():
                conn.exec_driver_sql(statement)
    yield result
    result.dispose()


def create(ledger):
    return ledger.create_intent(
        intent_id="intent-1",
        client_id="client-1",
        book="equities",
        idea_id="idea-1",
        asset_id="asset-1",
        symbol="ABC",
        purpose="entry",
        wire_side="buy",
        qty=D(10),
        multiplier=D(1),
        created_at=AT,
        detail={},
    )


def order(status="new", filled="0", minute=0):
    return {
        "id": "order-1",
        "client_order_id": "client-1",
        "asset_id": "asset-1",
        "symbol": "ABC",
        "side": "buy",
        "qty": "10",
        "filled_qty": filled,
        "status": status,
        "updated_at": (AT + timedelta(minutes=minute)).isoformat(),
    }


def test_crash_before_and_after_submission_recovers_identity(engine):
    first = Ledger(engine, ACCOUNT)
    assert create(first)
    restart = Ledger(engine, ACCOUNT)
    assert not create(restart)
    assert restart.begin_submission("client-1")
    # A lost response or even a crash before network transmission is unknown.
    # No second worker may POST; only original-order lookup can resolve it.
    assert not Ledger(engine, ACCOUNT).begin_submission("client-1")
    assert restart.observe_order("client-1", order("partially_filled", "4", 1))
    assert restart.observe_order("client-1", order("canceled", "4", 2))
    assert not restart.observe_order("client-1", order("new", "0", 0))
    with engine.connect() as conn:
        row = conn.execute(text("SELECT state,filled_quantity FROM alpaca_ledger_attempts")).one()
        assert tuple(row) == ("canceled", 4)
    assert not Ledger(engine, ACCOUNT).begin_submission("client-1")


def test_order_conflicts_fail_without_modifying_prior_state(engine):
    ledger = Ledger(engine, ACCOUNT)
    create(ledger)
    ledger.observe_order("client-1", order("partially_filled", "4", 1))
    for invalid in [
        order("new", "0", 2),
        order("filled", "4", 2),
        {**order("filled", "10", 2), "asset_id": "other"},
    ]:
        with pytest.raises(ValueError):
            ledger.observe_order("client-1", invalid)
    with engine.connect() as conn:
        assert (
            conn.execute(text("SELECT filled_quantity FROM alpaca_ledger_attempts")).scalar() == 4
        )


def test_activity_replay_and_revision_conflict(engine):
    ledger = Ledger(engine, ACCOUNT)
    create(ledger)
    activity = {"id": "fill-1", "activity_type": "FILL", "qty": "4"}
    with engine.begin() as conn:
        ingest_activity(conn, ACCOUNT, activity)
        ingest_activity(conn, ACCOUNT, activity)
    with pytest.raises(ValueError), engine.begin() as conn:
        ingest_activity(conn, ACCOUNT, {**activity, "qty": "5"})
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_activities")).scalar() == 1


def test_parse_fill_does_not_take_order_average_or_last_quote():
    original = {**order(), "side": "sell", "filled_avg_price": "99"}
    activity = {
        "id": "fill-1",
        "order_id": "order-1",
        "symbol": "ABC",
        "activity_type": "FILL",
        "side": "sell_short",
        "qty": "0.125",
        "price": "10",
        "transaction_time": AT.isoformat(),
    }
    parsed = parse_fill(activity, original)
    assert parsed.price == 10
    assert parsed.qty == D("0.125")
    assert parsed.fees is None
    assert parsed.side == "sell"
    with pytest.raises(ValueError):
        parse_fill({**activity, "price": None}, original)
