"""Real competing/crashing workers and independent inventory oracles (CL-0deu.1/18).

Only the explicitly disposable PostgreSQL fixture is used. These tests exercise
the persistence primitive, not an activated executor or a real broker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from migrations.run import _strip_sql_comments
from scripts.alpaca_recovery_report import internal_snapshot
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from src.execution.alpaca_entry_reservation import (
    EntryLimits,
    EntryProposal,
    EntrySnapshot,
    ReservationBlockedError,
    reserve_entry,
)
from src.execution.alpaca_fill_ledger import Ledger
from src.execution.alpaca_ledger_reconcile import ingest_snapshot
from tests.integration.test_alpaca_ledger_postgres import ledger_engine as ledger_engine
from tests.integration.test_recovery_postgres import recovery_engine as recovery_engine

ACCOUNT = "alpaca-paper:reservation-fixture"
AT = datetime(2026, 9, 9, 14, tzinfo=UTC)
LIMITS = EntryLimits(max_per_day=10, max_per_hour=2, max_positions_per_ticker=1)


def test_account_scoped_capture_before_ledger_migration_preserves_legacy_and_flags_unknown(
    recovery_engine: Engine,
) -> None:
    """A pre-migration baseline must not require the schema it will preserve."""
    with recovery_engine.begin() as conn:
        conn.execute(text("INSERT INTO trade_ideas(idea_id) VALUES ('legacy')"))
        conn.execute(text(
            "INSERT INTO alpaca_equity_orders(idea_id,ticker,qty,status,submitted_at) "
            "VALUES ('legacy','ABC',7,'submitted',:at),('unknown','XYZ',2,'reserved',:at)"
        ), {"at": AT})
    captured = internal_snapshot(recovery_engine, ACCOUNT)
    assert captured["reservation_coverage"] == "ledger_schema_absent"
    assert captured["unattributed_reserved_count"] == 1
    assert captured["idea_ids"] == ["legacy"]
    assert [(row["idea_id"], row["qty"]) for row in captured["equities"]] == [("legacy", 7)]
    assert captured["options"] == []


@pytest.fixture
def reservation_engine(ledger_engine: Engine) -> Engine:
    with ledger_engine.begin() as conn:
        sql = _strip_sql_comments(Path("migrations/024_trading_halt.sql").read_text())
        for statement in sql.split(";"):
            if statement.strip():
                conn.execute(text(statement))
        # Deliberate fixture-only activation; no application/default changes.
        conn.execute(text("UPDATE trading_halt_state SET mode='ACTIVE' WHERE id=1"))
        conn.execute(text(
            "INSERT INTO alpaca_ledger_accounts(account_scope,entries_paused) VALUES (:a,FALSE)"
        ), {"a": ACCOUNT})
        conn.execute(text("CREATE TABLE fixture_worker_ready(worker TEXT PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE fixture_broker_posts(client_id TEXT,broker_order TEXT)"))
        for idea in ("a", "b"):
            conn.execute(text(
                "INSERT INTO trade_ideas(idea_id,geo_event_id,ticker,action,created_at,"
                "status_updated_at) VALUES (:i,1,'ABC','buy_calls',:at,:at)"
            ), {"i": idea, "at": AT})
    return ledger_engine


def proposal(idea: str = "a") -> EntryProposal:
    return EntryProposal(
        book="equities", idea_id=idea, asset_id="fixture-asset", symbol="ABC",
        underlying="ABC", side="buy", qty=Decimal(10), multiplier=Decimal(1),
    )


def empty_snapshot(version: int = 0) -> EntrySnapshot:
    # A fixed fixture validity interval, not an operational freshness setting.
    return EntrySnapshot(ACCOUNT, version, [], [], AT, AT + timedelta(seconds=1))


def worker_engine() -> Engine:
    return create_engine(
        os.environ["TEST_URL"],
        connect_args={"options": "-csearch_path=" + os.environ["TEST_SCHEMA"]},
    )


def worker_environment(engine: Engine, idea: str = "a") -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"],
        "TEST_URL": os.environ["CURLIT_AUDIT_TEST_DB_URL"],
        "TEST_SCHEMA": engine.test_schema,  # type: ignore[attr-defined]  # Disposable fixture metadata.
        "TEST_IDEA": idea,
    }


def _competing_worker() -> None:
    engine = worker_engine()
    idea = os.environ["TEST_IDEA"]
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO fixture_worker_ready(worker) VALUES (:i)"), {"i": idea})
    deadline = time.monotonic() + 5  # Bounded local two-process rendezvous.
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            if conn.execute(text("SELECT count(*) FROM fixture_worker_ready")).scalar_one() == 2:
                break
        time.sleep(0.01)
    else:
        raise AssertionError("second worker never reached the fixture rendezvous")
    try:
        reserve_entry(Ledger(engine, ACCOUNT), proposal(idea), empty_snapshot(), LIMITS, now=AT)
    except ReservationBlockedError as exc:
        assert str(exc) == "account_paused_missing_or_snapshot_stale"
        raise SystemExit(23) from exc


def test_two_processes_cannot_consume_one_snapshot_capacity(reservation_engine: Engine) -> None:
    command = [sys.executable, "-c", (
        "from tests.integration.test_alpaca_entry_reservation_postgres import _competing_worker; "
        "_competing_worker()"
    )]
    workers = [
        subprocess.Popen(command, env=worker_environment(reservation_engine, idea))
        for idea in ("a", "b")
    ]
    try:
        assert sorted(worker.wait(timeout=15) for worker in workers) == [0, 23]
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)
    with reservation_engine.connect() as conn:
        assert conn.execute(text("SELECT version FROM alpaca_ledger_accounts")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_intents")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_attempts")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM alpaca_equity_orders")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM fixture_broker_posts")).scalar_one() == 0


def test_legacy_stub_collision_rolls_back_capacity_intent_and_attempt(
    reservation_engine: Engine,
) -> None:
    with reservation_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO alpaca_equity_orders(idea_id,ticker,side,qty,status,submitted_at) "
            "VALUES ('a','OTHER','buy',7,'submitted',:at)"
        ), {"at": AT})
    with pytest.raises(IntegrityError):
        reserve_entry(Ledger(reservation_engine, ACCOUNT), proposal(), empty_snapshot(), LIMITS, now=AT)
    with reservation_engine.connect() as conn:
        assert conn.execute(text("SELECT version FROM alpaca_ledger_accounts")).scalar_one() == 0
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_intents")).scalar_one() == 0
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_attempts")).scalar_one() == 0
        assert tuple(conn.execute(text(
            "SELECT ticker,qty,status FROM alpaca_equity_orders"
        )).one()) == ("OTHER", 7, "submitted")


def test_submission_unknown_without_broker_order_retains_all_capacity(
    reservation_engine: Engine,
) -> None:
    ledger = Ledger(reservation_engine, ACCOUNT)
    client = reserve_entry(ledger, proposal(), empty_snapshot(), LIMITS, now=AT)
    assert ledger.begin_submission(client)
    assert not Ledger(reservation_engine, ACCOUNT).begin_submission(client)
    with pytest.raises(ReservationBlockedError, match="ticker_capacity_exhausted"):
        reserve_entry(ledger, proposal("b"), empty_snapshot(1), LIMITS, now=AT)
    with reservation_engine.connect() as conn:
        assert tuple(conn.execute(text(
            "SELECT state,filled_quantity,broker_order_id FROM alpaca_ledger_attempts"
        )).one()) == ("submission_unknown", 0, None)
        assert conn.execute(text("SELECT version FROM alpaca_ledger_accounts")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM alpaca_equity_orders")).scalar_one() == 1


@pytest.mark.parametrize("orphan_book", ["equities", "options"])
def test_schema_present_orphan_is_visible_but_proven_other_account_is_not_adopted(
    reservation_engine: Engine, orphan_book: str,
) -> None:
    reserve_entry(
        Ledger(reservation_engine, ACCOUNT), proposal(), empty_snapshot(), LIMITS, now=AT,
    )
    other = internal_snapshot(reservation_engine, "alpaca-paper:other")
    assert other["reservation_coverage"] == "account_scoped"
    assert other["unattributed_reserved_count"] == 0
    assert other["equities"] == [] and other["options"] == []

    with reservation_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO alpaca_equity_orders(idea_id,ticker,qty,status,submitted_at) "
            "VALUES ('b','XYZ',2,'reserved',:at)"
            if orphan_book == "equities"
            else "INSERT INTO alpaca_option_orders(idea_id,ticker,occ_symbol,qty,status,submitted_at) "
            "VALUES ('b','XYZ','XYZ261218C00010000',2,'reserved',:at)"
        ), {"at": AT})
    for account in (ACCOUNT, "alpaca-paper:other"):
        internal = internal_snapshot(reservation_engine, account)
        assert internal["reservation_coverage"] == "account_scoped"
        assert internal["unattributed_reserved_count"] == 1
        assert all(row["idea_id"] != "b" for row in internal[orphan_book])
        with pytest.raises(ValueError, match="reserved entry ownership unavailable"):
            ingest_snapshot(reservation_engine, {
                "account_scope": account, "internal": internal,
                "orders": {"records": [], "exhausted": True},
                "activities": {"records": [], "exhausted": True},
            })
    with reservation_engine.connect() as conn:
        assert conn.execute(text("SELECT version FROM alpaca_ledger_accounts")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_allocations")).scalar_one() == 0


def _accept_entry_then_die() -> None:
    engine = worker_engine()
    ledger = Ledger(engine, ACCOUNT)
    client = reserve_entry(ledger, proposal(), empty_snapshot(), LIMITS, now=AT)
    assert ledger.begin_submission(client)
    broker_order = {
        "id": "fixture-accepted-entry", "client_order_id": client,
        "asset_id": "fixture-asset", "asset_class": "us_equity", "symbol": "ABC",
        "qty": "10", "filled_qty": "10", "side": "buy", "status": "filled",
        "submitted_at": AT.isoformat(), "updated_at": AT.isoformat(),
    }
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO fixture_broker_posts(client_id,broker_order) VALUES (:c,:o)"
        ), {"c": client, "o": json.dumps(broker_order)})
    # Exchange evidence survives; application acknowledgement/cleanup never runs.
    os._exit(77)


def test_reserved_entry_survives_acceptance_crash_and_replay_without_auto_management(
    reservation_engine: Engine,
) -> None:
    child = subprocess.run(
        [sys.executable, "-c", (
            "from tests.integration.test_alpaca_entry_reservation_postgres import _accept_entry_then_die; "
            "_accept_entry_then_die()"
        )],
        env=worker_environment(reservation_engine), timeout=15, check=False,
    )
    assert child.returncode == 77
    assert internal_snapshot(reservation_engine)["equities"] == []
    assert internal_snapshot(reservation_engine, "alpaca-paper:other")["equities"] == []
    internal = internal_snapshot(reservation_engine, ACCOUNT)
    assert len(internal["equities"]) == 1
    assert internal["equities"][0]["status"] == "reserved"
    with reservation_engine.connect() as conn:
        row = conn.execute(text("SELECT client_id,broker_order FROM fixture_broker_posts")).one()
        broker_order = json.loads(row.broker_order)
        assert tuple(conn.execute(text(
            "SELECT client_order_id,state,broker_order_id FROM alpaca_ledger_attempts"
        )).one()) == (row.client_id, "submission_unknown", None)
    assert not Ledger(reservation_engine, ACCOUNT).begin_submission(row.client_id)
    position = {"asset_id": "fixture-asset", "symbol": "ABC", "qty": "10"}
    captured: dict[str, Any] = {
        "account_scope": ACCOUNT, "internal": internal, "internal_after": internal,
        "positions_before": [position], "positions_after": [position],
        "orders": {"records": [broker_order], "exhausted": True},
        "activities": {"exhausted": True, "records": [{
            "id": "fixture-fill", "order_id": broker_order["id"], "symbol": "ABC",
            "activity_type": "FILL", "side": "buy", "qty": "10", "price": "12",
            "transaction_time": AT.isoformat(),
        }]},
    }
    for _ in range(2):
        ingest_snapshot(reservation_engine, captured)
    with reservation_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM fixture_broker_posts")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_fills")).scalar_one() == 1
        assert tuple(conn.execute(text(
            "SELECT signed_quantity,entry_average,gross_realized,net_realized,management_enabled "
            "FROM alpaca_ledger_allocations"
        )).one()) == (10, 12, 0, None, False)
        assert tuple(conn.execute(text(
            "SELECT client_order_id,state,broker_order_id FROM alpaca_ledger_attempts"
        )).one()) == (row.client_id, "filled", "fixture-accepted-entry")
