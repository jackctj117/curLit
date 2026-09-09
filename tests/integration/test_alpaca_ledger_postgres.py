"""Real transactional repair, replay, and crash oracles for CL-0deu.3/18."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from migrations.run import _strip_sql_comments
from scripts.alpaca_recovery_report import internal_snapshot
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.execution.alpaca_equity_exit import EquityExitConfig
from src.execution.alpaca_fill_ledger import Ledger
from src.execution.alpaca_ledger_exits import close_only_cycle, eligible_allocations, lock_key
from src.execution.alpaca_ledger_reconcile import ingest_snapshot, project_snapshot
from src.execution.alpaca_ledger_repair import apply_repairs
from src.execution.alpaca_ledger_reporting import closed_performance
from src.execution.alpaca_recovery import History

ACCOUNT = "alpaca-paper:integration"
AT = datetime(2026, 8, 1, 14, tzinfo=UTC)


@pytest.fixture
def ledger_engine():
    raw = os.environ.get("CURLIT_AUDIT_TEST_DB_URL")
    if not raw:
        pytest.skip("Explicit disposable PostgreSQL required")
    url = make_url(raw)
    if url.host != "127.0.0.1" or not (url.database or "").startswith("curlit_test_"):
        pytest.fail("Refusing non-disposable PostgreSQL")
    schema = "ledger_test_" + uuid4().hex
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        with engine.begin() as conn:
            for _ in range(2):
                for name in (
                    "005_geo_events.sql",
                    "007_trade_ideas.sql",
                    "008_trade_idea_levels.sql",
                    "014_alpaca_option_orders.sql",
                    "016_alpaca_option_exits.sql",
                    "018_option_entry_mid.sql",
                    "019_alpaca_equity_orders.sql",
                    "022_alpaca_fill_ledger.sql",
                ):
                    for statement in _strip_sql_comments(
                        (Path("migrations") / name).read_text()
                    ).split(";"):
                        if statement.strip():
                            conn.execute(text(statement))
        engine.test_schema = schema
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def seed(engine, idea="a", qty=10, closed_external=True):
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO trade_ideas(idea_id,geo_event_id,ticker,action,created_at,"
                "status_updated_at,time_stop_days) VALUES (:i,1,'ABC','buy_calls',:at,:at,10)"
            ),
            {"i": idea, "at": AT},
        )
        conn.execute(
            text(
                "INSERT INTO alpaca_equity_orders(idea_id,ticker,side,qty,entry_price,"
                "alpaca_order_id,status,submitted_at,exit_status,exit_reason) "
                "VALUES (:i,'ABC','buy',:q,10,:o,'submitted',:at,:s,:r)"
            ),
            {
                "i": idea,
                "q": qty,
                "o": "order-" + idea,
                "at": AT,
                "s": "closed" if closed_external else None,
                "r": "closed_external" if closed_external else None,
            },
        )


def order(idea="a", qty=10, closing=False, at=AT):
    return {
        "id": "order-" + ("exit-" if closing else "") + idea,
        "client_order_id": "curlit-eq-" + ("exit-" if closing else "") + idea,
        "asset_id": "asset",
        "asset_class": "us_equity",
        "symbol": "ABC",
        "qty": str(qty),
        "side": "sell" if closing else "buy",
        "filled_qty": str(qty),
        "status": "filled",
        "submitted_at": at.isoformat(),
        "updated_at": at.isoformat(),
    }


def activity(broker_order, price="10"):
    return {
        "id": "fill-" + broker_order["id"],
        "order_id": broker_order["id"],
        "symbol": "ABC",
        "activity_type": "FILL",
        "side": broker_order["side"],
        "qty": broker_order["qty"],
        "price": price,
        "transaction_time": broker_order["submitted_at"],
    }


def snapshot(engine, orders=None):
    orders = orders or [order()]
    qty = sum(Decimal(o["filled_qty"]) * (1 if o["side"] == "buy" else -1) for o in orders)
    positions = (
        []
        if not qty
        else [
            {
                "asset_id": "asset",
                "symbol": "ABC",
                "qty": str(qty),
                "current_price": "12",
                "avg_entry_price": "10",
            }
        ]
    )
    internal = internal_snapshot(engine)
    result = {
        "account_scope": ACCOUNT,
        "positions_before": positions,
        "positions_after": positions,
        "internal": internal,
        "internal_after": internal,
        "orders": {"records": orders, "exhausted": True},
        "activities": {"records": [activity(o) for o in orders], "exhausted": True},
    }
    return json.loads(json.dumps(result, default=str))


def test_real_postgres_repair_preserves_clock_estimates_and_is_idempotent(ledger_engine):
    seed(ledger_engine)
    before = snapshot(ledger_engine)
    result = apply_repairs(ledger_engine, before, before, restore_equities={"a"})
    assert result["applied"] == 1
    with ledger_engine.connect() as conn:
        row = conn.execute(text("SELECT submitted_at,exit_status FROM alpaca_equity_orders")).one()
        assert tuple(row) == (AT, None)
        assert (
            conn.execute(text("SELECT management_enabled FROM alpaca_ledger_allocations")).scalar()
            is True
        )
        assert (
            conn.execute(text("SELECT entries_paused FROM alpaca_ledger_accounts")).scalar() is True
        )
        audit = json.loads(
            conn.execute(text("SELECT before_payload FROM alpaca_ledger_repairs")).scalar_one()
        )
        assert audit["exit_reason"] == "closed_external"
    repeated = apply_repairs(ledger_engine, before, snapshot(ledger_engine), restore_equities={"a"})
    assert repeated["applied"] == 0 and repeated["replayed"] == 1


def test_repair_refuses_row_race_and_retains_old_status(ledger_engine):
    seed(ledger_engine)
    before = snapshot(ledger_engine)
    with ledger_engine.begin() as conn:
        conn.execute(text("UPDATE alpaca_equity_orders SET qty=11"))
    with pytest.raises(ValueError, match="changed"):
        apply_repairs(ledger_engine, before, before, restore_equities={"a"})
    with ledger_engine.connect() as conn:
        assert (
            conn.execute(text("SELECT exit_status FROM alpaca_equity_orders")).scalar() == "closed"
        )
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_repairs")).scalar() == 0


def test_real_workers_compete_for_one_submission_and_restart_keeps_unknown(ledger_engine):
    ledger = Ledger(ledger_engine, ACCOUNT)
    ledger.create_intent(
        intent_id="i",
        client_id="c",
        book="equities",
        idea_id="a",
        asset_id="asset",
        symbol="ABC",
        purpose="entry",
        wire_side="buy",
        qty=Decimal(10),
        multiplier=Decimal(1),
        created_at=AT,
        detail={},
    )
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(lambda _: Ledger(ledger_engine, ACCOUNT).begin_submission("c"), range(2))
        )
    assert sorted(results) == [False, True]
    # Separate process crashes after the durable unknown transition. Its death
    # cannot reset state or give a restarted process another POST opportunity.
    env = {
        "PATH": os.environ["PATH"],
        "TEST_URL": os.environ["CURLIT_AUDIT_TEST_DB_URL"],
        "TEST_SCHEMA": ledger_engine.test_schema,
    }
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
from sqlalchemy import create_engine
from src.execution.alpaca_fill_ledger import Ledger
engine=create_engine(os.environ['TEST_URL'],connect_args={'options':'-csearch_path='+os.environ['TEST_SCHEMA']})
assert not Ledger(engine,'alpaca-paper:integration').begin_submission('c')
os._exit(77)
""",
        ],
        env=env,
        timeout=10,
        check=False,
    )
    assert child.returncode == 77
    assert not Ledger(ledger_engine, ACCOUNT).begin_submission("c")


def test_lost_exit_response_reconciles_fill_without_duplicate_or_coowner_sale(
    ledger_engine, monkeypatch
):
    seed(ledger_engine, "a", 10, closed_external=False)
    seed(ledger_engine, "b", 5, closed_external=False)
    broker_orders = [order("a", 10), order("b", 5)]
    initial = snapshot(ledger_engine, broker_orders)
    apply_repairs(ledger_engine, initial, initial, restore_equities=set())
    posted = []

    class Evidence:
        def account_scope(self):
            return ACCOUNT

        def positions(self):
            return snapshot(ledger_engine, broker_orders)["positions_after"]

        def history(self, kind):
            return History(records=copy.deepcopy(broker_orders), exhausted=True)

    class Trading:
        def is_market_open(self):
            return True

        def get_stock_quote(self, symbol):
            return (12, 12)

        def _req(self, method, path, json_body):
            posted.append(json_body)
            assert method == "POST" and path == "/v2/orders"
            assert json_body["qty"] == "10"  # NOT the aggregate broker 15.
            broker_orders.append(order("a", 10, closing=True, at=AT + timedelta(days=30)))
            raise TimeoutError("broker accepted then response lost")

    monkeypatch.setattr(
        "src.execution.alpaca_ledger_exits.capture",
        lambda *args, **kwargs: snapshot(ledger_engine, broker_orders),
    )
    first = close_only_cycle(
        ledger_engine, Evidence(), Trading(), book="equities", cfg=EquityExitConfig()
    )
    assert first["pending"] == 1
    # Disable B only to isolate restart recovery of A; co-owner inventory stays 5.
    with ledger_engine.begin() as conn:
        conn.execute(
            text("UPDATE alpaca_ledger_allocations SET management_enabled=FALSE WHERE idea_id='b'")
        )
    second = close_only_cycle(
        ledger_engine, Evidence(), Trading(), book="equities", cfg=EquityExitConfig()
    )
    assert len(posted) == 1
    assert second["closed"] == 1
    with ledger_engine.connect() as conn:
        quantities = dict(
            conn.execute(
                text("SELECT idea_id,signed_quantity FROM alpaca_ledger_allocations")
            ).all()
        )
        assert quantities == {"a": 0, "b": 5}
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_fills")).scalar() == 3
    ingest_snapshot(ledger_engine, snapshot(ledger_engine, broker_orders))
    with ledger_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM alpaca_ledger_fills")).scalar() == 3


def test_closed_report_uses_actual_fills_and_unknown_costs_not_legacy_prices(ledger_engine):
    seed(ledger_engine)
    data = snapshot(ledger_engine, [order(), order(closing=True, at=AT + timedelta(days=1))])
    data["activities"]["records"][-1]["price"] = "12"
    with ledger_engine.begin() as conn:
        conn.execute(
            text("UPDATE alpaca_equity_orders SET entry_price=999,exit_price=1,pnl_pct=-0.999")
        )
    ingest_snapshot(ledger_engine, data)
    rows = closed_performance(ledger_engine, AT)
    assert len(rows) == 1
    assert rows[0]["gross_realized"] == 20  # (12 - 10) * 10, unrelated to the legacy 999/1.
    assert rows[0]["net_realized"] is None
    assert rows[0]["costs_status"] == "unknown"


def test_current_position_after_proven_flat_not_stranded_by_old_overclose(ledger_engine):
    seed(ledger_engine, "a", 10)
    seed(ledger_engine, "b", 5)
    seed(ledger_engine, "c", 7)
    later = AT + timedelta(days=2)
    with ledger_engine.begin() as conn:
        conn.execute(
            text("UPDATE alpaca_equity_orders SET submitted_at=:at WHERE idea_id='c'"),
            {"at": later},
        )
    data = snapshot(
        ledger_engine,
        [
            order("a", 10),
            order("b", 5),
            order("a", 15, closing=True, at=AT + timedelta(days=1)),
            order("c", 7, at=later),
        ],
    )
    projections = project_snapshot(data)
    assert next(p for p in projections if p["idea_id"] == "a")["evidence_status"] == "unresolved"
    eligible = eligible_allocations(projections, data)
    assert list(eligible) == ["equities:c"]
    assert eligible["equities:c"]["allocation"]["signed_quantity"] == 7
    # An unowned order in the current epoch must still stop automatic management.
    external = order("manual", 2, at=later + timedelta(minutes=1))
    external["client_order_id"] = "external-manual"
    data = snapshot(ledger_engine, [*data["orders"]["records"], external])
    assert "equities:c" not in eligible_allocations(project_snapshot(data), data)


def test_account_fence_blocks_other_book_without_broker_reads(ledger_engine):
    class Evidence:
        def account_scope(self):
            return ACCOUNT

        def positions(self):
            pytest.fail("second writer reached broker under held account fence")

    with ledger_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as owner:
        owner.execute(text("SELECT pg_advisory_lock(:k)"), {"k": lock_key(ACCOUNT)})
        try:
            result = close_only_cycle(
                ledger_engine, Evidence(), None, book="equities", cfg=EquityExitConfig()
            )
            assert result["blocked"] == 1
        finally:
            owner.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": lock_key(ACCOUNT)})
