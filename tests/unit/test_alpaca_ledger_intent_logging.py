"""CL-cw5x: replayed ledger intents must not claim an imminent broker submission.

Oracles are independent of the logging change: the broker orders in a
hand-built snapshot fix how many intents a replay touches, the DB rows are
compared against values written out by hand from that snapshot, and the SQL
statement stream is compared between submission and replay modes byte for byte.
No broker, network, or PostgreSQL: SQLite with the ledger migrations, and
the two advisory-lock functions stubbed so ``close_only_cycle`` can run.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, text

from src.execution.alpaca_equity_exit import EquityExitConfig
from src.execution.alpaca_fill_ledger import Ledger
from src.execution.alpaca_ledger_exits import close_only_cycle
from src.execution.alpaca_ledger_reconcile import ingest_snapshot
from src.execution.alpaca_recovery import History

ACCOUNT = "alpaca-paper:fixture"
AT = datetime(2026, 8, 1, 14, tzinfo=UTC)
SUBMIT_LINE = "Persisting order intent before broker submission"
REPLAY_LINE = "Persisting broker-observed order intent (replay, no submission)"
SUMMARY_LINE = "Replayed broker-observed order intents (no broker submission)"
LEDGER_TABLES = ("alpaca_ledger_accounts", "alpaca_ledger_intents", "alpaca_ledger_attempts")


@pytest.fixture
def engine():
    result = create_engine("sqlite://")

    @event.listens_for(result, "connect")
    def _advisory_lock_stubs(dbapi_conn, _record):
        # PostgreSQL session locks; one process in a unit test always wins.
        dbapi_conn.create_function("pg_try_advisory_lock", 1, lambda _k: 1)
        dbapi_conn.create_function("pg_advisory_unlock", 1, lambda _k: 1)

    statements = Path("migrations/022_alpaca_fill_ledger.sql").read_text().split(";")
    for statement in Path("migrations/023_alpaca_accounting_evidence.sql").read_text().split(";"):
        # SQLite lacks ADD COLUMN IF NOT EXISTS; the schema here is always fresh.
        statements.append(statement.replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN"))
    statements += [
        "CREATE TABLE geo_events (id INTEGER PRIMARY KEY, status TEXT)",
        "CREATE TABLE trade_ideas (idea_id TEXT PRIMARY KEY, geo_event_id INTEGER, "
        "status TEXT, time_stop_days INTEGER)",
    ]
    with result.begin() as conn:
        for statement in statements:
            lines = [ln for ln in statement.splitlines() if not ln.strip().startswith("--")]
            if "\n".join(lines).strip():
                conn.exec_driver_sql("\n".join(lines))
    yield result
    result.dispose()


def broker_order(idea, qty, *, closing=False, at=AT):
    return {
        "id": "order-" + ("exit-" if closing else "") + idea,
        "client_order_id": "curlit-eq-" + ("exit-" if closing else "") + idea,
        "asset_id": "asset-" + idea,
        "asset_class": "us_equity",
        "symbol": "SYM" + idea.upper(),
        "qty": str(qty),
        "side": "sell" if closing else "buy",
        "filled_qty": str(qty),
        "status": "filled",
        "submitted_at": at.isoformat(),
        "updated_at": at.isoformat(),
    }


def fill_activity(order):
    return {
        "id": "fill-" + order["id"],
        "order_id": order["id"],
        "symbol": order["symbol"],
        "activity_type": "FILL",
        "side": order["side"],
        "qty": order["qty"],
        "price": "10",
        "transaction_time": order["submitted_at"],
    }


def legacy(idea, qty):
    return {
        "idea_id": idea,
        "ticker": "SYM" + idea.upper(),
        "side": "buy",
        "qty": qty,
        "alpaca_order_id": "order-" + idea,
        "exit_order_id": None,
        "submitted_at": AT.isoformat(),
    }


def snapshot(orders, ideas):
    net: dict[str, Decimal] = {}
    for o in orders:
        signed = Decimal(o["filled_qty"]) * (1 if o["side"] == "buy" else -1)
        net[o["asset_id"]] = net.get(o["asset_id"], Decimal(0)) + signed
    by_asset = {o["asset_id"]: o["symbol"] for o in orders}
    positions = [
        {"asset_id": a, "symbol": by_asset[a], "qty": str(q), "avg_entry_price": "10"}
        for a, q in net.items()
        if q
    ]
    internal = {"options": [], "equities": [legacy(i, q) for i, q in ideas], "idea_ids": []}
    internal["idea_ids"] = [i for i, _ in ideas]
    return {
        "account_scope": ACCOUNT,
        "positions_before": positions,
        "positions_after": copy.deepcopy(positions),
        "internal": internal,
        "internal_after": copy.deepcopy(internal),
        "orders": {"records": orders, "exhausted": True},
        "activities": {"records": [fill_activity(o) for o in orders], "exhausted": True},
    }


def dump(engine):
    with engine.connect() as conn:
        return {
            t: sorted(tuple(r) for r in conn.execute(text(f"SELECT * FROM {t}")))  # noqa: S608
            for t in LEDGER_TABLES
        }


def messages(caplog, level, message):
    return [r for r in caplog.records if r.levelno == level and r.getMessage() == message]


# Closed idea "a" (entry + exit) and open idea "b" (entry only): 3 broker orders.
CLOSED_AND_OPEN = [
    broker_order("a", 10),
    broker_order("a", 10, closing=True, at=AT + timedelta(days=1)),
    broker_order("b", 5),
]


def test_replay_cycle_logs_no_submission_info_and_one_aggregate(engine, caplog):
    caplog.set_level(logging.DEBUG)
    data = snapshot(CLOSED_AND_OPEN, [("a", 10), ("b", 5)])
    for newly in (3, 0):  # First ingest records 3 new intents; replays record none.
        caplog.clear()
        ingest_snapshot(engine, copy.deepcopy(data))
        assert not [r for r in caplog.records if "before broker submission" in r.getMessage()]
        assert len(messages(caplog, logging.DEBUG, REPLAY_LINE)) == 3
        [summary] = messages(caplog, logging.INFO, SUMMARY_LINE)
        assert summary.extra_data == {"replayed": 3, "newly_recorded": newly}


def test_replay_persists_exact_backfill_rows_idempotently(engine):
    data = snapshot(CLOSED_AND_OPEN, [("a", 10), ("b", 5)])
    ingest_snapshot(engine, copy.deepcopy(data))
    first = dump(engine)
    backfill = json.dumps({"origin": "broker_backfill"}, sort_keys=True)
    # Expected rows written out from the snapshot itself, not from the code.
    entry_at, exit_at = str(AT), str(AT + timedelta(days=1))
    assert first["alpaca_ledger_intents"] == [
        (ACCOUNT, "curlit-eq-a", "equities", "a", "asset-a", "SYMA", "entry", "buy", 10, 1)
        + ("USD", entry_at, backfill),
        (ACCOUNT, "curlit-eq-b", "equities", "b", "asset-b", "SYMB", "entry", "buy", 5, 1)
        + ("USD", entry_at, backfill),
        (ACCOUNT, "curlit-eq-exit-a", "equities", "a", "asset-a", "SYMA", "exit", "sell", 10, 1)
        + ("USD", exit_at, backfill),
    ]
    assert [r[1:6] for r in first["alpaca_ledger_attempts"]] == [
        ("curlit-eq-a", "curlit-eq-a", "order-a", "filled", 10),
        ("curlit-eq-b", "curlit-eq-b", "order-b", "filled", 5),
        ("curlit-eq-exit-a", "curlit-eq-exit-a", "order-exit-a", "filled", 10),
    ]
    ingest_snapshot(engine, copy.deepcopy(data))
    second = dump(engine)
    # Only the account version counter moves on replay (observe_order + snapshot).
    assert {t: v for t, v in second.items() if t != "alpaca_ledger_accounts"} == {
        t: v for t, v in first.items() if t != "alpaca_ledger_accounts"
    }


def _create(ledger, **overrides):
    kwargs = {
        "intent_id": "intent-1",
        "client_id": "client-1",
        "book": "equities",
        "idea_id": "idea-1",
        "asset_id": "asset-1",
        "symbol": "ABC",
        "purpose": "exit",
        "wire_side": "sell",
        "qty": Decimal(10),
        "multiplier": Decimal(1),
        "created_at": AT,
        "detail": {"reason": "time_stop"},
    }
    return ledger.create_intent(**{**kwargs, **overrides})


@pytest.mark.parametrize("repeat", [1, 2])
def test_replay_flag_changes_no_sql_rows_or_return_value(repeat):
    """Same statements, same parameters, same order, same rows, same result.

    Five statements per call: account upsert, intent insert, intent read-back,
    attempt insert, attempt read-back (one transaction, alpaca_fill_ledger).
    """
    traces, dumps, returns = [], [], []
    for replay in (False, True):
        eng = create_engine("sqlite://")
        with eng.begin() as conn:
            for stmt in Path("migrations/022_alpaca_fill_ledger.sql").read_text().split(";"):
                if stmt.strip():
                    conn.exec_driver_sql(stmt)
        trace: list[tuple[str, object]] = []
        event.listen(
            eng,
            "before_cursor_execute",
            lambda _c, _cur, stmt, params, _ctx, _many, trace=trace: trace.append((stmt, params)),
        )
        ledger = Ledger(eng, ACCOUNT)
        returns.append([_create(ledger, replay=replay) for _ in range(repeat)])
        traces.append(list(trace))  # Before dump() adds its own SELECTs.
        dumps.append(dump(eng))
        eng.dispose()
    assert traces[0] == traces[1] and len(traces[0]) == 5 * repeat
    assert dumps[0] == dumps[1]
    assert returns[0] == returns[1] == [True, False][:repeat]


def test_submission_intent_logs_info_before_any_write(engine, caplog):
    caplog.set_level(logging.DEBUG)
    timeline: list[str] = []

    class Spy(logging.Handler):
        def emit(self, record):
            if record.getMessage() in {SUBMIT_LINE, REPLAY_LINE}:
                timeline.append(f"log:{record.levelname}")

    spy = Spy()
    logging.getLogger("src.execution.alpaca_fill_ledger").addHandler(spy)
    event.listen(
        engine,
        "before_cursor_execute",
        lambda _c, _cur, stmt, *_a: timeline.append("sql:" + stmt.split()[0].upper()),
    )
    try:
        assert _create(Ledger(engine, ACCOUNT))
    finally:
        logging.getLogger("src.execution.alpaca_fill_ledger").removeHandler(spy)
    assert timeline[0] == "log:INFO"
    assert timeline.count("log:INFO") == 1 and "log:DEBUG" not in timeline
    assert any(step.startswith("sql:INSERT") for step in timeline[1:])


class _Evidence:
    def __init__(self, orders):
        self.orders = orders

    def account_scope(self):
        return ACCOUNT

    def positions(self):
        return snapshot(self.orders, [])["positions_after"]

    def history(self, kind):
        return History(records=copy.deepcopy(self.orders), exhausted=True)

    def order(self, *, client_id):
        return next(o for o in self.orders if o["client_order_id"] == client_id)


class _Trading:
    def __init__(self, orders, caplog):
        self.orders, self.caplog, self.posted = orders, caplog, []
        self.logged_before_post: list[list[tuple[str, int]]] = []

    def is_market_open(self):
        return True

    def get_stock_quote(self, symbol):
        return (10.0, 10.0)

    def _req(self, method, path, json_body):
        # Snapshot what had been logged at the moment of the broker call.
        self.logged_before_post.append([(r.getMessage(), r.levelno) for r in self.caplog.records])
        self.posted.append(json_body)
        fill = broker_order("b", 5, closing=True, at=AT + timedelta(days=30))
        self.orders.append(fill)
        return fill


def _seed_ideas(engine, ideas):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO geo_events(id,status) VALUES (1,'CONFIRMED')"))
        for idea in ideas:
            conn.execute(
                text(
                    "INSERT INTO trade_ideas(idea_id,geo_event_id,status,time_stop_days) "
                    "VALUES (:i,1,'open',10)"
                ),
                {"i": idea},
            )


def _run_cycle(engine, orders, ideas, caplog, monkeypatch):
    trading = _Trading(orders, caplog)
    monkeypatch.setattr(
        "src.execution.alpaca_ledger_exits.capture",
        lambda *a, **k: snapshot(copy.deepcopy(orders), ideas),
    )
    counts = close_only_cycle(
        engine, _Evidence(orders), trading, book="equities", cfg=EquityExitConfig()
    )
    return counts, trading


def test_close_only_cycle_without_submission_emits_no_submission_info(engine, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    orders = copy.deepcopy(CLOSED_AND_OPEN[:2])  # Only the closed idea "a".
    _seed_ideas(engine, ["a"])
    ingest_snapshot(engine, snapshot(copy.deepcopy(orders), [("a", 10)]))
    with engine.begin() as conn:
        conn.execute(text("UPDATE alpaca_ledger_allocations SET management_enabled=1"))
    caplog.clear()
    counts, trading = _run_cycle(engine, orders, [("a", 10)], caplog, monkeypatch)
    assert counts["submitted"] == 0 and counts["closed"] == 1 and not trading.posted
    assert not messages(caplog, logging.INFO, SUBMIT_LINE)
    assert len(messages(caplog, logging.DEBUG, REPLAY_LINE)) == 2
    [summary] = messages(caplog, logging.INFO, SUMMARY_LINE)
    assert summary.extra_data == {"replayed": 2, "newly_recorded": 0}


def test_real_exit_submission_still_logs_info_before_broker_call(engine, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    orders = [broker_order("b", 5)]
    ideas = [("b", 5)]
    _seed_ideas(engine, ["b"])
    ingest_snapshot(engine, snapshot(copy.deepcopy(orders), ideas))
    with engine.begin() as conn:
        conn.execute(text("UPDATE alpaca_ledger_allocations SET management_enabled=1"))
    caplog.clear()
    counts, trading = _run_cycle(engine, orders, ideas, caplog, monkeypatch)
    assert counts["submitted"] == 1
    assert trading.posted == [
        {
            "symbol": "SYMB",
            "qty": "5",
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": "curlit-eq-exit-b",
        }
    ]
    [before_post] = trading.logged_before_post
    submit = [m for m in before_post if m == (SUBMIT_LINE, logging.INFO)]
    assert len(submit) == 1  # Exactly the one genuinely submitted intent, at INFO.
    replay = [m for m in before_post if m[0] == REPLAY_LINE]
    assert replay == [(REPLAY_LINE, logging.DEBUG)]  # The entry order, demoted.
    assert before_post.index((SUBMIT_LINE, logging.INFO)) < before_post.index(
        ("Submitting verified allocation reduction", logging.INFO)
    )
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT i.purpose,i.side,i.quantity,a.state FROM alpaca_ledger_intents i "
                "JOIN alpaca_ledger_attempts a ON a.intent_id=i.intent_id "
                "WHERE i.intent_id='curlit-eq-exit-b'"
            )
        ).one()
    assert tuple(row) == ("exit", "sell", 5, "filled")
