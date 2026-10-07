"""CL-o9sq / CL-pksi: emergency-order fences resolve only from broker evidence
and survive restarts.

Oracles (independent of the implementation):

* Broker truth is a scripted mock broker whose ``net`` book and order records
  are the ground truth of what "really" happened at the venue; a duplicate
  close is an extra entry in ``broker.orders`` for the same symbol.
* Fill accounting is checked against the arithmetic sum of DISTINCT venue
  transaction ids — the definition of an idempotent fill ledger.
* Persistence is checked by reading the sqlite rows of the production
  migration (025) directly, not through the store under test.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from src.execution.broker import (
    BrokerOrderNotFoundError,
    Order,
    OrderStatus,
    OrderType,
    Position,
)
from src.execution.oanda_broker import OandaBroker
from src.execution.oms import OrderIntent, OrderManager, SubmissionStatus
from src.risk.emergency_attempts import (
    EVIDENCE_FILL_EVENT,
    NOT_FOUND_GRACE_SEC,
    AttemptStatus,
    DeriskEvidence,
    EmergencyAttempt,
    InMemoryEmergencyAttemptStore,
    SqlEmergencyAttemptStore,
    apply_evidence,
)
from src.risk.kill_switches import KillSwitchManager

DD = {"portfolio_dd": -0.5}  # drawdown_limit -> flatten_all
VIX = {"vix_level": 40.0, "vix_change_1d": 0.6}  # vix_spike -> reduce_50pct
UNRESOLVED_CAUSE = "external:unresolved_emergency_orders"
T0 = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@dataclass
class Clock:
    now: datetime = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class VenueBroker:
    """Scripted broker. ``script`` drives successive place_order calls:

    fill          FILLED synchronously, book updated
    reject        REJECTED synchronously, book untouched
    working       PENDING (acknowledged, not filled)
    lost_filled   order FILLED at the venue but the response is lost
    lost_stale    as lost_filled, but get_positions keeps the OLD book
    lost_unfilled request lost; nothing created at the venue
    """

    net: dict[str, float]
    script: list[str] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    lookups: dict[str, Order | Exception] = field(default_factory=dict)
    lookup_calls: list[str] = field(default_factory=list)
    on_place: Callable[[Order], None] | None = None
    stale_book: dict[str, float] | None = None

    def get_positions(self) -> list[Position]:
        book = self.stale_book if self.stale_book is not None else self.net
        return [Position(s, q, 1.0) for s, q in book.items() if q]

    def place_order(self, order: Order) -> Order:
        self.orders.append(order)
        if self.on_place is not None:
            self.on_place(order)
        mode = self.script.pop(0) if self.script else "fill"
        signed = order.quantity if order.side == "buy" else -order.quantity
        if mode in ("lost_filled", "lost_stale"):
            if mode == "lost_stale":
                self.stale_book = dict(self.net)
            self.net[order.symbol] = self.net.get(order.symbol, 0.0) + signed
            raise TimeoutError("read timeout: response lost")
        if mode == "lost_unfilled":
            raise TimeoutError("connect timeout: request lost")
        if mode == "fill":
            self.net[order.symbol] = self.net.get(order.symbol, 0.0) + signed
            order.status = OrderStatus.FILLED
            order.order_id = f"F{len(self.orders)}"
        elif mode == "reject":
            order.status = OrderStatus.REJECTED
            order.reject_reason = "INSUFFICIENT_LIQUIDITY"
        elif mode == "working":
            order.status = OrderStatus.PENDING
            order.order_id = f"O{len(self.orders)}"
        else:  # pragma: no cover - test wiring error
            raise AssertionError(mode)
        return order

    def get_order_by_client_id(self, client_order_id: str) -> Order:
        self.lookup_calls.append(client_order_id)
        found = self.lookups.get(client_order_id)
        if found is None:
            raise BrokerOrderNotFoundError(client_order_id)
        if isinstance(found, Exception):
            raise found
        return found


def mk(
    broker: VenueBroker,
    *,
    store: Any = None,
    clock: Clock | None = None,
) -> KillSwitchManager:
    return KillSwitchManager(
        broker,
        OrderManager(broker),  # type: ignore[arg-type]
        {},
        clock=clock or Clock(),
        trailing_state_path=None,
        attempt_store=store,
    )


def install_attempts(engine: Any) -> None:
    """Apply the PRODUCTION migration 025 to sqlite (TIMESTAMPTZ shimmed)."""
    from migrations.run import _strip_sql_comments

    sql = _strip_sql_comments((REPO / "migrations" / "025_fx_emergency_attempts.sql").read_text())
    sql = sql.replace("TIMESTAMPTZ", "TEXT")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))


def rows(engine: Any) -> dict[str, tuple[Any, ...]]:
    with engine.connect() as conn:
        out = conn.execute(
            text(
                "SELECT client_order_id, symbol, status, cumulative_fill_qty, target, "
                "requested_qty FROM fx_emergency_attempts"
            )
        ).all()
    return {str(r[0]): tuple(r[1:]) for r in out}


def fill_event(order: Order, units: float, txn: str) -> dict[str, Any]:
    return {
        "type": "ORDER_FILL",
        "transaction_id": txn,
        "order_id": order.order_id,
        "client_order_id": order.client_order_id,
        "instrument": order.symbol,
        "units": units,
        "price": 1.1,
    }


def attempt_for(risk: KillSwitchManager, client_id: str | None) -> EmergencyAttempt:
    return next(a for a in risk._attempts.values() if a.client_order_id == client_id)


def lookup(order: Order, status: OrderStatus, filled: float | None, txn: str | None) -> Order:
    return Order(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        order_type=OrderType.MARKET,
        order_id=order.order_id or "X1",
        client_order_id=order.client_order_id,
        status=status,
        filled_quantity=filled,
        fill_transaction_id=txn,
    )


# --------------------------------------------------------------------------- #
# Fence resolution (CL-o9sq reviewer finding)
# --------------------------------------------------------------------------- #


def test_lost_response_then_fill_event_resolves_without_duplicate_close() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["lost_filled"])
    risk = mk(broker)
    risk.check(DD)
    assert len(broker.orders) == 1
    assert broker.net["EURUSD"] == 0.0  # it really filled; we just never heard
    assert risk.unresolved_derisk_symbols() == ["EURUSD"]

    # The flat book alone is not evidence: not complete, nothing resent.
    risk.check(DD)
    assert len(broker.orders) == 1
    assert "drawdown_limit" not in risk._triggered_today
    assert risk.unresolved_derisk_symbols() == ["EURUSD"]

    # The delayed fill arrives on the transaction stream via the real OMS.
    assert risk.oms.on_fill(fill_event(broker.orders[0], -1000.0, "T9")) is True
    assert risk.unresolved_derisk_symbols() == []
    assert attempt_for(risk, broker.orders[0].client_order_id).status is AttemptStatus.FILLED

    risk.check(DD)
    assert len(broker.orders) == 1  # never a second close
    assert "drawdown_limit" in risk._triggered_today


def test_fill_event_with_lagging_snapshot_never_resends_the_close() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["lost_stale"])
    risk = mk(broker)
    risk.check(DD)
    risk.oms.on_fill(fill_event(broker.orders[0], -1000.0, "T1"))
    assert risk.unresolved_derisk_symbols() == []
    for _ in range(3):  # the position feed still shows the pre-fill 1000
        risk.check(DD)
    assert len(broker.orders) == 1
    assert "drawdown_limit" not in risk._triggered_today  # needs reconciliation


def test_fill_racing_ahead_of_the_placement_response_is_not_overwritten() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["working"])
    risk = mk(broker)

    def stream_fill_first(order: Order) -> None:
        # The venue fills and streams the fill before the HTTP response lands;
        # the position feed has not caught up yet.
        broker.stale_book = dict(broker.net)
        broker.net["EURUSD"] = 0.0
        risk.oms.on_fill(fill_event(order, -1000.0, "T1"))

    broker.on_place = stream_fill_first
    risk.check(DD)
    broker.on_place = None
    assert attempt_for(risk, broker.orders[0].client_order_id).status is AttemptStatus.FILLED
    for _ in range(3):
        risk.check(DD)
    assert len(broker.orders) == 1


def test_flat_snapshot_alone_does_not_clear_a_working_fence() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["working"])
    risk = mk(broker)
    risk.check(DD)
    order = broker.orders[0]
    broker.net["EURUSD"] = 0.0  # book now looks flat
    broker.lookups[str(order.client_order_id)] = lookup(order, OrderStatus.PENDING, None, None)
    for _ in range(3):
        assert risk.refresh_unresolved_derisk() == 1
        risk.check(DD)
    assert len(broker.orders) == 1
    assert risk.unresolved_derisk_symbols() == ["EURUSD"]
    assert attempt_for(risk, order.client_order_id).status is AttemptStatus.WORKING
    assert "drawdown_limit" not in risk._triggered_today


def test_partial_fill_then_cancel_records_cumulative_keeps_fence_no_resubmit() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["working"])
    risk = mk(broker)
    risk.check(DD)
    order = broker.orders[0]
    cid = str(order.client_order_id)

    risk.oms.on_fill(fill_event(order, -400.0, "T1"))
    broker.net["EURUSD"] = 600.0
    partial = attempt_for(risk, cid)
    assert (partial.status, partial.cumulative_fill_qty) == (AttemptStatus.WORKING, 400.0)

    # Venue: order cancelled after the 400 fill (same fill transaction T1).
    broker.lookups[cid] = lookup(order, OrderStatus.CANCELLED, 400.0, "T1")
    assert risk.refresh_unresolved_derisk() == 1
    terminal = attempt_for(risk, cid)
    assert terminal.status is AttemptStatus.PARTIAL_TERMINAL
    assert terminal.cumulative_fill_qty == 400.0  # not 800: same txn counted once

    for _ in range(3):
        risk.check(DD)
    assert len(broker.orders) == 1  # no resubmission over a partial fill
    assert "drawdown_limit" not in risk._triggered_today
    risk.reset_daily(clear_causes=True)  # operator resume cannot clear it
    assert UNRESOLVED_CAUSE in risk.active_halt_causes()

    released = risk.release_derisk_fence("EUR_USD", changed_by="jack", reason="600 residual")
    assert released == [order.client_order_id]
    risk.check(DD)
    assert len(broker.orders) == 2
    assert (broker.orders[1].side, broker.orders[1].quantity) == ("sell", 600.0)
    assert broker.net["EURUSD"] == 0.0


def test_delayed_and_duplicate_fill_events_are_idempotent() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["working"])
    risk = mk(broker)
    risk.check(DD)
    order = broker.orders[0]
    first = fill_event(order, -500.0, "T1")
    assert risk.oms.on_fill(first) is True
    assert risk.oms.on_fill(first) is False  # OMS transaction-id dedup
    risk.on_broker_fill(first)  # redelivery past the OMS (e.g. after restart)
    assert attempt_for(risk, order.client_order_id).cumulative_fill_qty == 500.0
    assert risk.unresolved_derisk_symbols() == ["EURUSD"]

    second = fill_event(order, -500.0, "T2")
    risk.on_broker_fill(second)
    risk.on_broker_fill(second)
    done = attempt_for(risk, order.client_order_id)
    assert (done.status, done.cumulative_fill_qty) == (AttemptStatus.FILLED, 1000.0)
    assert risk.unresolved_derisk_symbols() == []


def test_foreign_fill_is_not_evidence() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["working"])
    risk = mk(broker)
    risk.check(DD)
    risk.oms.on_fill({"transaction_id": "T7", "client_order_id": "someone-else", "units": -1000.0})
    assert risk.unresolved_derisk_symbols() == ["EURUSD"]


# --------------------------------------------------------------------------- #
# Bounded retry against the ORIGINAL target
# --------------------------------------------------------------------------- #


def test_verified_rejection_retries_exactly_once_per_tick_at_original_target() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["reject", "reject", "fill"])
    risk = mk(broker)
    risk.check(VIX)  # reduce_50pct: original target 500
    assert len(broker.orders) == 1
    assert risk.unresolved_derisk_symbols() == []  # verified terminal, zero fill

    broker.net["EURUSD"] = 800.0  # an unrelated reduction landed meanwhile
    risk.check(VIX)
    assert len(broker.orders) == 2
    assert broker.orders[1].quantity == 300.0  # toward 500, not 400 (=800/2)

    risk.check(VIX)
    assert len(broker.orders) == 3
    assert broker.net["EURUSD"] == 500.0
    assert "vix_spike" in risk._triggered_today
    assert [o.quantity for o in broker.orders] == [500.0, 300.0, 300.0]


def test_two_actions_in_one_tick_send_at_most_one_order_per_symbol() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["reject"] * 6)
    risk = mk(broker)
    for tick in range(1, 4):
        risk.check({**DD, **VIX})  # flatten AND reduce both fire
        assert len(broker.orders) == tick


def test_lookup_verified_zero_fill_cancel_permits_one_retry() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["lost_unfilled"])
    risk = mk(broker)
    risk.check(DD)
    order = broker.orders[0]
    cid = str(order.client_order_id)
    # Terminal, but zero fill NOT verified (amount unknown): stays fenced.
    broker.lookups[cid] = lookup(order, OrderStatus.CANCELLED, None, None)
    risk.refresh_unresolved_derisk()
    risk.check(DD)
    assert len(broker.orders) == 1
    # Verified zero fill: one retry next tick.
    broker.lookups[cid] = lookup(order, OrderStatus.CANCELLED, 0.0, None)
    risk.refresh_unresolved_derisk()
    assert attempt_for(risk, cid).status is AttemptStatus.CANCELLED
    risk.check(DD)
    assert len(broker.orders) == 2
    assert broker.net["EURUSD"] == 0.0


def test_not_found_is_trusted_only_after_the_grace_window() -> None:
    clock = Clock()
    broker = VenueBroker({"EURUSD": 1000.0}, script=["lost_unfilled"])
    risk = mk(broker, clock=clock)
    risk.check(DD)
    cid = str(broker.orders[0].client_order_id)
    clock.advance(NOT_FOUND_GRACE_SEC - 1)
    assert risk.refresh_unresolved_derisk() == 1
    risk.check(DD)
    assert len(broker.orders) == 1
    clock.advance(2)
    assert risk.refresh_unresolved_derisk() == 0
    assert attempt_for(risk, cid).status is AttemptStatus.NOT_FOUND
    risk.check(DD)
    assert len(broker.orders) == 2


# --------------------------------------------------------------------------- #
# Writers, persistence, restart and resume
# --------------------------------------------------------------------------- #


def test_fence_blocks_every_other_writer_through_the_oms() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["working"])
    risk = mk(broker)
    risk.check(DD)
    out = risk.oms.submit_intent_result(
        OrderIntent(strategy_id="rate_diff", symbol="EUR_USD", target_position=0.0)
    )
    assert out.status is SubmissionStatus.BLOCKED
    assert len(broker.orders) == 1


def test_unpersistable_attempt_is_never_submitted() -> None:
    class BrokenStore(InMemoryEmergencyAttemptStore):
        def insert(self, attempt: EmergencyAttempt) -> None:
            raise OSError("database unavailable")

    broker = VenueBroker({"EURUSD": 1000.0})
    risk = mk(broker, store=BrokenStore())
    risk.check(DD)
    assert broker.orders == []
    assert "external:emergency_attempts_unavailable" in risk.active_halt_causes()
    assert risk.oms.is_halted()


def test_crash_restart_refences_and_resolves_from_broker_evidence(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'attempts.db'}"
    engine = create_engine(url)
    install_attempts(engine)
    broker1 = VenueBroker(
        {"EURUSD": 1000.0, "USDCAD": -2000.0}, script=["lost_filled", "lost_unfilled"]
    )
    seen_before_send: list[tuple[Any, ...]] = []
    broker1.on_place = lambda o: seen_before_send.append(rows(engine)[str(o.client_order_id)])
    clock = Clock()
    risk1 = mk(broker1, store=SqlEmergencyAttemptStore(engine), clock=clock)
    risk1.check(DD)
    # Persisted BEFORE each broker call, with the fixed target.
    assert [r[1] for r in seen_before_send] == ["SUBMITTING", "SUBMITTING"]
    assert sorted((r[0], r[3]) for r in seen_before_send) == [("EURUSD", 0.0), ("USDCAD", 0.0)]
    eur, cad = (str(o.client_order_id) for o in broker1.orders)
    assert {k: v[1] for k, v in rows(engine).items()} == {eur: "UNKNOWN", cad: "UNKNOWN"}

    # --- process dies; a new one starts against the same database ---------
    broker2 = VenueBroker(dict(broker1.net))  # EURUSD really closed; USDCAD not
    eur_order = broker1.orders[0]
    broker2.lookups[eur] = lookup(eur_order, OrderStatus.FILLED, 1000.0, "T5")
    clock.advance(10)  # USDCAD "not found" is still inside the grace window
    risk2 = mk(broker2, store=SqlEmergencyAttemptStore(create_engine(url)), clock=clock)
    recovered = risk2.recover_emergency_attempts()
    assert sorted(a.client_order_id for a in recovered) == sorted([eur, cad])
    assert risk2.unresolved_derisk_symbols() == ["USDCAD"]
    assert UNRESOLVED_CAUSE in risk2.active_halt_causes()
    assert risk2.oms.is_halted()
    assert rows(engine)[eur][1:3] == ("FILLED", 1000.0)

    risk2.check(DD)
    assert broker2.orders == []  # no duplicate close on either symbol
    blocked = risk2.oms.submit_intent_result(
        OrderIntent(strategy_id="reconciler", symbol="USDCAD", target_position=0.0)
    )
    assert blocked.status is SubmissionStatus.BLOCKED

    clock.advance(NOT_FOUND_GRACE_SEC)
    assert risk2.refresh_unresolved_derisk() == 0
    assert rows(engine)[cad][1] == "NOT_FOUND"
    risk2.check(DD)
    assert [(o.symbol, o.side, o.quantity) for o in broker2.orders] == [("USDCAD", "buy", 2000.0)]
    assert broker2.net == {"EURUSD": 0.0, "USDCAD": 0.0}


def test_restart_with_unreachable_broker_keeps_fence_and_halt(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'a.db'}")
    install_attempts(engine)
    broker1 = VenueBroker({"EURUSD": 1000.0}, script=["lost_filled"])
    mk(broker1, store=SqlEmergencyAttemptStore(engine)).check(DD)
    broker2 = VenueBroker(dict(broker1.net))
    broker2.lookups[str(broker1.orders[0].client_order_id)] = ConnectionError("down")
    risk2 = mk(broker2, store=SqlEmergencyAttemptStore(engine))
    risk2.recover_emergency_attempts()
    assert risk2.unresolved_derisk_symbols() == ["EURUSD"]
    risk2.check(DD)
    assert broker2.orders == []
    assert risk2.attempt_auto_resume({"price_stream_age_sec": 1.0}) is False
    assert risk2.oms.is_halted()


def test_auto_resume_and_operator_resume_cannot_lift_unresolved_halt() -> None:
    broker = VenueBroker({"EURUSD": 1000.0}, script=["working"])
    risk = mk(broker)
    risk.check({**DD, "price_stream_age_sec": 9999.0})  # stale_prices also fires
    assert {"stale_prices", UNRESOLVED_CAUSE} <= risk.active_halt_causes()

    assert risk.attempt_auto_resume({"price_stream_age_sec": 1.0}) is False
    assert risk.active_halt_causes() == {UNRESOLVED_CAUSE}
    assert risk.oms.is_halted()

    risk.oms.resume_trades()  # what /api/system/resume does before re-arming
    risk.reset_daily(clear_causes=True)
    assert risk.active_halt_causes() == {UNRESOLVED_CAUSE}
    assert risk.oms.is_halted()

    risk.on_broker_fill(fill_event(broker.orders[0], -1000.0, "T1"))
    assert risk.unresolved_derisk_symbols() == []
    assert risk.attempt_auto_resume({"price_stream_age_sec": 1.0}) is False  # still sticky
    risk.reset_daily(clear_causes=True)  # operator resume after resolution
    assert risk.active_halt_causes() == frozenset()


def test_api_reports_fences_and_refuses_resume_until_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from src.web import api

    monkeypatch.setenv("WEB_API_SECRET", "test-secret-pksi")
    auth = {"X-API-Key": "test-secret-pksi"}
    broker = VenueBroker({"EURUSD": 1000.0}, script=["working"])
    risk = mk(broker)
    risk.check(DD)
    saved = dict(api._runtime)
    api._runtime.clear()
    api._runtime.update(
        {"broker": broker, "oms": risk.oms, "strategies": [], "kill_switch_manager": risk}
    )
    try:
        client = TestClient(api.app)
        status = client.get("/api/system", headers=auth).json()
        assert status["derisk_fences"]["count"] == 1
        assert status["derisk_fences"]["symbols"] == ["EURUSD"]
        assert status["derisk_fences"]["attempts"][0]["status"] == "WORKING"
        assert client.post("/api/system/resume", headers=auth).status_code == 409
        assert risk.oms.is_halted()
        r = client.post(
            "/api/system/derisk-fences/release",
            headers=auth,
            json={"symbol": "EURUSD", "reason": "verified flat at broker", "changed_by": "jack"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["derisk_fences"]["count"] == 0
        assert client.post("/api/system/resume", headers=auth).status_code == 200
        assert not risk.oms.is_halted()
    finally:
        api._runtime.clear()
        api._runtime.update(saved)


def test_listener_failure_never_breaks_fill_journaling() -> None:
    oms = OrderManager(VenueBroker({}))  # type: ignore[arg-type]

    def boom(fill: dict[str, Any]) -> None:
        raise RuntimeError("listener bug")

    oms.add_fill_listener(boom)
    assert oms.on_fill({"transaction_id": "T1", "client_order_id": "c1", "units": 1.0}) is True


# --------------------------------------------------------------------------- #
# Pure evidence model
# --------------------------------------------------------------------------- #


def _attempt(requested: float = 1000.0) -> EmergencyAttempt:
    return EmergencyAttempt(
        intent_id="i1",
        client_order_id="i1",
        action="kill_switch_flatten",
        symbol="EURUSD",
        route_symbol="EURUSD",
        original_qty=requested,
        target=0.0,
        requested_qty=requested,
        status=AttemptStatus.WORKING,
        created_at=T0,
        updated_at=T0,
    )


@settings(max_examples=200, deadline=None)
@given(
    units=st.dictionaries(
        st.sampled_from([f"T{i}" for i in range(8)]),
        st.integers(min_value=1, max_value=400),
        min_size=1,
    ),
    data=st.data(),
)
def test_cumulative_fill_is_sum_of_distinct_transactions(
    units: dict[str, int], data: st.DataObject
) -> None:
    deliveries = data.draw(
        st.lists(st.sampled_from(sorted(units)), min_size=len(units), max_size=3 * len(units))
    )
    deliveries += sorted(units)  # every fill delivered at least once
    attempt = _attempt()
    seen: list[float] = []
    for txn in deliveries:
        attempt = apply_evidence(
            attempt,
            DeriskEvidence(
                source=EVIDENCE_FILL_EVENT,
                client_order_id="i1",
                fill_transaction_id=txn,
                fill_qty=float(units[txn]),
            ),
            now=T0,
        )
        seen.append(attempt.cumulative_fill_qty)
    expected = float(sum(units.values()))
    assert attempt.cumulative_fill_qty == expected
    assert seen == sorted(seen)  # monotone: duplicates never subtract or add
    assert (attempt.status is AttemptStatus.FILLED) == (expected >= 999.0)


def test_lookup_after_terminal_never_regresses_status() -> None:
    done = replace_status(_attempt(), AttemptStatus.FILLED)
    after = apply_evidence(
        done,
        DeriskEvidence(source="order_lookup", client_order_id="i1", not_found=True),
        now=T0 + timedelta(hours=1),
    )
    assert after.status is AttemptStatus.FILLED


def replace_status(a: EmergencyAttempt, status: AttemptStatus) -> EmergencyAttempt:
    from dataclasses import replace

    return replace(a, status=status)


# --------------------------------------------------------------------------- #
# OandaBroker.get_order through a mocked transport
# --------------------------------------------------------------------------- #


class _Transport:
    def __init__(self, routes: dict[str, tuple[int, dict[str, Any]]]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, path: str, **_: Any) -> SimpleNamespace:
        self.calls.append(path)
        status, payload = self.routes.get(path, (404, {"errorMessage": "not found"}))

        def raise_for_status() -> None:
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")

        return SimpleNamespace(
            status_code=status,
            json=lambda: payload,
            raise_for_status=raise_for_status,
            headers={},
            text="",
        )


def _oanda(routes: dict[str, tuple[int, dict[str, Any]]]) -> OandaBroker:
    b = OandaBroker.__new__(OandaBroker)
    b.account_id = "ACC"
    b.api_key = "secret"
    b.client = _Transport(routes)  # type: ignore[assignment]
    return b


def test_oanda_get_order_by_client_id_reads_fill_transaction() -> None:
    b = _oanda(
        {
            "/v3/accounts/ACC/orders/@cid-1": (
                200,
                {
                    "order": {
                        "id": "101",
                        "type": "MARKET",
                        "instrument": "EUR_USD",
                        "units": "-1000",
                        "state": "FILLED",
                        "fillingTransactionID": "102",
                        "clientExtensions": {"id": "cid-1"},
                    }
                },
            ),
            "/v3/accounts/ACC/transactions/102": (
                200,
                {"transaction": {"id": "102", "type": "ORDER_FILL", "units": "-1000"}},
            ),
        }
    )
    order = b.get_order_by_client_id("cid-1")
    assert (order.status, order.filled_quantity, order.fill_transaction_id) == (
        OrderStatus.FILLED,
        1000.0,
        "102",
    )
    assert (order.symbol, order.side, order.quantity, order.order_id, order.client_order_id) == (
        "EURUSD",
        "sell",
        1000.0,
        "101",
        "cid-1",
    )


def test_oanda_get_order_cancelled_without_fill_reports_zero() -> None:
    b = _oanda(
        {
            "/v3/accounts/ACC/orders/77": (
                200,
                {
                    "order": {
                        "id": "77",
                        "instrument": "USD_CAD",
                        "units": "500",
                        "state": "CANCELLED",
                    }
                },
            )
        }
    )
    order = b.get_order("77")
    assert (order.status, order.filled_quantity) == (OrderStatus.CANCELLED, 0.0)


def test_oanda_get_order_unreadable_fill_amount_is_unknown_not_zero() -> None:
    b = _oanda(
        {
            "/v3/accounts/ACC/orders/9": (
                200,
                {
                    "order": {
                        "id": "9",
                        "instrument": "EUR_USD",
                        "units": "100",
                        "state": "CANCELLED",
                        "fillingTransactionID": "10",
                    }
                },
            ),
            "/v3/accounts/ACC/transactions/10": (500, {}),
        }
    )
    order = b.get_order("9")
    assert order.status is OrderStatus.CANCELLED
    assert order.filled_quantity is None


def test_oanda_get_order_404_is_not_found_and_bad_state_raises() -> None:
    b = _oanda(
        {"/v3/accounts/ACC/orders/5": (200, {"order": {"id": "5", "state": "WEIRD", "units": "1"}})}
    )
    with pytest.raises(BrokerOrderNotFoundError):
        b.get_order_by_client_id("missing")
    with pytest.raises(ValueError):
        b.get_order("5")
    with pytest.raises(ValueError):
        b.get_order("../positions")
    assert all(path.startswith("/v3/accounts/ACC/orders/") for path in b.client.calls)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Engine wiring: recovery precedes the cold-start reconciler; ticks refresh
# --------------------------------------------------------------------------- #


def _context_stub() -> Any:
    from unittest.mock import Mock

    return Mock(build=Mock(return_value={}), consume_day_rollover=lambda: False)


def test_engine_startup_refences_before_reconciler_can_submit() -> None:
    from unittest.mock import Mock

    from src.runtime.live_engine import LiveEngine

    store = InMemoryEmergencyAttemptStore()
    first = VenueBroker({"EURUSD": 1000.0}, script=["lost_unfilled"])
    mk(first, store=store).check(DD)
    broker = VenueBroker({"EURUSD": 1000.0})
    broker.lookups[str(first.orders[0].client_order_id)] = ConnectionError("down")
    risk = mk(broker, store=store)
    seen: list[Any] = []

    def reconcile() -> SimpleNamespace:
        seen.append(risk.unresolved_derisk_symbols())
        seen.append(
            risk.oms.submit_intent_result(
                OrderIntent(strategy_id="reconciler", symbol="EURUSD", target_position=0.0)
            ).status
        )
        return SimpleNamespace(entries=[], has_mismatches=False)

    engine = LiveEngine(
        [],
        risk.oms,
        broker,  # type: ignore[arg-type]
        cold_start_reconciler=Mock(reconcile=reconcile),
        kill_switch_manager=risk,
        risk_context_builder=_context_stub(),
    )
    engine._reconcile_startup()
    assert seen == [["EURUSD"], SubmissionStatus.BLOCKED]
    assert broker.orders == []
    assert UNRESOLVED_CAUSE in risk.active_halt_causes()


def test_health_tick_resolves_fences_from_broker_lookups() -> None:
    from src.runtime.live_engine import LiveEngine

    broker = VenueBroker({"EURUSD": 1000.0}, script=["lost_filled"])
    broker.get_account = lambda: SimpleNamespace(equity=100_000.0)  # type: ignore[attr-defined]
    risk = mk(broker)
    risk.check(DD)
    order = broker.orders[0]
    broker.lookups[str(order.client_order_id)] = lookup(order, OrderStatus.FILLED, 1000.0, "T3")
    engine = LiveEngine(
        [],
        risk.oms,
        broker,  # type: ignore[arg-type]
        kill_switch_manager=risk,
        risk_context_builder=_context_stub(),
    )
    engine._health_tick()
    assert risk.unresolved_derisk_symbols() == []
    assert broker.lookup_calls == [order.client_order_id]
    assert len(broker.orders) == 1
