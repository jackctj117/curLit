"""Durable account-wide entry halt (CL-0deu.2).

Store contract (fail-closed reads, explicit attributed resume, audit trail,
restart persistence, optimistic concurrency, acknowledgement accounting) and
the integration boundary: both Alpaca entry functions honor the halt at the
top of a cycle AND at the final pre-submit decision point, including a halt
that lands mid-cycle (the racing-worker case).

Oracle: the requirement text of CL-0deu.2's acceptance criteria — zero new
exposure while halted/unknown, halts survive restarts, resume is explicit,
"applied" only after every path acknowledges or is visibly unavailable.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.execution.alpaca_equity_executor import execute_pending_equities
from src.execution.alpaca_options_executor import execute_pending_options
from src.risk.trading_halt import (
    EXECUTION_PATHS,
    PATH_ALPACA_EQUITIES,
    PATH_ALPACA_OPTIONS,
    PATH_FX_OMS,
    REASON_ACTIVE,
    REASON_HALTED,
    REASON_UNAVAILABLE,
    HaltMode,
    HaltStateUnavailableError,
    TradingHaltStore,
)
from tests.unit import test_alpaca_equity_executor as eq_t
from tests.unit import test_alpaca_options_executor as opt_t
from tests.unit._trading_halt_fixture import install_trading_halt


@pytest.fixture
def halt_engine(tmp_path):  # type: ignore[no-untyped-def]
    return create_engine(f"sqlite:///{tmp_path / 'halt.db'}")


def _pause(store: TradingHaltStore, why: str = "operator test halt") -> None:
    store.request(HaltMode.PAUSE_ENTRIES, reason=why, source="test", changed_by="pytest")


# --------------------------------------------------------------------------- #
# store contract
# --------------------------------------------------------------------------- #


class TestStoreContract:
    def test_fresh_migration_is_paused_not_active(self, halt_engine: Any) -> None:
        # Deploying migration 024 must never silently unpause a book.
        store = install_trading_halt(halt_engine, active=False)
        state = store.read()
        assert state.mode is HaltMode.PAUSE_ENTRIES
        assert state.version == 1
        decision = store.entry_decision()
        assert decision.allowed is False
        assert decision.reason_code == REASON_HALTED

    def test_missing_table_fails_closed(self, halt_engine: Any) -> None:
        decision = TradingHaltStore(halt_engine).entry_decision()
        assert decision.allowed is False
        assert decision.reason_code == REASON_UNAVAILABLE
        with pytest.raises(HaltStateUnavailableError):
            TradingHaltStore(halt_engine).read()

    def test_missing_row_fails_closed(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        with halt_engine.begin() as c:
            c.execute(text("DELETE FROM trading_halt_state"))
        assert store.entry_decision().reason_code == REASON_UNAVAILABLE

    def test_resume_is_explicit_attributed_and_audited(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine, active=False)
        with pytest.raises(ValueError, match="reason"):
            store.resume(reason="  ", source="api", changed_by="op")
        with pytest.raises(ValueError, match="changed_by"):
            store.resume(reason="checked", source="api", changed_by="")
        state = store.resume(reason="reconciled; ok to trade", source="api", changed_by="jack")
        assert state.mode is HaltMode.ACTIVE and state.version == 2
        assert store.entry_decision().reason_code == REASON_ACTIVE
        with halt_engine.connect() as c:
            events = c.execute(
                text("SELECT version, mode, changed_by, reason FROM trading_halt_events")
            ).all()
        assert [tuple(e) for e in events] == [(2, "ACTIVE", "jack", "reconciled; ok to trade")]

    def test_same_mode_request_does_not_churn_version(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)  # ACTIVE v2
        _pause(store)
        v = store.read().version
        _pause(store, "kill switch fired again")
        assert store.read().version == v

    def test_emergency_flatten_is_refused_until_implemented(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        with pytest.raises(ValueError, match="EMERGENCY_FLATTEN"):
            store.request(
                HaltMode.EMERGENCY_FLATTEN, reason="x", source="test", changed_by="pytest"
            )
        assert store.read().mode is HaltMode.ACTIVE

    def test_restart_cannot_clear_a_halt(self, tmp_path: Any) -> None:
        url = f"sqlite:///{tmp_path / 'restart.db'}"
        store = install_trading_halt(create_engine(url))
        _pause(store)
        # A brand-new process: new engine, new store, no shared memory.
        reborn = TradingHaltStore(create_engine(url))
        assert reborn.entry_decision().allowed is False
        assert reborn.read().mode is HaltMode.PAUSE_ENTRIES

    def test_concurrent_writer_is_detected_and_retried(
        self, halt_engine: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = install_trading_halt(halt_engine)  # ACTIVE v2
        rival = TradingHaltStore(halt_engine)
        real_read = TradingHaltStore.read
        calls = {"n": 0}

        def read_then_race(self: TradingHaltStore) -> Any:
            state = real_read(self)
            calls["n"] += 1
            if calls["n"] == 1:
                # Between our read (v2) and our UPDATE, a rival bumps to v3.
                monkeypatch.setattr(TradingHaltStore, "read", real_read)
                rival.request(
                    HaltMode.CLOSE_ONLY, reason="rival", source="test", changed_by="rival"
                )
                monkeypatch.setattr(TradingHaltStore, "read", read_then_race)
            return state

        monkeypatch.setattr(TradingHaltStore, "read", read_then_race)
        final = store.request(
            HaltMode.PAUSE_ENTRIES, reason="ours", source="test", changed_by="pytest"
        )
        # Our stale-version UPDATE matched nothing; the retry built on v3.
        assert (final.mode, final.version) == (HaltMode.PAUSE_ENTRIES, 4)
        with halt_engine.connect() as c:
            versions = [r[0] for r in c.execute(text("SELECT version FROM trading_halt_events"))]
        assert sorted(versions) == [2, 3, 4]


class TestApplicationStatus:
    def test_not_applied_until_every_path_acknowledges(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        _pause(store)
        status = store.application_status()
        assert status.applied is False
        assert set(status.paths.values()) == {"missing"}
        store.observe(PATH_ALPACA_OPTIONS)
        store.observe(PATH_ALPACA_EQUITIES)
        assert store.application_status().applied is False  # fx_oms missing
        store.observe(PATH_FX_OMS)
        assert store.application_status().applied is True

    def test_older_ack_is_lagging_after_a_new_request(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        for path in EXECUTION_PATHS:
            store.observe(path)
        _pause(store)
        status = store.application_status()
        assert status.applied is False
        assert status.paths == {p: "lagging" for p in EXECUTION_PATHS}

    def test_visibly_unavailable_path_counts_as_accounted_for(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        _pause(store)
        store.observe(PATH_FX_OMS)
        store.observe(PATH_ALPACA_OPTIONS)
        store.acknowledge(
            PATH_ALPACA_EQUITIES, status="unavailable", state=None, detail="daemon disabled"
        )
        status = store.application_status()
        assert status.paths[PATH_ALPACA_EQUITIES] == "unavailable"
        assert status.applied is True


# --------------------------------------------------------------------------- #
# integration: the actual Alpaca entry functions
# --------------------------------------------------------------------------- #


class _HaltingOptionsClient(opt_t._FakeClient):
    """Accepts the first order, and while that submit is 'in flight' an
    operator halt lands — the racing-worker case."""

    def __init__(self, store: TradingHaltStore) -> None:
        super().__init__(ask=2.0)
        self.store = store

    def find_contracts(self, underlying: str, *a: Any, **k: Any) -> list[dict[str, Any]]:
        # One listed contract PER underlying, so the second idea is a genuine
        # independent entry (not blocked by the CL-0deu.1 underlying check).
        return [{**opt_t._CONTRACT, "symbol": f"{underlying}260821C00105000"}]

    def submit_option_order(self, *a: Any, **k: Any) -> dict[str, Any]:
        order = super().submit_option_order(*a, **k)
        if len(self.orders) == 1:
            _pause(self.store, "halt raced an in-flight entry")
        return order


class TestOptionsBoundary:
    @pytest.fixture
    def engine(self, tmp_path: Any) -> Any:
        return opt_t.engine.__wrapped__(tmp_path)  # type: ignore[attr-defined]

    def test_halt_blocks_every_entry_and_acknowledges(self, engine: Any) -> None:
        store = TradingHaltStore(engine)
        _pause(store)
        opt_t._seed(engine, "a")
        client = opt_t._FakeClient(ask=2.0)
        counts = execute_pending_options(
            engine, client, opt_t._price, now=opt_t.NOW, technicals_fn=opt_t._no_tech
        )
        assert counts["halted"] == 1 and counts["submitted"] == 0
        assert client.orders == []
        assert store.application_status().paths[PATH_ALPACA_OPTIONS] == "applied"

    def test_halt_is_acknowledged_even_when_market_closed(self, engine: Any) -> None:
        store = TradingHaltStore(engine)
        _pause(store)
        client = opt_t._FakeClient(market_open=False)
        execute_pending_options(
            engine, client, opt_t._price, now=opt_t.NOW, technicals_fn=opt_t._no_tech
        )
        assert store.application_status().paths[PATH_ALPACA_OPTIONS] == "applied"

    def test_unreadable_halt_state_blocks_entries(self, engine: Any) -> None:
        with engine.begin() as c:
            c.execute(text("DROP TABLE trading_halt_state"))
        opt_t._seed(engine, "a")
        client = opt_t._FakeClient(ask=2.0)
        counts = execute_pending_options(
            engine, client, opt_t._price, now=opt_t.NOW, technicals_fn=opt_t._no_tech
        )
        assert counts["halted"] == 1 and client.orders == []

    def test_halt_landing_mid_cycle_stops_the_next_submit(self, engine: Any) -> None:
        store = TradingHaltStore(engine)
        opt_t._seed(engine, "first", ticker="RTX")
        opt_t._seed(engine, "second", ticker="LMT")
        client = _HaltingOptionsClient(store)
        counts = execute_pending_options(
            engine, client, opt_t._price, now=opt_t.NOW, technicals_fn=opt_t._no_tech
        )
        # The already-sent order stands (it is documented residual exposure);
        # nothing is sent after the halt is observable.
        assert len(client.orders) == 1
        assert counts["submitted"] == 1 and counts["halted_at_submit"] == 1
        # The path acked the PRE-halt version at cycle start -> lagging until
        # its next quiescent point, then applied.
        assert store.application_status().paths[PATH_ALPACA_OPTIONS] == "lagging"
        execute_pending_options(
            engine, client, opt_t._price, now=opt_t.NOW, technicals_fn=opt_t._no_tech
        )
        assert store.application_status().paths[PATH_ALPACA_OPTIONS] == "applied"
        assert len(client.orders) == 1

    def test_explicit_resume_restores_entries(self, engine: Any) -> None:
        store = TradingHaltStore(engine)
        _pause(store)
        opt_t._seed(engine, "a")
        client = opt_t._FakeClient(ask=2.0)
        execute_pending_options(
            engine, client, opt_t._price, now=opt_t.NOW, technicals_fn=opt_t._no_tech
        )
        assert client.orders == []
        store.resume(reason="checked", source="test", changed_by="pytest")
        counts = execute_pending_options(
            engine, client, opt_t._price, now=opt_t.NOW, technicals_fn=opt_t._no_tech
        )
        assert counts["submitted"] == 1


class TestEquitiesBoundary:
    @pytest.fixture
    def engine(self, tmp_path: Any) -> Any:
        return eq_t.engine.__wrapped__(tmp_path)  # type: ignore[attr-defined]

    def test_halt_blocks_long_and_short_entries(self, engine: Any) -> None:
        store = TradingHaltStore(engine)
        _pause(store)
        eq_t._seed(engine, "long", ticker="RTX")
        eq_t._seed(engine, "short", ticker="LMT", action="buy_puts")
        client = eq_t._FakeClient()
        counts = execute_pending_equities(
            engine, client, eq_t._price, now=eq_t.NOW, technicals_fn=eq_t._no_tech
        )
        assert counts["halted"] == 1 and counts["submitted"] == 0
        assert client.orders == []
        assert store.application_status().paths[PATH_ALPACA_EQUITIES] == "applied"

    def test_unreadable_halt_state_blocks_entries(self, engine: Any) -> None:
        with engine.begin() as c:
            c.execute(text("DELETE FROM trading_halt_state"))
        eq_t._seed(engine, "long", ticker="RTX")
        client = eq_t._FakeClient()
        counts = execute_pending_equities(
            engine, client, eq_t._price, now=eq_t.NOW, technicals_fn=eq_t._no_tech
        )
        assert counts["halted"] == 1 and client.orders == []

    def test_halt_landing_mid_cycle_stops_the_next_submit(self, engine: Any) -> None:
        store = TradingHaltStore(engine)
        eq_t._seed(engine, "first", ticker="RTX")
        eq_t._seed(engine, "second", ticker="LMT")
        client = eq_t._FakeClient()
        real_submit = client.submit_equity_order

        def racing_submit(*a: Any, **k: Any) -> dict[str, Any]:
            order = real_submit(*a, **k)
            if len(client.orders) == 1:
                _pause(store, "halt raced an in-flight entry")
            return order

        client.submit_equity_order = racing_submit  # type: ignore[method-assign]
        counts = execute_pending_equities(
            engine, client, eq_t._price, now=eq_t.NOW, technicals_fn=eq_t._no_tech
        )
        assert len(client.orders) == 1
        assert counts["halted_at_submit"] == 1


# --------------------------------------------------------------------------- #
# FX path: the OMS gate and its quiescent acknowledgement
# --------------------------------------------------------------------------- #

from types import SimpleNamespace  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from src.execution.broker import OrderStatus, Position  # noqa: E402
from src.execution.oms import OrderIntent, OrderManager  # noqa: E402
from src.runtime.live_engine import LiveEngine  # noqa: E402
from src.web import api  # noqa: E402


class _FxBroker:
    def __init__(self, positions: list[Position] | None = None) -> None:
        self._positions = positions or []
        self.orders: list[tuple[str, str, float]] = []
        self.on_place: Any = None

    def get_positions(self) -> list[Position]:
        return self._positions

    def place_order(self, order: Any) -> Any:
        if self.on_place is not None:
            self.on_place()
        self.orders.append((order.symbol, order.side, order.quantity))
        order.status = OrderStatus.FILLED
        order.order_id = f"t{len(self.orders)}"
        return order


def _fx(symbol: str, target: float, strategy: str = "ev") -> OrderIntent:
    return OrderIntent(strategy_id=strategy, symbol=symbol, target_position=target)


class TestFxOms:
    def test_durable_halt_blocks_new_fx_exposure(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        _pause(store)
        broker = _FxBroker()
        oms = OrderManager(broker, halt_store=store)  # type: ignore[arg-type]
        oms.submit_intent(_fx("USD_JPY", 1000.0))
        assert broker.orders == []
        assert oms._halted is False  # durable halt, not the local flag

    def test_durable_halt_allows_genuine_reduction(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        _pause(store)
        broker = _FxBroker([Position(symbol="USDCAD", quantity=-8916.0, avg_price=1.41)])
        oms = OrderManager(broker, halt_store=store)  # type: ignore[arg-type]
        oms.submit_intent(_fx("USD_CAD", 0.0))
        assert broker.orders == [("USD_CAD", "buy", 8916.0)]

    def test_durable_halt_blocks_a_flip(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        _pause(store)
        broker = _FxBroker([Position(symbol="EURUSD", quantity=-100.0, avg_price=1.1)])
        oms = OrderManager(broker, halt_store=store)  # type: ignore[arg-type]
        oms.submit_intent(_fx("EUR_USD", 100.0))  # through zero = new exposure
        assert broker.orders == []

    def test_unreadable_state_blocks_fx_entries(self, halt_engine: Any) -> None:
        broker = _FxBroker()
        oms = OrderManager(broker, halt_store=TradingHaltStore(halt_engine))  # type: ignore[arg-type]
        oms.submit_intent(_fx("USD_JPY", 1000.0))
        assert broker.orders == []

    def test_active_state_allows_entries(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        broker = _FxBroker()
        oms = OrderManager(broker, halt_store=store)  # type: ignore[arg-type]
        oms.submit_intent(_fx("USD_JPY", 1000.0))
        assert broker.orders == [("USD_JPY", "buy", 1000.0)]

    def test_ack_is_deferred_while_an_entry_is_in_flight(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        broker = _FxBroker()
        oms = OrderManager(broker, halt_store=store)  # type: ignore[arg-type]
        seen: list[bool] = []
        # While place_order is running the entry is in flight: an ack
        # attempted now must be refused (RLock lets this same thread in).
        broker.on_place = lambda: seen.append(oms.acknowledge_portfolio_halt())
        oms.submit_intent(_fx("USD_JPY", 1000.0))
        assert seen == [False]
        assert store.application_status().paths[PATH_FX_OMS] == "missing"
        assert oms.acknowledge_portfolio_halt() is True
        assert store.application_status().paths[PATH_FX_OMS] == "applied"

    def test_health_tick_acks_before_any_broker_call(self, halt_engine: Any) -> None:
        store = install_trading_halt(halt_engine)
        _pause(store)
        oms = OrderManager(_FxBroker(), halt_store=store)  # type: ignore[arg-type]

        class _DeadBroker:
            def get_account(self) -> Any:
                raise ConnectionError("broker unreachable")

        fake_engine = SimpleNamespace(oms=oms, broker=_DeadBroker(), kill_switch_manager=None)
        with pytest.raises(ConnectionError):
            LiveEngine._health_tick(fake_engine)  # type: ignore[arg-type]
        assert store.application_status().paths[PATH_FX_OMS] == "applied"


# --------------------------------------------------------------------------- #
# web control plane: durable, attributed halt/resume; manual trades gated
# --------------------------------------------------------------------------- #

SECRET = "test-secret-halt"


@pytest.fixture
def web(halt_engine: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("WEB_API_SECRET", SECRET)
    saved = dict(api._runtime)
    store = install_trading_halt(halt_engine)
    broker = _FxBroker()
    oms = OrderManager(broker, halt_store=store)  # type: ignore[arg-type]
    api._runtime.clear()
    api._runtime.update(
        {"broker": broker, "oms": oms, "strategies": [], "kill_switch_manager": None}
    )
    api._runtime["halt_store"] = store
    yield SimpleNamespace(
        client=TestClient(api.app), store=store, oms=oms, broker=broker, engine=halt_engine
    )
    api._runtime.clear()
    api._runtime.update(saved)


AUTH = {"X-API-Key": SECRET}


class TestWebControl:
    def test_halt_is_durable_and_attributed(self, web: Any) -> None:
        r = web.client.post(
            "/api/system/halt",
            headers=AUTH,
            json={"reason": "news blackout", "changed_by": "jack"},
        )
        assert r.status_code == 200, r.text
        state = web.store.read()
        assert (state.mode, state.reason, state.changed_by) == (
            HaltMode.PAUSE_ENTRIES,
            "news blackout",
            "jack",
        )
        body = r.json()["account_halt"]
        assert body["entries_allowed"] is False and body["applied"] is False

    def test_bodyless_halt_still_works(self, web: Any) -> None:
        # An emergency halt must never fail on a missing field.
        r = web.client.post("/api/system/halt", headers=AUTH)
        assert r.status_code == 200
        assert web.store.read().mode is HaltMode.PAUSE_ENTRIES

    def test_resume_without_reason_is_refused_and_stays_halted(self, web: Any) -> None:
        web.client.post("/api/system/halt", headers=AUTH)
        r = web.client.post("/api/system/resume", headers=AUTH)
        assert r.status_code == 400
        assert web.store.read().mode is HaltMode.PAUSE_ENTRIES
        assert web.oms._halted is True

    def test_attributed_resume_clears_both_and_is_audited(self, web: Any) -> None:
        web.client.post("/api/system/halt", headers=AUTH)
        r = web.client.post(
            "/api/system/resume",
            headers=AUTH,
            json={"reason": "reconciled", "changed_by": "jack"},
        )
        assert r.status_code == 200, r.text
        assert web.store.read().mode is HaltMode.ACTIVE
        assert web.oms._halted is False
        with web.engine.connect() as c:
            last = c.execute(
                text("SELECT mode, changed_by FROM trading_halt_events ORDER BY version DESC")
            ).first()
        assert tuple(last) == ("ACTIVE", "jack")

    def test_unrecordable_halt_is_a_loud_503_but_fx_is_still_halted(self, web: Any) -> None:
        with web.engine.begin() as c:
            c.execute(text("DROP TABLE trading_halt_state"))
        r = web.client.post("/api/system/halt", headers=AUTH)
        assert r.status_code == 503
        assert "could NOT be recorded" in r.json()["detail"]
        assert web.oms._halted is True

    def test_emergency_flatten_request_is_rejected(self, web: Any) -> None:
        r = web.client.post("/api/system/halt", headers=AUTH, json={"mode": "EMERGENCY_FLATTEN"})
        assert r.status_code == 400

    def test_manual_trade_uses_the_same_gate(self, web: Any) -> None:
        web.store.request(HaltMode.PAUSE_ENTRIES, reason="x", source="test", changed_by="t")
        r = web.client.post(
            "/api/trade", headers=AUTH, json={"symbol": "USD_JPY", "target_position": 100}
        )
        assert r.status_code == 200, r.text
        assert web.broker.orders == []

    def test_halt_status_reports_per_path_application(self, web: Any) -> None:
        web.client.post("/api/system/halt", headers=AUTH)
        web.oms.acknowledge_portfolio_halt()
        body = web.client.get("/api/system/halt-status", headers=AUTH).json()
        assert body["mode"] == "PAUSE_ENTRIES"
        assert body["paths"][PATH_FX_OMS] == "applied"
        assert body["paths"][PATH_ALPACA_OPTIONS] == "missing"
        assert body["applied"] is False
