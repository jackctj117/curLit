"""CL-80tv: broker position reads must not run under the OMS state lock.

Oracle: the requirement in CL-80tv — a slow ``get_positions`` must not block a
concurrent ``halt_new_trades()`` nor another symbol's submit; a fill processed
while a read is in flight must be detected (no delta from a stale book); and
fill handling stays exactly-once (one journal row per venue transaction id).
The broker is faked at its external boundary (``get_positions`` /
``place_order``); timing is controlled with threading.Event, never sleeps.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from src.execution.broker import Order, OrderStatus, Position
from src.execution.oms import OrderIntent, OrderManager, SubmissionStatus
from src.execution.trade_journal import EventType

#: Upper bound for "did not block": generous for a loaded CI box, far below
#: the indefinite wait the old lock scope produced.
_NOT_BLOCKED_SEC = 5.0


class _Journal:
    def __init__(self, fail_times: int = 0) -> None:
        self.rows: list[SimpleNamespace] = []
        self._fail = fail_times
        self._lock = threading.Lock()

    def record(
        self,
        *,
        event_type: EventType,
        payload: dict[str, Any],
        intent_id: str | None,
        strategy_id: str | None,
        symbol: str | None,
    ) -> None:
        with self._lock:
            if self._fail > 0:
                self._fail -= 1
                raise RuntimeError("journal down")
            self.rows.append(
                SimpleNamespace(event_type=event_type, payload=payload, intent_id=intent_id)
            )

    def query_by_intent(self, intent_id: str) -> list[SimpleNamespace]:
        with self._lock:
            return [r for r in self.rows if r.intent_id == intent_id]

    def fills(self) -> list[SimpleNamespace]:
        return [r for r in self.rows if r.event_type is EventType.ORDER_FILLED]


class _Broker:
    """get_positions blocks on ``release`` for calls made from a thread named
    ``slow``; ``on_read`` (if set) runs inside the read and may return the
    snapshot to report."""

    def __init__(self) -> None:
        self.book: dict[str, float] = {}
        self.entered = threading.Event()
        self.release = threading.Event()
        self.reads = 0
        self.on_read: Any = None
        self.placed: list[Order] = []
        self._lock = threading.Lock()

    def get_positions(self) -> list[Position]:
        with self._lock:
            self.reads += 1
            n = self.reads
        if threading.current_thread().name == "slow":
            self.entered.set()
            assert self.release.wait(_NOT_BLOCKED_SEC * 2), "test never released the read"
        if self.on_read is not None:
            snap = self.on_read(n)
            if snap is not None:
                return snap
        return [Position(symbol=s, quantity=q, avg_price=1.0) for s, q in self.book.items() if q]

    def place_order(self, order: Order) -> Order:
        with self._lock:
            self.placed.append(order)
            signed = order.quantity if order.side == "buy" else -order.quantity
            self.book[order.symbol] = self.book.get(order.symbol, 0.0) + signed
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        return order


def _fill(txn: str, client: str | None = "c1", instrument: str = "EURUSD") -> dict[str, Any]:
    return {
        "type": "ORDER_FILL",
        "transaction_id": txn,
        "order_id": "o-" + txn,
        "client_order_id": client,
        "instrument": instrument,
        "units": 1000.0,
        "price": 1.1,
    }


def _run(name: str, fn: Any) -> tuple[threading.Thread, dict[str, Any]]:
    box: dict[str, Any] = {}

    def body() -> None:
        box["result"] = fn()

    t = threading.Thread(target=body, name=name, daemon=True)
    t.start()
    return t, box


class TestReadOutsideLock:
    def test_slow_read_does_not_block_halt_and_halt_is_honored(self) -> None:
        broker = _Broker()
        oms = OrderManager(broker)  # type: ignore[arg-type]
        intent = OrderIntent(strategy_id="s", symbol="EURUSD", target_position=1000.0)
        slow, box = _run("slow", lambda: oms.submit_intent_result(intent))
        try:
            assert broker.entered.wait(_NOT_BLOCKED_SEC)
            halter, _ = _run("halter", oms.halt_new_trades)
            halter.join(_NOT_BLOCKED_SEC)
            assert not halter.is_alive(), "halt_new_trades blocked behind a broker read"
            assert oms.is_halted()
        finally:
            broker.release.set()
        slow.join(_NOT_BLOCKED_SEC)
        # CL-8lv6: the halt that landed during the read still gates the entry.
        assert box["result"].status is SubmissionStatus.BLOCKED
        assert broker.placed == []

    def test_slow_read_does_not_block_other_symbol_submit(self) -> None:
        broker = _Broker()
        oms = OrderManager(broker)  # type: ignore[arg-type]
        slow_intent = OrderIntent(strategy_id="s", symbol="EURUSD", target_position=1000.0)
        other = OrderIntent(strategy_id="s", symbol="USDJPY", target_position=2000.0)
        slow, slow_box = _run("slow", lambda: oms.submit_intent_result(slow_intent))
        try:
            assert broker.entered.wait(_NOT_BLOCKED_SEC)
            fast, box = _run("fast", lambda: oms.submit_intent_result(other))
            fast.join(_NOT_BLOCKED_SEC)
            assert not fast.is_alive(), "other-symbol submit blocked behind a broker read"
            assert box["result"].status is SubmissionStatus.FILLED
            assert [o.symbol for o in broker.placed] == ["USDJPY"]
        finally:
            broker.release.set()
        slow.join(_NOT_BLOCKED_SEC)
        assert slow_box["result"].status is SubmissionStatus.FILLED
        assert sorted(o.symbol for o in broker.placed) == ["EURUSD", "USDJPY"]

    def test_fill_during_read_is_detected_and_book_reread(self) -> None:
        broker = _Broker()
        oms = OrderManager(broker)  # type: ignore[arg-type]

        def on_read(n: int) -> list[Position] | None:
            if n == 1:
                # A stream fill for the same symbol lands mid-read, on another
                # thread (as the engine's fill task does). The venue snapshot
                # we return predates it.
                t, _ = _run("fill", lambda: oms.on_fill(_fill("t-mid", client="earlier")))
                t.join(_NOT_BLOCKED_SEC)
                assert not t.is_alive(), "on_fill blocked behind a broker read"
                return []
            return [Position(symbol="EURUSD", quantity=1000.0, avg_price=1.0)]

        broker.on_read = on_read
        intent = OrderIntent(strategy_id="s", symbol="EURUSD", target_position=1000.0)
        res = oms.submit_intent_result(intent)
        assert broker.reads == 2  # stale snapshot discarded, book re-read
        # Sized from the post-fill book: already at target, nothing sent. The
        # stale snapshot would have bought another 1000 (doubling the leg).
        assert res.status is SubmissionStatus.AT_TARGET
        assert broker.placed == []

    def test_book_moving_on_every_read_is_refused_not_sized(self) -> None:
        broker = _Broker()
        oms = OrderManager(broker)  # type: ignore[arg-type]

        def on_read(n: int) -> list[Position]:
            t, _ = _run("fill", lambda: oms.on_fill(_fill(f"t{n}", client=None)))
            t.join(_NOT_BLOCKED_SEC)
            return []

        broker.on_read = on_read
        res = oms.submit_intent_result(
            OrderIntent(strategy_id="s", symbol="EURUSD", target_position=1000.0)
        )
        assert res.status is SubmissionStatus.BLOCKED
        assert broker.placed == []
        assert broker.reads == OrderManager._POSITION_READ_ATTEMPTS

    def test_other_symbol_fill_during_read_does_not_force_reread(self) -> None:
        broker = _Broker()
        oms = OrderManager(broker)  # type: ignore[arg-type]

        def on_read(n: int) -> None:
            if n == 1:
                t, _ = _run("fill", lambda: oms.on_fill(_fill("t-jpy", instrument="USDJPY")))
                t.join(_NOT_BLOCKED_SEC)

        broker.on_read = on_read
        res = oms.submit_intent_result(
            OrderIntent(strategy_id="s", symbol="EURUSD", target_position=1000.0)
        )
        assert res.status is SubmissionStatus.FILLED
        assert broker.reads == 1

    def test_fence_set_during_read_is_honored(self) -> None:
        broker = _Broker()
        oms = OrderManager(broker)  # type: ignore[arg-type]
        intent = OrderIntent(strategy_id="s", symbol="EURUSD", target_position=1000.0)
        slow, box = _run("slow", lambda: oms.submit_intent_result(intent))
        try:
            assert broker.entered.wait(_NOT_BLOCKED_SEC)
            oms.fence_symbol("EUR_USD", "emergency order unresolved", owner="someone-else")
        finally:
            broker.release.set()
        slow.join(_NOT_BLOCKED_SEC)
        assert box["result"].status is SubmissionStatus.BLOCKED
        assert broker.placed == []


class _RetryOnce:
    """Duck-typed RejectionHandler: retry every failure at full size."""

    def handle(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            halt_strategy=False,
            should_retry=True,
            sleep_sec=0.0,
            next_size_fraction=1.0,
            final_resolution=SimpleNamespace(value="retry"),
        )

    def sleep(self, _sec: float) -> None:
        return None


class TestRetryReread:
    def test_fill_during_retry_reread_is_detected(self) -> None:
        broker = _Broker()
        oms = OrderManager(broker, rejection_handler=_RetryOnce())  # type: ignore[arg-type]
        calls = {"place": 0}

        def place(order: Order) -> Order:
            calls["place"] += 1
            raise TimeoutError("read timed out after the venue accepted it")

        broker.place_order = place  # type: ignore[method-assign]

        def on_read(n: int) -> list[Position] | None:
            if n == 2:  # the retry re-read; the prior attempt's fill streams in
                t, _ = _run("fill", lambda: oms.on_fill(_fill("t-prior", client=None)))
                t.join(_NOT_BLOCKED_SEC)
                return []
            if n >= 3:
                return [Position(symbol="EURUSD", quantity=1000.0, avg_price=1.0)]
            return []

        broker.on_read = on_read
        res = oms.submit_intent_result(
            OrderIntent(strategy_id="s", symbol="EURUSD", target_position=1000.0)
        )
        # The stale re-read (flat) would have re-sent and doubled the position.
        assert res.status is SubmissionStatus.AT_TARGET
        assert calls["place"] == 1


class TestFillExactlyOnce:
    def test_concurrent_duplicate_delivery_journals_once(self) -> None:
        journal = _Journal()
        oms = OrderManager(_Broker(), journal=journal)  # type: ignore[arg-type]
        seen: list[dict[str, Any]] = []
        oms.add_fill_listener(seen.append)
        start = threading.Barrier(8)
        results: list[bool] = []
        lock = threading.Lock()

        def deliver() -> None:
            start.wait(_NOT_BLOCKED_SEC)
            r = oms.on_fill(_fill("dup-1"))
            with lock:
                results.append(r)

        threads = [threading.Thread(target=deliver) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(_NOT_BLOCKED_SEC)
        assert results.count(True) == 1
        assert len(journal.fills()) == 1
        assert len(seen) == 1

    def test_failed_journal_is_not_durable_and_retry_journals_once(self) -> None:
        journal = _Journal(fail_times=1)
        oms = OrderManager(_Broker(), journal=journal)  # type: ignore[arg-type]
        intent = OrderIntent(strategy_id="strat-x", symbol="EURUSD", target_position=1000.0)
        oms._pending[intent.intent_id] = []
        oms._pending_intents[intent.intent_id] = intent
        seen: list[dict[str, Any]] = []
        oms.add_fill_listener(seen.append)
        first = oms.process_fill(_fill("t1", client=intent.intent_id))
        assert first.newly_processed and not first.durable
        assert journal.fills() == []
        second = oms.process_fill(_fill("t1", client=intent.intent_id))
        assert second.durable
        third = oms.process_fill(_fill("t1", client=intent.intent_id))
        assert not third.newly_processed and third.durable
        rows = journal.fills()
        assert len(rows) == 1
        # Attribution survives the retry; listeners fired exactly once.
        assert rows[0].intent_id == intent.intent_id
        assert len(seen) == 1

    def test_restart_replay_skips_fill_already_in_journal(self) -> None:
        journal = _Journal()
        first = OrderManager(_Broker(), journal=journal)  # type: ignore[arg-type]
        assert first.on_fill(_fill("t7", client="c7"))
        restarted = OrderManager(_Broker(), journal=journal)  # type: ignore[arg-type]
        out = restarted.process_fill(_fill("t7", client="c7"), check_journal=True)
        assert not out.newly_processed and out.durable
        assert len(journal.fills()) == 1

    def test_restart_replay_journal_lookup_failure_propagates(self) -> None:
        journal = _Journal()

        def broken(_iid: str) -> list[SimpleNamespace]:
            raise RuntimeError("db down")

        journal.query_by_intent = broken  # type: ignore[method-assign]
        oms = OrderManager(_Broker(), journal=journal)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError):
            oms.process_fill(_fill("t8"), check_journal=True)
        assert journal.fills() == []

    def test_sync_fill_records_fill_id_for_restart_dedup(self) -> None:
        journal = _Journal()
        broker = _Broker()
        oms = OrderManager(broker, journal=journal)  # type: ignore[arg-type]
        intent = OrderIntent(strategy_id="s", symbol="EURUSD", target_position=1000.0)

        def place(order: Order) -> Order:
            order.status = OrderStatus.FILLED
            order.order_id = "fill-txn-55"
            order.filled_quantity = order.quantity
            return order

        broker.place_order = place  # type: ignore[method-assign]
        oms.submit_intent_result(intent)
        restarted = OrderManager(_Broker(), journal=journal)  # type: ignore[arg-type]
        out = restarted.process_fill(
            _fill("fill-txn-55", client=intent.intent_id), check_journal=True
        )
        assert not out.newly_processed
        assert len(journal.fills()) == 1
