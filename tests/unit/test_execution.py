"""Unit tests — execution: PaperBroker, OMS."""

import threading

from src.execution.broker import (
    Account,
    Broker,
    Order,
    OrderStatus,
    OrderType,
    Position,
)
from src.execution.oms import OrderIntent, OrderManager
from src.execution.paper_broker import PaperBroker


class TestPaperBroker:
    def test_buy_sell_roundtrip(self) -> None:
        broker = PaperBroker(initial_capital=10000)
        broker.set_price("EURUSD", 1.1000, 1.1002)

        order = Order(symbol="EURUSD", side="buy", quantity=1000, order_type=OrderType.MARKET)
        placed = broker.place_order(order)
        assert placed.status == OrderStatus.FILLED
        assert len(broker.get_positions()) == 1

        sell = Order(symbol="EURUSD", side="sell", quantity=1000, order_type=OrderType.MARKET)
        broker.place_order(sell)
        assert len(broker.get_positions()) == 1
        assert abs(broker.get_positions()[0].quantity) < 0.01


class TestOrderManager:
    def test_submit_intent_places_order(self) -> None:
        broker = PaperBroker()
        broker.set_price("EURUSD", 1.1000, 1.1002)
        oms = OrderManager(broker)
        intent = OrderIntent(strategy_id="test", symbol="EURUSD", target_position=500)
        oms.submit_intent(intent)
        assert len(broker.get_positions()) > 0

    def test_halt_blocks_new_intents(self) -> None:
        broker = PaperBroker()
        oms = OrderManager(broker)
        oms.halt_new_trades()
        intent = OrderIntent(strategy_id="test", symbol="EURUSD", target_position=100)
        oms.submit_intent(intent)
        assert len(broker.get_positions()) == 0

    def test_bypass_halt_allows_risk_layer_derisking(self) -> None:
        # CL-i4tx: kill-switch flatten/reduce intents must not be blocked by
        # a prior halt_new — "halt NEW trades" cannot veto emergency closes.
        broker = PaperBroker()
        broker.set_price("EURUSD", 1.1000, 1.1002)
        oms = OrderManager(broker)
        oms.submit_intent(
            OrderIntent(strategy_id="test", symbol="EURUSD", target_position=500),
        )
        oms.halt_new_trades()
        oms.submit_intent(
            OrderIntent(
                strategy_id="kill_switch_flatten", symbol="EURUSD",
                target_position=0,
            ),
            bypass_halt=True,
        )
        assert abs(broker.get_positions()[0].quantity) < 0.01


class TestOrderManagerAsync:
    def test_submit_intent_async_offloads_broker_io(self) -> None:
        """CL-xdnh: the async wrapper must run the blocking submit path
        (broker get_positions + place_order) in a worker thread, never on
        the event-loop thread."""
        import asyncio
        import threading

        broker = PaperBroker()
        broker.set_price("EURUSD", 1.1000, 1.1002)
        call_threads: list[int] = []
        original = broker.place_order

        def spy(order):  # noqa: ANN001, ANN202
            call_threads.append(threading.get_ident())
            return original(order)

        broker.place_order = spy  # type: ignore[method-assign]
        oms = OrderManager(broker)
        intent = OrderIntent(strategy_id="t", symbol="EURUSD", target_position=500)

        async def run() -> tuple[int, str]:
            loop_thread = threading.get_ident()
            intent_id = await oms.submit_intent_async(intent)
            return loop_thread, intent_id

        loop_thread, intent_id = asyncio.run(run())
        assert intent_id == intent.intent_id
        assert len(broker.get_positions()) == 1
        assert call_threads and all(t != loop_thread for t in call_threads)

    def test_submit_intent_async_respects_halt_and_bypass(self) -> None:
        import asyncio

        broker = PaperBroker()
        broker.set_price("EURUSD", 1.1000, 1.1002)
        oms = OrderManager(broker)

        async def run() -> None:
            await oms.submit_intent_async(
                OrderIntent(strategy_id="t", symbol="EURUSD", target_position=500),
            )
            oms.halt_new_trades()
            # Halted: plain async submit is rejected...
            await oms.submit_intent_async(
                OrderIntent(strategy_id="t", symbol="EURUSD", target_position=900),
            )
            # ...but bypass_halt (risk-layer de-risking) still flows.
            await oms.submit_intent_async(
                OrderIntent(
                    strategy_id="kill_switch_flatten", symbol="EURUSD",
                    target_position=0,
                ),
                bypass_halt=True,
            )

        asyncio.run(run())
        assert abs(broker.get_positions()[0].quantity) < 0.01


class _BlockingBroker(Broker):
    """Broker whose place_order for NON-emergency orders blocks on an event.

    Emergency orders (bypass_halt → Order.emergency=True) fill immediately.
    Lets a test prove the OMS lock is NOT held across the blocking
    place_order call (CL-8s2a): an emergency de-risk must complete while a
    slow normal order is still mid-flight.
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self.slow_started = threading.Event()
        self.emergency_done = threading.Event()

    def place_order(self, order: Order) -> Order:
        if order.emergency:
            order.status = OrderStatus.FILLED
            self.emergency_done.set()
            return order
        # Slow NORMAL order: signal we've entered, then block until released.
        self.slow_started.set()
        # If the OMS still held the RLock here, the emergency submit below
        # could never acquire it — the test would deadlock/time out.
        released = self.release.wait(timeout=5.0)
        assert released, "slow order was never released"
        order.status = OrderStatus.FILLED
        return order

    def cancel_order(self, order_id: str) -> bool:  # noqa: ARG002
        return True

    def get_order(self, order_id: str) -> Order:  # noqa: ARG002
        raise NotImplementedError

    def get_positions(self) -> list[Position]:
        return []

    def get_account(self) -> Account:
        return Account(balance=100_000.0, equity=100_000.0)

    def get_price(self, symbol: str) -> tuple[float, float]:  # noqa: ARG002
        return (1.1000, 1.1002)

    async def stream_prices(self, symbols):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN201, ARG002
        if False:
            yield {}


class TestOmsLockWidth:
    def test_bypass_halt_not_blocked_by_slow_normal_submit(self) -> None:
        """CL-8s2a: shrinking the OMS critical section means a slow strategy
        submit (blocked in place_order + retry) must NOT hold the RLock, so an
        emergency bypass_halt de-risk placed concurrently completes without
        waiting for the slow order to be released."""
        broker = _BlockingBroker()
        oms = OrderManager(broker)

        # Thread 1: a normal strategy submit that blocks inside place_order.
        def slow_normal() -> None:
            oms.submit_intent(
                OrderIntent(strategy_id="strat", symbol="EURUSD",
                            target_position=1000.0),
            )

        t_slow = threading.Thread(target=slow_normal)
        t_slow.start()
        # Wait until the slow order is actually inside place_order (i.e. past
        # the guarded decision section and into the released blocking region).
        assert broker.slow_started.wait(timeout=5.0), "slow order never started"

        # Thread 2: emergency de-risk. If the lock were still held across the
        # slow place_order, this submit would block until release.set().
        def emergency() -> None:
            oms.submit_intent(
                OrderIntent(strategy_id="kill_switch_flatten", symbol="GBPUSD",
                            target_position=500.0),
                bypass_halt=True,
            )

        t_emerg = threading.Thread(target=emergency)
        t_emerg.start()

        # The emergency order must fill while the slow one is STILL blocked.
        assert broker.emergency_done.wait(timeout=5.0), (
            "emergency de-risk was blocked behind the slow normal submit — "
            "the OMS lock is still held across place_order (CL-8s2a regressed)"
        )
        assert not broker.release.is_set(), "test bug: slow order released early"
        t_emerg.join(timeout=5.0)
        assert not t_emerg.is_alive()

        # Release the slow order and let it finish cleanly.
        broker.release.set()
        t_slow.join(timeout=5.0)
        assert not t_slow.is_alive()

    def test_halt_decision_still_serialized_under_lock(self) -> None:
        """The halt GATE stays inside the lock (halt-TOCTOU, CL-8lv6): a
        halted OMS still rejects a non-reducing normal intent even though the
        blocking submit now runs lock-free."""
        broker = PaperBroker()
        broker.set_price("EURUSD", 1.1000, 1.1002)
        oms = OrderManager(broker)
        oms.halt_new_trades()
        oms.submit_intent(
            OrderIntent(strategy_id="strat", symbol="EURUSD",
                        target_position=1000.0),
        )
        assert broker.get_positions() == []  # blocked, nothing placed

    def test_pending_cleared_on_synchronous_fill_lock_free_path(self) -> None:
        """_pending FILLED-cleanup (CL-8lv6) stays correct with the lock now
        re-acquired only around the _pending write (CL-8s2a): a synchronously
        FILLED order must not linger in _pending."""
        broker = PaperBroker()
        broker.set_price("EURUSD", 1.1000, 1.1002)
        oms = OrderManager(broker)
        oms.submit_intent(
            OrderIntent(strategy_id="strat", symbol="EURUSD",
                        target_position=1000.0),
        )
        assert oms.has_pending() is False
