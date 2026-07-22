"""Unit tests — execution: PaperBroker, OMS."""

from src.execution.broker import Order, OrderStatus, OrderType
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
