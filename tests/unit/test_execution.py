"""Unit tests — execution: PaperBroker, OMS."""

import pytest

from src.execution.broker import Order, OrderType, OrderStatus
from src.execution.paper_broker import PaperBroker
from src.execution.oms import OrderManager, OrderIntent


class TestPaperBroker:
    def test_buy_sell_roundtrip(self) -> None:
        broker = PaperBroker(initial_capital=10000)
        broker.set_price("EURUSD", 1.1000, 1.1002)
        start_eq = broker.get_account().equity

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
