"""OMS symbol-form regression tests (CL-qqra).

The live incident: broker positions key compact ("USDCAD") while event
intents use OANDA-underscore ("USD_CAD"); the raw dict lookup missed, so
EXITS computed delta 0 and never closed at the broker, and ENTRIES saw
"flat" and could stack. These tests pin the canonical-key fix.
"""

from __future__ import annotations

from src.execution.broker import Position, canonical_symbol
from src.execution.oms import OrderIntent, OrderManager


class _Broker:
    """Fake broker holding compact-symbol positions; records orders."""

    def __init__(self, positions: list[Position]) -> None:
        self._positions = positions
        self.orders: list[tuple[str, str, float]] = []

    def get_positions(self) -> list[Position]:
        return self._positions

    def place_order(self, order):  # noqa: ANN001, ANN201
        self.orders.append((order.symbol, order.side, order.quantity))
        from src.execution.broker import OrderStatus
        order.status = OrderStatus.FILLED
        order.order_id = "t1"
        return order


def _intent(symbol: str, target: float) -> OrderIntent:
    return OrderIntent(strategy_id="ev", symbol=symbol, target_position=target)


def test_canonical_symbol_bridges_dialects():
    assert canonical_symbol("USD_CAD") == "USDCAD"
    assert canonical_symbol("usdcad") == "USDCAD"
    assert canonical_symbol("EUR/USD") == "EURUSD"
    assert canonical_symbol("XAU_USD") == "XAUUSD"


def test_exit_closes_position_across_symbol_dialects():
    """THE incident: target 0 for USD_CAD vs broker USDCAD -8916 must emit a
    closing BUY 8916 — before the fix, delta was 0 and nothing was sent."""
    broker = _Broker([Position(symbol="USDCAD", quantity=-8916.0, avg_price=1.411)])
    oms = OrderManager(broker)  # type: ignore[arg-type]
    oms.submit_intent(_intent("USD_CAD", 0.0))
    assert broker.orders == [("USD_CAD", "buy", 8916.0)]


def test_entry_does_not_stack_on_existing_position():
    """Target -8916 while the broker already holds -8916 must be a no-op —
    before the fix, the lookup missed and it doubled the position."""
    broker = _Broker([Position(symbol="USDNOK", quantity=-1292.0, avg_price=9.63)])
    oms = OrderManager(broker)  # type: ignore[arg-type]
    oms.submit_intent(_intent("USD_NOK", -1292.0))
    assert broker.orders == []  # delta 0 → nothing submitted


def test_partial_resize_uses_true_delta():
    broker = _Broker([Position(symbol="EURUSD", quantity=1000.0, avg_price=1.1)])
    oms = OrderManager(broker)  # type: ignore[arg-type]
    oms.submit_intent(_intent("EUR_USD", 1500.0))
    assert broker.orders == [("EUR_USD", "buy", 500.0)]


def test_compact_intents_unaffected():
    broker = _Broker([Position(symbol="EURUSD", quantity=1000.0, avg_price=1.1)])
    oms = OrderManager(broker)  # type: ignore[arg-type]
    oms.submit_intent(_intent("EURUSD", 0.0))
    assert broker.orders == [("EURUSD", "sell", 1000.0)]
