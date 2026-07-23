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


# --------------------------------------------------------------------------- #
# REJECTED-status routing (ultrareview #2)
# --------------------------------------------------------------------------- #


class _RejectingBroker:
    """Returns a REJECTED order (no exception) — the OANDA reject shape."""

    def __init__(self) -> None:
        self.calls = 0

    def get_positions(self):
        return []

    def place_order(self, order):  # noqa: ANN001, ANN201
        from src.execution.broker import OrderStatus

        self.calls += 1
        order.status = OrderStatus.REJECTED
        order.reject_reason = "INSUFFICIENT_MARGIN"
        return order


class _RecordingRejectionHandler:
    def __init__(self) -> None:
        self.handled: list[str] = []

    def handle(self, intent, order, exc, attempt, response_text=None):  # noqa: ANN001, ANN201
        from types import SimpleNamespace

        self.handled.append(str(exc))
        return SimpleNamespace(
            should_retry=False,
            halt_strategy=False,
            sleep_sec=0.0,
            next_size_fraction=1.0,
            final_resolution=SimpleNamespace(value="abort"),
        )


def test_rejected_status_routes_through_rejection_handler():
    """Before the fix: a REJECTED order entered _pending forever, was
    journaled ORDER_PLACED, and never reached the RejectionHandler."""
    broker = _RejectingBroker()
    handler = _RecordingRejectionHandler()
    oms = OrderManager(broker, rejection_handler=handler)  # type: ignore[arg-type]
    oms.submit_intent(_intent("USD_CAD", -8916.0))
    assert handler.handled == ["INSUFFICIENT_MARGIN"]  # policy path fired
    assert oms.has_pending() is False  # no poisoned pending


def test_rejected_status_without_handler_drops_cleanly():
    broker = _RejectingBroker()
    oms = OrderManager(broker)  # type: ignore[arg-type]
    oms.submit_intent(_intent("USD_CAD", -8916.0))
    assert oms.has_pending() is False
