"""Unit tests — execution.rejection: classification + handler resolutions + OMS wiring (CL-yteo)."""

from __future__ import annotations

from typing import Any

import pytest

from src.execution.broker import Order, OrderType
from src.execution.oms import OrderIntent, OrderManager
from src.execution.paper_broker import PaperBroker
from src.execution.rejection import (
    ClassPolicy,
    RejectionClass,
    RejectionHandler,
    RejectionPolicy,
    RejectionResolution,
    classify_exception,
)

# =============================================================================
# Classification
# =============================================================================


class TestClassification:
    def test_margin_classified(self) -> None:
        assert classify_exception(Exception("INSUFFICIENT_MARGIN")) == RejectionClass.MARGIN
        assert classify_exception(Exception("Insufficient margin to place order")) == RejectionClass.MARGIN

    def test_liquidity_classified(self) -> None:
        assert classify_exception(Exception("FOK fill failed")) == RejectionClass.LIQUIDITY
        assert classify_exception(Exception("no liquidity at requested price")) == RejectionClass.LIQUIDITY

    def test_halt_classified(self) -> None:
        assert classify_exception(Exception("instrument halted")) == RejectionClass.HALT
        assert classify_exception(Exception("market closed for maintenance")) == RejectionClass.HALT

    def test_transient_classified(self) -> None:
        assert classify_exception(Exception("connection reset")) == RejectionClass.TRANSIENT
        assert classify_exception(Exception("503 Service Unavailable")) == RejectionClass.TRANSIENT
        assert classify_exception(Exception("timeout reading from broker")) == RejectionClass.TRANSIENT

    def test_malformed_classified(self) -> None:
        assert classify_exception(Exception("400 Bad Request")) == RejectionClass.MALFORMED
        assert classify_exception(Exception("invalid json payload")) == RejectionClass.MALFORMED

    def test_unknown_default(self) -> None:
        assert classify_exception(Exception("something completely random")) == RejectionClass.UNKNOWN

    def test_response_text_used_when_provided(self) -> None:
        # Exception message is innocuous; rejection signal lives in body.
        cls = classify_exception(
            Exception("HTTPError"),
            response_text='{"errorMessage": "INSUFFICIENT_MARGIN"}',
        )
        assert cls == RejectionClass.MARGIN


# =============================================================================
# Handler resolutions per class
# =============================================================================


def _make_intent_and_order(qty: float = 1000.0) -> tuple[OrderIntent, Order]:
    intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=qty)
    order = Order(symbol="EURUSD", side="buy", quantity=qty, order_type=OrderType.MARKET)
    return intent, order


class TestHandlerLiquidity:
    def test_first_attempt_retry_smaller(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        out = handler.handle(intent, order, Exception("FOK fill failed"), attempt=1)
        assert out.should_retry is True
        # Halve once for attempt=1 → next_size_fraction = 0.5.
        assert out.next_size_fraction == pytest.approx(0.5)
        assert out.halt_strategy is False

    def test_exhausts_attempts(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        out = handler.handle(intent, order, Exception("FOK fill failed"), attempt=3)
        assert out.should_retry is False
        assert out.halt_strategy is False

    def test_below_min_fraction_aborts(self) -> None:
        # With max_attempts=10 (override) and halving, attempt 4 → 0.0625 < min 0.10 → abort.
        policy = RejectionPolicy(by_class={
            RejectionClass.LIQUIDITY: ClassPolicy(
                resolution=RejectionResolution.RETRY_SMALLER,
                max_attempts=10,
            ),
            **{
                k: v for k, v in RejectionPolicy.default().by_class.items()
                if k != RejectionClass.LIQUIDITY
            },
        })
        handler = RejectionHandler(policy)
        intent, order = _make_intent_and_order()
        out = handler.handle(intent, order, Exception("FOK fill failed"), attempt=4)
        assert out.should_retry is False


class TestHandlerMargin:
    def test_aborts_immediately(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        out = handler.handle(intent, order, Exception("INSUFFICIENT_MARGIN"), attempt=1)
        assert out.should_retry is False
        assert out.halt_strategy is False
        assert out.final_resolution == RejectionResolution.ABORT


class TestHandlerHalt:
    def test_halts_strategy(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        out = handler.handle(intent, order, Exception("instrument halted"), attempt=1)
        assert out.should_retry is False
        assert out.halt_strategy is True
        assert out.final_resolution == RejectionResolution.ABORT_HALT_STRATEGY


class TestHandlerTransient:
    def test_first_attempt_retries_with_backoff(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        out = handler.handle(intent, order, Exception("connection reset"), attempt=1)
        assert out.should_retry is True
        # First attempt sleep = initial (1.0s).
        assert out.sleep_sec == pytest.approx(1.0)
        assert out.next_size_fraction == pytest.approx(1.0)

    def test_backoff_doubles(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        out2 = handler.handle(intent, order, Exception("connection reset"), attempt=2)
        out3 = handler.handle(intent, order, Exception("connection reset"), attempt=3)
        assert out2.sleep_sec == pytest.approx(2.0)
        assert out3.sleep_sec == pytest.approx(4.0)

    def test_exhausts_attempts(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        # default TRANSIENT max_attempts = 4
        out = handler.handle(intent, order, Exception("connection reset"), attempt=4)
        assert out.should_retry is False


class TestHandlerMalformed:
    def test_aborts_no_retry(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        out = handler.handle(intent, order, Exception("400 Bad Request"), attempt=1)
        assert out.should_retry is False
        assert out.halt_strategy is False


class TestHandlerUnknown:
    def test_aborts_conservatively(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        out = handler.handle(intent, order, Exception("strange new error"), attempt=1)
        assert out.should_retry is False


class TestEventLog:
    def test_each_handle_logs_event(self) -> None:
        handler = RejectionHandler()
        intent, order = _make_intent_and_order()
        handler.handle(intent, order, Exception("INSUFFICIENT_MARGIN"), attempt=1)
        handler.handle(intent, order, Exception("FOK fill failed"), attempt=1)
        events = handler.events
        assert len(events) == 2
        assert events[0].rejection_class == RejectionClass.MARGIN
        assert events[1].rejection_class == RejectionClass.LIQUIDITY


# =============================================================================
# OMS integration
# =============================================================================


class _AlwaysFailBroker(PaperBroker):
    """Broker that raises a configurable exception on every place_order."""

    def __init__(self, exc_factory: Any) -> None:
        super().__init__(initial_capital=100_000)
        self.set_price("EURUSD", 1.0999, 1.1001)
        self._exc_factory = exc_factory
        self.attempts = 0

    def place_order(self, order: Order) -> Order:
        self.attempts += 1
        raise self._exc_factory()


class _SecondAttemptSuccessBroker(PaperBroker):
    """Fails once with TRANSIENT, succeeds on retry — verifies retry path."""

    def __init__(self) -> None:
        super().__init__(initial_capital=100_000)
        self.set_price("EURUSD", 1.0999, 1.1001)
        self.attempts = 0

    def place_order(self, order: Order) -> Order:
        self.attempts += 1
        if self.attempts == 1:
            raise ConnectionError("connection reset")
        # Defer to PaperBroker for actual fill on retry.
        return super().place_order(order)


class TestOMSRejectionWiring:
    def test_legacy_no_handler_logs_and_drops(self) -> None:
        broker = _AlwaysFailBroker(lambda: Exception("INSUFFICIENT_MARGIN"))
        oms = OrderManager(broker)  # no handler
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)
        oms.submit_intent(intent)
        # Single attempt without handler — legacy behavior.
        assert broker.attempts == 1

    def test_margin_aborts_after_one_attempt(self) -> None:
        # Patch out actual sleeping for fast tests.
        handler = RejectionHandler()
        handler.sleep = lambda _s: None  # type: ignore[method-assign]
        broker = _AlwaysFailBroker(lambda: Exception("INSUFFICIENT_MARGIN"))
        oms = OrderManager(broker, rejection_handler=handler)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)
        oms.submit_intent(intent)
        # MARGIN max_attempts=1 → exactly 1 broker call.
        assert broker.attempts == 1
        assert handler.events[0].rejection_class == RejectionClass.MARGIN

    def test_liquidity_retries_with_smaller_size(self) -> None:
        handler = RejectionHandler()
        handler.sleep = lambda _s: None  # type: ignore[method-assign]
        broker = _AlwaysFailBroker(lambda: Exception("FOK fill failed"))
        oms = OrderManager(broker, rejection_handler=handler)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)
        oms.submit_intent(intent)
        # LIQUIDITY max_attempts=3 → 3 broker calls.
        assert broker.attempts == 3
        # All 3 should be classified LIQUIDITY.
        assert all(e.rejection_class == RejectionClass.LIQUIDITY for e in handler.events)

    def test_transient_recovers_after_retry(self) -> None:
        handler = RejectionHandler()
        handler.sleep = lambda _s: None  # type: ignore[method-assign]
        broker = _SecondAttemptSuccessBroker()
        oms = OrderManager(broker, rejection_handler=handler)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=500)
        oms.submit_intent(intent)
        # Failed once + succeeded once.
        assert broker.attempts == 2
        # Position present on broker post-retry.
        assert any(p.symbol == "EURUSD" for p in broker.get_positions())

    def test_halt_invokes_strategy_halt_callback(self) -> None:
        halted: list[tuple[str, str]] = []

        def on_halt(sid: str, sym: str) -> None:
            halted.append((sid, sym))

        handler = RejectionHandler()
        handler.sleep = lambda _s: None  # type: ignore[method-assign]
        broker = _AlwaysFailBroker(lambda: Exception("instrument halted"))
        oms = OrderManager(broker, rejection_handler=handler, on_strategy_halt=on_halt)
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)
        oms.submit_intent(intent)
        assert halted == [("s1", "EURUSD")]

    def test_halted_oms_does_not_attempt(self) -> None:
        # Pre-existing OMS halt path should still take precedence over rejection retry.
        broker = _AlwaysFailBroker(lambda: Exception("INSUFFICIENT_MARGIN"))
        oms = OrderManager(broker, rejection_handler=RejectionHandler())
        oms.halt_new_trades()
        intent = OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)
        oms.submit_intent(intent)
        assert broker.attempts == 0
