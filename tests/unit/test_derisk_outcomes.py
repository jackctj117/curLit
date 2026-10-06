"""CL-o9sq: risk actions use real OMS outcomes, not a fake that always raises."""

import pytest

from src.execution.broker import Order, OrderStatus, Position
from src.execution.oms import OrderIntent, OrderManager, SubmissionStatus
from src.risk.kill_switches import KillSwitchManager


class BrokerFixture:
    def __init__(self, status: OrderStatus = OrderStatus.REJECTED) -> None:
        self.status = status
        self.net = {"EURUSD": 1000.0, "USDCAD": -2000.0}
        self.orders: list[Order] = []
        self.fail_second = False
        self.unknown = False

    def get_positions(self) -> list[Position]:
        return [Position(symbol, qty, 1.0) for symbol, qty in self.net.items() if qty]

    def place_order(self, order: Order) -> Order:
        self.orders.append(order)
        if self.unknown:
            raise TimeoutError("response lost; acceptance unknown")
        order.status = (
            OrderStatus.REJECTED if self.fail_second and order.symbol == "USDCAD" else self.status
        )
        if order.status == OrderStatus.FILLED:
            self.net[order.symbol] += order.quantity * (1 if order.side == "buy" else -1)
        elif order.status == OrderStatus.REJECTED:
            order.reject_reason = "PRICE_BOUND_EXCEEDED"
        return order


def manager(broker: BrokerFixture) -> KillSwitchManager:
    return KillSwitchManager(broker, OrderManager(broker), {}, trailing_state_path=None)  # type: ignore[arg-type]


def test_rejected_flatten_does_not_spend_trigger_with_real_oms() -> None:
    broker = BrokerFixture()
    risk = manager(broker)
    fired = risk.check({"portfolio_dd": -0.5})
    assert len(broker.orders) == 2
    assert broker.net == {"EURUSD": 1000.0, "USDCAD": -2000.0}
    assert risk.oms.is_halted
    assert "drawdown_limit" not in risk._triggered_today
    assert fired[0]["effective"] is False
    broker.status = OrderStatus.FILLED
    risk.check({"portfolio_dd": -0.5})
    assert broker.net == {"EURUSD": 0.0, "USDCAD": 0.0}
    assert "drawdown_limit" in risk._triggered_today


def test_partial_reduction_retries_original_target_not_another_half() -> None:
    broker = BrokerFixture(OrderStatus.FILLED)
    broker.fail_second = True
    risk = manager(broker)
    context = {"vix_level": 40, "vix_change_1d": 0.6}
    risk.check(context)
    assert broker.net == {"EURUSD": 500.0, "USDCAD": -2000.0}
    assert "vix_spike" not in risk._triggered_today
    broker.fail_second = False
    risk.check(context)
    assert broker.net == {"EURUSD": 500.0, "USDCAD": -1000.0}
    assert "vix_spike" in risk._triggered_today
    assert len([order for order in broker.orders if order.symbol == "EURUSD"]) == 1


def test_previous_fill_cannot_hide_new_or_stale_exposure() -> None:
    broker = BrokerFixture(OrderStatus.FILLED)
    broker.fail_second = True
    risk = manager(broker)
    risk.check({"portfolio_dd": -0.5})
    broker.net["EURUSD"] = 1000.0  # position snapshot now disagrees with prior fill
    broker.fail_second = False
    risk.check({"portfolio_dd": -0.5})
    assert "drawdown_limit" not in risk._triggered_today
    assert len([order for order in broker.orders if order.symbol == "EURUSD"]) == 1


def test_uncertain_submission_is_not_success_or_blindly_retried() -> None:
    broker = BrokerFixture()
    broker.unknown = True
    risk = manager(broker)
    risk.check({"portfolio_dd": -0.5})
    risk.check({"portfolio_dd": -0.5})
    assert "drawdown_limit" not in risk._triggered_today
    assert len(broker.orders) == 2


def test_working_order_is_not_success_or_resubmitted() -> None:
    broker = BrokerFixture(OrderStatus.PENDING)
    risk = manager(broker)
    risk.check({"portfolio_dd": -0.5})
    risk.check({"portfolio_dd": -0.5})
    assert "drawdown_limit" not in risk._triggered_today
    assert len(broker.orders) == 2


@pytest.mark.parametrize("status", [OrderStatus.PARTIAL, OrderStatus.CANCELLED])
def test_partial_or_cancelled_close_requires_reconciliation(status: OrderStatus) -> None:
    broker = BrokerFixture(status)
    risk = manager(broker)
    risk.check({"portfolio_dd": -0.5})
    risk.reset_daily(clear_causes=False)
    risk.check({"portfolio_dd": -0.5})
    assert "drawdown_limit" not in risk._triggered_today
    assert len(broker.orders) == 2  # rollover cannot discard uncertain-order fences
    assert risk.oms.is_halted


@pytest.mark.parametrize("quantity", [float("nan"), float("inf"), True])
def test_invalid_position_snapshot_never_submits_a_partial_flatten(quantity: float) -> None:
    broker = BrokerFixture(OrderStatus.FILLED)
    broker.net["USDCAD"] = quantity
    risk = manager(broker)
    risk.check({"portfolio_dd": -0.5})
    assert not broker.orders
    assert "drawdown_limit" not in risk._triggered_today
    assert risk.oms.is_halted


@pytest.mark.parametrize("target", [1500.0, -500.0])
def test_emergency_label_cannot_add_or_reverse_exposure(target: float) -> None:
    broker = BrokerFixture(OrderStatus.FILLED)
    oms = OrderManager(broker)  # type: ignore[arg-type]
    oms.halt_new_trades()
    result = oms.submit_intent_result(
        OrderIntent(strategy_id="kill_switch_reduce", symbol="EURUSD", target_position=target),
        bypass_halt=True,
    )
    assert result.status is SubmissionStatus.BLOCKED
    assert not result.target_reached
    assert not broker.orders
