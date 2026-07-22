"""Tests for the last code-review residual fixes (CL-r8gv, CL-a0sv).

- Kill-switch construction failure must STOP the engine boot, not launch
  an unprotected engine.
- OMS reuses a coordinator-provided positions snapshot instead of one
  get_positions round-trip per intent.
"""

from __future__ import annotations

import pytest

from src.execution.broker import Position
from src.execution.oms import OrderIntent, OrderManager
from src.execution.paper_broker import PaperBroker


class _CountingBroker(PaperBroker):
    def __init__(self) -> None:
        super().__init__(initial_capital=100_000)
        self.position_fetches = 0

    def get_positions(self):  # type: ignore[override]
        self.position_fetches += 1
        return super().get_positions()


def test_kill_switch_build_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.risk.risk_profile as risk_profile
    from src.runtime.run_engine import build_kill_switch_manager

    def _boom() -> None:
        raise ValueError("corrupt risk profile")

    monkeypatch.setattr(risk_profile, "load_active_profile", _boom)
    broker = PaperBroker(initial_capital=100_000)
    oms = OrderManager(broker)
    with pytest.raises(RuntimeError, match="refusing to start"):
        build_kill_switch_manager(broker, oms)


def test_submit_intent_reuses_positions_snapshot() -> None:
    broker = _CountingBroker()
    oms = OrderManager(broker)
    snapshot = [Position(symbol="EURUSD", quantity=0.0, avg_price=1.1)]
    oms.submit_intent(
        OrderIntent(strategy_id="s", symbol="EURUSD", target_position=0.0),
        positions=snapshot,
    )
    assert broker.position_fetches == 0  # snapshot reused, no re-fetch


def test_submit_intent_fetches_when_no_snapshot() -> None:
    broker = _CountingBroker()
    oms = OrderManager(broker)
    oms.submit_intent(
        OrderIntent(strategy_id="s", symbol="EURUSD", target_position=0.0),
    )
    assert broker.position_fetches == 1
