"""Tests for multi-strategy intent semantics (CL-8lv6 P0).

The OMS deltas against the ABSOLUTE broker position; the coordinator must
therefore aggregate every strategy's remembered share of a symbol, not
just this tick's intents — plus self-sized pass-through, book seeding,
halt-allows-reducing, and _pending lifecycle.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from src.execution.broker import Position
from src.execution.oms import OrderIntent, OrderManager
from src.execution.paper_broker import PaperBroker
from src.portfolio.coordinator import PortfolioCoordinator, StrategyAllocation


@dataclass
class _Strat:
    id: str
    self_sized: bool = False
    open_positions: dict[str, Any] = field(default_factory=dict)


class _RecOMS:
    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    def submit_intent(self, intent: OrderIntent, **kw: object) -> str:
        self.submitted.append(intent)
        return intent.intent_id


class _NullState:
    def record_portfolio_order(self, *a: object, **k: object) -> None: ...
    def record_reallocation(self, *a: object, **k: object) -> None: ...
    def get_positions_by_strategy(self, sid: str) -> list[Any]:
        return []
    def load_strategy_returns_history(self, *a: object, **k: object):  # noqa: ANN201
        import pandas as pd
        return pd.DataFrame()


def _coord(strats: list[_Strat]):  # noqa: ANN202
    broker = PaperBroker(initial_capital=100_000)
    broker.set_price("EURUSD", 1.0999, 1.1001)
    oms = _RecOMS()
    coord = PortfolioCoordinator(
        strategies=strats,  # type: ignore[arg-type]
        oms=oms,  # type: ignore[arg-type]
        broker=broker,
        state=_NullState(),  # type: ignore[arg-type]
    )
    for s in strats:
        coord.allocations[s.id] = StrategyAllocation(
            strategy_id=s.id, target_weight=1.0 / len(strats),
        )
    return coord, oms


def _run(coord, intents):  # noqa: ANN001, ANN202
    return asyncio.run(coord.process_intents(intents))


# --------------------------------------------------------------------------- #
# cross-tick target memory
# --------------------------------------------------------------------------- #


def test_shared_symbol_preserves_other_strategys_share() -> None:
    a, b = _Strat("a", self_sized=True), _Strat("b", self_sized=True)
    coord, oms = _coord([a, b])
    _run(coord, {"a": [OrderIntent(strategy_id="a", symbol="EURUSD",
                                   target_position=100.0)]})
    # Tick 2: only b speaks — a's remembered 100 must still be counted.
    _run(coord, {"b": [OrderIntent(strategy_id="b", symbol="EURUSD",
                                   target_position=50.0)]})
    assert oms.submitted[-1].target_position == 150.0


def test_exit_to_zero_does_not_flatten_everyone() -> None:
    a, b = _Strat("a", self_sized=True), _Strat("b", self_sized=True)
    coord, oms = _coord([a, b])
    _run(coord, {
        "a": [OrderIntent(strategy_id="a", symbol="EURUSD",
                          target_position=100.0)],
        "b": [OrderIntent(strategy_id="b", symbol="EURUSD",
                          target_position=50.0)],
    })
    assert oms.submitted[-1].target_position == 150.0
    # a exits: aggregate must fall to b's 50, NOT to 0.
    _run(coord, {"a": [OrderIntent(strategy_id="a", symbol="EURUSD",
                                   target_position=0.0)]})
    assert oms.submitted[-1].target_position == 50.0


def test_restart_seeds_targets_from_books() -> None:
    @dataclass
    class _Pos:
        quantity: float

    a = _Strat("a", self_sized=True,
               open_positions={"EUR_USD": _Pos(quantity=-15_462.0)})
    b = _Strat("b", self_sized=True)
    coord, oms = _coord([a, b])
    # First tick after "restart": only b trades the shared symbol — a's
    # book share must be included via seeding (canonical match EUR_USD ==
    # EURUSD).
    _run(coord, {"b": [OrderIntent(strategy_id="b", symbol="EURUSD",
                                   target_position=1_000.0)]})
    assert oms.submitted[-1].target_position == -14_462.0


# --------------------------------------------------------------------------- #
# self-sized pass-through
# --------------------------------------------------------------------------- #


def test_self_sized_strategy_not_scaled() -> None:
    ev = _Strat("event", self_sized=True)
    other = _Strat("mr")
    coord, oms = _coord([ev, other])  # equal weight would be 0.5
    _run(coord, {"event": [OrderIntent(strategy_id="event", symbol="EURUSD",
                                       target_position=1_000.0)]})
    assert oms.submitted[-1].target_position == 1_000.0  # NOT 500


def test_scaled_strategy_still_scaled() -> None:
    ev = _Strat("event", self_sized=True)
    other = _Strat("mr")
    coord, oms = _coord([ev, other])
    _run(coord, {"mr": [OrderIntent(strategy_id="mr", symbol="EURUSD",
                                    target_position=1_000.0)]})
    assert oms.submitted[-1].target_position == 500.0  # 0.5 weight applies


# --------------------------------------------------------------------------- #
# OMS: halt allows reducing; _pending lifecycle
# --------------------------------------------------------------------------- #


def _oms_with_position(qty: float = 100.0):  # noqa: ANN202
    broker = PaperBroker(initial_capital=100_000)
    broker.set_price("EURUSD", 1.0999, 1.1001)
    oms = OrderManager(broker)
    oms.submit_intent(OrderIntent(strategy_id="s", symbol="EURUSD",
                                  target_position=qty))
    assert broker.get_positions()[0].quantity == qty
    return oms, broker


def test_halted_oms_allows_reducing_exit() -> None:
    oms, broker = _oms_with_position(100.0)
    oms.halt_new_trades()
    oms.submit_intent(OrderIntent(strategy_id="s", symbol="EURUSD",
                                  target_position=0.0))
    # Exit went through the halt (zero-qty row may or may not linger —
    # representation detail owned by the broker).
    assert sum(p.quantity for p in broker.get_positions()) == 0.0


def test_halted_oms_blocks_adds_and_flips() -> None:
    oms, broker = _oms_with_position(100.0)
    oms.halt_new_trades()
    oms.submit_intent(OrderIntent(strategy_id="s", symbol="EURUSD",
                                  target_position=200.0))  # add
    oms.submit_intent(OrderIntent(strategy_id="s", symbol="EURUSD",
                                  target_position=-50.0))  # flip
    oms.submit_intent(OrderIntent(strategy_id="s", symbol="GBPUSD",
                                  target_position=10.0))  # new position
    assert [p.quantity for p in broker.get_positions()] == [100.0]


def test_pending_cleared_on_synchronous_fill() -> None:
    oms, _broker = _oms_with_position(100.0)
    # PaperBroker fills synchronously — nothing may linger in _pending.
    assert not oms.has_pending()


def test_emergency_flag_set_from_bypass_halt() -> None:
    broker = PaperBroker(initial_capital=100_000)
    broker.set_price("EURUSD", 1.0999, 1.1001)
    placed: list[Any] = []
    original = broker.place_order

    def spy(order):  # noqa: ANN001, ANN202
        placed.append(order)
        return original(order)

    broker.place_order = spy  # type: ignore[method-assign]
    oms = OrderManager(broker)
    oms.submit_intent(OrderIntent(strategy_id="s", symbol="EURUSD",
                                  target_position=10.0))
    oms.submit_intent(OrderIntent(strategy_id="risk", symbol="EURUSD",
                                  target_position=0.0), bypass_halt=True)
    assert placed[0].emergency is False
    assert placed[1].emergency is True


def test_position_seed_positions_param_respected() -> None:
    # sanity: snapshot path still works alongside the new halt logic
    broker = PaperBroker(initial_capital=100_000)
    broker.set_price("EURUSD", 1.0999, 1.1001)
    oms = OrderManager(broker)
    snap = [Position(symbol="EURUSD", quantity=0.0, avg_price=1.1)]
    oms.submit_intent(OrderIntent(strategy_id="s", symbol="EURUSD",
                                  target_position=5.0), positions=snap)
    assert broker.get_positions()[0].quantity == 5.0
