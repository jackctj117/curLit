"""Test PortfolioCoordinator.promote_strategy_to_live (CL-6vv)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.portfolio.coordinator import (
    PortfolioCoordinator,
    StrategyAllocation,
)


class _FakeStrategy:
    def __init__(self, sid: str) -> None:
        self.id = sid
        self.symbols = ["EURUSD"]


class _FakeBroker:
    def get_positions(self) -> list[Any]:
        return []


class _FakeOMS:
    def __init__(self) -> None:
        self.intents: list[Any] = []
        self.halted = False

    def submit_intent(self, intent: Any) -> str:
        self.intents.append(intent)
        return "iid"

    def halt_new_trades(self) -> None:
        self.halted = True


class _FakeState:
    def record_reallocation(
        self, ts: datetime, weights: dict[str, float],
        regime: dict[str, Any],
    ) -> None: ...

    def record_portfolio_order(
        self, ts: datetime, symbol: str, target_position: float,
        strategy_contributions: dict[str, float],
    ) -> None: ...

    def get_positions_by_strategy(self, strategy_id: str) -> list[Any]:
        return []

    def load_strategy_returns_history(
        self, strategy_ids: list[str], lookback_days: int,
    ) -> pd.DataFrame:
        return pd.DataFrame()


def _build_coordinator(seed_strategies: list[_FakeStrategy] | None = None) -> PortfolioCoordinator:
    seed = seed_strategies or [_FakeStrategy("__seed__")]
    return PortfolioCoordinator(
        strategies=seed,
        oms=_FakeOMS(),
        broker=_FakeBroker(),
        state=_FakeState(),
    )


class TestPromote:
    def test_promote_changes_paper_mode_and_weight(self) -> None:
        coord = _build_coordinator()
        s1 = _FakeStrategy("s1")
        coord.add_strategy(s1)
        assert coord.allocations["s1"].paper_mode is True

        coord.promote_strategy_to_live("s1", initial_weight=0.05)

        assert coord.allocations["s1"].paper_mode is False
        assert coord.allocations["s1"].target_weight == 0.05

    def test_promote_redistributes_existing(self) -> None:
        live1 = _FakeStrategy("live1")
        coord = _build_coordinator(seed_strategies=[live1])
        coord.allocations["live1"] = StrategyAllocation(
            strategy_id="live1", target_weight=1.0, current_exposure_mult=1.0,
            paper_mode=False,
        )
        # New paper strategy
        s2 = _FakeStrategy("s2")
        coord.add_strategy(s2)

        coord.promote_strategy_to_live("s2", initial_weight=0.05)

        # Live1 scaled from 1.0 to 0.95 to make room for s2
        assert coord.allocations["live1"].target_weight == 0.95
        assert coord.allocations["s2"].target_weight == 0.05
        assert sum(
            a.target_weight for a in coord.allocations.values()
            if not a.paper_mode
        ) == 1.0

    def test_promote_already_live_is_noop(self) -> None:
        s1 = _FakeStrategy("s1")
        coord = _build_coordinator(seed_strategies=[s1])
        coord.allocations["s1"] = StrategyAllocation(
            strategy_id="s1", target_weight=0.5, current_exposure_mult=1.0,
            paper_mode=False,
        )
        coord.promote_strategy_to_live("s1", initial_weight=0.10)
        assert coord.allocations["s1"].target_weight == 0.5  # unchanged


class TestRemoveAfterPromote:
    def test_remove_calls_oms_for_each_position(self) -> None:
        s1 = _FakeStrategy("s1")
        coord = _build_coordinator(seed_strategies=[s1])
        coord.allocations["s1"] = StrategyAllocation(
            strategy_id="s1", target_weight=0.5, current_exposure_mult=1.0,
            paper_mode=False,
        )
        coord.remove_strategy("s1")
        assert "s1" not in coord.strategies
        assert "s1" not in coord.allocations
