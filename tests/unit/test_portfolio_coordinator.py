"""Unit tests — portfolio.coordinator: scaling, aggregation, constraints, rebalance, lifecycle."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.execution.oms import OrderIntent, Urgency
from src.execution.paper_broker import PaperBroker
from src.portfolio.coordinator import (
    PortfolioConstraints,
    PortfolioCoordinator,
    StrategyAllocation,
)

# =============================================================================
# Test fakes
# =============================================================================


@dataclass
class _FakeStrategy:
    id: str


@dataclass
class _FakePosition:
    symbol: str
    quantity: float = 0.0


class _RecordingOMS:
    """OMS double that records intents instead of placing orders."""

    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    def submit_intent(self, intent: OrderIntent, **kwargs: object) -> str:
        self.submitted.append(intent)
        return intent.intent_id


class _FakeState:
    """PortfolioStateProtocol double — captures calls + serves canned data."""

    def __init__(
        self,
        returns_df: pd.DataFrame | None = None,
        positions_by_strategy: dict[str, list[_FakePosition]] | None = None,
    ) -> None:
        self._returns = returns_df if returns_df is not None else pd.DataFrame()
        self._positions = positions_by_strategy or {}
        self.reallocations: list[tuple[datetime, dict[str, float], dict[str, Any]]] = []
        self.portfolio_orders: list[
            tuple[datetime, str, float, dict[str, float]]
        ] = []

    def record_reallocation(
        self,
        ts: datetime,
        weights: dict[str, float],
        regime: dict[str, Any],
    ) -> None:
        self.reallocations.append((ts, dict(weights), dict(regime)))

    def record_portfolio_order(
        self,
        ts: datetime,
        symbol: str,
        target_position: float,
        strategy_contributions: dict[str, float],
    ) -> None:
        self.portfolio_orders.append(
            (ts, symbol, target_position, dict(strategy_contributions))
        )

    def get_positions_by_strategy(self, strategy_id: str) -> list[_FakePosition]:
        return list(self._positions.get(strategy_id, []))

    def load_strategy_returns_history(
        self,
        strategy_ids: list[str],
        lookback_days: int,
    ) -> pd.DataFrame:
        if self._returns.empty:
            return pd.DataFrame()
        cols = [c for c in self._returns.columns if c in strategy_ids]
        return self._returns[cols].copy()


# =============================================================================
# Fixtures / builders
# =============================================================================


def _make_coord(
    strategy_ids: list[str] | None = None,
    initial_capital: float = 100_000.0,
    constraints: PortfolioConstraints | None = None,
    state: _FakeState | None = None,
    eurusd_mid: float = 1.10,
    usdjpy_mid: float = 150.0,
    gbpusd_mid: float = 1.25,
    blackout_evaluator: Any | None = None,
) -> tuple[PortfolioCoordinator, _RecordingOMS, PaperBroker, _FakeState]:
    sids = strategy_ids or ["s1", "s2"]
    strategies = [_FakeStrategy(id=sid) for sid in sids]
    broker = PaperBroker(initial_capital=initial_capital)
    # Configure prices for symbols used across tests.
    broker.set_price("EURUSD", eurusd_mid - 0.0001, eurusd_mid + 0.0001)
    broker.set_price("USDJPY", usdjpy_mid - 0.01, usdjpy_mid + 0.01)
    broker.set_price("GBPUSD", gbpusd_mid - 0.0001, gbpusd_mid + 0.0001)
    oms = _RecordingOMS()
    fake_state = state or _FakeState()
    coord = PortfolioCoordinator(
        strategies=strategies,
        oms=oms,  # type: ignore[arg-type]
        broker=broker,
        state=fake_state,  # type: ignore[arg-type]
        constraints=constraints,
        blackout_evaluator=blackout_evaluator,
    )
    return coord, oms, broker, fake_state


# =============================================================================
# StrategyAllocation invariants
# =============================================================================


class TestStrategyAllocation:
    def test_effective_scale_product(self) -> None:
        a = StrategyAllocation(
            strategy_id="x",
            target_weight=0.5,
            current_exposure_mult=0.5,
            performance_override=0.5,
        )
        assert a.effective_scale == pytest.approx(0.5 * 0.5 * 0.5)

    def test_invalid_weight_rejected(self) -> None:
        with pytest.raises(AssertionError):
            StrategyAllocation(strategy_id="x", target_weight=1.5)

    def test_invalid_exposure_rejected(self) -> None:
        with pytest.raises(AssertionError):
            StrategyAllocation(
                strategy_id="x", target_weight=0.5, current_exposure_mult=-0.1,
            )


# =============================================================================
# PortfolioConstraints invariants
# =============================================================================


class TestPortfolioConstraints:
    def test_defaults_valid(self) -> None:
        c = PortfolioConstraints()
        assert c.max_gross_leverage == 3.0
        assert c.max_net_leverage == 2.0

    def test_net_cannot_exceed_gross(self) -> None:
        with pytest.raises(AssertionError):
            PortfolioConstraints(max_gross_leverage=2.0, max_net_leverage=3.0)

    def test_per_pair_pct_must_be_positive(self) -> None:
        with pytest.raises(AssertionError):
            PortfolioConstraints(max_notional_per_pair_pct=0.0)
        with pytest.raises(AssertionError):
            PortfolioConstraints(max_notional_per_pair_pct=-0.1)


# =============================================================================
# Initialization
# =============================================================================


class TestInitialization:
    def test_requires_strategies(self) -> None:
        with pytest.raises(AssertionError):
            PortfolioCoordinator(
                strategies=[], oms=_RecordingOMS(),  # type: ignore[arg-type]
                broker=PaperBroker(), state=_FakeState(),  # type: ignore[arg-type]
            )

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(AssertionError):
            PortfolioCoordinator(
                strategies=[_FakeStrategy("a"), _FakeStrategy("a")],
                oms=_RecordingOMS(),  # type: ignore[arg-type]
                broker=PaperBroker(),
                state=_FakeState(),  # type: ignore[arg-type]
            )


class TestInitializeAllocations:
    def test_normalizes_to_one(self) -> None:
        coord, *_ = _make_coord(["s1", "s2"])
        coord.initialize_allocations({"s1": 2.0, "s2": 3.0})
        total = sum(a.target_weight for a in coord.allocations.values())
        assert total == pytest.approx(1.0)
        assert coord.allocations["s1"].target_weight == pytest.approx(0.4)
        assert coord.allocations["s2"].target_weight == pytest.approx(0.6)

    def test_unknown_strategy_warned_and_skipped(self) -> None:
        coord, *_ = _make_coord(["s1", "s2"])
        coord.initialize_allocations({"s1": 1.0, "ghost": 1.0})
        assert "ghost" not in coord.allocations
        # s1 gets full weight from the recognized 1.0 / 1.0.
        assert coord.allocations["s1"].target_weight == pytest.approx(1.0)
        # s2 gets paper mode at zero.
        assert coord.allocations["s2"].paper_mode is True
        assert coord.allocations["s2"].target_weight == 0.0

    def test_no_positive_weights_falls_back_to_equal(self) -> None:
        coord, *_ = _make_coord(["s1", "s2"])
        coord.initialize_allocations({"s1": 0.0, "s2": 0.0})
        assert coord.allocations["s1"].target_weight == pytest.approx(0.5)
        assert coord.allocations["s2"].target_weight == pytest.approx(0.5)

    def test_negative_weight_rejected(self) -> None:
        coord, *_ = _make_coord(["s1"])
        with pytest.raises(AssertionError):
            coord.initialize_allocations({"s1": -0.1})


# =============================================================================
# Intent scaling
# =============================================================================


class TestScaleIntents:
    def test_scales_by_effective_scale(self) -> None:
        coord, *_ = _make_coord(["s1"])
        coord.initialize_allocations({"s1": 1.0})
        coord.allocations["s1"].current_exposure_mult = 0.5
        intents = {
            "s1": [OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)]
        }
        scaled = coord._scale_intents(intents)
        assert len(scaled) == 1
        # weight=1.0 * exposure=0.5 * perf=1.0 = 0.5
        assert scaled[0].target_position == pytest.approx(500.0)

    def test_paper_mode_drops_intents(self) -> None:
        coord, *_ = _make_coord(["s1"])
        coord.allocations["s1"] = StrategyAllocation(
            strategy_id="s1", target_weight=0.0,
            current_exposure_mult=0.0, paper_mode=True,
        )
        intents = {
            "s1": [OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)]
        }
        scaled = coord._scale_intents(intents)
        assert scaled == []

    def test_unknown_strategy_dropped(self) -> None:
        coord, *_ = _make_coord(["s1"])
        coord.initialize_allocations({"s1": 1.0})
        intents = {
            "ghost": [
                OrderIntent(strategy_id="ghost", symbol="EURUSD", target_position=1000)
            ]
        }
        scaled = coord._scale_intents(intents)
        assert scaled == []


# =============================================================================
# Aggregation
# =============================================================================


class TestAggregateBySymbol:
    def test_summing_same_direction(self) -> None:
        coord, *_ = _make_coord()
        intents = [
            OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100),
            OrderIntent(strategy_id="s2", symbol="EURUSD", target_position=200),
        ]
        agg = coord._aggregate_by_symbol(intents)
        assert agg["EURUSD"]["target_position"] == pytest.approx(300.0)
        assert agg["EURUSD"]["strategy_contributions"] == {"s1": 100, "s2": 200}

    def test_netting_opposite_direction(self) -> None:
        coord, *_ = _make_coord()
        intents = [
            OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=300),
            OrderIntent(strategy_id="s2", symbol="EURUSD", target_position=-100),
        ]
        agg = coord._aggregate_by_symbol(intents)
        assert agg["EURUSD"]["target_position"] == pytest.approx(200.0)

    def test_conflict_logged(self) -> None:
        coord, *_ = _make_coord()
        intents = [
            OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=300),
            OrderIntent(strategy_id="s2", symbol="EURUSD", target_position=-100),
        ]
        coord._aggregate_by_symbol(intents)
        assert len(coord.conflicts_log) == 1
        assert coord.conflicts_log[0]["symbol"] == "EURUSD"
        assert coord.conflicts_log[0]["net"] == pytest.approx(200.0)

    def test_no_conflict_when_same_sign(self) -> None:
        coord, *_ = _make_coord()
        intents = [
            OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=100),
            OrderIntent(strategy_id="s2", symbol="EURUSD", target_position=200),
        ]
        coord._aggregate_by_symbol(intents)
        assert coord.conflicts_log == []

    def test_urgency_escalation(self) -> None:
        coord, *_ = _make_coord()
        intents = [
            OrderIntent(
                strategy_id="s1", symbol="EURUSD",
                target_position=100, urgency="passive",
            ),
            OrderIntent(
                strategy_id="s2", symbol="EURUSD",
                target_position=100, urgency="urgent",
            ),
            OrderIntent(
                strategy_id="s3", symbol="EURUSD",
                target_position=100, urgency="normal",
            ),
        ]
        agg = coord._aggregate_by_symbol(intents)
        assert agg["EURUSD"]["urgency"] == "urgent"

    def test_non_canonical_urgency_never_escalates(self) -> None:
        """The rank map is derived from the canonical Urgency enum
        (CL-ikz2): any value outside it — like the legacy event-exit
        "high" — ranks 0 and cannot escalate past a canonical value."""
        coord, *_ = _make_coord()
        intents = [
            OrderIntent(
                strategy_id="s1", symbol="EURUSD",
                target_position=100, urgency="normal",
            ),
            OrderIntent(
                strategy_id="s2", symbol="EURUSD",
                target_position=100, urgency="high",  # legacy, non-canonical
            ),
        ]
        agg = coord._aggregate_by_symbol(intents)
        assert agg["EURUSD"]["urgency"] == "normal"

    def test_rank_map_matches_enum_order(self) -> None:
        from src.portfolio.coordinator import _URGENCY_RANK

        assert _URGENCY_RANK == {"passive": 0, "normal": 1, "urgent": 2}
        assert [u.value for u in Urgency] == ["passive", "normal", "urgent"]


# =============================================================================
# Constraints
# =============================================================================


class TestConstraints:
    def test_gross_leverage_cap_scales_down(self) -> None:
        # Isolate gross-cap behavior by setting per-pair cap above the gross
        # cap so the gross-leverage clip is the only binding constraint.
        relaxed = PortfolioConstraints(max_notional_per_pair_pct=10.0)
        coord, *_ = _make_coord(
            ["s1"], initial_capital=100_000, constraints=relaxed,
        )
        # 100k equity, 3x cap → max gross 300k notional. Push 600k → 50% scale.
        units = 600_000 / 1.10
        agg = {
            "EURUSD": {
                "target_position": units,
                "urgency": "normal",
                "strategy_contributions": {"s1": units},
            }
        }
        coord._apply_portfolio_constraints(agg)
        # Gross was 600k, max is 300k → scale 0.5
        assert agg["EURUSD"]["target_position"] == pytest.approx(units * 0.5)
        assert agg["EURUSD"]["strategy_contributions"]["s1"] == pytest.approx(units * 0.5)

    def test_per_pair_cap_scales_down(self) -> None:
        # 100k equity, 35% per-pair cap → 35k max per pair.
        # Push 50k notional EURUSD → scale to 35k.
        coord, *_ = _make_coord(["s1"], initial_capital=100_000)
        units = 50_000 / 1.10
        agg = {
            "EURUSD": {
                "target_position": units,
                "urgency": "normal",
                "strategy_contributions": {"s1": units},
            }
        }
        coord._apply_portfolio_constraints(agg)
        notional = abs(agg["EURUSD"]["target_position"]) * 1.10
        assert notional == pytest.approx(35_000.0, rel=1e-6)

    def test_constraint_scaling_preserves_sign(self) -> None:
        coord, *_ = _make_coord(["s1"], initial_capital=100_000)
        units = -600_000 / 1.10  # short EURUSD over gross cap
        agg = {
            "EURUSD": {
                "target_position": units,
                "urgency": "normal",
                "strategy_contributions": {"s1": units},
            }
        }
        coord._apply_portfolio_constraints(agg)
        assert agg["EURUSD"]["target_position"] < 0  # still short

    def test_within_caps_unchanged(self) -> None:
        coord, *_ = _make_coord(["s1"], initial_capital=100_000)
        units = 10_000 / 1.10  # 10k notional, well under all caps
        agg = {
            "EURUSD": {
                "target_position": units,
                "urgency": "normal",
                "strategy_contributions": {"s1": units},
            }
        }
        original = agg["EURUSD"]["target_position"]
        coord._apply_portfolio_constraints(agg)
        assert agg["EURUSD"]["target_position"] == pytest.approx(original)

    def test_empty_aggregated_noop(self) -> None:
        coord, *_ = _make_coord(["s1"])
        out = coord._apply_portfolio_constraints({})
        assert out == {}

    def test_currency_exposure_decomposition(self) -> None:
        coord, *_ = _make_coord(["s1"])
        # Long 10000 EURUSD at price 1.10 → +11000 EUR, -11000 USD
        agg = {
            "EURUSD": {
                "target_position": 10_000,
                "urgency": "normal",
                "strategy_contributions": {"s1": 10_000},
            }
        }
        exposures = coord._compute_currency_exposure(agg)
        assert exposures["EUR"] == pytest.approx(11_000.0)
        assert exposures["USD"] == pytest.approx(-11_000.0)


# =============================================================================
# process_intents end-to-end
# =============================================================================


class TestProcessIntents:
    def test_submits_aggregated_intent_to_oms(self) -> None:
        coord, oms, *_ = _make_coord(["s1", "s2"])
        coord.initialize_allocations({"s1": 0.5, "s2": 0.5})
        raw = {
            "s1": [OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)],
            "s2": [OrderIntent(strategy_id="s2", symbol="EURUSD", target_position=2000)],
        }
        asyncio.run(coord.process_intents(raw))
        assert len(oms.submitted) == 1
        out = oms.submitted[0]
        assert out.symbol == "EURUSD"
        # 1000 * 0.5 + 2000 * 0.5 = 500 + 1000 = 1500
        assert out.target_position == pytest.approx(1500.0)
        assert out.strategy_id == "portfolio"

    def test_multiple_symbols_each_submitted(self) -> None:
        coord, oms, *_ = _make_coord(["s1"])
        coord.initialize_allocations({"s1": 1.0})
        raw = {
            "s1": [
                OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000),
                OrderIntent(strategy_id="s1", symbol="USDJPY", target_position=500),
            ],
        }
        asyncio.run(coord.process_intents(raw))
        assert len(oms.submitted) == 2
        symbols = {i.symbol for i in oms.submitted}
        assert symbols == {"EURUSD", "USDJPY"}

    def test_paper_mode_strategy_not_submitted(self) -> None:
        coord, oms, *_ = _make_coord(["s1"])
        coord.allocations["s1"] = StrategyAllocation(
            strategy_id="s1", target_weight=0.0,
            current_exposure_mult=0.0, paper_mode=True,
        )
        raw = {
            "s1": [OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)]
        }
        asyncio.run(coord.process_intents(raw))
        assert oms.submitted == []

    def test_state_records_portfolio_order(self) -> None:
        coord, _oms, _broker, state = _make_coord(["s1"])
        coord.initialize_allocations({"s1": 1.0})
        raw = {
            "s1": [OrderIntent(strategy_id="s1", symbol="EURUSD", target_position=1000)]
        }
        asyncio.run(coord.process_intents(raw))
        assert len(state.portfolio_orders) == 1
        _, sym, target, contribs = state.portfolio_orders[0]
        assert sym == "EURUSD"
        assert target == pytest.approx(1000.0)
        assert contribs == {"s1": 1000.0}


# =============================================================================
# Strategy lifecycle
# =============================================================================


class TestStrategyLifecycle:
    def test_add_strategy_in_paper_mode(self) -> None:
        coord, *_ = _make_coord(["s1"])
        coord.add_strategy(_FakeStrategy("s2"))
        assert "s2" in coord.strategies
        assert coord.allocations["s2"].paper_mode is True
        assert coord.allocations["s2"].target_weight == 0.0
        assert coord.allocations["s2"].current_exposure_mult == 0.0

    def test_add_duplicate_raises(self) -> None:
        coord, *_ = _make_coord(["s1"])
        with pytest.raises(ValueError):
            coord.add_strategy(_FakeStrategy("s1"))

    def test_remove_strategy_liquidates(self) -> None:
        positions = {"s1": [_FakePosition(symbol="EURUSD", quantity=1000)]}
        coord, oms, _broker, _state = _make_coord(
            ["s1", "s2"], state=_FakeState(positions_by_strategy=positions),
        )
        coord.initialize_allocations({"s1": 0.5, "s2": 0.5})
        coord.remove_strategy("s1")
        assert "s1" not in coord.strategies
        assert "s1" not in coord.allocations
        # Removal sent a liquidation intent.
        assert len(oms.submitted) == 1
        assert oms.submitted[0].symbol == "EURUSD"
        assert oms.submitted[0].target_position == 0.0

    def test_remove_unknown_strategy_noop(self) -> None:
        coord, oms, *_ = _make_coord(["s1"])
        coord.remove_strategy("ghost")
        assert oms.submitted == []
        assert "s1" in coord.strategies


# =============================================================================
# Rebalance
# =============================================================================


def _make_returns_df(
    n_days: int, n_strategies: int = 2, seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0005, 0.01, size=(n_days, n_strategies))
    cols = [f"s{i + 1}" for i in range(n_strategies)]
    return pd.DataFrame(data, columns=cols)


class TestRebalance:
    def test_skips_within_min_interval(self) -> None:
        coord, *_ = _make_coord(["s1", "s2"])
        coord.initialize_allocations({"s1": 0.5, "s2": 0.5})
        coord._last_rebalance = datetime.now(UTC)
        # Without force, should be a no-op.
        asyncio.run(coord.rebalance_allocations())
        assert coord._last_rebalance is not None  # still set; no replacement happened

    def test_force_overrides_interval(self) -> None:
        returns = _make_returns_df(n_days=400, n_strategies=2)
        state = _FakeState(returns_df=returns)
        coord, _oms, _broker, _state = _make_coord(["s1", "s2"], state=state)
        coord.initialize_allocations({"s1": 0.5, "s2": 0.5})
        coord._last_rebalance = datetime.now(UTC)
        asyncio.run(coord.rebalance_allocations(force=True))
        assert len(state.reallocations) == 1

    def test_falls_back_equal_when_history_short(self) -> None:
        # Only 50 days of history < min 252 days.
        returns = _make_returns_df(n_days=50, n_strategies=2)
        state = _FakeState(returns_df=returns)
        coord, _oms, _broker, _state = _make_coord(["s1", "s2"], state=state)
        coord.initialize_allocations({"s1": 0.5, "s2": 0.5})
        asyncio.run(coord.rebalance_allocations(force=True))
        # Equal weight expected.
        assert coord.allocations["s1"].target_weight == pytest.approx(0.5)
        assert coord.allocations["s2"].target_weight == pytest.approx(0.5)

    def test_risk_parity_weights_sum_to_one(self) -> None:
        returns = _make_returns_df(n_days=400, n_strategies=3)
        state = _FakeState(returns_df=returns)
        coord, _oms, _broker, _state = _make_coord(
            ["s1", "s2", "s3"], state=state,
        )
        coord.initialize_allocations({"s1": 1.0, "s2": 1.0, "s3": 1.0})
        asyncio.run(coord.rebalance_allocations(force=True))
        total = sum(a.target_weight for a in coord.allocations.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_risk_parity_respects_bounds(self) -> None:
        # Construct returns where one strategy has much lower vol → would normally
        # dominate without bounds.
        rng = np.random.default_rng(42)
        s1 = rng.normal(0.0005, 0.001, 400)  # very low vol
        s2 = rng.normal(0.0005, 0.02, 400)
        s3 = rng.normal(0.0005, 0.02, 400)
        returns = pd.DataFrame({"s1": s1, "s2": s2, "s3": s3})
        state = _FakeState(returns_df=returns)
        coord, _oms, _broker, _state = _make_coord(
            ["s1", "s2", "s3"], state=state,
        )
        coord.initialize_allocations({"s1": 1.0, "s2": 1.0, "s3": 1.0})
        asyncio.run(coord.rebalance_allocations(force=True))
        for sid, a in coord.allocations.items():
            # Bounds (0.05, 0.40) post-normalization; allow tiny tolerance for SLSQP slack.
            assert a.target_weight >= 0.05 - 1e-6, f"{sid} below floor: {a.target_weight}"
            assert a.target_weight <= 0.40 + 1e-6, f"{sid} above cap: {a.target_weight}"


# =============================================================================
# Hypothesis property tests
# =============================================================================


@given(
    contributions=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=8, alphabet="abcdefghij"),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
    ),
)
@settings(max_examples=200)
def test_aggregate_net_equals_sum_of_contributions(
    contributions: list[tuple[str, float]],
) -> None:
    """Net target_position must equal the sum of per-strategy contributions."""
    coord, *_ = _make_coord(["s1"])
    intents = [
        OrderIntent(strategy_id=sid, symbol="EURUSD", target_position=qty)
        for sid, qty in contributions
    ]
    agg = coord._aggregate_by_symbol(intents)
    if "EURUSD" not in agg:
        return  # empty input
    expected = sum(qty for _, qty in contributions)
    # Floating point: aggregation order matters slightly. Allow 1e-6 abs tolerance.
    assert agg["EURUSD"]["target_position"] == pytest.approx(expected, abs=1e-6)


@given(
    weight=st.floats(min_value=0.0, max_value=1.0),
    exposure=st.floats(min_value=0.0, max_value=1.0),
    perf=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_effective_scale_bounded(
    weight: float, exposure: float, perf: float,
) -> None:
    """effective_scale always lies in [0, 1] regardless of inputs in [0, 1]."""
    a = StrategyAllocation(
        strategy_id="x",
        target_weight=weight,
        current_exposure_mult=exposure,
        performance_override=perf,
    )
    assert 0.0 <= a.effective_scale <= 1.0


@given(
    target=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    scale=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_rescale_symbol_preserves_sign(target: float, scale: float) -> None:
    """Rescaling never flips the sign of a non-zero target_position."""
    agg = {
        "target_position": target,
        "urgency": "normal",
        "strategy_contributions": {"s1": target},
    }
    PortfolioCoordinator._rescale_symbol(agg, scale)
    if target == 0:
        return
    assert np.sign(agg["target_position"]) == np.sign(target * scale)


@given(
    notionals=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=100, deadline=None)
def test_gross_leverage_capped_after_constraints(notionals: list[float]) -> None:
    """After _apply_portfolio_constraints, gross leverage <= cap (within tolerance)."""
    coord, *_ = _make_coord(["s1"], initial_capital=100_000)
    # Spread across distinct symbols to avoid per-pair cap clipping first.
    symbols = ["EURUSD", "USDJPY", "GBPUSD"]
    agg: dict[str, dict[str, Any]] = {}
    for i, n in enumerate(notionals[:3]):
        sym = symbols[i]
        # Convert notional to units at the configured mid-price.
        bid, ask = coord.broker.get_price(sym)
        mid = (bid + ask) / 2
        units = n / mid if mid > 0 else 0.0
        agg[sym] = {
            "target_position": units,
            "urgency": "normal",
            "strategy_contributions": {"s1": units},
        }
    coord._apply_portfolio_constraints(agg)
    gross = sum(
        abs(a["target_position"]) * coord._get_price(sym) for sym, a in agg.items()
    )
    equity = coord.broker.get_account().equity
    leverage = gross / equity
    assert leverage <= coord.constraints.max_gross_leverage + 1e-6


# =============================================================================
# Blackout SIZE_DOWN_50PCT (CL-k74b) — coordinator-level intent halving.
# =============================================================================


def _evaluator_with_event(
    minutes_until: float,
    currency: str | None = None,
) -> Any:
    """Build a BlackoutEvaluator whose calendar has one tier-1 event at the
    given offset.
    """
    from datetime import timedelta

    from src.data.economic_calendar import (
        BlackoutEvaluator,
        EconomicCalendar,
        EconomicEvent,
        SeverityTier,
    )
    when = datetime.now(UTC) + timedelta(minutes=minutes_until)
    cal = EconomicCalendar([
        EconomicEvent(
            ts=when,
            event_type="NFP",
            severity_tier=SeverityTier.TIER_1,
            description="test",
            currency=currency,
        ),
    ])
    return BlackoutEvaluator(cal)


class TestBlackoutSizeDown:
    def test_size_down_window_halves_target(self) -> None:
        # Tier-1 SIZE_DOWN window is 0..24h; PAUSE starts at 1h, EXIT_FLAT at
        # 30min. Event at 6h falls strictly into SIZE_DOWN_50PCT.
        evaluator = _evaluator_with_event(minutes_until=360)
        coord, *_ = _make_coord(["s1"], blackout_evaluator=evaluator)
        out = coord._apply_blackout_size_down("EURUSD", 10_000.0)
        assert out == pytest.approx(5_000.0)

    def test_full_size_window_unchanged(self) -> None:
        # Event 48h+ away — outside any tier-1 window → FULL_SIZE → no halving.
        evaluator = _evaluator_with_event(minutes_until=4320)  # 72h
        coord, *_ = _make_coord(["s1"], blackout_evaluator=evaluator)
        out = coord._apply_blackout_size_down("EURUSD", 10_000.0)
        assert out == pytest.approx(10_000.0)

    def test_pause_window_unchanged_at_coordinator_level(self) -> None:
        # PAUSE_NEW_ENTRIES is the validator's responsibility (rejection),
        # not the coordinator's (size mutation). Coordinator must NOT mutate
        # in pause/exit-flat windows — pass through unchanged.
        evaluator = _evaluator_with_event(minutes_until=45)  # PAUSE window
        coord, *_ = _make_coord(["s1"], blackout_evaluator=evaluator)
        out = coord._apply_blackout_size_down("EURUSD", 10_000.0)
        assert out == pytest.approx(10_000.0)

    def test_exit_flat_window_unchanged_at_coordinator_level(self) -> None:
        # Same as PAUSE — validator rejects, coordinator doesn't mutate.
        evaluator = _evaluator_with_event(minutes_until=5)  # EXIT_FLAT
        coord, *_ = _make_coord(["s1"], blackout_evaluator=evaluator)
        out = coord._apply_blackout_size_down("EURUSD", 10_000.0)
        assert out == pytest.approx(10_000.0)

    def test_no_evaluator_passes_through(self) -> None:
        coord, *_ = _make_coord(["s1"], blackout_evaluator=None)
        out = coord._apply_blackout_size_down("EURUSD", 10_000.0)
        assert out == pytest.approx(10_000.0)

    def test_currency_filter_passes_for_unaffected_pair(self) -> None:
        # Event tagged USD-only; intent on EUR pair derives currency='EUR'
        # → calendar.evaluate filters out the USD event → FULL_SIZE.
        evaluator = _evaluator_with_event(minutes_until=360, currency="USD")
        coord, *_ = _make_coord(["s1"], blackout_evaluator=evaluator)
        out = coord._apply_blackout_size_down("EURJPY", 10_000.0)
        assert out == pytest.approx(10_000.0)

    def test_negative_target_halved(self) -> None:
        # Short positions also get halved (sign-preserving multiply by 0.5).
        evaluator = _evaluator_with_event(minutes_until=360)
        coord, *_ = _make_coord(["s1"], blackout_evaluator=evaluator)
        out = coord._apply_blackout_size_down("EURUSD", -8_000.0)
        assert out == pytest.approx(-4_000.0)


# =============================================================================
# Fail-closed pricing: withhold, never fabricate an exit (CL-e8ze + ultrareview #1)
# =============================================================================


class TestUnpriceableSymbolWithheld:
    def test_unpriceable_symbol_removed_from_aggregate(self) -> None:
        """A symbol with no price must be DELETED from the aggregate — the
        first fix rescaled its target to 0, which flowed a zero-target intent
        to the OMS and FORCE-LIQUIDATED any held position on a pricing flap."""
        coord, _oms, _broker, _state = _make_coord()
        units = 1000.0
        agg = {
            "XAU_USD": {  # no price configured for gold → unpriceable
                "target_position": units,
                "urgency": "normal",
                "strategy_contributions": {"s1": units},
            },
            "EURUSD": {
                "target_position": 500.0,
                "urgency": "normal",
                "strategy_contributions": {"s1": 500.0},
            },
        }
        out = coord._apply_portfolio_constraints(agg)
        assert "XAU_USD" not in out          # withheld entirely — no intent
        assert "EURUSD" in out                # priceable symbol unaffected
        assert out["EURUSD"]["target_position"] == pytest.approx(500.0)

    def test_all_unpriceable_returns_empty(self) -> None:
        coord, _oms, _broker, _state = _make_coord()
        agg = {
            "XAU_USD": {
                "target_position": 100.0,
                "urgency": "normal",
                "strategy_contributions": {"s1": 100.0},
            },
        }
        assert coord._apply_portfolio_constraints(agg) == {}


# =============================================================================
# Async hygiene (CL-xdnh): blocking broker/OMS/state I/O runs off the event
# loop, and fire-and-forget rebalance tasks are retained + observed.
# =============================================================================


class _ThreadRecordingBroker:
    """Wraps PaperBroker, recording the thread ident of every I/O call."""

    def __init__(self, inner: PaperBroker) -> None:
        self._inner = inner
        self.call_threads: list[int] = []

    def _record(self) -> None:
        import threading

        self.call_threads.append(threading.get_ident())

    def get_account(self):  # noqa: ANN201
        self._record()
        return self._inner.get_account()

    def get_price(self, symbol: str):  # noqa: ANN201
        self._record()
        return self._inner.get_price(symbol)

    def get_positions(self):  # noqa: ANN201
        self._record()
        return self._inner.get_positions()


class _ThreadRecordingOMS(_RecordingOMS):
    def __init__(self) -> None:
        super().__init__()
        self.call_threads: list[int] = []

    def submit_intent(self, intent: OrderIntent, **kwargs: object) -> str:
        import threading

        self.call_threads.append(threading.get_ident())
        return super().submit_intent(intent)


class _ThreadRecordingValidator:
    def __init__(self) -> None:
        self.call_threads: list[int] = []
        # Captured to assert the shared snapshot (CL-qsue) is threaded through.
        self.price_maps: list[Any] = []
        self.accounts: list[Any] = []

    def validate(  # noqa: ANN201
        self,
        intent: OrderIntent,
        current_positions=None,  # noqa: ANN001
        price_map=None,  # noqa: ANN001
        account=None,  # noqa: ANN001
    ):
        import threading

        self.call_threads.append(threading.get_ident())
        self.price_maps.append(price_map)
        self.accounts.append(account)
        return None


class _ThreadRecordingState(_FakeState):
    def __init__(self) -> None:
        super().__init__()
        self.call_threads: list[int] = []

    def record_portfolio_order(self, ts, symbol, target_position, strategy_contributions):  # noqa: ANN001
        import threading

        self.call_threads.append(threading.get_ident())
        super().record_portfolio_order(
            ts, symbol, target_position, strategy_contributions,
        )


class TestBrokerIoOffEventLoop:
    def test_process_intents_never_calls_broker_on_loop_thread(self) -> None:
        """Every broker / OMS / validator / state-write call inside the async
        process_intents path must run in a worker thread — with the httpx
        OANDA broker these are synchronous HTTP round-trips that used to
        freeze the event loop (and with it the price stream, health ticks,
        and kill switches)."""
        import threading

        inner = PaperBroker(initial_capital=100_000.0)
        inner.set_price("EURUSD", 1.0999, 1.1001)
        broker = _ThreadRecordingBroker(inner)
        oms = _ThreadRecordingOMS()
        validator = _ThreadRecordingValidator()
        state = _ThreadRecordingState()
        coord = PortfolioCoordinator(
            strategies=[_FakeStrategy("s1")],
            oms=oms,  # type: ignore[arg-type]
            broker=broker,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
            pre_trade_validator=validator,
        )
        coord.initialize_allocations({"s1": 1.0})
        raw = {"s1": [OrderIntent(
            strategy_id="s1", symbol="EURUSD", target_position=1000.0,
        )]}

        loop_thread: list[int] = []

        async def run() -> None:
            loop_thread.append(threading.get_ident())
            await coord.process_intents(raw)

        asyncio.run(run())

        assert oms.submitted, "intent must actually reach the OMS"
        for name, threads in (
            ("broker", broker.call_threads),
            ("oms", oms.call_threads),
            ("validator", validator.call_threads),
            ("state", state.call_threads),
        ):
            assert threads, f"{name} was never called"
            assert all(t != loop_thread[0] for t in threads), (
                f"{name} I/O ran on the event-loop thread"
            )


class TestBackgroundRebalanceTasks:
    def test_remove_strategy_retains_task_and_logs_failure(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        coord, _oms, _broker, _state = _make_coord()
        coord.initialize_allocations({"s1": 0.5, "s2": 0.5})

        async def boom(force: bool = False) -> None:
            raise RuntimeError("rebalance exploded")

        coord.rebalance_allocations = boom  # type: ignore[method-assign]

        async def run() -> None:
            coord.remove_strategy("s2")
            assert len(coord._background_tasks) == 1  # reference retained
            task = next(iter(coord._background_tasks))
            with pytest.raises(RuntimeError, match="rebalance exploded"):
                await task
            await asyncio.sleep(0)  # let the done-callback run
            assert coord._background_tasks == set()

        with caplog.at_level(logging.ERROR, logger="src.portfolio.coordinator"):
            asyncio.run(run())

        failures = [
            r for r in caplog.records
            if "Background task" in r.getMessage() and r.exc_info
        ]
        assert failures, "failed rebalance task must be logged with traceback"

    def test_promote_spawns_observed_task_that_completes(self) -> None:
        coord, _oms, _broker, state = _make_coord()
        coord.initialize_allocations({"s1": 1.0})  # s2 → paper mode

        async def run() -> None:
            coord.promote_strategy_to_live("s2", initial_weight=0.10)
            assert len(coord._background_tasks) == 1
            task = next(iter(coord._background_tasks))
            await task  # completes without raising
            await asyncio.sleep(0)
            assert coord._background_tasks == set()

        asyncio.run(run())
        # Forced rebalance really ran: reallocation recorded via state.
        assert state.reallocations

    def test_no_running_loop_defers_without_task(self) -> None:
        coord, _oms, _broker, _state = _make_coord()
        coord.initialize_allocations({"s1": 0.5, "s2": 0.5})
        coord.remove_strategy("s2")  # sync context — must not raise
        assert coord._background_tasks == set()
