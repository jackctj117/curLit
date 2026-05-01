"""Tests for src/monitoring/portfolio_metrics.py emission helper (CL-5lq)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.monitoring.metrics import (
    portfolio_conflicts_rate,
    portfolio_correlation_max,
    portfolio_gross_leverage,
    portfolio_net_leverage,
    strategy_allocation,
    strategy_attributed_pnl_usd,
    strategy_attributed_sharpe,
    strategy_exposure_mult,
)
from src.monitoring.portfolio_metrics import emit_portfolio_metrics


@dataclass
class _FakePos:
    symbol: str
    quantity: float
    avg_price: float


@dataclass
class _FakePnL:
    realized: float
    unrealized: float


def _gauge_value(gauge) -> float:  # type: ignore[no-untyped-def]
    """Pull the current value from a Prometheus gauge with no labels."""
    return float(gauge._value.get())  # type: ignore[attr-defined]


def _labeled(gauge, **labels) -> float:  # type: ignore[no-untyped-def]
    return float(gauge.labels(**labels)._value.get())  # type: ignore[attr-defined]


class TestLeverage:
    def test_long_only_leverage(self) -> None:
        positions = [
            _FakePos("EURUSD", 1000.0, 1.10),  # 1100 notional
            _FakePos("GBPUSD", 500.0, 1.30),   # 650 notional
        ]
        emit_portfolio_metrics(
            allocations={}, exposure_multipliers={},
            positions=positions, equity=10_000,
        )
        assert _gauge_value(portfolio_gross_leverage) == pytest.approx(0.175)
        assert _gauge_value(portfolio_net_leverage) == pytest.approx(0.175)

    def test_long_short_offsets_in_net(self) -> None:
        positions = [
            _FakePos("EURUSD", 1000.0, 1.0),
            _FakePos("GBPUSD", -500.0, 1.0),
        ]
        emit_portfolio_metrics(
            allocations={}, exposure_multipliers={},
            positions=positions, equity=10_000,
        )
        # Gross = 1500/10000 = 0.15; Net = 500/10000 = 0.05
        assert _gauge_value(portfolio_gross_leverage) == pytest.approx(0.15)
        assert _gauge_value(portfolio_net_leverage) == pytest.approx(0.05)

    def test_zero_equity_no_op(self) -> None:
        # Should not crash or emit infinity.
        emit_portfolio_metrics(
            allocations={}, exposure_multipliers={},
            positions=[_FakePos("EURUSD", 1000, 1.0)],
            equity=0,
        )
        # Gauge retains its previous value rather than going to infinity.
        # (We just verify no exception escaped.)


class TestStrategyLevel:
    def test_allocations_and_multipliers(self) -> None:
        emit_portfolio_metrics(
            allocations={"s1": 0.6, "s2": 0.4},
            exposure_multipliers={"s1": 1.0, "s2": 0.5},
            positions=[], equity=10_000,
        )
        assert _labeled(strategy_allocation, strategy="s1") == pytest.approx(0.6)
        assert _labeled(strategy_exposure_mult, strategy="s2") == pytest.approx(0.5)

    def test_attributed_pnl(self) -> None:
        emit_portfolio_metrics(
            allocations={}, exposure_multipliers={},
            positions=[], equity=10_000,
            attributed_pnls={"s1": _FakePnL(realized=100.0, unrealized=20.0)},
        )
        assert _labeled(strategy_attributed_pnl_usd, strategy="s1") == pytest.approx(120.0)

    def test_sharpe_emission(self) -> None:
        # Construct returns with mean 0.001/day, std 0.01/day.
        # Annualized Sharpe ≈ 0.001/0.01 * sqrt(252) ≈ 1.587.
        returns = pd.Series([0.001, 0.011, -0.009, 0.005, -0.002])
        emit_portfolio_metrics(
            allocations={}, exposure_multipliers={},
            positions=[], equity=10_000,
            daily_returns={"s1": returns},
        )
        s = _labeled(strategy_attributed_sharpe, strategy="s1")
        assert -10 < s < 10  # Sane range


class TestConflicts:
    def test_rate_zero_when_no_conflicts(self) -> None:
        emit_portfolio_metrics(
            allocations={}, exposure_multipliers={},
            positions=[], equity=10_000,
            conflicts=0, n_intents=10,
        )
        assert _gauge_value(portfolio_conflicts_rate) == 0.0

    def test_rate_fraction(self) -> None:
        emit_portfolio_metrics(
            allocations={}, exposure_multipliers={},
            positions=[], equity=10_000,
            conflicts=2, n_intents=8,
        )
        assert _gauge_value(portfolio_conflicts_rate) == pytest.approx(0.25)


class TestCorrelation:
    def test_max_pair_corr(self) -> None:
        emit_portfolio_metrics(
            allocations={}, exposure_multipliers={},
            positions=[], equity=10_000,
            max_pair_corr=0.85,
        )
        assert _gauge_value(portfolio_correlation_max) == pytest.approx(0.85)
