"""Unit tests — risk: kill switches, sizing, correlation monitor."""

import pytest

from src.risk.sizing import PositionSizer
from src.risk.kill_switches import KillSwitchManager
from src.risk.regime_sizing import RegimeAwareSizer


class TestPositionSizer:
    def test_kelly_never_negative(self) -> None:
        for edge, odds in [(0.6, 2.0), (0.5, 1.5), (0.3, 3.0)]:
            k = PositionSizer.kelly(edge, odds)
            assert 0 <= k <= 1

    def test_fixed_fractional(self) -> None:
        size = PositionSizer.fixed_fractional(10000, 0.01, 0.005, 1.10)
        assert size > 0


class _FakeOMS:
    def __init__(self) -> None:
        self.halts: list[str] = []
        self.strategy_halts: list[str] = []

    def halt_new_trades(self) -> None:
        self.halts.append("all")

    def halt_strategy(self, sid: str) -> None:
        self.strategy_halts.append(sid)


class _FakeBroker:
    def get_positions(self) -> list[object]:
        return []


def _baseline_ctx() -> dict[str, object]:
    """Context that triggers nothing — overlay one bad value per test."""
    return {
        "portfolio_dd": -0.05,
        "daily_pnl_pct": 0.01,
        "vix_level": 22,
        "vix_change_1d": 0.1,
        "max_price_age_sec": 5,
        "cvix_zscore": 0.5,
        "position_mismatch": False,
        "correlation_regime": "normal",
        "max_pair_corr": 0.3,
        "strategy_drawdowns": {"s1": -0.05},
    }


class TestKillSwitchManager:
    def test_drawdown_limit_triggers(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {})
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.25
        triggered = mgr.check(ctx)
        names = {t["switch"] for t in triggered}
        assert "drawdown_limit" in names

    def test_correlation_crisis_triggers_reduce(self) -> None:
        oms = _FakeOMS()
        mgr = KillSwitchManager(_FakeBroker(), oms, {})
        ctx = _baseline_ctx()
        ctx["correlation_regime"] = "crisis"
        triggered = mgr.check(ctx)
        names = {t["switch"] for t in triggered}
        assert "portfolio_correlation_crisis" in names

    def test_correlation_spike_triggers(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {})
        ctx = _baseline_ctx()
        ctx["max_pair_corr"] = 0.92
        triggered = mgr.check(ctx)
        names = {t["switch"] for t in triggered}
        assert "strategy_correlation_spike" in names

    def test_single_strategy_dd_calls_halt_strategy(self) -> None:
        oms = _FakeOMS()
        mgr = KillSwitchManager(_FakeBroker(), oms, {})
        ctx = _baseline_ctx()
        ctx["strategy_drawdowns"] = {"s1": -0.05, "s2": -0.30}
        triggered = mgr.check(ctx)
        assert {"single_strategy_drawdown"} <= {t["switch"] for t in triggered}
        assert oms.strategy_halts == ["s2"]

    def test_normal_context_triggers_nothing(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {})
        triggered = mgr.check(_baseline_ctx())
        assert triggered == []


class TestRegimeAwareSizer:
    def test_normal_returns_full_multiplier(self) -> None:
        sizer = RegimeAwareSizer()
        adj = sizer.compute_adjustment(vix=15, portfolio_dd=-0.02, correlation_regime="normal", max_pair_corr=0.3)
        assert adj.final_multiplier == 1.0

    def test_extreme_crisis_returns_near_zero(self) -> None:
        sizer = RegimeAwareSizer()
        adj = sizer.compute_adjustment(vix=45, portfolio_dd=-0.22, correlation_regime="crisis", max_pair_corr=0.9)
        assert adj.final_multiplier < 0.1
