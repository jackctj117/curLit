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


class TestKillSwitchManager:
    def test_drawdown_limit_triggers(self) -> None:
        class FakeOMS:
            def halt_new_trades(self): pass

        class FakeBroker:
            def get_positions(self): return []

        mgr = KillSwitchManager(FakeBroker(), FakeOMS(), {})
        triggered = mgr.check({"portfolio_dd": -0.25, "daily_pnl_pct": 0.01, "vix_level": 22, "vix_change_1d": 0.1, "max_price_age_sec": 5})
        assert len(triggered) == 1
        assert triggered[0]["switch"] == "drawdown_limit"


class TestRegimeAwareSizer:
    def test_normal_returns_full_multiplier(self) -> None:
        sizer = RegimeAwareSizer()
        adj = sizer.compute_adjustment(vix=15, portfolio_dd=-0.02, correlation_regime="normal", max_pair_corr=0.3)
        assert adj.final_multiplier == 1.0

    def test_extreme_crisis_returns_near_zero(self) -> None:
        sizer = RegimeAwareSizer()
        adj = sizer.compute_adjustment(vix=45, portfolio_dd=-0.22, correlation_regime="crisis", max_pair_corr=0.9)
        assert adj.final_multiplier < 0.1
