"""Unit tests — risk: kill switches, sizing, correlation monitor.

CL-ep0c adds coverage for the equity-curve trailing stop (threshold,
state-file round-trip across manager instances, cooldown across
restarts with an injectable clock, post-expiry resume + peak reset)
and the open-position correlation switch (correlated shorts fire,
hedged long/short doesn't, single position / missing data never fire),
plus yaml -> KillSwitchConfig -> manager plumb-through.
"""

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.execution.broker import Position
from src.risk.kill_switches import KillSwitchManager
from src.risk.regime_sizing import RegimeAwareSizer
from src.risk.risk_profile import load_active_profile
from src.risk.sizing import PositionSizer

_T0 = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


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
    def __init__(self, positions: list[Position] | None = None) -> None:
        self.positions = positions or []

    def get_positions(self) -> list[Position]:
        return self.positions


class _FakeClock:
    """Injectable clock — same pattern as test_polymarket_loss_caps."""

    def __init__(self, now: datetime = _T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class _FakeDataProvider:
    def __init__(self, df: pd.DataFrame | None) -> None:
        self.df = df

    def get_aligned_series(
        self, symbols: list[str], start: datetime, end: datetime,
    ) -> pd.DataFrame | None:
        if self.df is None:
            return None
        cols = [c for c in self.df.columns if c in symbols]
        return self.df[cols] if cols else None


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


def _ctx(equity: float | None = None) -> dict[str, object]:
    ctx = _baseline_ctx()
    if equity is not None:
        ctx["equity"] = equity
    return ctx


class TestKillSwitchManager:
    def test_drawdown_limit_triggers(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.25
        triggered = mgr.check(ctx)
        names = {t["switch"] for t in triggered}
        assert "drawdown_limit" in names

    def test_correlation_crisis_triggers_reduce(self) -> None:
        oms = _FakeOMS()
        mgr = KillSwitchManager(_FakeBroker(), oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["correlation_regime"] = "crisis"
        triggered = mgr.check(ctx)
        names = {t["switch"] for t in triggered}
        assert "portfolio_correlation_crisis" in names

    def test_correlation_spike_triggers(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["max_pair_corr"] = 0.92
        triggered = mgr.check(ctx)
        names = {t["switch"] for t in triggered}
        assert "strategy_correlation_spike" in names

    def test_single_strategy_dd_calls_halt_strategy(self) -> None:
        oms = _FakeOMS()
        mgr = KillSwitchManager(_FakeBroker(), oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["strategy_drawdowns"] = {"s1": -0.05, "s2": -0.30}
        triggered = mgr.check(ctx)
        assert {"single_strategy_drawdown"} <= {t["switch"] for t in triggered}
        assert oms.strategy_halts == ["s2"]

    def test_normal_context_triggers_nothing(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {}, trailing_state_path=None)
        triggered = mgr.check(_baseline_ctx())
        assert triggered == []


# --------------------------------------------------------------------- #
# CL-ep0c — equity-curve trailing stop
# --------------------------------------------------------------------- #

def _trailing_mgr(
    tmp_path: Path,
    oms: _FakeOMS | None = None,
    clock: _FakeClock | None = None,
    config: dict[str, object] | None = None,
) -> KillSwitchManager:
    return KillSwitchManager(
        _FakeBroker(), oms or _FakeOMS(), config or {},
        clock=clock or _FakeClock(),
        trailing_state_path=tmp_path / "trailing_state.json",
    )


class TestEquityTrailingStop:
    def test_fires_exactly_at_threshold(self, tmp_path: Path) -> None:
        oms = _FakeOMS()
        mgr = _trailing_mgr(tmp_path, oms=oms)
        assert mgr.check(_ctx(equity=100_000.0)) == []  # establishes the peak
        triggered = mgr.check(_ctx(equity=90_000.0))    # exactly -10%
        assert {t["switch"] for t in triggered} == {"equity_trailing_stop"}
        assert triggered[0]["action"] == "halt_new"
        assert oms.halts == ["all"]

    def test_does_not_fire_above_threshold(self, tmp_path: Path) -> None:
        mgr = _trailing_mgr(tmp_path)
        mgr.check(_ctx(equity=100_000.0))
        assert mgr.check(_ctx(equity=90_001.0)) == []

    def test_missing_equity_never_fires(self, tmp_path: Path) -> None:
        mgr = _trailing_mgr(tmp_path)
        mgr.check(_ctx(equity=100_000.0))
        assert mgr.check(_ctx()) == []  # no equity in ctx, no broker account

    def test_peak_persists_across_manager_instances(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        _trailing_mgr(tmp_path, clock=clock).check(_ctx(equity=100_000.0))
        # Fresh manager, same state file: the peak survives the "restart".
        oms = _FakeOMS()
        mgr2 = _trailing_mgr(tmp_path, oms=oms, clock=clock)
        triggered = mgr2.check(_ctx(equity=89_000.0))
        assert {t["switch"] for t in triggered} == {"equity_trailing_stop"}
        assert oms.halts == ["all"]

    def test_cooldown_blocks_across_restart_even_after_recovery(
        self, tmp_path: Path,
    ) -> None:
        clock = _FakeClock()
        mgr1 = _trailing_mgr(tmp_path, clock=clock)
        mgr1.check(_ctx(equity=100_000.0))
        assert mgr1.check(_ctx(equity=85_000.0))  # trips, cooldown starts
        # Two days later, equity fully recovered, process restarted:
        # the persisted cooldown still keeps firing (idempotent halt).
        clock.now = _T0 + timedelta(days=2)
        oms = _FakeOMS()
        mgr2 = _trailing_mgr(tmp_path, oms=oms, clock=clock)
        triggered = mgr2.check(_ctx(equity=100_000.0))
        assert {t["switch"] for t in triggered} == {"equity_trailing_stop"}
        assert oms.halts == ["all"]

    def test_resumes_after_expiry_and_peak_resets(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        mgr1 = _trailing_mgr(tmp_path, clock=clock)
        mgr1.check(_ctx(equity=100_000.0))
        mgr1.check(_ctx(equity=85_000.0))  # trips; cooldown until T0+7d
        clock.now = _T0 + timedelta(days=8)
        mgr2 = _trailing_mgr(tmp_path, clock=clock)
        # First post-expiry mark-to-market: resumes, peak resets to it.
        assert mgr2.check(_ctx(equity=95_000.0)) == []
        state = json.loads((tmp_path / "trailing_state.json").read_text())
        assert state["peak_equity"] == 95_000.0
        assert state["cooldown_until"] is None
        # Drawdown is measured from the NEW peak: -10.6% from 95k re-fires.
        triggered = mgr2.check(_ctx(equity=84_900.0))
        assert {t["switch"] for t in triggered} == {"equity_trailing_stop"}

    def test_config_thresholds_apply(self, tmp_path: Path) -> None:
        cfg = {"trailing_stop_pct": 0.30, "trailing_stop_cooldown_days": 1}
        mgr = _trailing_mgr(tmp_path, config=cfg)
        mgr.check(_ctx(equity=100_000.0))
        assert mgr.check(_ctx(equity=75_000.0)) == []       # -25% < 30% limit
        assert mgr.check(_ctx(equity=70_000.0))             # -30% fires

    def test_corrupt_state_raises_at_construction(self, tmp_path: Path) -> None:
        (tmp_path / "trailing_state.json").write_text("{not json")
        with pytest.raises(ValueError, match="corrupt"):
            _trailing_mgr(tmp_path)


# --------------------------------------------------------------------- #
# CL-ep0c — open-position correlation
# --------------------------------------------------------------------- #

def _correlated_frame(n: int = 80) -> pd.DataFrame:
    """Two price series whose daily returns correlate ~0.99."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2026-03-02", periods=n)
    shocks = rng.normal(0, 0.004, n)
    a = 1.10 * np.exp(np.cumsum(shocks))
    b = 0.85 * np.exp(np.cumsum(shocks + rng.normal(0, 0.0004, n)))
    return pd.DataFrame({"EURUSD": a, "GBPUSD": b}, index=idx)


def _corr_mgr(
    positions: list[Position],
    df: pd.DataFrame | None,
    config: dict[str, object] | None = None,
    oms: _FakeOMS | None = None,
) -> KillSwitchManager:
    return KillSwitchManager(
        _FakeBroker(positions=positions), oms or _FakeOMS(), config or {},
        data_provider=_FakeDataProvider(df),
        clock=_FakeClock(),
        trailing_state_path=None,
    )


class TestOpenPositionCorrelation:
    def test_correlated_shorts_fire(self) -> None:
        # Two shorts in ~perfectly correlated pairs ARE the same bet.
        oms = _FakeOMS()
        mgr = _corr_mgr(
            [Position("EURUSD", -100_000, 1.10), Position("GBPUSD", -100_000, 0.85)],
            _correlated_frame(), oms=oms,
        )
        triggered = mgr.check(_baseline_ctx())
        assert {t["switch"] for t in triggered} == {"open_position_correlation"}
        assert triggered[0]["action"] == "reduce_50pct"
        # Minimal-safe reduce_50pct: halt new trades + evidence retained
        # for the CRITICAL operator log.
        assert oms.halts == ["all"]
        assert mgr.open_position_corr.last_mean_adjusted > 0.85
        assert mgr.open_position_corr.last_matrix is not None
        assert mgr.open_position_corr.last_directions == {"EURUSD": -1, "GBPUSD": -1}

    def test_hedged_long_short_does_not_fire(self) -> None:
        # Long+short in the same correlated pairs hedge each other.
        mgr = _corr_mgr(
            [Position("EURUSD", 100_000, 1.10), Position("GBPUSD", -100_000, 0.85)],
            _correlated_frame(),
        )
        assert mgr.check(_baseline_ctx()) == []

    def test_single_position_never_fires(self) -> None:
        mgr = _corr_mgr([Position("EURUSD", -100_000, 1.10)], _correlated_frame())
        assert mgr.check(_baseline_ctx()) == []

    def test_missing_data_never_fires(self) -> None:
        positions = [
            Position("EURUSD", -100_000, 1.10),
            Position("GBPUSD", -100_000, 0.85),
        ]
        # Provider returns nothing.
        assert _corr_mgr(positions, None).check(_baseline_ctx()) == []
        # Too few overlapping observations for a meaningful correlation.
        assert _corr_mgr(positions, _correlated_frame(n=5)).check(_baseline_ctx()) == []
        # No data provider wired at all (None-safe param).
        mgr = KillSwitchManager(
            _FakeBroker(positions=positions), _FakeOMS(), {},
            trailing_state_path=None,
        )
        assert mgr.check(_baseline_ctx()) == []

    def test_config_threshold_applies(self) -> None:
        cfg = {"open_position_corr_threshold": 0.9999}
        mgr = _corr_mgr(
            [Position("EURUSD", -100_000, 1.10), Position("GBPUSD", -100_000, 0.85)],
            _correlated_frame(), config=cfg,
        )
        assert mgr.check(_baseline_ctx()) == []


# --------------------------------------------------------------------- #
# CL-ep0c — yaml -> KillSwitchConfig -> manager plumb-through
# --------------------------------------------------------------------- #

class TestConfigPlumbThrough:
    def test_yaml_reaches_manager_evaluators(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "risk_profile.yaml"
        yaml_path.write_text("""
active: custom
profiles:
  custom:
    kill_switches:
      trailing_stop_pct: 0.33
      trailing_stop_cooldown_days: 2
      open_position_corr_threshold: 0.5
      open_position_corr_lookback_days: 10
""")
        with patch.dict(os.environ, {"CURLIT_RISK_PROFILE": ""}):
            profile = load_active_profile(yaml_path)
        ks = profile.kill_switches
        assert ks.trailing_stop_pct == 0.33
        assert ks.trailing_stop_cooldown_days == 2
        assert ks.open_position_corr_threshold == 0.5
        assert ks.open_position_corr_lookback_days == 10
        # The manager consumes the block as a plain dict (asdict) — the
        # same shape run_engine.build_kill_switch_manager passes it.
        mgr = KillSwitchManager(
            _FakeBroker(), _FakeOMS(), asdict(ks),
            trailing_state_path=tmp_path / "state.json",
        )
        assert mgr.trailing_stop.trailing_stop_pct == 0.33
        assert mgr.trailing_stop.cooldown_days == 2
        assert mgr.open_position_corr.threshold == 0.5
        assert mgr.open_position_corr.lookback_days == 10


class TestRegimeAwareSizer:
    def test_normal_returns_full_multiplier(self) -> None:
        sizer = RegimeAwareSizer()
        adj = sizer.compute_adjustment(vix=15, portfolio_dd=-0.02, correlation_regime="normal", max_pair_corr=0.3)
        assert adj.final_multiplier == 1.0

    def test_extreme_crisis_returns_near_zero(self) -> None:
        sizer = RegimeAwareSizer()
        adj = sizer.compute_adjustment(vix=45, portfolio_dd=-0.22, correlation_regime="crisis", max_pair_corr=0.9)
        assert adj.final_multiplier < 0.1
