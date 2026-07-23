"""Unit tests — risk: kill switches, sizing, correlation monitor.

CL-ep0c adds coverage for the equity-curve trailing stop (threshold,
state-file round-trip across manager instances, cooldown across
restarts with an injectable clock, post-expiry resume + peak reset)
and the open-position correlation switch (correlated shorts fire,
hedged long/short doesn't, single position / missing data never fire),
plus yaml -> KillSwitchConfig -> manager plumb-through.

CL-i4tx adds coverage for the de-facaded subsystem: profile-driven
daily-loss/drawdown thresholds, real flatten_all / reduce_50pct via OMS
intents (canonical-symbol netted, bypassing a prior halt), fail-closed
after three consecutive evaluation failures of the same switch,
reset_daily re-arming, and boot-time ARMED/UNARMED logging.
"""

import json
import logging
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
        # (symbol, target_position, bypass_halt) per submitted intent.
        self.intents: list[tuple[str, float, bool]] = []

    def halt_new_trades(self) -> None:
        self.halts.append("all")

    def submit_intent(self, intent, *, bypass_halt: bool = False) -> str:  # noqa: ANN001
        self.intents.append((intent.symbol, intent.target_position, bypass_halt))
        return intent.intent_id


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
        "price_stream_age_sec": 5,
        "cvix_zscore": 0.5,
        "position_mismatch": False,
    }


def _ctx(equity: float | None = None) -> dict[str, object]:
    ctx = _baseline_ctx()
    if equity is not None:
        ctx["equity"] = equity
    return ctx


class TestKillSwitchManager:
    def test_drawdown_limit_triggers_and_flattens(self) -> None:
        oms = _FakeOMS()
        broker = _FakeBroker([
            Position("EURUSD", 5_000.0, 1.10),
            Position("USDCAD", -8_000.0, 1.41),
        ])
        mgr = KillSwitchManager(broker, oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.25
        triggered = mgr.check(ctx)
        names = {t["switch"] for t in triggered}
        assert "drawdown_limit" in names
        # CL-i4tx: flatten_all is real — one target-0 intent per position,
        # bypassing the halt gate, then new trades halted.
        assert sorted(oms.intents) == [
            ("EURUSD", 0.0, True), ("USDCAD", 0.0, True),
        ]
        assert oms.halts == ["all"]

    def test_daily_loss_limit_uses_profile_threshold(self) -> None:
        # Aggressive-style profile: -10% daily budget. A -5% day (would fire
        # the old hardcoded -3%) must NOT fire; -12% must.
        cfg = {"daily_loss_limit_pct": -0.10}
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), cfg,
                                trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["daily_pnl_pct"] = -0.05
        assert mgr.check(ctx) == []
        ctx["daily_pnl_pct"] = -0.12
        names = {t["switch"] for t in mgr.check(ctx)}
        assert "daily_loss_limit" in names

    def test_drawdown_limit_uses_profile_threshold(self) -> None:
        cfg = {"drawdown_limit_pct": -0.40}
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), cfg,
                                trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.25  # would fire the old hardcoded -20%
        assert mgr.check(ctx) == []
        ctx["portfolio_dd"] = -0.45
        names = {t["switch"] for t in mgr.check(ctx)}
        assert "drawdown_limit" in names

    def test_vix_spike_requires_level_and_change(self) -> None:
        oms = _FakeOMS()
        broker = _FakeBroker([Position("EURUSD", 4_000.0, 1.10)])
        mgr = KillSwitchManager(broker, oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["vix_level"] = 40  # level alone insufficient
        assert mgr.check(ctx) == []
        ctx["vix_change_1d"] = 0.6
        triggered = mgr.check(ctx)
        assert {t["switch"] for t in triggered} == {"vix_spike"}
        assert triggered[0]["action"] == "reduce_50pct"
        # CL-i4tx: reduce_50pct is real — halved target via OMS, then halt.
        assert oms.intents == [("EURUSD", 2_000.0, True)]
        assert oms.halts == ["all"]

    def test_fx_vol_spike_triggers(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {},
                                trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["cvix_zscore"] = 3.5
        names = {t["switch"] for t in mgr.check(ctx)}
        assert "fx_vol_spike" in names

    def test_stale_prices_triggers(self) -> None:
        oms = _FakeOMS()
        mgr = KillSwitchManager(_FakeBroker(), oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["price_stream_age_sec"] = 700
        names = {t["switch"] for t in mgr.check(ctx)}
        assert "stale_prices" in names
        assert oms.halts == ["all"]


class _ResumableOMS(_FakeOMS):
    def __init__(self) -> None:
        super().__init__()
        self.resumes = 0

    def resume_trades(self) -> None:
        self.resumes += 1


class TestStalePricesAutoResume:
    """CL-nxjx: stale_prices (a data-availability gate) auto-lifts when the
    stream recovers; risk switches stay sticky for human review."""

    def _mgr(self):  # noqa: ANN202
        oms = _ResumableOMS()
        return KillSwitchManager(_FakeBroker(), oms, {},
                                 trailing_state_path=None), oms

    def test_auto_resumes_when_stream_recovers(self) -> None:
        mgr, oms = self._mgr()
        stale = _baseline_ctx()
        stale["price_stream_age_sec"] = 700
        mgr.check(stale)                       # fires stale_prices -> halt
        assert oms.halts == ["all"]
        fresh = _baseline_ctx()                # price_stream_age_sec = 5
        assert mgr.attempt_auto_resume(fresh) is True
        assert oms.resumes == 1
        # re-armed: can fire again if it goes stale later
        assert "stale_prices" not in mgr._triggered_today
        mgr.check(stale)
        assert oms.halts == ["all", "all"]

    def test_risk_switch_stays_sticky(self) -> None:
        mgr, oms = self._mgr()
        bad = _baseline_ctx()
        bad["daily_pnl_pct"] = -0.50  # daily_loss_limit
        mgr.check(bad)
        assert oms.halts == ["all"]
        recovered = _baseline_ctx()  # pnl back to +1%
        assert mgr.attempt_auto_resume(recovered) is False
        assert oms.resumes == 0  # risk halt NEVER auto-lifts

    def test_stale_plus_risk_no_resume(self) -> None:
        mgr, oms = self._mgr()
        both = _baseline_ctx()
        both["price_stream_age_sec"] = 700
        both["daily_pnl_pct"] = -0.50
        mgr.check(both)
        fresh = _baseline_ctx()  # stale cleared, but the risk cause remains
        assert mgr.attempt_auto_resume(fresh) is False
        assert oms.resumes == 0

    def test_manual_halt_not_auto_resumed(self) -> None:
        # No switch fired → no active causes → auto-resume must not touch a
        # (hypothetical) manual halt.
        mgr, oms = self._mgr()
        assert mgr.attempt_auto_resume(_baseline_ctx()) is False
        assert oms.resumes == 0

    def test_reset_daily_clears_causes(self) -> None:
        mgr, _oms = self._mgr()
        stale = _baseline_ctx()
        stale["price_stream_age_sec"] = 700
        mgr.check(stale)
        assert mgr._active_halt_causes
        mgr.reset_daily()  # manual-resume default clears
        assert not mgr._active_halt_causes

    def test_rollover_keeps_causes_no_deadlock(self) -> None:
        # CL-ssoh (P1): an AUTOMATIC UTC rollover (clear_causes=False) must NOT
        # strand a stale_prices halt. Old bug: it cleared causes with the OMS
        # still halted, so auto-resume saw an empty set, thought "manual halt",
        # and never lifted it — permanent weekend deadlock.
        mgr, oms = self._mgr()
        stale = _baseline_ctx()
        stale["price_stream_age_sec"] = 700
        mgr.check(stale)
        assert mgr._active_halt_causes == {"stale_prices"}
        mgr.reset_daily(clear_causes=False)  # the rollover path
        assert mgr._active_halt_causes == {"stale_prices"}  # NOT stranded
        assert "stale_prices" not in mgr._triggered_today   # but re-armed
        # Stream recovers -> auto-resume lifts it (no deadlock).
        assert mgr.attempt_auto_resume(_baseline_ctx()) is True
        assert oms.resumes == 1

    def test_eval_failclosed_halt_is_sticky_vs_auto_resume(self) -> None:
        # CL-7zwp (P1): a fail-closed eval-failure halt registers a sticky
        # cause, so even when a co-active stale_prices gate clears, auto-resume
        # must NOT lift trading while the evaluator is still broken.
        mgr, oms = self._mgr()
        for sw in mgr.switches:
            if sw.name == "daily_loss_limit":
                def _boom(ctx: dict[str, object]) -> bool:
                    raise RuntimeError("broken")
                sw.condition = _boom
        stale = _baseline_ctx()
        stale["price_stream_age_sec"] = 700
        for _ in range(3):
            mgr.check(stale)  # 3 eval failures -> fail-closed; stale also fires
        assert "daily_loss_limit" in mgr._active_halt_causes  # sticky
        # Stream recovers, but the broken-evaluator halt holds trading closed.
        assert mgr.attempt_auto_resume(_baseline_ctx()) is False
        assert oms.resumes == 0

    def test_reconciliation_failure_triggers(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {},
                                trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["position_mismatch"] = True
        names = {t["switch"] for t in mgr.check(ctx)}
        assert "reconciliation_failure" in names

    def test_normal_context_triggers_nothing(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {}, trailing_state_path=None)
        triggered = mgr.check(_baseline_ctx())
        assert triggered == []

    def test_missing_keys_trigger_nothing(self) -> None:
        # Fail-soft: an omitted input is "no opinion", never a trigger.
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {}, trailing_state_path=None)
        assert mgr.check({"equity": 100_000.0}) == []

    def test_deleted_switches_are_gone(self) -> None:
        # CL-i4tx honesty: switches without a real input feed were deleted.
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {}, trailing_state_path=None)
        names = {sw.name for sw in mgr.switches}
        assert names == {
            "daily_loss_limit", "drawdown_limit", "vix_spike", "fx_vol_spike",
            "reconciliation_failure", "stale_prices", "equity_trailing_stop",
            "open_position_correlation",
        }


class TestFlattenAndReduce:
    def test_flatten_nets_by_canonical_symbol(self) -> None:
        # Duplicate dialects of the same instrument must yield ONE intent —
        # two close orders for one position would double-close.
        oms = _FakeOMS()
        broker = _FakeBroker([
            Position("USD_CAD", -3_000.0, 1.41),
            Position("USDCAD", -1_000.0, 1.41),
        ])
        mgr = KillSwitchManager(broker, oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.25
        mgr.check(ctx)
        assert oms.intents == [("USD_CAD", 0.0, True)]

    def test_flatten_skips_dust_positions(self) -> None:
        oms = _FakeOMS()
        broker = _FakeBroker([Position("EURUSD", 1e-9, 1.10)])
        mgr = KillSwitchManager(broker, oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.25
        mgr.check(ctx)
        assert oms.intents == []
        assert oms.halts == ["all"]  # still halts even with nothing to close

    def test_flatten_survives_broker_failure(self) -> None:
        class _BrokenBroker:
            def get_positions(self):  # noqa: ANN202
                raise ConnectionError("api down")

        oms = _FakeOMS()
        mgr = KillSwitchManager(_BrokenBroker(), oms, {},
                                trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.25
        triggered = mgr.check(ctx)
        # Trigger still recorded and new trades still halted.
        assert {t["switch"] for t in triggered} == {"drawdown_limit"}
        assert oms.halts == ["all"]

    def test_partial_flatten_does_not_spend_trigger(self) -> None:
        # CL-xh6g (P1): one leg's flatten is rejected by the broker -> the
        # de-risk is INCOMPLETE, so the once-per-day trigger is NOT spent and
        # the switch re-fires next tick to retry the unclosed leg (residual
        # open risk must not go quiet for the rest of the UTC day).
        class _PartialFailOMS(_FakeOMS):
            def submit_intent(self, intent, *, bypass_halt: bool = False):  # noqa: ANN001, ANN201
                if intent.symbol == "USDCAD":
                    raise RuntimeError("broker rejected")
                return super().submit_intent(intent, bypass_halt=bypass_halt)

        oms = _PartialFailOMS()
        broker = _FakeBroker([
            Position("EURUSD", 5_000.0, 1.10),
            Position("USDCAD", -8_000.0, 1.41),
        ])
        mgr = KillSwitchManager(broker, oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.50  # drawdown -> flatten
        mgr.check(ctx)
        assert ("EURUSD", 0.0, True) in oms.intents      # good leg closed
        assert "drawdown_limit" not in mgr._triggered_today  # NOT spent
        # Next tick re-fires and retries (the good leg is re-submitted, the
        # bad leg re-attempted) — still incomplete, still not spent.
        mgr.check(ctx)
        assert "drawdown_limit" not in mgr._triggered_today

    def test_complete_flatten_spends_trigger(self) -> None:
        # Companion: when EVERY leg closes, the trigger IS spent (no re-fire).
        oms = _FakeOMS()
        broker = _FakeBroker([Position("EURUSD", 5_000.0, 1.10)])
        mgr = KillSwitchManager(broker, oms, {}, trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["portfolio_dd"] = -0.50
        mgr.check(ctx)
        assert "drawdown_limit" in mgr._triggered_today


class TestFailClosedOnEvalErrors:
    @staticmethod
    def _break_switch(mgr: KillSwitchManager, name: str) -> None:
        for sw in mgr.switches:
            if sw.name == name:
                def _boom(ctx: dict[str, object]) -> bool:
                    raise RuntimeError("data path broken")
                sw.condition = _boom
                return
        raise AssertionError(f"no switch named {name}")

    def test_three_consecutive_failures_halt(self) -> None:
        oms = _FakeOMS()
        mgr = KillSwitchManager(_FakeBroker(), oms, {}, trailing_state_path=None)
        self._break_switch(mgr, "daily_loss_limit")
        mgr.check(_baseline_ctx())
        mgr.check(_baseline_ctx())
        assert oms.halts == []  # two failures: logged, not yet fail-closed
        mgr.check(_baseline_ctx())
        assert oms.halts == ["all"]  # third consecutive failure fails CLOSED
        # CL-7zwp (P1): >= not == — a still-broken evaluator RE-halts every
        # subsequent tick (halt_new is idempotent) so that if anything ever
        # resumes trading, the next failing tick corrects it. It also
        # registers a sticky halt cause so auto-resume never lifts it.
        mgr.check(_baseline_ctx())
        assert oms.halts == ["all", "all"]  # re-halts on the 4th (idempotent)
        assert "daily_loss_limit" in mgr._active_halt_causes  # sticky cause

    def test_success_resets_failure_streak(self) -> None:
        oms = _FakeOMS()
        mgr = KillSwitchManager(_FakeBroker(), oms, {}, trailing_state_path=None)
        original = next(
            sw for sw in mgr.switches if sw.name == "daily_loss_limit"
        ).condition
        self._break_switch(mgr, "daily_loss_limit")
        mgr.check(_baseline_ctx())
        mgr.check(_baseline_ctx())
        # Recovers: streak resets, so two MORE failures don't halt.
        for sw in mgr.switches:
            if sw.name == "daily_loss_limit":
                sw.condition = original
        mgr.check(_baseline_ctx())
        self._break_switch(mgr, "daily_loss_limit")
        mgr.check(_baseline_ctx())
        mgr.check(_baseline_ctx())
        assert oms.halts == []


class TestResetDaily:
    def test_fired_switch_rearms_after_reset(self) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {},
                                trailing_state_path=None)
        ctx = _baseline_ctx()
        ctx["daily_pnl_pct"] = -0.05
        assert {t["switch"] for t in mgr.check(ctx)} == {"daily_loss_limit"}
        assert mgr.check(ctx) == []  # deduped for the rest of the day
        mgr.reset_daily()
        assert {t["switch"] for t in mgr.check(ctx)} == {"daily_loss_limit"}


class TestLogArming:
    def test_armed_and_unarmed_lines(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {},
                                trailing_state_path=None)
        with caplog.at_level(logging.INFO, logger="src.risk.kill_switches"):
            mgr.log_arming({
                "equity", "daily_pnl_pct", "portfolio_dd", "vix_level",
                "vix_change_1d", "cvix_zscore", "price_stream_age_sec",
                "position_mismatch",
            })
        text = caplog.text
        assert "daily_loss_limit: ARMED" in text
        assert "equity_trailing_stop: ARMED" in text
        # No data_provider wired -> the corr switch honestly reports UNARMED.
        assert "open_position_correlation: UNARMED" in text

    def test_missing_inputs_log_unarmed(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mgr = KillSwitchManager(_FakeBroker(), _FakeOMS(), {},
                                trailing_state_path=None)
        with caplog.at_level(logging.INFO, logger="src.risk.kill_switches"):
            mgr.log_arming({"equity"})
        assert "daily_loss_limit: UNARMED" in caplog.text
        assert "stale_prices: UNARMED" in caplog.text


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
