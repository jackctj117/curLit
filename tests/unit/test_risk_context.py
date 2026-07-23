"""Unit tests — RiskContextBuilder (CL-i4tx).

The builder assembles the REAL kill-switch context each health tick:
daily PnL vs persisted UTC day-start equity, drawdown vs persisted
running peak, VIX level/1d change, CVIX z-score, price-stream age
(trading-window gated), and the position-mismatch flag. Fail-soft:
an unavailable input omits its key; a CORRUPT state file raises at
construction (fail loud — mirrors EquityTrailingStop).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.risk.risk_context import RiskContextBuilder

_T0 = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


class _FakeClock:
    def __init__(self, now: datetime = _T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class _FakeProvider:
    """Minimal DataProvider double: get_series + get_latest_value."""

    def __init__(
        self,
        series: dict[str, list[float]] | None = None,
        latest: dict[str, float] | None = None,
    ) -> None:
        self.series = series or {}
        self.latest = latest or {}

    def get_series(
        self, series_id: str, start: datetime, end: datetime,
    ) -> pd.Series:
        return pd.Series(self.series.get(series_id, []))

    def get_latest_value(self, series_id: str, as_of: datetime) -> float | None:
        return self.latest.get(series_id)


def _builder(tmp_path: Path | None = None, **kwargs: object) -> RiskContextBuilder:
    state = tmp_path / "risk_context_state.json" if tmp_path is not None else None
    kwargs.setdefault("clock", _FakeClock())
    return RiskContextBuilder(state_path=state, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# daily_pnl_pct + UTC day rollover
# --------------------------------------------------------------------- #


class TestDailyPnl:
    def test_first_tick_sets_day_start_and_zero_pnl(self) -> None:
        ctx = _builder().build(100_000.0)
        assert ctx["equity"] == 100_000.0
        assert ctx["daily_pnl_pct"] == 0.0

    def test_intraday_loss_measured_from_day_start(self) -> None:
        b = _builder()
        b.build(100_000.0)
        ctx = b.build(96_000.0)
        assert ctx["daily_pnl_pct"] == pytest.approx(-0.04)

    def test_day_start_persists_across_restart(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        _builder(tmp_path, clock=clock).build(100_000.0)
        # Same UTC day, new process: day-start must survive — a redeploy
        # must NOT grant a fresh daily-loss budget.
        b2 = _builder(tmp_path, clock=clock)
        ctx = b2.build(95_000.0)
        assert ctx["daily_pnl_pct"] == pytest.approx(-0.05)
        assert b2.consume_day_rollover() is False

    def test_utc_rollover_resets_day_start_and_flags_once(self) -> None:
        clock = _FakeClock()
        b = _builder(clock=clock)
        b.build(100_000.0)
        b.build(97_000.0)
        clock.now = _T0 + timedelta(days=1)
        ctx = b.build(97_000.0)
        assert ctx["daily_pnl_pct"] == 0.0  # new day, new baseline
        assert b.consume_day_rollover() is True
        assert b.consume_day_rollover() is False  # consumed exactly once

    def test_rollover_detected_across_restart(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        _builder(tmp_path, clock=clock).build(100_000.0)
        clock.now = _T0 + timedelta(days=1)
        b2 = _builder(tmp_path, clock=clock)
        ctx = b2.build(90_000.0)
        assert ctx["daily_pnl_pct"] == 0.0
        assert b2.consume_day_rollover() is True

    def test_nonpositive_equity_does_not_poison_state(self) -> None:
        b = _builder()
        b.build(100_000.0)
        ctx = b.build(0.0)  # broken mark: no pnl/dd fabricated from it
        assert ctx["equity"] == 0.0
        # Day-start survives the bad mark.
        assert b.build(98_000.0)["daily_pnl_pct"] == pytest.approx(-0.02)

    def test_nonpositive_equity_omits_pnl_and_dd_keys(self) -> None:
        # CL-ubo0 (P0): a zero/negative mark must NOT fabricate a -100%
        # daily_pnl_pct / portfolio_dd (which would trip daily_loss_limit +
        # drawdown_limit → flatten_all on a single feed glitch). The keys are
        # omitted entirely — "no opinion" — until a good mark returns.
        b = _builder()
        b.build(110_000.0)  # establish a good day-start + peak
        for bad in (0.0, -5.0):
            ctx = b.build(bad)
            assert "daily_pnl_pct" not in ctx, f"fabricated pnl at eq={bad}"
            assert "portfolio_dd" not in ctx, f"fabricated dd at eq={bad}"
        # The next good mark restores correct values against the prior state.
        good = b.build(104_500.0)
        assert good["daily_pnl_pct"] == pytest.approx(-0.05)
        assert good["portfolio_dd"] == pytest.approx(-0.05)


# --------------------------------------------------------------------- #
# portfolio_dd (running peak)
# --------------------------------------------------------------------- #


class TestPortfolioDrawdown:
    def test_dd_measured_from_running_peak(self) -> None:
        b = _builder()
        b.build(100_000.0)
        b.build(110_000.0)  # new peak
        ctx = b.build(88_000.0)
        assert ctx["portfolio_dd"] == pytest.approx(-0.20)

    def test_peak_persists_across_restart(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        _builder(tmp_path, clock=clock).build(120_000.0)
        ctx = _builder(tmp_path, clock=clock).build(90_000.0)
        assert ctx["portfolio_dd"] == pytest.approx(-0.25)

    def test_peak_survives_day_rollover(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        b = _builder(tmp_path, clock=clock)
        b.build(120_000.0)
        clock.now = _T0 + timedelta(days=3)
        ctx = b.build(96_000.0)
        # Daily pnl resets, drawdown does NOT — it tracks the all-time peak.
        assert ctx["daily_pnl_pct"] == 0.0
        assert ctx["portfolio_dd"] == pytest.approx(-0.20)


# --------------------------------------------------------------------- #
# Vol indices (VIX / CVIX) — fail-soft omission
# --------------------------------------------------------------------- #


class TestVolInputs:
    def test_no_provider_omits_vol_keys(self) -> None:
        ctx = _builder().build(100_000.0)
        assert "vix_level" not in ctx
        assert "vix_change_1d" not in ctx
        assert "cvix_zscore" not in ctx

    def test_vix_level_and_change_from_series(self) -> None:
        provider = _FakeProvider(series={"VIX": [20.0, 24.0, 37.0]})
        ctx = _builder(data_provider=provider).build(100_000.0)
        assert ctx["vix_level"] == 37.0
        assert ctx["vix_change_1d"] == pytest.approx(37.0 / 24.0 - 1.0)

    def test_single_vix_row_omits_change(self) -> None:
        provider = _FakeProvider(series={"VIX": [22.0]})
        ctx = _builder(data_provider=provider).build(100_000.0)
        assert ctx["vix_level"] == 22.0
        assert "vix_change_1d" not in ctx

    def test_empty_vix_series_omits_both(self) -> None:
        ctx = _builder(data_provider=_FakeProvider()).build(100_000.0)
        assert "vix_level" not in ctx

    def test_cvix_zscore_present_when_series_populated(self) -> None:
        # Calm baseline then a spike: z-score must be large and positive.
        history = [8.0, 8.1, 7.9, 8.0, 8.2, 8.1, 8.0, 7.8, 8.1, 8.0] * 6
        history[-1] = 15.0
        provider = _FakeProvider(
            series={"CVIX": history}, latest={"CVIX": 15.0},
        )
        ctx = _builder(
            data_provider=provider, cvix_z_lookback_days=20,
        ).build(100_000.0)
        assert ctx["cvix_zscore"] > 3.0

    def test_cvix_absent_latest_omits_key(self) -> None:
        provider = _FakeProvider(series={"CVIX": [8.0] * 60}, latest={})
        ctx = _builder(data_provider=provider).build(100_000.0)
        assert "cvix_zscore" not in ctx

    def test_provider_error_is_fail_soft(self) -> None:
        class _Boom:
            def get_series(self, *a: object, **k: object) -> None:
                raise ConnectionError("db down")

            def get_latest_value(self, *a: object, **k: object) -> None:
                raise ConnectionError("db down")

        ctx = _builder(data_provider=_Boom()).build(100_000.0)
        assert ctx["equity"] == 100_000.0
        assert "vix_level" not in ctx
        assert "cvix_zscore" not in ctx


# --------------------------------------------------------------------- #
# price_stream_age_sec
# --------------------------------------------------------------------- #


class TestPriceStreamAge:
    def test_fresh_tick_small_age(self) -> None:
        prices = {"EURUSD": {"ts": (_T0 - timedelta(seconds=5)).isoformat()}}
        ctx = _builder(last_prices=prices).build(100_000.0)
        assert ctx["price_stream_age_sec"] == pytest.approx(5.0)

    def test_freshest_tick_wins(self) -> None:
        # One dead symbol must not trip staleness while another still ticks.
        prices = {
            "EURUSD": {"ts": (_T0 - timedelta(hours=3)).isoformat()},
            "USDJPY": {"ts": (_T0 - timedelta(seconds=30)).isoformat()},
        }
        ctx = _builder(last_prices=prices).build(100_000.0)
        assert ctx["price_stream_age_sec"] == pytest.approx(30.0)

    def test_oanda_nanosecond_z_timestamp_parses(self) -> None:
        stamp = "2026-07-14T11:59:20.123456789Z"  # 40s before _T0
        ctx = _builder(last_prices={"EURUSD": {"ts": stamp}}).build(100_000.0)
        assert ctx["price_stream_age_sec"] == pytest.approx(39.876543, abs=0.01)

    def test_no_ticks_measures_from_builder_start(self) -> None:
        clock = _FakeClock()
        b = _builder(last_prices={}, clock=clock)
        clock.now = _T0 + timedelta(seconds=700)
        # Stream never connected: staleness must STILL be measurable.
        assert b.build(100_000.0)["price_stream_age_sec"] == pytest.approx(700.0)

    def test_outside_trading_window_omits_key(self) -> None:
        prices = {"EURUSD": {"ts": (_T0 - timedelta(hours=40)).isoformat()}}
        ctx = _builder(
            last_prices=prices, in_trading_window=lambda ts: False,
        ).build(100_000.0)
        # Weekend close is not a dead stream.
        assert "price_stream_age_sec" not in ctx

    def test_no_last_prices_wiring_omits_key(self) -> None:
        ctx = _builder().build(100_000.0)
        assert "price_stream_age_sec" not in ctx


# --------------------------------------------------------------------- #
# position_mismatch
# --------------------------------------------------------------------- #


class TestPositionMismatch:
    def test_supplier_true_false(self) -> None:
        assert _builder(position_mismatch=lambda: True).build(1.0)[
            "position_mismatch"] is True
        assert _builder(position_mismatch=lambda: False).build(1.0)[
            "position_mismatch"] is False

    def test_supplier_none_omits_key(self) -> None:
        ctx = _builder(position_mismatch=lambda: None).build(1.0)
        assert "position_mismatch" not in ctx

    def test_no_supplier_omits_key(self) -> None:
        assert "position_mismatch" not in _builder().build(1.0)


# --------------------------------------------------------------------- #
# provided_keys — feeds the boot-time ARMED/UNARMED log
# --------------------------------------------------------------------- #


class TestProvidedKeys:
    def test_minimal_wiring(self) -> None:
        assert _builder().provided_keys() == {
            "equity", "daily_pnl_pct", "portfolio_dd",
        }

    def test_full_wiring(self) -> None:
        keys = _builder(
            data_provider=_FakeProvider(),
            last_prices={},
            position_mismatch=lambda: None,
        ).provided_keys()
        assert keys == {
            "equity", "daily_pnl_pct", "portfolio_dd", "vix_level",
            "vix_change_1d", "cvix_zscore", "price_stream_age_sec",
            "position_mismatch",
        }


# --------------------------------------------------------------------- #
# State-file discipline
# --------------------------------------------------------------------- #


class TestStateFile:
    def test_corrupt_state_raises_at_construction(self, tmp_path: Path) -> None:
        (tmp_path / "risk_context_state.json").write_text("{not json")
        with pytest.raises(ValueError, match="corrupt"):
            _builder(tmp_path)

    def test_state_written_atomically_no_tmp_left(self, tmp_path: Path) -> None:
        _builder(tmp_path).build(100_000.0)
        state_file = tmp_path / "risk_context_state.json"
        assert state_file.exists()
        assert not (tmp_path / "risk_context_state.json.tmp").exists()
        raw = json.loads(state_file.read_text())
        assert raw["day"] == "2026-07-14"
        assert raw["day_start_equity"] == 100_000.0
        assert raw["peak_equity"] == 100_000.0
