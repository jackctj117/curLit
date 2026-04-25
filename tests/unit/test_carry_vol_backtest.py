"""Unit tests — backtest.carry_vol_backtest (CL-2hs / A6).

Acceptance:
- Backtest runs on multi-year data
- Sharpe improvement with vol filter visible (in adversarial regime)
- Max DD reduced (in adversarial regime)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.carry_vol_backtest import (
    BacktestRun,
    CarryVolBacktester,
    CarryVolBacktestResult,
)
from src.backtest.cost_model import CostModel
from src.strategies.carry_vol_filter import CarryVolFilterConfig

# Trading-day frequency: business-day resampling for synthetic test data.
_BDAY = "B"


# =============================================================================
# Test data builders
# =============================================================================


def _build_rates(
    n_days: int, currencies: list[str], rate_map: dict[str, float],
) -> pd.DataFrame:
    """Constant rates per currency over n_days."""
    idx = pd.date_range("2010-01-04", periods=n_days, freq=_BDAY)
    return pd.DataFrame({c: [rate_map[c]] * n_days for c in currencies}, index=idx)


def _build_fx_returns(
    rates_index: pd.DatetimeIndex,
    currencies: list[str],
    seed: int = 0,
    drift_per_ccy: dict[str, float] | None = None,
    vol_per_ccy: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Random daily returns per currency vs USD."""
    rng = np.random.default_rng(seed)
    drift_per_ccy = drift_per_ccy or {c: 0.0 for c in currencies}
    vol_per_ccy = vol_per_ccy or {c: 0.005 for c in currencies}
    return pd.DataFrame(
        {
            c: rng.normal(drift_per_ccy[c], vol_per_ccy[c], len(rates_index))
            for c in currencies
        },
        index=rates_index,
    )


def _build_vol_series(
    rates_index: pd.DatetimeIndex,
    base_level: float = 10.0,
    spike_window: tuple[int, int] | None = None,
    spike_level: float = 25.0,
) -> pd.Series:
    """Vol level mostly flat + optional spike window."""
    arr = np.full(len(rates_index), base_level, dtype=float)
    # Add a tiny amount of noise so std > 0 in the trailing window.
    rng = np.random.default_rng(7)
    arr += rng.normal(0, 0.5, len(rates_index))
    if spike_window:
        start_i, end_i = spike_window
        arr[start_i:end_i] = spike_level
    return pd.Series(arr, index=rates_index)


# =============================================================================
# Smoke tests
# =============================================================================


class TestSmoke:
    def test_simple_run_produces_runs(self) -> None:
        currencies = ["EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD", "NOK"]
        rates = _build_rates(
            n_days=252,
            currencies=currencies,
            rate_map={
                "EUR": 0.04, "GBP": 0.045, "AUD": 0.05, "NZD": 0.055,
                "JPY": 0.001, "CHF": 0.005, "CAD": 0.04, "NOK": 0.04,
            },
        )
        fx = _build_fx_returns(rates.index, currencies, seed=1)
        vol = _build_vol_series(rates.index, base_level=10.0)
        result = CarryVolBacktester(
            CarryVolFilterConfig(top_k=3, bottom_k=3),
        ).run(rates, fx, vol)
        assert isinstance(result, CarryVolBacktestResult)
        assert isinstance(result.filtered, BacktestRun)
        assert isinstance(result.unfiltered, BacktestRun)
        assert len(result.filtered.returns) == 252
        assert len(result.unfiltered.returns) == 252

    def test_metrics_present(self) -> None:
        currencies = ["EUR", "GBP", "AUD", "NZD", "JPY", "CHF"]
        rates = _build_rates(
            n_days=252,
            currencies=currencies,
            rate_map={
                "EUR": 0.04, "GBP": 0.045, "AUD": 0.05, "NZD": 0.055,
                "JPY": 0.001, "CHF": 0.005,
            },
        )
        fx = _build_fx_returns(rates.index, currencies, seed=2)
        vol = _build_vol_series(rates.index)
        result = CarryVolBacktester(
            CarryVolFilterConfig(top_k=2, bottom_k=2),
        ).run(rates, fx, vol)
        for key in ("sharpe", "max_drawdown", "cagr"):
            assert key in result.filtered.metrics
            assert key in result.unfiltered.metrics


# =============================================================================
# Filter actually filters
# =============================================================================


class TestFilterEffect:
    def test_filtered_exposure_drops_during_vol_spike(self) -> None:
        currencies = ["EUR", "GBP", "AUD", "NZD", "JPY", "CHF"]
        n = 1500  # ~6 years business days
        rates = _build_rates(
            n_days=n,
            currencies=currencies,
            rate_map={
                "EUR": 0.04, "GBP": 0.045, "AUD": 0.05, "NZD": 0.055,
                "JPY": 0.001, "CHF": 0.005,
            },
        )
        fx = _build_fx_returns(rates.index, currencies, seed=5)
        # Spike vol from day 500 to 700 (way above the rolling baseline).
        vol = _build_vol_series(
            rates.index, base_level=10.0,
            spike_window=(500, 700), spike_level=30.0,
        )
        result = CarryVolBacktester(
            CarryVolFilterConfig(top_k=2, bottom_k=2),
        ).run(rates, fx, vol)

        # In the spike window, filtered exposure should drop below 1.0.
        spike_window_exposure = result.filtered.exposure_history.iloc[500:700]
        assert (spike_window_exposure < 1.0).any(), (
            "filtered run never reduced exposure during vol spike"
        )
        # Unfiltered always 1.0.
        assert (result.unfiltered.exposure_history == 1.0).all()

    def test_unfiltered_returns_unaffected_by_vol(self) -> None:
        # Returns of the unfiltered run should be the same regardless of vol
        # input — just a sanity check on the filter wiring.
        currencies = ["EUR", "GBP", "AUD", "JPY"]
        rates = _build_rates(
            n_days=300,
            currencies=currencies,
            rate_map={"EUR": 0.04, "GBP": 0.045, "AUD": 0.05, "JPY": 0.001},
        )
        fx = _build_fx_returns(rates.index, currencies, seed=8)
        vol_low = _build_vol_series(rates.index, base_level=10.0)
        vol_high = _build_vol_series(rates.index, base_level=80.0)
        bt = CarryVolBacktester(
            CarryVolFilterConfig(top_k=1, bottom_k=1),
        )
        r1 = bt.run(rates, fx, vol_low)
        r2 = bt.run(rates, fx, vol_high)
        # Unfiltered runs should produce identical returns regardless of vol.
        pd.testing.assert_series_equal(
            r1.unfiltered.returns, r2.unfiltered.returns, check_names=False,
        )


class TestDrawdownReduction:
    def test_filter_reduces_drawdown_in_adversarial_regime(self) -> None:
        """Construct a regime where vol spikes coincide with adverse returns —
        the filter should clearly reduce drawdown vs unfiltered."""
        currencies = ["EUR", "GBP", "AUD", "NZD", "JPY", "CHF"]
        n = 1500
        idx = pd.date_range("2010-01-04", periods=n, freq=_BDAY)
        rates = pd.DataFrame(
            {c: [r] * n for c, r in zip(
                currencies,
                [0.04, 0.045, 0.05, 0.055, 0.001, 0.005],
                strict=True,
            )},
            index=idx,
        )
        # Background returns are mildly positive (carry pays).
        rng = np.random.default_rng(13)
        fx_arr = rng.normal(0.00005, 0.005, (n, len(currencies)))
        # Inject a 200-day adverse window where high-yielders crash hard.
        crash_start, crash_end = 600, 800
        fx_arr[crash_start:crash_end, [3, 2]] = rng.normal(
            -0.005, 0.01, (crash_end - crash_start, 2),
        )  # NZD, AUD crash
        fx = pd.DataFrame(fx_arr, index=idx, columns=currencies)
        # Vol spikes during the same window (filter should detect + cut).
        vol = _build_vol_series(
            idx, base_level=10.0, spike_window=(crash_start, crash_end),
            spike_level=30.0,
        )

        result = CarryVolBacktester(
            CarryVolFilterConfig(top_k=2, bottom_k=2),
        ).run(rates, fx, vol)

        # Filter should cut the max drawdown (max_dd is negative; filtered > unfiltered means closer to 0).
        assert result.filtered.metrics["max_drawdown"] > result.unfiltered.metrics["max_drawdown"], (
            f"filter did not reduce drawdown: filtered={result.filtered.metrics['max_dd']:.4f} "
            f"unfiltered={result.unfiltered.metrics['max_dd']:.4f}"
        )


# =============================================================================
# Construction validation
# =============================================================================


class TestConstruction:
    def test_empty_rates_rejected(self) -> None:
        with pytest.raises(AssertionError):
            CarryVolBacktester().run(
                pd.DataFrame(),
                pd.DataFrame({"EUR": [0.0]}, index=[pd.Timestamp("2010-01-04")]),
                pd.Series([10.0], index=[pd.Timestamp("2010-01-04")]),
            )

    def test_too_few_overlapping_days_rejected(self) -> None:
        idx = pd.date_range("2020-01-01", periods=30, freq=_BDAY)
        rates = pd.DataFrame({"EUR": [0.04] * 30}, index=idx)
        fx = pd.DataFrame({"EUR": [0.0] * 30}, index=idx)
        vol = pd.Series([10.0] * 30, index=idx)
        with pytest.raises(AssertionError, match="overlapping"):
            CarryVolBacktester().run(rates, fx, vol)


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_summary_includes_metrics(self) -> None:
        currencies = ["EUR", "GBP", "AUD", "JPY"]
        rates = _build_rates(
            n_days=300,
            currencies=currencies,
            rate_map={"EUR": 0.04, "GBP": 0.045, "AUD": 0.05, "JPY": 0.001},
        )
        fx = _build_fx_returns(rates.index, currencies, seed=20)
        vol = _build_vol_series(rates.index)
        result = CarryVolBacktester(
            CarryVolFilterConfig(top_k=1, bottom_k=1),
        ).run(rates, fx, vol)
        s = result.summary()
        assert "Sharpe" in s
        assert "max_dd" in s

    def test_run_to_dict_keys(self) -> None:
        currencies = ["EUR", "GBP", "AUD", "JPY"]
        rates = _build_rates(
            n_days=300,
            currencies=currencies,
            rate_map={"EUR": 0.04, "GBP": 0.045, "AUD": 0.05, "JPY": 0.001},
        )
        fx = _build_fx_returns(rates.index, currencies, seed=21)
        vol = _build_vol_series(rates.index)
        result = CarryVolBacktester(
            CarryVolFilterConfig(top_k=1, bottom_k=1),
        ).run(rates, fx, vol)
        d = result.filtered.to_dict()
        assert d["label"] == "filtered"
        assert "metrics" in d
        assert "n_days" in d


# =============================================================================
# Cost model integration
# =============================================================================


class TestCostModel:
    def test_higher_costs_reduce_returns(self) -> None:
        currencies = ["EUR", "GBP", "AUD", "NZD", "JPY", "CHF"]
        rates = _build_rates(
            n_days=500,
            currencies=currencies,
            rate_map={
                "EUR": 0.04, "GBP": 0.045, "AUD": 0.05, "NZD": 0.055,
                "JPY": 0.001, "CHF": 0.005,
            },
        )
        fx = _build_fx_returns(rates.index, currencies, seed=33)
        vol = _build_vol_series(rates.index)
        cheap = CarryVolBacktester(
            CarryVolFilterConfig(top_k=2, bottom_k=2),
            cost_model=CostModel(spread_bps=0.1, slippage_bps=0.1),
        ).run(rates, fx, vol)
        expensive = CarryVolBacktester(
            CarryVolFilterConfig(top_k=2, bottom_k=2),
            cost_model=CostModel(spread_bps=10.0, slippage_bps=5.0),
        ).run(rates, fx, vol)
        # Expensive run nets less than cheap run on the SAME data.
        assert expensive.filtered.metrics["cagr"] < cheap.filtered.metrics["cagr"]
