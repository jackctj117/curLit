"""Tests — CL-x50g cost realism in the walk-forward backtest.

Coverage:
    - CostModel.overnight_funding_daily (annualized bps → per-bar fraction)
    - per-pair costs via get_cost_per_turn(pair) in _trades_for_signals
      (Series path via inferred pair; DataFrame path per column)
    - overnight funding charged on held positions, reflected in net_return
    - legacy_flat_costs flag reproduces the pre-CL-x50g flat behavior
    - bare duck-typed cost doubles (cost_per_turn only) still work
    - pair inference from strategy .symbols / .config.pair
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.cost_model import CostModel
from src.backtest.walkforward import WalkForwardConfig, WalkForwardRunner

_FUNDING_DAILY_15BPS = 15.0 / 10_000.0 / 252.0


def _data(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": 1.0 + 0.01 * np.arange(n)}, index=idx)


def _signals(n: int = 10) -> pd.Series:
    vals = [0, 1, 1, 1, 0, 1, 1, 0, 0, 1][:n]
    return pd.Series(vals, index=_data(n).index, dtype=float)


# =============================================================================
# CostModel
# =============================================================================


class TestCostModelFunding:
    def test_default_overnight_funding_daily(self) -> None:
        assert CostModel().overnight_funding_daily == pytest.approx(
            _FUNDING_DAILY_15BPS,
        )

    def test_zeroed_funding_is_zero(self) -> None:
        cm = CostModel(overnight_funding_annual_bps=0.0)
        assert cm.overnight_funding_daily == 0.0

    def test_per_pair_cost_per_turn(self) -> None:
        cm = CostModel()
        # Known pair: per-pair spread (0.3) + slippage (0.3) = 0.6 bps.
        assert cm.get_cost_per_turn("EURUSD") == pytest.approx(0.6e-4)
        # Unknown cross: default spread (1.5) + slippage (0.3) = 1.8 bps.
        assert cm.get_cost_per_turn("EURNOK") == pytest.approx(1.8e-4)
        # Legacy flat: spread (0.5) + slippage (0.3) = 0.8 bps.
        assert cm.cost_per_turn == pytest.approx(0.8e-4)


# =============================================================================
# _trades_for_signals — single-asset (Series) path
# =============================================================================


class TestSingleAssetCosts:
    def test_per_pair_cost_applied_when_pair_known(self) -> None:
        cm = CostModel(overnight_funding_annual_bps=0.0)
        df = WalkForwardRunner._trades_for_signals(
            signals=_signals(), data=_data(), cost_model=cm, pair="EURUSD",
        )
        expected = df["position_change"] * 0.6e-4
        pd.testing.assert_series_equal(
            df["cost"], expected, check_names=False,
        )
        assert (df["funding"] == 0.0).all()

    def test_flat_cost_when_pair_unknown(self) -> None:
        cm = CostModel(overnight_funding_annual_bps=0.0)
        df = WalkForwardRunner._trades_for_signals(
            signals=_signals(), data=_data(), cost_model=cm, pair=None,
        )
        expected = df["position_change"] * cm.cost_per_turn
        pd.testing.assert_series_equal(
            df["cost"], expected, check_names=False,
        )

    def test_overnight_funding_charged_on_held_positions(self) -> None:
        cm = CostModel()  # 15 bps/yr funding
        df = WalkForwardRunner._trades_for_signals(
            signals=_signals(), data=_data(), cost_model=cm, pair="EURUSD",
        )
        expected_funding = df["position"].abs() * _FUNDING_DAILY_15BPS
        pd.testing.assert_series_equal(
            df["funding"], expected_funding, check_names=False,
        )
        # Funding is part of cost → net_return drops by exactly funding
        # vs the zero-funding run.
        cm0 = CostModel(overnight_funding_annual_bps=0.0)
        df0 = WalkForwardRunner._trades_for_signals(
            signals=_signals(), data=_data(), cost_model=cm0, pair="EURUSD",
        )
        assert np.allclose(
            df["net_return"], df0["net_return"] - df["funding"],
        )
        # Sanity: holding costs money — total net is strictly lower.
        assert df["net_return"].sum() < df0["net_return"].sum()

    def test_legacy_flag_restores_flat_costs_no_funding(self) -> None:
        cm = CostModel()  # funding + per-pair spreads configured
        df = WalkForwardRunner._trades_for_signals(
            signals=_signals(), data=_data(), cost_model=cm, pair="EURUSD",
            legacy_flat_costs=True,
        )
        expected = df["position_change"] * cm.cost_per_turn  # flat 0.8 bps
        pd.testing.assert_series_equal(
            df["cost"], expected, check_names=False,
        )
        assert (df["funding"] == 0.0).all()

    def test_bare_cost_double_still_works(self) -> None:
        # Pre-CL-x50g test doubles expose only cost_per_turn — must not break.
        flat = SimpleNamespace(cost_per_turn=1e-4)
        df = WalkForwardRunner._trades_for_signals(
            signals=_signals(), data=_data(), cost_model=flat, pair="EURUSD",
        )
        expected = df["position_change"] * 1e-4
        pd.testing.assert_series_equal(
            df["cost"], expected, check_names=False,
        )
        assert (df["funding"] == 0.0).all()


# =============================================================================
# _trades_for_signals — multi-asset (DataFrame) path
# =============================================================================


class TestMultiAssetCosts:
    @staticmethod
    def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        data = pd.DataFrame(
            {
                "EURUSD": [1.10, 1.11, 1.12, 1.11],
                "USDJPY": [150.0, 151.0, 150.5, 151.5],
                "close": [1.10, 1.11, 1.12, 1.11],
            },
            index=idx,
        )
        signals = pd.DataFrame(
            {"EURUSD": [0.0, 1.0, 1.0, 0.0], "USDJPY": [0.0, -1.0, -1.0, 0.0]},
            index=idx,
        )
        return signals, data

    def test_per_column_pair_costs_and_funding(self) -> None:
        signals, data = self._panel()
        cm = CostModel()
        df = WalkForwardRunner._trades_for_signals(
            signals=signals, data=data, cost_model=cm,
        )
        fd = _FUNDING_DAILY_15BPS
        # Positions (lagged): both flat, flat, ±1, ±1.
        # Row 2: one turn each pair → 0.6 + 0.7 bps + funding on 2 units.
        assert df.loc[data.index[2], "cost"] == pytest.approx(
            0.6e-4 + 0.7e-4 + 2 * fd,
        )
        assert df.loc[data.index[2], "funding"] == pytest.approx(2 * fd)
        assert df.loc[data.index[3], "funding"] == pytest.approx(2 * fd)
        assert df.loc[data.index[0], "cost"] == 0.0

    def test_legacy_flag_matches_pre_change_multi_asset(self) -> None:
        signals, data = self._panel()
        cm = CostModel()
        df = WalkForwardRunner._trades_for_signals(
            signals=signals, data=data, cost_model=cm, legacy_flat_costs=True,
        )
        # Old behavior: same flat cost for every pair, no funding.
        expected = df["position_change"] * cm.cost_per_turn
        pd.testing.assert_series_equal(
            df["cost"], expected, check_names=False,
        )
        assert (df["funding"] == 0.0).all()


# =============================================================================
# Pair inference
# =============================================================================


class TestInferPair:
    def test_from_single_symbols(self) -> None:
        strat = SimpleNamespace(symbols=["EURUSD"])
        assert WalkForwardRunner._infer_pair(strat) == "EURUSD"

    def test_from_config_pair(self) -> None:
        strat = SimpleNamespace(config=SimpleNamespace(pair="USDJPY"))
        assert WalkForwardRunner._infer_pair(strat) == "USDJPY"

    def test_multi_symbol_returns_none(self) -> None:
        strat = SimpleNamespace(symbols=["EURUSD", "USDJPY"])
        assert WalkForwardRunner._infer_pair(strat) is None

    def test_bare_strategy_returns_none(self) -> None:
        assert WalkForwardRunner._infer_pair(object()) is None


# =============================================================================
# End-to-end: runner threads pair + flags through
# =============================================================================


class _AlwaysLongEURUSD:
    """Series-signal strategy exposing the traded pair via .symbols."""

    symbols = ["EURUSD"]

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def generate_signals(self, test: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=test.index)


class TestRunnerEndToEnd:
    @staticmethod
    def _panel(n: int = 220) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        close = 1.1 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n)))
        return pd.DataFrame({"close": close}, index=idx)

    @staticmethod
    def _run(cfg: WalkForwardConfig, cm: object) -> pd.DataFrame:
        runner = WalkForwardRunner(cfg)
        result = runner.run(
            TestRunnerEndToEnd._panel(), lambda: _AlwaysLongEURUSD(), cm,
        )
        assert not result.trades.empty
        return result.trades

    _CFG = {
        "is_window_days": 100, "oos_window_days": 50,
        "step_days": 50, "min_history": 100,
    }

    def test_per_pair_cost_via_inferred_symbol(self) -> None:
        trades = self._run(
            WalkForwardConfig(**self._CFG),
            CostModel(overnight_funding_annual_bps=0.0),
        )
        # EURUSD per-pair (0.6 bps), not the flat 0.8 bps.
        expected = trades["position_change"] * 0.6e-4
        pd.testing.assert_series_equal(
            trades["cost"], expected, check_names=False,
        )

    def test_legacy_flag_reproduces_old_numbers(self) -> None:
        cm = CostModel()
        trades = self._run(
            WalkForwardConfig(**self._CFG, legacy_flat_costs=True), cm,
        )
        # Exactly the pre-CL-x50g formula: flat cost_per_turn, no funding.
        expected_cost = trades["position_change"] * cm.cost_per_turn
        pd.testing.assert_series_equal(
            trades["cost"], expected_cost, check_names=False,
        )
        assert (trades["funding"] == 0.0).all()
        expected_net = trades["strategy_return"] - expected_cost
        pd.testing.assert_series_equal(
            trades["net_return"], expected_net, check_names=False,
        )

    def test_funding_lowers_net_returns_as_expected(self) -> None:
        cfg = WalkForwardConfig(**self._CFG)
        no_funding = self._run(cfg, CostModel(overnight_funding_annual_bps=0.0))
        with_funding = self._run(cfg, CostModel(overnight_funding_annual_bps=15.0))
        # Same alpha, same turn costs — the delta is exactly the funding.
        assert np.allclose(
            with_funding["net_return"],
            no_funding["net_return"] - with_funding["funding"],
        )
        held_days = int((with_funding["position"].abs() > 0).sum())
        assert held_days > 0
        assert with_funding["net_return"].sum() == pytest.approx(
            no_funding["net_return"].sum() - held_days * _FUNDING_DAILY_15BPS,
        )
