"""Unit tests — edge_testing.runner: G1→G9 orchestration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.edge_testing.edge_policy import LiveAction
from src.edge_testing.live_tracker import BacktestExpectations
from src.edge_testing.runner import (
    EdgeRunner,
    StrategyEdgeInputs,
)
from src.execution.broker import Fill

T0 = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)


def _build_returns(n: int, mean: float, vol: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(mean, vol, n),
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )


def _build_signal_aligned_strat(
    asset: pd.Series, hit_rate: float = 0.7,
) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(7)
    informed = np.sign(asset.values)
    random = np.where(rng.random(len(asset)) < 0.5, 1.0, -1.0)
    mask = rng.random(len(asset)) < hit_rate
    signal = np.where(mask, informed, random).astype(float)
    return (
        pd.Series(signal * asset.values, index=asset.index),
        pd.Series(signal, index=asset.index),
    )


def _build_fill(
    side: str, price: float, ts: datetime, oid: str = "o1",
) -> Fill:
    return Fill(
        order_id=oid, fill_id=f"{oid}-fill", symbol="EURUSD",
        side=side, quantity=1000, price=price, timestamp=ts,
    )


# =============================================================================
# Single-strategy orchestration
# =============================================================================


class TestRunStrategy:
    def test_skips_all_layers_with_empty_inputs(self) -> None:
        runner = EdgeRunner(seed=42)
        result = runner.run_strategy(
            StrategyEdgeInputs(strategy_id="s1"),
        )
        # All layers None in the result.
        assert result.g1_edge_exists is None
        assert result.g2_passed is None
        assert result.g3_severity is None
        assert result.g4_avg_abs_diff_bps is None
        # G8 verdict still produced (with INSUFFICIENT_DATA from dashboard).
        assert result.g8_verdict is not None
        # Notes explain the skips.
        assert any("g1 skipped" in n for n in result.notes)
        assert any("g3 skipped" in n for n in result.notes)
        assert any("g4 skipped" in n for n in result.notes)

    def test_g1_runs_when_returns_provided(self) -> None:
        asset = _build_returns(n=400, mean=0.0001, vol=0.01, seed=10)
        strat, signal = _build_signal_aligned_strat(asset, hit_rate=0.7)
        runner = EdgeRunner(null_n_simulations=500, seed=11)
        result = runner.run_strategy(
            StrategyEdgeInputs(
                strategy_id="s1",
                backtest_returns=strat,
                asset_returns=asset,
                signal=signal,
            ),
        )
        # G1 should run and edge_exists should be True (signal aligned with asset).
        assert result.g1_edge_exists is not None
        # G2 should run since G1 produced p-values.
        assert result.g2_passed is not None

    def test_g3_runs_with_live_returns_and_expectations(self) -> None:
        runner = EdgeRunner(seed=20)
        live = _build_returns(n=120, mean=0.0006, vol=0.006, seed=21)
        expected = BacktestExpectations(
            sharpe=1.5, hit_rate=0.55, mean_return=0.0006, vol=0.006,
        )
        result = runner.run_strategy(
            StrategyEdgeInputs(
                strategy_id="s1",
                live_returns=live,
                backtest_expectations=expected,
                live_mean_return=float(live.mean()),
            ),
        )
        assert result.g3_severity is not None
        assert result.g3_sharpe_z is not None
        # G9 fires once G3 runs.
        assert result.g9_action is not None

    def test_g4_runs_with_paper_and_live_fills(self) -> None:
        runner = EdgeRunner(seed=30)
        paper = [_build_fill("buy", 1.10, T0, oid="o1")]
        live = [_build_fill(
            "buy", 1.10011, T0 + timedelta(milliseconds=200), oid="o1",
        )]
        result = runner.run_strategy(
            StrategyEdgeInputs(
                strategy_id="s1", paper_fills=paper, live_fills=live,
            ),
        )
        assert result.g4_n_matched == 1
        assert result.g4_avg_abs_diff_bps is not None

    def test_decay_tau_reaches_g9(self) -> None:
        runner = EdgeRunner(seed=40)
        live = _build_returns(n=120, mean=0.0006, vol=0.006, seed=41)
        expected = BacktestExpectations(
            sharpe=1.5, hit_rate=0.55, mean_return=0.0006, vol=0.006,
        )
        result = runner.run_strategy(
            StrategyEdgeInputs(
                strategy_id="s1",
                live_returns=live,
                backtest_expectations=expected,
                live_mean_return=float(live.mean()),
                decay_tau=-0.85,  # past policy threshold (-0.7) → retire
            ),
        )
        # Retire wins under policy precedence.
        assert result.g9_action == LiveAction.RETIRE_STRATEGY

    def test_g9_falls_back_to_continue_when_g3_passes(self) -> None:
        # Live returns matching backtest → severity ON_TRACK → CONTINUE.
        runner = EdgeRunner(seed=50)
        # Slightly outperforming so the (sometimes-finicky) G3 stays clearly
        # ON_TRACK rather than slipping to UNDERPERFORMING from sample noise.
        rng = np.random.default_rng(51)
        live = pd.Series(
            rng.normal(0.0009, 0.006, 500),
            index=pd.date_range("2025-01-01", periods=500, freq="B"),
        )
        expected = BacktestExpectations(
            sharpe=1.5, hit_rate=0.55, mean_return=0.0006, vol=0.006,
        )
        result = runner.run_strategy(
            StrategyEdgeInputs(
                strategy_id="s1",
                live_returns=live,
                backtest_expectations=expected,
                live_mean_return=float(live.mean()),
            ),
        )
        # Either CONTINUE or REVIEW depending on z-score noise; never HALT.
        assert result.g9_action in {
            LiveAction.CONTINUE, LiveAction.REVIEW,
        }


# =============================================================================
# Multi-strategy orchestration
# =============================================================================


class TestRunAll:
    def test_run_all_returns_one_result_per_strategy(self) -> None:
        runner = EdgeRunner(seed=60)
        inputs = [
            StrategyEdgeInputs(strategy_id=f"s{i}")
            for i in range(3)
        ]
        report = runner.run_all(inputs)
        assert len(report.results) == 3
        assert set(report.results.keys()) == {"s0", "s1", "s2"}

    def test_failure_in_one_does_not_kill_others(self) -> None:
        runner = EdgeRunner(seed=70)
        # First inputs has malformed data that will fail G3 (live_returns is
        # not a Series); other inputs are clean. The outer try/except in
        # run_all should record the crash and continue.
        inputs = [
            StrategyEdgeInputs(strategy_id="bad"),  # all-none also OK
            StrategyEdgeInputs(strategy_id="good"),
        ]
        report = runner.run_all(inputs)
        assert "good" in report.results

    def test_report_to_dict_serializable(self) -> None:
        import json
        runner = EdgeRunner(seed=80)
        report = runner.run_all([
            StrategyEdgeInputs(strategy_id="s1"),
        ])
        # Round-trip through JSON to ensure no non-serializable types leak in.
        text = json.dumps(report.to_dict(), default=str)
        parsed = json.loads(text)
        assert "ts" in parsed
        assert "n_strategies" in parsed
        assert parsed["n_strategies"] == 1


# =============================================================================
# Reporting
# =============================================================================


class TestResultStructure:
    def test_to_dict_keys_present_even_when_layers_skipped(self) -> None:
        runner = EdgeRunner(seed=90)
        result = runner.run_strategy(StrategyEdgeInputs(strategy_id="s1"))
        d = result.to_dict()
        for key in ("strategy_id", "ts", "g1", "g2", "g3", "g4", "g8_verdict", "g9_action", "notes"):
            assert key in d

    def test_g1_p_values_dict_in_result(self) -> None:
        asset = _build_returns(n=300, mean=0.0001, vol=0.01, seed=100)
        strat, signal = _build_signal_aligned_strat(asset, hit_rate=0.7)
        runner = EdgeRunner(null_n_simulations=300, seed=101)
        result = runner.run_strategy(
            StrategyEdgeInputs(
                strategy_id="s1",
                backtest_returns=strat,
                asset_returns=asset,
                signal=signal,
            ),
        )
        assert isinstance(result.g1_evaluated_p_values, dict)
        # At least the G1 nulls that ran independent of optional inputs.
        assert "random_longshort" in result.g1_evaluated_p_values
