"""Edge test runner — orchestrates G1→G9 (reference/14_edge_testing.md).

Without this orchestrator, the Phase G modules (NullHypothesisFramework,
MultipleTestingCorrection, LiveEdgeTracker, PaperLiveDivergence,
EdgeDashboard, EdgePolicy) are inert library code — they have no entrypoint
that runs them on a schedule, composes their outputs, and acts on the result.

This module defines:

    StrategyEdgeInputs    raw inputs for one strategy: backtest returns,
                          live returns, market data, paper/live fills,
                          backtest metrics + paper days, etc. The runner
                          tolerates None for any field — the corresponding
                          test is then SKIPPED with a note.

    StrategyEdgeResult    structured output: per-layer results + the G8
                          dashboard verdict + the G9 policy action.

    EdgeRunner            orchestrator. Run on a single strategy via
                          run_strategy(); over a list of strategies via
                          run_all().

The CLI entrypoint (scripts/run_edge_tests.py) wires this to the live
StrategyStateStore for production weekly cron use. Tests exercise the
runner directly with synthetic data so the orchestration logic is
covered independently of DB state.

G5 (FeatureEdgeAttributor), G6 (RegimeEdgeAnalyzer), G7 (EdgeDecayMonitor)
are not yet implemented (CL-m0h, CL-lx4, CL-675 still open) — the runner
records them as not_evaluated. The G8 dashboard handles the partial input
gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.edge_testing.dashboard import (
    EdgeDashboard,
    EdgeLayerInputs,
    StrategyVerdict,
)
from src.edge_testing.edge_policy import (
    EdgePolicy,
    LiveAction,
    LiveSnapshot,
    load_edge_policy,
)
from src.edge_testing.live_tracker import (
    BacktestExpectations,
    LiveEdgeTracker,
)
from src.edge_testing.multiple_testing import MultipleTestingCorrection
from src.edge_testing.null_hypothesis import (
    NullHypothesisFramework,
)
from src.edge_testing.paper_live_divergence import PaperLiveDivergence
from src.execution.broker import Fill

logger = logging.getLogger(__name__)


# Default n_simulations for the per-strategy null-hypothesis run from the
# weekly cron — lower than the unit-test default to keep runtime reasonable
# when iterating across many strategies.
_RUNNER_NULL_N_SIMULATIONS: int = 5_000

# Minimum bars of live history before the LiveEdgeTracker is invoked.
# Below this we skip with a note rather than emitting a noisy verdict.
_MIN_LIVE_BARS_FOR_TRACKER: int = 30

# Minimum backtest history before G1 is invoked.
_MIN_BACKTEST_BARS_FOR_G1: int = 100

# Default G2 alpha (matches MultipleTestingCorrection default).
_G2_ALPHA: float = 0.05


# =============================================================================
# Inputs
# =============================================================================


@dataclass
class StrategyEdgeInputs:
    """Inputs for one strategy's edge test pass.

    All fields optional except strategy_id — the runner tolerates missing
    inputs and SKIPS the affected layer. Wire from production via the
    runner script's data adapter.
    """

    strategy_id: str

    # G1 / G2 inputs.
    backtest_returns: pd.Series | None = None
    asset_returns: pd.Series | None = None
    signal: pd.Series | None = None
    basket_returns: pd.DataFrame | None = None
    yields: pd.DataFrame | None = None

    # G3 inputs.
    live_returns: pd.Series | None = None
    backtest_expectations: BacktestExpectations | None = None

    # G4 inputs.
    paper_fills: list[Fill] | None = None
    live_fills: list[Fill] | None = None

    # Decay (G7) input — fitted decay tau, when available. Until G7 lands
    # this stays None and the policy gate skips the decay precedence.
    decay_tau: float | None = None

    # Live-mean-return for the policy's negative-mean override.
    live_mean_return: float | None = None


# =============================================================================
# Result
# =============================================================================


@dataclass
class StrategyEdgeResult:
    """One strategy's full pass output."""

    strategy_id: str
    ts: datetime
    g1_edge_exists: bool | None = None
    g1_strategy_sharpe: float | None = None
    g1_evaluated_p_values: dict[str, float] = field(default_factory=dict)
    g2_passed: bool | None = None
    g2_failed_metrics: list[str] = field(default_factory=list)
    g3_severity: str | None = None
    g3_sharpe_z: float | None = None
    g4_avg_abs_diff_bps: float | None = None
    g4_n_matched: int | None = None
    g8_verdict: StrategyVerdict | None = None
    g9_action: LiveAction | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "ts": self.ts.isoformat(),
            "g1": {
                "edge_exists": self.g1_edge_exists,
                "strategy_sharpe": self.g1_strategy_sharpe,
                "evaluated_p_values": dict(self.g1_evaluated_p_values),
            },
            "g2": {
                "passed": self.g2_passed,
                "failed_metrics": list(self.g2_failed_metrics),
            },
            "g3": {
                "severity": self.g3_severity,
                "sharpe_z": self.g3_sharpe_z,
            },
            "g4": {
                "avg_abs_diff_bps": self.g4_avg_abs_diff_bps,
                "n_matched": self.g4_n_matched,
            },
            "g8_verdict": (self.g8_verdict.to_dict() if self.g8_verdict else None),
            "g9_action": self.g9_action.value if self.g9_action else None,
            "notes": list(self.notes),
        }


@dataclass
class EdgeRunReport:
    """Report covering all strategies in one run."""

    ts: datetime
    results: dict[str, StrategyEdgeResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "n_strategies": len(self.results),
            "results": {sid: r.to_dict() for sid, r in self.results.items()},
        }


# =============================================================================
# Runner
# =============================================================================


class EdgeRunner:
    """Run G1→G9 on one or many strategies and produce a structured report."""

    def __init__(
        self,
        policy: EdgePolicy | None = None,
        null_n_simulations: int = _RUNNER_NULL_N_SIMULATIONS,
        seed: int | None = None,
    ) -> None:
        self.policy = policy or load_edge_policy()
        self.null_framework = NullHypothesisFramework(
            n_simulations=null_n_simulations,
            seed=seed,
        )
        self.multiple_testing = MultipleTestingCorrection(alpha=_G2_ALPHA)
        self.divergence = PaperLiveDivergence()
        self.dashboard = EdgeDashboard()

    def run_strategy(self, inputs: StrategyEdgeInputs) -> StrategyEdgeResult:
        """Run all available layers for one strategy."""
        ts = datetime.now(UTC)
        result = StrategyEdgeResult(strategy_id=inputs.strategy_id, ts=ts)

        layer_inputs = EdgeLayerInputs()

        # ----- G1: null hypothesis framework -----
        if (
            inputs.backtest_returns is not None
            and inputs.asset_returns is not None
            and len(inputs.backtest_returns) >= _MIN_BACKTEST_BARS_FOR_G1
        ):
            try:
                g1_report = self.null_framework.test_strategy(
                    strategy_returns=inputs.backtest_returns,
                    asset_returns=inputs.asset_returns,
                    signal=inputs.signal,
                    basket_returns=inputs.basket_returns,
                    yields=inputs.yields,
                )
                result.g1_edge_exists = g1_report.edge_exists
                result.g1_strategy_sharpe = g1_report.strategy_sharpe
                result.g1_evaluated_p_values = {
                    name: r.p_value
                    for name, r in g1_report.results.items()
                    if r.evaluated and r.p_value is not None
                }
                layer_inputs.null_hypothesis_passed = g1_report.edge_exists
                layer_inputs.null_hypothesis_detail = (
                    f"Sharpe={g1_report.strategy_sharpe:.2f}, "
                    f"{len(g1_report.passed_nulls())}/"
                    f"{len(g1_report.passed_nulls()) + len(g1_report.failed_nulls())} nulls passed"
                )
            except Exception as exc:
                result.notes.append(f"g1 failed: {exc}")
                logger.exception("G1 null hypothesis failed for %s", inputs.strategy_id)
        else:
            result.notes.append(
                "g1 skipped: insufficient backtest_returns or asset_returns",
            )

        # ----- G2: multiple-testing correction -----
        # Pulled from the G1 evaluated p-values when available.
        if result.g1_evaluated_p_values:
            try:
                g2_report = self.multiple_testing.correct(
                    p_values=list(result.g1_evaluated_p_values.values()),
                )
                # Pass = NOT all rejected by Bonferroni's strict gate.
                # We're lenient: any single Bonferroni-passing test means edge.
                g2_passed = any(g2_report.bonferroni_reject)
                result.g2_passed = g2_passed
                result.g2_failed_metrics = [] if g2_passed else ["bonferroni_all_rejected"]
                layer_inputs.multiple_testing_passed = g2_passed
                layer_inputs.multiple_testing_detail = (
                    f"{int(sum(g2_report.bonferroni_reject))} of "
                    f"{len(g2_report.bonferroni_reject)} survive Bonferroni"
                )
            except Exception as exc:
                result.notes.append(f"g2 failed: {exc}")
                logger.exception("G2 multiple testing failed for %s", inputs.strategy_id)
        else:
            result.notes.append("g2 skipped: no G1 p-values to correct")

        # ----- G3: live edge tracker -----
        if (
            inputs.live_returns is not None
            and len(inputs.live_returns) >= _MIN_LIVE_BARS_FOR_TRACKER
            and inputs.backtest_expectations is not None
        ):
            try:
                tracker = LiveEdgeTracker(
                    strategy_id=inputs.strategy_id,
                    backtest=inputs.backtest_expectations,
                )
                g3_assessment = tracker.assess(inputs.live_returns)
                result.g3_severity = g3_assessment.severity.value
                result.g3_sharpe_z = g3_assessment.sharpe_z_score
                layer_inputs.live_tracker_severity = g3_assessment.severity.value
                layer_inputs.live_tracker_detail = f"sharpe_z={g3_assessment.sharpe_z_score:.2f}"
            except Exception as exc:
                result.notes.append(f"g3 failed: {exc}")
                logger.exception("G3 live tracker failed for %s", inputs.strategy_id)
        else:
            result.notes.append(
                "g3 skipped: insufficient live_returns or no backtest_expectations",
            )

        # ----- G4: paper-live divergence -----
        if inputs.paper_fills is not None and inputs.live_fills is not None:
            try:
                g4_report = self.divergence.compare(
                    inputs.paper_fills,
                    inputs.live_fills,
                )
                result.g4_avg_abs_diff_bps = g4_report.avg_abs_price_diff_bps
                result.g4_n_matched = g4_report.n_matched
                layer_inputs.paper_live_avg_abs_diff_bps = g4_report.avg_abs_price_diff_bps
                layer_inputs.paper_live_detail = (
                    f"{g4_report.n_matched} matched, {g4_report.n_flagged} flagged"
                )
            except Exception as exc:
                result.notes.append(f"g4 failed: {exc}")
                logger.exception("G4 divergence failed for %s", inputs.strategy_id)
        else:
            result.notes.append("g4 skipped: paper_fills or live_fills missing")

        # ----- G5/G6/G7: stubs (modules not yet built) -----
        result.notes.append("g5/g6/g7 not_evaluated: modules not yet implemented")
        if inputs.decay_tau is not None:
            layer_inputs.edge_decayed = inputs.decay_tau <= self.policy.live.decay_tau_retire
            layer_inputs.edge_decay_detail = f"decay_tau={inputs.decay_tau:.2f}"

        # ----- G8: compose verdict -----
        result.g8_verdict = self.dashboard.assess(
            inputs.strategy_id,
            layer_inputs,
            ts=ts,
        )

        # ----- G9: pre-committed lifecycle action -----
        if result.g3_sharpe_z is not None:
            snap = LiveSnapshot(
                sharpe_z=result.g3_sharpe_z,
                live_mean_return=(
                    inputs.live_mean_return if inputs.live_mean_return is not None else 0.0
                ),
                decay_tau=inputs.decay_tau,
            )
            result.g9_action = self.policy.evaluate_live(snap)
        else:
            result.notes.append("g9 skipped: no G3 sharpe_z to evaluate")

        return result

    def run_all(
        self,
        all_inputs: list[StrategyEdgeInputs],
    ) -> EdgeRunReport:
        """Run G1-G9 across many strategies. Independent failures are isolated."""
        report = EdgeRunReport(ts=datetime.now(UTC))
        for inputs in all_inputs:
            try:
                report.results[inputs.strategy_id] = self.run_strategy(inputs)
            except Exception:
                logger.exception(
                    "EdgeRunner crashed on %s — continuing",
                    inputs.strategy_id,
                )
        return report


__all__ = [
    "EdgeRunReport",
    "EdgeRunner",
    "StrategyEdgeInputs",
    "StrategyEdgeResult",
]
