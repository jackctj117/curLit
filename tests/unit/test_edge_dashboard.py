"""Unit tests — edge_testing.dashboard (G8 / CL-4lp)."""

from __future__ import annotations

from datetime import UTC, datetime

from src.edge_testing.dashboard import (
    EdgeDashboard,
    EdgeLayerInputs,
    EdgeVerdict,
    LayerStatus,
)

T0 = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Insufficient data
# =============================================================================


class TestInsufficientData:
    def test_no_layers_reporting_returns_insufficient(self) -> None:
        verdict = EdgeDashboard().assess("strat_a", EdgeLayerInputs(), ts=T0)
        assert verdict.verdict == EdgeVerdict.INSUFFICIENT_DATA

    def test_one_layer_reporting_returns_insufficient(self) -> None:
        inputs = EdgeLayerInputs(null_hypothesis_passed=True)
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict == EdgeVerdict.INSUFFICIENT_DATA

    def test_two_layers_can_yield_verdict(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict != EdgeVerdict.INSUFFICIENT_DATA


# =============================================================================
# Decay precedence
# =============================================================================


class TestDecayOverride:
    def test_decay_overrides_strong(self) -> None:
        # Everything else is strong, but G7 says decayed.
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            live_tracker_severity="on_track",
            paper_live_avg_abs_diff_bps=1.0,
            feature_attribution_passed=True,
            regime_edge_diversified=True,
            edge_decayed=True,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict == EdgeVerdict.EDGE_DECAYED

    def test_no_decay_does_not_force_decay_verdict(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            edge_decayed=False,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict != EdgeVerdict.EDGE_DECAYED


# =============================================================================
# Foundational fails
# =============================================================================


class TestFoundationalFail:
    def test_g1_fail_returns_no_edge(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=False,
            multiple_testing_passed=True,
            live_tracker_severity="on_track",
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict == EdgeVerdict.NO_EDGE_DETECTED

    def test_g2_fail_returns_no_edge(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=False,
            live_tracker_severity="on_track",
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict == EdgeVerdict.NO_EDGE_DETECTED


# =============================================================================
# Strong / weak partition
# =============================================================================


class TestStrongWeakPartition:
    def test_all_pass_returns_strong(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            live_tracker_severity="on_track",
            paper_live_avg_abs_diff_bps=1.0,  # below default 5 bp threshold
            feature_attribution_passed=True,
            regime_edge_diversified=True,
            edge_decayed=False,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict == EdgeVerdict.STRONG_EDGE
        assert verdict.n_passing == 7

    def test_one_warn_returns_strong_when_threshold_met(self) -> None:
        # 6 PASS + 1 WARN = 6/7 PASS = 0.857 ≥ STRONG threshold (0.85).
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            live_tracker_severity="underperforming",  # WARN
            paper_live_avg_abs_diff_bps=1.0,
            feature_attribution_passed=True,
            regime_edge_diversified=True,
            edge_decayed=False,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        # Pass-fraction is 6/7=0.857 ≥ 0.85 strong threshold.
        assert verdict.verdict == EdgeVerdict.STRONG_EDGE

    def test_one_fail_returns_weak(self) -> None:
        # G1+G2 pass (so foundational gate doesn't fire). Optional layer fails.
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            live_tracker_severity="on_track",
            paper_live_avg_abs_diff_bps=1.0,
            feature_attribution_passed=False,  # fail
            regime_edge_diversified=True,
            edge_decayed=False,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict == EdgeVerdict.WEAK_EDGE
        assert verdict.n_failing == 1

    def test_partial_layers_can_be_strong(self) -> None:
        # Only G1+G2 reporting; both pass → STRONG (no failures).
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        assert verdict.verdict == EdgeVerdict.STRONG_EDGE


# =============================================================================
# Per-layer signal mapping
# =============================================================================


class TestLayerSignals:
    def test_all_signals_emitted_even_when_not_reported(self) -> None:
        verdict = EdgeDashboard().assess("strat_a", EdgeLayerInputs(), ts=T0)
        names = {s.name for s in verdict.signals}
        # All 7 layers tracked.
        assert names == {
            "G1_null_hypothesis",
            "G2_multiple_testing",
            "G3_live_tracker",
            "G4_paper_live",
            "G5_feature_attribution",
            "G6_regime_decomposition",
            "G7_decay_monitor",
        }

    def test_g3_severely_maps_to_fail(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            live_tracker_severity="severely_underperforming",
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        g3 = next(s for s in verdict.signals if s.name == "G3_live_tracker")
        assert g3.status == LayerStatus.FAIL

    def test_g3_significantly_maps_to_fail(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            live_tracker_severity="significantly_underperforming",
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        g3 = next(s for s in verdict.signals if s.name == "G3_live_tracker")
        assert g3.status == LayerStatus.FAIL

    def test_g4_thresholds(self) -> None:
        # PASS at avg diff <= threshold.
        inputs_pass = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            paper_live_avg_abs_diff_bps=4.0,
            paper_live_threshold_bps=5.0,
        )
        v = EdgeDashboard().assess("strat_a", inputs_pass, ts=T0)
        g4 = next(s for s in v.signals if s.name == "G4_paper_live")
        assert g4.status == LayerStatus.PASS

        # WARN at threshold < diff <= 2× threshold.
        inputs_warn = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            paper_live_avg_abs_diff_bps=8.0,
            paper_live_threshold_bps=5.0,
        )
        v = EdgeDashboard().assess("strat_a", inputs_warn, ts=T0)
        g4 = next(s for s in v.signals if s.name == "G4_paper_live")
        assert g4.status == LayerStatus.WARN

        # FAIL at diff > 2× threshold.
        inputs_fail = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            paper_live_avg_abs_diff_bps=15.0,
            paper_live_threshold_bps=5.0,
        )
        v = EdgeDashboard().assess("strat_a", inputs_fail, ts=T0)
        g4 = next(s for s in v.signals if s.name == "G4_paper_live")
        assert g4.status == LayerStatus.FAIL


# =============================================================================
# StrategyVerdict properties
# =============================================================================


class TestStrategyVerdictProperties:
    def test_n_passing_failing_reporting_counts(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
            feature_attribution_passed=False,
            regime_edge_diversified=True,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        # 3 PASS + 1 FAIL + 3 NOT_REPORTED.
        assert verdict.n_passing == 3
        assert verdict.n_failing == 1
        assert verdict.n_reporting == 4

    def test_to_dict_contains_required_fields(self) -> None:
        inputs = EdgeLayerInputs(
            null_hypothesis_passed=True,
            multiple_testing_passed=True,
        )
        verdict = EdgeDashboard().assess("strat_a", inputs, ts=T0)
        d = verdict.to_dict()
        for key in ("strategy_id", "verdict", "ts", "signals", "summary"):
            assert key in d
        assert isinstance(d["signals"], list)
        assert len(d["signals"]) == 7
