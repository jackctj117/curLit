"""Unit tests — edge_testing.edge_policy (G9 / CL-337)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest

from src.edge_testing.edge_policy import (
    DEFAULT_POLICY_PATH,
    LiveAction,
    LiveSnapshot,
    PromotionVerdict,
    StrategyMetrics,
    load_edge_policy,
)

# =============================================================================
# YAML loading
# =============================================================================


class TestLoading:
    def test_default_yaml_parses(self) -> None:
        policy = load_edge_policy()
        assert policy.paper.min_days_before_live == 90
        assert policy.paper.null_p_max == 0.10
        assert policy.paper.reality_check_p_max == 0.15
        assert policy.paper.edge_concentration_max == 0.60
        assert policy.live.sharpe_z_halt == -3.0
        assert policy.retirement.cooldown_days == 180
        assert policy.retirement.requires_new_research is True

    def test_alert_severity_loaded(self) -> None:
        policy = load_edge_policy()
        assert policy.alert_severity_for(LiveAction.HALT_STRATEGY) == "critical"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_edge_policy(tmp_path / "missing.yaml")

    def test_malformed_missing_keys_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("paper_trading: {}\n")
        with pytest.raises(ValueError, match="missing"):
            load_edge_policy(bad)

    def test_malformed_wrong_type_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(dedent("""
            paper_trading:
              min_days_before_live: "ninety"
              required_tests:
                null_p_max: 0.10
                reality_check_p_max: 0.15
                edge_concentration_max: 0.60
            live_trading:
              degradation_triggers:
                sharpe_z_review: -1.5
                sharpe_z_reduce_size_50pct: -2.0
                sharpe_z_halt: -3.0
                decay_tau_retire: -0.7
                live_mean_return_floor: 0.0
            retirement:
              cooldown_days: 180
              requires_new_research: true
        """))
        with pytest.raises(ValueError, match="wrong type"):
            load_edge_policy(bad)

    def test_default_path_exists(self) -> None:
        # Sanity: the file we ship is at the documented path.
        assert DEFAULT_POLICY_PATH.exists(), (
            f"shipped policy missing at {DEFAULT_POLICY_PATH}"
        )


# =============================================================================
# Promotion gate
# =============================================================================


class TestPromotion:
    def setup_method(self) -> None:
        self.policy = load_edge_policy()

    def test_all_gates_pass_approves(self) -> None:
        metrics = StrategyMetrics(
            paper_days=120, null_max_p=0.05, reality_check_p=0.10,
            edge_concentration=0.40,
        )
        decision = self.policy.evaluate_promotion(metrics)
        assert decision.verdict == PromotionVerdict.APPROVED
        assert decision.failed_gates == []

    def test_paper_days_below_minimum_rejects(self) -> None:
        metrics = StrategyMetrics(
            paper_days=30, null_max_p=0.05, reality_check_p=0.10,
            edge_concentration=0.40,
        )
        decision = self.policy.evaluate_promotion(metrics)
        assert decision.verdict == PromotionVerdict.REJECTED
        assert "min_days_before_live" in decision.failed_gates

    def test_null_p_too_high_rejects(self) -> None:
        metrics = StrategyMetrics(
            paper_days=120, null_max_p=0.20, reality_check_p=0.10,
            edge_concentration=0.40,
        )
        decision = self.policy.evaluate_promotion(metrics)
        assert decision.verdict == PromotionVerdict.REJECTED
        assert "null_p_max" in decision.failed_gates

    def test_reality_check_p_too_high_rejects(self) -> None:
        metrics = StrategyMetrics(
            paper_days=120, null_max_p=0.05, reality_check_p=0.30,
            edge_concentration=0.40,
        )
        decision = self.policy.evaluate_promotion(metrics)
        assert decision.verdict == PromotionVerdict.REJECTED
        assert "reality_check_p_max" in decision.failed_gates

    def test_edge_concentration_too_high_rejects(self) -> None:
        metrics = StrategyMetrics(
            paper_days=120, null_max_p=0.05, reality_check_p=0.10,
            edge_concentration=0.85,
        )
        decision = self.policy.evaluate_promotion(metrics)
        assert decision.verdict == PromotionVerdict.REJECTED
        assert "edge_concentration_max" in decision.failed_gates

    def test_multiple_failures_listed(self) -> None:
        metrics = StrategyMetrics(
            paper_days=10, null_max_p=0.50, reality_check_p=0.50,
            edge_concentration=0.95,
        )
        decision = self.policy.evaluate_promotion(metrics)
        assert decision.verdict == PromotionVerdict.REJECTED
        assert set(decision.failed_gates) == {
            "min_days_before_live",
            "null_p_max",
            "reality_check_p_max",
            "edge_concentration_max",
        }

    def test_invalid_metrics_rejected_at_construction(self) -> None:
        with pytest.raises(AssertionError):
            StrategyMetrics(
                paper_days=120, null_max_p=1.5, reality_check_p=0.10,
                edge_concentration=0.40,
            )


# =============================================================================
# Live degradation triggers
# =============================================================================


class TestLiveDegradation:
    def setup_method(self) -> None:
        self.policy = load_edge_policy()

    def test_continue_when_no_trigger_fires(self) -> None:
        snap = LiveSnapshot(sharpe_z=0.0, live_mean_return=0.001)
        assert self.policy.evaluate_live(snap) == LiveAction.CONTINUE

    def test_review_at_mild_z_threshold(self) -> None:
        snap = LiveSnapshot(sharpe_z=-1.5, live_mean_return=0.001)
        assert self.policy.evaluate_live(snap) == LiveAction.REVIEW

    def test_reduce_size_at_z_minus_2(self) -> None:
        snap = LiveSnapshot(sharpe_z=-2.0, live_mean_return=0.001)
        assert self.policy.evaluate_live(snap) == LiveAction.REDUCE_SIZE_50PCT

    def test_halt_at_z_minus_3(self) -> None:
        snap = LiveSnapshot(sharpe_z=-3.0, live_mean_return=0.001)
        assert self.policy.evaluate_live(snap) == LiveAction.HALT_STRATEGY

    def test_retire_at_decay_threshold(self) -> None:
        snap = LiveSnapshot(sharpe_z=-1.0, live_mean_return=0.001, decay_tau=-0.8)
        assert self.policy.evaluate_live(snap) == LiveAction.RETIRE_STRATEGY

    def test_decay_overrides_other_triggers(self) -> None:
        # Even with a halt-worthy z, decay-retire wins.
        snap = LiveSnapshot(sharpe_z=-5.0, live_mean_return=-0.005, decay_tau=-0.9)
        assert self.policy.evaluate_live(snap) == LiveAction.RETIRE_STRATEGY

    def test_negative_mean_return_forces_size_down(self) -> None:
        # Mild z, but negative mean return → REDUCE_SIZE_50PCT (the floor trigger).
        snap = LiveSnapshot(sharpe_z=-0.5, live_mean_return=-0.001)
        assert self.policy.evaluate_live(snap) == LiveAction.REDUCE_SIZE_50PCT

    def test_negative_mean_with_z_at_reduce_threshold_halts(self) -> None:
        snap = LiveSnapshot(sharpe_z=-2.0, live_mean_return=-0.001)
        assert self.policy.evaluate_live(snap) == LiveAction.HALT_STRATEGY


# =============================================================================
# Retirement cooldown
# =============================================================================


class TestRetirement:
    def setup_method(self) -> None:
        self.policy = load_edge_policy()

    def test_within_cooldown_not_eligible(self) -> None:
        retired = datetime(2026, 1, 1)
        today = datetime(2026, 4, 1)  # 90 days later, < 180
        assert self.policy.is_retirement_eligible(retired, today) is False

    def test_after_cooldown_eligible(self) -> None:
        retired = datetime(2026, 1, 1)
        today = retired + timedelta(days=181)
        assert self.policy.is_retirement_eligible(retired, today) is True

    def test_exactly_at_cooldown_eligible(self) -> None:
        retired = datetime(2026, 1, 1)
        today = retired + timedelta(days=180)
        assert self.policy.is_retirement_eligible(retired, today) is True


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_promotion_decision_to_dict(self) -> None:
        policy = load_edge_policy()
        metrics = StrategyMetrics(
            paper_days=120, null_max_p=0.05, reality_check_p=0.10,
            edge_concentration=0.40,
        )
        d = policy.evaluate_promotion(metrics).to_dict()
        assert d["verdict"] == "approved"
        assert isinstance(d["reasons"], list)
        assert d["failed_gates"] == []
