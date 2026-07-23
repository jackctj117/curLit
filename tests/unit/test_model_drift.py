"""Unit tests — models.drift: DriftDetector, RetrainGate, RetrainSchedule (CL-ih7m)."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.models.drift import (
    DriftDetector,
    DriftSeverity,
    DriftThresholds,
    ModelMetrics,
    RetrainGate,
    RetrainGateThresholds,
    RetrainSchedule,
)

# =============================================================================
# DriftDetector
# =============================================================================


class TestDriftDetector:
    def setup_method(self) -> None:
        self.detector = DriftDetector()
        self.regression_baseline = ModelMetrics(r_squared=0.40, mse=0.001, n_samples=1000)
        self.classifier_baseline = ModelMetrics(accuracy=0.85, f1=0.83, n_samples=1000)

    def test_normal_when_metrics_match(self) -> None:
        # Identical current to baseline → NORMAL.
        assessment = self.detector.evaluate(
            "rate_diff",
            self.regression_baseline,
            self.regression_baseline,
        )
        assert assessment.severity == DriftSeverity.NORMAL
        assert assessment.triggers == []

    def test_normal_within_tolerance(self) -> None:
        # Tiny degradation below the degraded threshold → still NORMAL.
        current = ModelMetrics(r_squared=0.39, mse=0.0011)
        assessment = self.detector.evaluate("rate_diff", self.regression_baseline, current)
        assert assessment.severity == DriftSeverity.NORMAL

    def test_degraded_on_r_squared_drop(self) -> None:
        # R² drops 0.12 from 0.40 → just past degraded threshold (0.10).
        current = ModelMetrics(r_squared=0.28, mse=0.001)
        assessment = self.detector.evaluate("rate_diff", self.regression_baseline, current)
        assert assessment.severity == DriftSeverity.DEGRADED
        assert any("r_squared" in t for t in assessment.triggers)

    def test_alarm_on_r_squared_drop(self) -> None:
        # R² drops 0.25 → past alarm threshold (0.20).
        current = ModelMetrics(r_squared=0.15, mse=0.001)
        assessment = self.detector.evaluate("rate_diff", self.regression_baseline, current)
        assert assessment.severity == DriftSeverity.ALARM

    def test_degraded_on_mse_increase(self) -> None:
        # MSE 30% higher → past degraded (25%).
        current = ModelMetrics(r_squared=0.40, mse=0.0013)
        assessment = self.detector.evaluate("rate_diff", self.regression_baseline, current)
        assert assessment.severity == DriftSeverity.DEGRADED
        assert any("mse" in t for t in assessment.triggers)

    def test_alarm_on_mse_increase(self) -> None:
        # MSE 60% higher → past alarm (50%).
        current = ModelMetrics(r_squared=0.40, mse=0.0016)
        assessment = self.detector.evaluate("rate_diff", self.regression_baseline, current)
        assert assessment.severity == DriftSeverity.ALARM

    def test_classifier_accuracy_drop(self) -> None:
        # 6% drop → DEGRADED. 11% → ALARM.
        c1 = ModelMetrics(accuracy=0.79, f1=0.83)
        c2 = ModelMetrics(accuracy=0.74, f1=0.83)
        assert (
            self.detector.evaluate("finbert", self.classifier_baseline, c1).severity
            == DriftSeverity.DEGRADED
        )
        assert (
            self.detector.evaluate("finbert", self.classifier_baseline, c2).severity
            == DriftSeverity.ALARM
        )

    def test_classifier_f1_drop(self) -> None:
        c1 = ModelMetrics(accuracy=0.85, f1=0.77)
        c2 = ModelMetrics(accuracy=0.85, f1=0.72)
        assert (
            self.detector.evaluate("finbert", self.classifier_baseline, c1).severity
            == DriftSeverity.DEGRADED
        )
        assert (
            self.detector.evaluate("finbert", self.classifier_baseline, c2).severity
            == DriftSeverity.ALARM
        )

    def test_worst_severity_wins(self) -> None:
        # R² mild drop (DEGRADED) + MSE huge drop (ALARM) → ALARM.
        current = ModelMetrics(r_squared=0.28, mse=0.0017)
        assessment = self.detector.evaluate("rate_diff", self.regression_baseline, current)
        assert assessment.severity == DriftSeverity.ALARM

    def test_skips_metrics_when_one_side_none(self) -> None:
        # Baseline has no F1 → don't compare it.
        baseline = ModelMetrics(accuracy=0.85, f1=None)
        current = ModelMetrics(accuracy=0.84, f1=0.50)
        assessment = self.detector.evaluate("finbert", baseline, current)
        # Should NOT trigger F1 alarm since baseline F1 is None.
        assert "f1" not in " ".join(assessment.triggers)
        assert assessment.severity == DriftSeverity.NORMAL

    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(AssertionError):
            DriftThresholds(
                r_squared_drop_alarm=0.05,  # less than degraded
                r_squared_drop_degraded=0.10,
            )


# =============================================================================
# RetrainGate
# =============================================================================


class TestRetrainGate:
    def setup_method(self) -> None:
        self.gate = RetrainGate()

    def test_approve_when_candidate_better(self) -> None:
        prod = ModelMetrics(r_squared=0.30, mse=0.002)
        cand = ModelMetrics(r_squared=0.40, mse=0.001)
        decision = self.gate.evaluate(prod, cand)
        assert decision.approve is True
        assert decision.failed_metrics == []

    def test_approve_when_candidate_within_tolerance(self) -> None:
        # Candidate slightly worse but within max_r_squared_regression (0.02).
        prod = ModelMetrics(r_squared=0.40, mse=0.002)
        cand = ModelMetrics(r_squared=0.39, mse=0.0021)  # 0.01 R² drop, 5% MSE
        decision = self.gate.evaluate(prod, cand)
        assert decision.approve is True

    def test_reject_when_r_squared_drops_too_much(self) -> None:
        prod = ModelMetrics(r_squared=0.40, mse=0.002)
        cand = ModelMetrics(r_squared=0.30, mse=0.002)  # 0.10 R² drop
        decision = self.gate.evaluate(prod, cand)
        assert decision.approve is False
        assert "r_squared" in decision.failed_metrics

    def test_reject_when_mse_jumps(self) -> None:
        prod = ModelMetrics(r_squared=0.40, mse=0.002)
        cand = ModelMetrics(r_squared=0.40, mse=0.003)  # 50% MSE increase
        decision = self.gate.evaluate(prod, cand)
        assert decision.approve is False
        assert "mse" in decision.failed_metrics

    def test_reject_classifier_accuracy_regression(self) -> None:
        prod = ModelMetrics(accuracy=0.85, f1=0.83)
        cand = ModelMetrics(accuracy=0.80, f1=0.83)  # 0.05 drop > 0.01 tolerance
        decision = self.gate.evaluate(prod, cand)
        assert decision.approve is False
        assert "accuracy" in decision.failed_metrics

    def test_multiple_failures_listed(self) -> None:
        prod = ModelMetrics(r_squared=0.40, mse=0.002, accuracy=0.85, f1=0.83)
        cand = ModelMetrics(r_squared=0.20, mse=0.005, accuracy=0.70, f1=0.65)
        decision = self.gate.evaluate(prod, cand)
        assert decision.approve is False
        assert set(decision.failed_metrics) == {"r_squared", "mse", "accuracy", "f1"}

    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(AssertionError):
            RetrainGateThresholds(max_r_squared_regression=-0.01)

    def test_decision_to_dict(self) -> None:
        prod = ModelMetrics(r_squared=0.40, mse=0.002)
        cand = ModelMetrics(r_squared=0.42, mse=0.002)
        d = self.gate.evaluate(prod, cand).to_dict()
        assert d["approve"] is True
        assert "reasons" in d


# =============================================================================
# RetrainSchedule
# =============================================================================


class TestRetrainSchedule:
    def test_due_when_never_retrained(self) -> None:
        sched = RetrainSchedule(model_id="finbert", cadence_days=30)
        assert sched.is_due(datetime(2026, 4, 25)) is True

    def test_not_due_within_cadence(self) -> None:
        sched = RetrainSchedule(
            model_id="finbert",
            cadence_days=30,
            last_retrain=datetime(2026, 4, 1),
        )
        assert sched.is_due(datetime(2026, 4, 15)) is False

    def test_due_after_cadence(self) -> None:
        sched = RetrainSchedule(
            model_id="finbert",
            cadence_days=30,
            last_retrain=datetime(2026, 4, 1),
        )
        assert sched.is_due(datetime(2026, 5, 5)) is True

    def test_exactly_at_cadence_due(self) -> None:
        sched = RetrainSchedule(
            model_id="rate_diff",
            cadence_days=90,
            last_retrain=datetime(2026, 1, 1),
        )
        assert sched.is_due(datetime(2026, 4, 1)) is True

    def test_days_until_due(self) -> None:
        sched = RetrainSchedule(
            model_id="finbert",
            cadence_days=30,
            last_retrain=datetime(2026, 4, 1),
        )
        assert sched.days_until_due(datetime(2026, 4, 15)) == 16
        assert sched.days_until_due(datetime(2026, 5, 1)) == 0

    def test_invalid_cadence_rejected(self) -> None:
        with pytest.raises(AssertionError):
            RetrainSchedule(model_id="x", cadence_days=0)

    def test_empty_model_id_rejected(self) -> None:
        with pytest.raises(AssertionError):
            RetrainSchedule(model_id="", cadence_days=30)


# =============================================================================
# Reporting
# =============================================================================


class TestReporting:
    def test_assessment_to_dict(self) -> None:
        baseline = ModelMetrics(r_squared=0.40, mse=0.001)
        current = ModelMetrics(r_squared=0.20, mse=0.002)
        assessment = DriftDetector().evaluate("m", baseline, current)
        d = assessment.to_dict()
        assert d["model_id"] == "m"
        assert d["severity"] == DriftSeverity.ALARM.value
        assert isinstance(d["triggers"], list)
        assert d["baseline"]["r_squared"] == 0.40

    def test_metrics_to_dict(self) -> None:
        m = ModelMetrics(r_squared=0.5, mse=0.001, n_samples=100)
        d = m.to_dict()
        assert d["r_squared"] == 0.5
        assert d["accuracy"] is None
