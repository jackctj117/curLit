"""Model drift detection + retraining gate (CL-ih7m).

Two distinct decisions, kept structurally separate so they can be wired
independently:

    DriftDetector — compares CURRENT (live) model metrics against a stored
                    baseline. Returns a DriftSeverity that downstream
                    monitoring uses to alert on degradation regardless of the
                    scheduled retrain cadence.

    RetrainGate   — compares a CANDIDATE (newly-fitted) model's hold-out
                    metrics against the running PRODUCTION model. Must clear
                    a no-worse-than threshold to promote, otherwise the
                    deploy is rejected and alerted.

    RetrainSchedule — declarative per-model cadence (e.g. monthly FinBERT,
                      quarterly rate-diff). is_due() answers "should the
                      scheduled retrain Airflow DAG run today?"

The Airflow DAG glue is operational and can be added later — the logic in
this module is dependency-free and unit-testable. The Prometheus gauge
fx_model_drift_severity surfaces DriftDetector output to dashboards.

Reference: reference/02_features_and_models.md, reference/14_edge_testing.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# Default tolerances calibrated to "noticeable" but not "single-bad-sample"
# drift. Override per model when ranges differ (e.g. tight R² regimes).
_DEFAULT_R_SQUARED_DROP_DEGRADED: float = 0.10
_DEFAULT_R_SQUARED_DROP_ALARM: float = 0.20
_DEFAULT_MSE_PCT_INCREASE_DEGRADED: float = 0.25
_DEFAULT_MSE_PCT_INCREASE_ALARM: float = 0.50
_DEFAULT_ACCURACY_DROP_DEGRADED: float = 0.05
_DEFAULT_ACCURACY_DROP_ALARM: float = 0.10
_DEFAULT_F1_DROP_DEGRADED: float = 0.05
_DEFAULT_F1_DROP_ALARM: float = 0.10


class DriftSeverity(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    ALARM = "alarm"


_SEVERITY_RANK: dict[DriftSeverity, int] = {
    DriftSeverity.NORMAL: 0,
    DriftSeverity.DEGRADED: 1,
    DriftSeverity.ALARM: 2,
}


def _escalate(current: DriftSeverity, candidate: DriftSeverity) -> DriftSeverity:
    """Return the more-severe of two DriftSeverity values."""
    return candidate if _SEVERITY_RANK[candidate] > _SEVERITY_RANK[current] else current


@dataclass
class ModelMetrics:
    """Subset of model performance metrics used by drift + gate logic.

    A model only populates the fields it produces — regression models set
    r_squared + mse; classifiers set accuracy + f1. None means "not
    applicable" and that metric isn't part of the comparison.
    """

    r_squared: float | None = None
    mse: float | None = None
    accuracy: float | None = None
    f1: float | None = None
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_squared": self.r_squared,
            "mse": self.mse,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "n_samples": self.n_samples,
        }


@dataclass
class DriftThresholds:
    r_squared_drop_degraded: float = _DEFAULT_R_SQUARED_DROP_DEGRADED
    r_squared_drop_alarm: float = _DEFAULT_R_SQUARED_DROP_ALARM
    mse_pct_increase_degraded: float = _DEFAULT_MSE_PCT_INCREASE_DEGRADED
    mse_pct_increase_alarm: float = _DEFAULT_MSE_PCT_INCREASE_ALARM
    accuracy_drop_degraded: float = _DEFAULT_ACCURACY_DROP_DEGRADED
    accuracy_drop_alarm: float = _DEFAULT_ACCURACY_DROP_ALARM
    f1_drop_degraded: float = _DEFAULT_F1_DROP_DEGRADED
    f1_drop_alarm: float = _DEFAULT_F1_DROP_ALARM

    def __post_init__(self) -> None:
        # Alarm thresholds must be at least as conservative as degraded.
        assert self.r_squared_drop_alarm >= self.r_squared_drop_degraded
        assert self.mse_pct_increase_alarm >= self.mse_pct_increase_degraded
        assert self.accuracy_drop_alarm >= self.accuracy_drop_degraded
        assert self.f1_drop_alarm >= self.f1_drop_degraded


@dataclass
class DriftAssessment:
    """Output of DriftDetector.evaluate."""

    model_id: str
    severity: DriftSeverity
    triggers: list[str] = field(default_factory=list)
    baseline: ModelMetrics | None = None
    current: ModelMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "severity": self.severity.value,
            "triggers": list(self.triggers),
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "current": self.current.to_dict() if self.current else None,
        }


# =============================================================================
# Drift detection
# =============================================================================


class DriftDetector:
    """Compares current model metrics against a stored baseline.

    A model's drift severity escalates per metric:
        - DEGRADED if any metric drops past its `*_degraded` threshold.
        - ALARM if any metric drops past its `*_alarm` threshold.
    The reported severity is the worst across all evaluated metrics.

    Thread-safe at the level of single evaluate() calls. State (baseline)
    is owned by the caller — the detector itself is stateless.
    """

    def __init__(
        self,
        thresholds: DriftThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or DriftThresholds()

    def evaluate(
        self,
        model_id: str,
        baseline: ModelMetrics,
        current: ModelMetrics,
    ) -> DriftAssessment:
        """Compare current metrics to baseline; return DriftAssessment.

        Each metric pair is compared independently; the worst severity wins.
        Metrics where either side is None are skipped (not all models
        produce all metrics).
        """
        triggers: list[str] = []
        worst = DriftSeverity.NORMAL

        # R²: regression goodness-of-fit. Higher is better → DROP triggers drift.
        if baseline.r_squared is not None and current.r_squared is not None:
            drop = baseline.r_squared - current.r_squared
            if drop >= self.thresholds.r_squared_drop_alarm:
                triggers.append(
                    f"r_squared dropped {drop:.3f} (>= {self.thresholds.r_squared_drop_alarm})",
                )
                worst = DriftSeverity.ALARM
            elif drop >= self.thresholds.r_squared_drop_degraded:
                triggers.append(
                    f"r_squared dropped {drop:.3f} (>= {self.thresholds.r_squared_drop_degraded})",
                )
                worst = _escalate(worst, DriftSeverity.DEGRADED)

        # MSE: regression error. Lower is better → INCREASE triggers drift.
        if (
            baseline.mse is not None
            and current.mse is not None
            and baseline.mse > 0
        ):
            pct = (current.mse - baseline.mse) / baseline.mse
            if pct >= self.thresholds.mse_pct_increase_alarm:
                triggers.append(
                    f"mse increased {pct:.1%} (>= {self.thresholds.mse_pct_increase_alarm:.0%})",
                )
                worst = DriftSeverity.ALARM
            elif pct >= self.thresholds.mse_pct_increase_degraded:
                triggers.append(
                    f"mse increased {pct:.1%} (>= {self.thresholds.mse_pct_increase_degraded:.0%})",
                )
                worst = _escalate(worst, DriftSeverity.DEGRADED)

        # Accuracy: classifier. Higher is better.
        if baseline.accuracy is not None and current.accuracy is not None:
            drop = baseline.accuracy - current.accuracy
            if drop >= self.thresholds.accuracy_drop_alarm:
                triggers.append(
                    f"accuracy dropped {drop:.3f} (>= {self.thresholds.accuracy_drop_alarm})",
                )
                worst = DriftSeverity.ALARM
            elif drop >= self.thresholds.accuracy_drop_degraded:
                triggers.append(
                    f"accuracy dropped {drop:.3f} (>= {self.thresholds.accuracy_drop_degraded})",
                )
                worst = _escalate(worst, DriftSeverity.DEGRADED)

        # F1: classifier balance. Higher is better.
        if baseline.f1 is not None and current.f1 is not None:
            drop = baseline.f1 - current.f1
            if drop >= self.thresholds.f1_drop_alarm:
                triggers.append(
                    f"f1 dropped {drop:.3f} (>= {self.thresholds.f1_drop_alarm})",
                )
                worst = DriftSeverity.ALARM
            elif drop >= self.thresholds.f1_drop_degraded:
                triggers.append(
                    f"f1 dropped {drop:.3f} (>= {self.thresholds.f1_drop_degraded})",
                )
                worst = _escalate(worst, DriftSeverity.DEGRADED)

        return DriftAssessment(
            model_id=model_id,
            severity=worst,
            triggers=triggers,
            baseline=baseline,
            current=current,
        )


# =============================================================================
# Retrain gate
# =============================================================================


@dataclass
class RetrainGateThresholds:
    """How much worse a candidate may be on hold-out before we reject deploy.

    A small tolerance buffers against random hold-out noise — without it,
    every retrain that produces a 0.001-worse R² would be rejected.
    """

    max_r_squared_regression: float = 0.02   # candidate may be up to 0.02 lower
    max_mse_pct_regression: float = 0.05      # candidate may be up to 5% higher MSE
    max_accuracy_regression: float = 0.01    # ditto for accuracy
    max_f1_regression: float = 0.01          # ditto for F1

    def __post_init__(self) -> None:
        assert self.max_r_squared_regression >= 0
        assert self.max_mse_pct_regression >= 0
        assert self.max_accuracy_regression >= 0
        assert self.max_f1_regression >= 0


@dataclass
class RetrainGateDecision:
    approve: bool
    reasons: list[str] = field(default_factory=list)
    failed_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approve": self.approve,
            "reasons": list(self.reasons),
            "failed_metrics": list(self.failed_metrics),
        }


class RetrainGate:
    """Decide whether a candidate model may replace production.

    Production wins ties — candidates must clear the no-worse-than tolerance
    on every applicable metric to promote.
    """

    def __init__(
        self,
        thresholds: RetrainGateThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or RetrainGateThresholds()

    def evaluate(
        self,
        production: ModelMetrics,
        candidate: ModelMetrics,
    ) -> RetrainGateDecision:
        reasons: list[str] = []
        failed: list[str] = []

        if production.r_squared is not None and candidate.r_squared is not None:
            drop = production.r_squared - candidate.r_squared
            if drop > self.thresholds.max_r_squared_regression:
                failed.append("r_squared")
                reasons.append(
                    f"r_squared regressed {drop:.3f} > "
                    f"{self.thresholds.max_r_squared_regression}",
                )
            else:
                reasons.append(
                    f"r_squared OK (Δ={drop:.3f} ≤ "
                    f"{self.thresholds.max_r_squared_regression}) ✓",
                )

        if (
            production.mse is not None
            and candidate.mse is not None
            and production.mse > 0
        ):
            pct = (candidate.mse - production.mse) / production.mse
            if pct > self.thresholds.max_mse_pct_regression:
                failed.append("mse")
                reasons.append(
                    f"mse regressed {pct:.1%} > "
                    f"{self.thresholds.max_mse_pct_regression:.0%}",
                )
            else:
                reasons.append(
                    f"mse OK (Δ={pct:.1%} ≤ "
                    f"{self.thresholds.max_mse_pct_regression:.0%}) ✓",
                )

        if production.accuracy is not None and candidate.accuracy is not None:
            drop = production.accuracy - candidate.accuracy
            if drop > self.thresholds.max_accuracy_regression:
                failed.append("accuracy")
                reasons.append(
                    f"accuracy regressed {drop:.3f} > "
                    f"{self.thresholds.max_accuracy_regression}",
                )
            else:
                reasons.append(
                    f"accuracy OK (Δ={drop:.3f} ≤ "
                    f"{self.thresholds.max_accuracy_regression}) ✓",
                )

        if production.f1 is not None and candidate.f1 is not None:
            drop = production.f1 - candidate.f1
            if drop > self.thresholds.max_f1_regression:
                failed.append("f1")
                reasons.append(
                    f"f1 regressed {drop:.3f} > {self.thresholds.max_f1_regression}",
                )
            else:
                reasons.append(
                    f"f1 OK (Δ={drop:.3f} ≤ {self.thresholds.max_f1_regression}) ✓",
                )

        approve = len(failed) == 0
        return RetrainGateDecision(
            approve=approve, reasons=reasons, failed_metrics=failed,
        )


# =============================================================================
# Retrain schedule
# =============================================================================


@dataclass
class RetrainSchedule:
    """Declarative per-model retrain cadence.

    Used by the scheduling layer (Airflow DAG / cron) to decide when to
    trigger a retrain. Production typical cadences:
        FinBERT          monthly (~30 days)
        rate-diff OLS    quarterly (~90 days)
    """

    model_id: str
    cadence_days: int
    last_retrain: datetime | None = None

    def __post_init__(self) -> None:
        assert self.model_id, "model_id must be non-empty"
        assert self.cadence_days >= 1, "cadence_days must be >= 1"

    def is_due(self, today: datetime) -> bool:
        """True iff cadence has elapsed since the last retrain (or never)."""
        if self.last_retrain is None:
            return True
        return today - self.last_retrain >= timedelta(days=self.cadence_days)

    def days_until_due(self, today: datetime) -> int:
        if self.last_retrain is None:
            return 0
        elapsed = (today - self.last_retrain).days
        return max(0, self.cadence_days - elapsed)
