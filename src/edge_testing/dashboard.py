"""Meta-edge dashboard (G8 / CL-4lp) — unified verdict from all edge-testing layers.

The Phase G layers each answer a narrow question:

    G1 NullHypothesisFramework      Does the strategy beat random baselines?
    G2 MultipleTestingCorrection    Does it survive data-snooping correction?
    G3 LiveEdgeTracker              Does live performance match backtest?
    G4 PaperLiveDivergence          Is execution quality acceptable?
    G5 FeatureEdgeAttributor        Are individual features signal-positive?
    G6 RegimeEdgeAnalyzer           Is edge spread across regimes (not lucky)?
    G7 EdgeDecayMonitor             Is edge stable over time?

EdgeDashboard collects whichever layer outputs are available for one strategy
and assigns a single verdict that the policy enforcer (G9 edge_policy.yaml)
or the human operator can act on:

    STRONG_EDGE         every available layer passes confidently
    WEAK_EDGE           passes more layers than it fails, but mixed
    NO_EDGE_DETECTED    the foundational tests (G1, G2) fail
    EDGE_DECAYED        G7 decay signal overrides everything else
    INSUFFICIENT_DATA   too few layers reporting to make a call

Layers are independently optional — the dashboard reports verdicts based on
what's available, with INSUFFICIENT_DATA when fewer than 2 layers report.
This lets the dashboard be useful before all 7 layers exist (currently
G1/G2/G3/G4 are built; G5/G6/G7 are still upcoming).

Reference: CL-4lp; reference/14_edge_testing.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.monitoring.metrics import edge_verdict

logger = logging.getLogger(__name__)


# Minimum number of layers reporting before we'll assign anything other than
# INSUFFICIENT_DATA. 2 = at least two independent statistical tests said
# something. Below that we don't trust the verdict.
_MIN_LAYERS_FOR_VERDICT: int = 2

# Threshold for STRONG vs WEAK at the same pass-count: at least this fraction
# of available layers must pass for STRONG. 0.85 keeps a single weak layer
# from blocking strong classification when 6 of 7 pass.
_STRONG_PASS_FRACTION: float = 0.85


class EdgeVerdict(Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    NO_EDGE_DETECTED = "no_edge_detected"
    EDGE_DECAYED = "edge_decayed"
    WEAK_EDGE = "weak_edge"
    STRONG_EDGE = "strong_edge"


# Numeric encoding for the Prometheus gauge — readable thresholds in alerts
# (e.g. "verdict <= 1" means trouble).
_VERDICT_RANK: dict[EdgeVerdict, int] = {
    EdgeVerdict.INSUFFICIENT_DATA: 0,
    EdgeVerdict.NO_EDGE_DETECTED: 1,
    EdgeVerdict.EDGE_DECAYED: 2,
    EdgeVerdict.WEAK_EDGE: 3,
    EdgeVerdict.STRONG_EDGE: 4,
}


# =============================================================================
# Per-layer signals
# =============================================================================


class LayerStatus(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_REPORTED = "not_reported"


@dataclass
class LayerSignal:
    """One layer's contribution to the overall verdict."""

    name: str
    status: LayerStatus
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
        }


@dataclass
class EdgeLayerInputs:
    """Container for raw per-layer outputs.

    All fields optional so the dashboard works even before all 7 layers are
    built. Each field type matches the corresponding Phase G module's output.
    """

    # G1: True iff the strategy beat all evaluated null baselines at α.
    # The G1 NullHypothesisReport carries `edge_exists`.
    null_hypothesis_passed: bool | None = None
    null_hypothesis_detail: str = ""

    # G2: Bonferroni / BH / Reality Check passed?
    # `passed` here means "approve at alpha after multiple-testing correction".
    multiple_testing_passed: bool | None = None
    multiple_testing_detail: str = ""

    # G3: Severity from LiveEdgeAssessment (severity attribute, .value).
    live_tracker_severity: str | None = None
    live_tracker_detail: str = ""

    # G4: avg_abs_price_diff_bps from DivergenceReport. Lower = better quality.
    paper_live_avg_abs_diff_bps: float | None = None
    paper_live_threshold_bps: float = 5.0   # passes if avg < this
    paper_live_detail: str = ""

    # G5 (TBD): feature edge attribution result. True iff features carry signal.
    feature_attribution_passed: bool | None = None
    feature_attribution_detail: str = ""

    # G6 (TBD): edge concentrated in a single regime → fail. Pass if spread.
    regime_edge_diversified: bool | None = None
    regime_edge_detail: str = ""

    # G7 (TBD): edge_decayed=True when decay signal fires.
    edge_decayed: bool | None = None
    edge_decay_detail: str = ""


# =============================================================================
# Verdict
# =============================================================================


@dataclass
class StrategyVerdict:
    """Final per-strategy assessment with all reporting layer signals."""

    strategy_id: str
    verdict: EdgeVerdict
    ts: datetime
    signals: list[LayerSignal] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "verdict": self.verdict.value,
            "ts": self.ts.isoformat(),
            "signals": [s.to_dict() for s in self.signals],
            "summary": self.summary,
        }

    @property
    def n_passing(self) -> int:
        return sum(1 for s in self.signals if s.status == LayerStatus.PASS)

    @property
    def n_failing(self) -> int:
        return sum(1 for s in self.signals if s.status == LayerStatus.FAIL)

    @property
    def n_reporting(self) -> int:
        return sum(
            1 for s in self.signals if s.status != LayerStatus.NOT_REPORTED
        )


# =============================================================================
# Dashboard
# =============================================================================


class EdgeDashboard:
    """Compose Phase G layer outputs into a single per-strategy verdict.

    Stateless — call assess() per strategy with whatever inputs are available.
    Emits the fx_edge_verdict Prometheus gauge as a side effect.
    """

    def assess(
        self,
        strategy_id: str,
        inputs: EdgeLayerInputs,
        ts: datetime | None = None,
    ) -> StrategyVerdict:
        ts = ts or datetime.now(UTC)
        signals = self._collect_signals(inputs)
        n_reporting = sum(
            1 for s in signals if s.status != LayerStatus.NOT_REPORTED
        )

        if n_reporting < _MIN_LAYERS_FOR_VERDICT:
            verdict = EdgeVerdict.INSUFFICIENT_DATA
            summary = f"only {n_reporting} layer(s) reporting (min {_MIN_LAYERS_FOR_VERDICT})"
        else:
            verdict, summary = self._classify(signals)

        result = StrategyVerdict(
            strategy_id=strategy_id,
            verdict=verdict,
            ts=ts,
            signals=signals,
            summary=summary,
        )
        self._publish_metric(strategy_id, verdict)
        logger.info(
            "Edge verdict %s: %s (%s)",
            strategy_id, verdict.value, summary,
        )
        return result

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_signals(inputs: EdgeLayerInputs) -> list[LayerSignal]:
        signals: list[LayerSignal] = []

        # G1 null hypothesis.
        if inputs.null_hypothesis_passed is None:
            signals.append(LayerSignal("G1_null_hypothesis", LayerStatus.NOT_REPORTED))
        elif inputs.null_hypothesis_passed:
            signals.append(LayerSignal(
                "G1_null_hypothesis", LayerStatus.PASS,
                detail=inputs.null_hypothesis_detail or "beats all evaluated nulls",
            ))
        else:
            signals.append(LayerSignal(
                "G1_null_hypothesis", LayerStatus.FAIL,
                detail=inputs.null_hypothesis_detail or "fails one or more nulls",
            ))

        # G2 multiple testing.
        if inputs.multiple_testing_passed is None:
            signals.append(LayerSignal("G2_multiple_testing", LayerStatus.NOT_REPORTED))
        elif inputs.multiple_testing_passed:
            signals.append(LayerSignal(
                "G2_multiple_testing", LayerStatus.PASS,
                detail=inputs.multiple_testing_detail or "survives correction",
            ))
        else:
            signals.append(LayerSignal(
                "G2_multiple_testing", LayerStatus.FAIL,
                detail=inputs.multiple_testing_detail
                or "rejected after multiple-testing correction",
            ))

        # G3 live tracker — translate severity to pass/warn/fail.
        if inputs.live_tracker_severity is None:
            signals.append(LayerSignal("G3_live_tracker", LayerStatus.NOT_REPORTED))
        else:
            sev = inputs.live_tracker_severity
            if sev == "on_track":
                status = LayerStatus.PASS
            elif sev in ("underperforming",):
                status = LayerStatus.WARN
            else:
                # significantly_/severely_underperforming
                status = LayerStatus.FAIL
            signals.append(LayerSignal(
                "G3_live_tracker", status,
                detail=inputs.live_tracker_detail or f"severity={sev}",
            ))

        # G4 paper-live divergence — compare avg |bps| to threshold.
        if inputs.paper_live_avg_abs_diff_bps is None:
            signals.append(LayerSignal("G4_paper_live", LayerStatus.NOT_REPORTED))
        else:
            diff = inputs.paper_live_avg_abs_diff_bps
            threshold = inputs.paper_live_threshold_bps
            if diff <= threshold:
                status = LayerStatus.PASS
            elif diff <= threshold * 2:
                status = LayerStatus.WARN
            else:
                status = LayerStatus.FAIL
            signals.append(LayerSignal(
                "G4_paper_live", status,
                detail=(
                    inputs.paper_live_detail
                    or f"avg |diff|={diff:.2f}bps vs threshold {threshold}bps"
                ),
            ))

        # G5 feature attribution.
        if inputs.feature_attribution_passed is None:
            signals.append(LayerSignal(
                "G5_feature_attribution", LayerStatus.NOT_REPORTED,
            ))
        elif inputs.feature_attribution_passed:
            signals.append(LayerSignal(
                "G5_feature_attribution", LayerStatus.PASS,
                detail=inputs.feature_attribution_detail or "features carry signal",
            ))
        else:
            signals.append(LayerSignal(
                "G5_feature_attribution", LayerStatus.FAIL,
                detail=inputs.feature_attribution_detail
                or "features not significant after ablation",
            ))

        # G6 regime decomposition.
        if inputs.regime_edge_diversified is None:
            signals.append(LayerSignal("G6_regime_decomposition", LayerStatus.NOT_REPORTED))
        elif inputs.regime_edge_diversified:
            signals.append(LayerSignal(
                "G6_regime_decomposition", LayerStatus.PASS,
                detail=inputs.regime_edge_detail or "edge spread across regimes",
            ))
        else:
            signals.append(LayerSignal(
                "G6_regime_decomposition", LayerStatus.FAIL,
                detail=inputs.regime_edge_detail
                or "edge concentrated in a single regime",
            ))

        # G7 decay monitor — special-cased upstream as a hard verdict override.
        if inputs.edge_decayed is None:
            signals.append(LayerSignal("G7_decay_monitor", LayerStatus.NOT_REPORTED))
        elif inputs.edge_decayed:
            signals.append(LayerSignal(
                "G7_decay_monitor", LayerStatus.FAIL,
                detail=inputs.edge_decay_detail or "decay signal fired",
            ))
        else:
            signals.append(LayerSignal(
                "G7_decay_monitor", LayerStatus.PASS,
                detail=inputs.edge_decay_detail or "no decay detected",
            ))

        return signals

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(signals: list[LayerSignal]) -> tuple[EdgeVerdict, str]:
        """Map collected signals to a verdict.

        Precedence:
            1. G7 decay → EDGE_DECAYED (terminal — overrides everything)
            2. G1 OR G2 fail → NO_EDGE_DETECTED (foundational)
            3. STRONG / WEAK based on pass-fraction over reporting layers
        """
        reporting = [s for s in signals if s.status != LayerStatus.NOT_REPORTED]

        # 1. Decay overrides.
        for s in signals:
            if s.name == "G7_decay_monitor" and s.status == LayerStatus.FAIL:
                return EdgeVerdict.EDGE_DECAYED, "G7 decay signal fired"

        # 2. Foundational tests fail → NO_EDGE.
        for s in signals:
            if (
                s.name in ("G1_null_hypothesis", "G2_multiple_testing")
                and s.status == LayerStatus.FAIL
            ):
                return (
                    EdgeVerdict.NO_EDGE_DETECTED,
                    f"{s.name} failed: {s.detail or 'no detail'}",
                )

        # 3. Pass fraction over reporting layers.
        n_reporting = len(reporting)
        n_passing = sum(1 for s in reporting if s.status == LayerStatus.PASS)
        n_failing = sum(1 for s in reporting if s.status == LayerStatus.FAIL)

        if n_failing > 0:
            return (
                EdgeVerdict.WEAK_EDGE,
                f"{n_passing}/{n_reporting} layers PASS, {n_failing} FAIL",
            )

        pass_fraction = n_passing / n_reporting if n_reporting > 0 else 0.0
        if pass_fraction >= _STRONG_PASS_FRACTION:
            return (
                EdgeVerdict.STRONG_EDGE,
                f"{n_passing}/{n_reporting} PASS (≥{_STRONG_PASS_FRACTION:.0%})",
            )
        return (
            EdgeVerdict.WEAK_EDGE,
            f"{n_passing}/{n_reporting} PASS (<{_STRONG_PASS_FRACTION:.0%})",
        )

    @staticmethod
    def _publish_metric(strategy_id: str, verdict: EdgeVerdict) -> None:
        try:
            edge_verdict.labels(strategy_id=strategy_id).set(_VERDICT_RANK[verdict])
        except Exception:
            logger.exception("edge_verdict gauge set failed")


__all__ = [
    "EdgeDashboard",
    "EdgeLayerInputs",
    "EdgeVerdict",
    "LayerSignal",
    "LayerStatus",
    "StrategyVerdict",
]
