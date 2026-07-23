"""Edge policy loader + evaluator (G9 / CL-337).

Reads `configs/edge_policy.yaml` and exposes structured decisions:

    EdgePolicy.evaluate_promotion(metrics) → PromotionDecision
        Should this paper-mode strategy be promoted to live? All gates must
        pass: minimum paper days, null hypothesis p-value (G1), Reality Check
        p-value (G2), edge concentration (G5).

    EdgePolicy.evaluate_live(assessment) → LiveAction
        Given a LiveEdgeAssessment from G3 (and optionally a fitted decay
        tau from G7), what action should fire? Pre-committed mapping from
        z-score thresholds to {continue, review, reduce_size_50pct,
        halt_strategy, retire_strategy}.

    EdgePolicy.is_retirement_eligible(retired_at, today) → bool
        Has the post-retirement cooldown elapsed? If `requires_new_research`
        is true, callers must additionally check that a new-research flag
        is present before re-promoting.

The PORTFOLIO COORDINATOR is the executor — given a LiveAction, it pauses,
sizes-down, or removes the strategy. This module ONLY decides; it does not
mutate any strategy state.

Reference: configs/edge_policy.yaml (the canonical policy text, with the
"DO NOT EDIT DURING DRAWDOWNS" warning).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Project-relative default — the canonical policy file lives in `configs/`.
DEFAULT_POLICY_PATH: Path = Path(__file__).parents[2] / "configs" / "edge_policy.yaml"


# =============================================================================
# Decision types
# =============================================================================


class PromotionVerdict(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class PromotionDecision:
    """Output of evaluate_promotion(). Always carries the reasons examined."""

    verdict: PromotionVerdict
    reasons: list[str] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "failed_gates": list(self.failed_gates),
        }


class LiveAction(Enum):
    CONTINUE = "continue"
    REVIEW = "review"
    REDUCE_SIZE_50PCT = "reduce_size_50pct"
    HALT_STRATEGY = "halt_strategy"
    RETIRE_STRATEGY = "retire_strategy"


@dataclass
class StrategyMetrics:
    """Inputs to evaluate_promotion().

    Aggregated upstream from the edge-testing framework: G1 supplies
    null_max_p, G2 supplies reality_check_p, G5 supplies edge_concentration,
    paper-mode tracking supplies paper_days.
    """

    paper_days: int
    null_max_p: float
    reality_check_p: float
    edge_concentration: float

    def __post_init__(self) -> None:
        assert self.paper_days >= 0, "paper_days must be non-negative"
        assert 0.0 <= self.null_max_p <= 1.0, "null_max_p must be in [0,1]"
        assert 0.0 <= self.reality_check_p <= 1.0, "reality_check_p must be in [0,1]"
        assert 0.0 <= self.edge_concentration <= 1.0, "edge_concentration must be in [0,1]"


@dataclass
class LiveSnapshot:
    """Inputs to evaluate_live().

    sharpe_z is the primary G3 output. live_mean_return is the rolling 30-day
    mean. decay_tau is from G7 (CL-675) when available; pass None pre-G7.
    """

    sharpe_z: float
    live_mean_return: float
    decay_tau: float | None = None


# =============================================================================
# Policy
# =============================================================================


@dataclass
class PaperGates:
    min_days_before_live: int
    null_p_max: float
    reality_check_p_max: float
    edge_concentration_max: float


@dataclass
class DegradationTriggers:
    sharpe_z_review: float
    sharpe_z_reduce_size_50pct: float
    sharpe_z_halt: float
    decay_tau_retire: float
    live_mean_return_floor: float


@dataclass
class RetirementPolicy:
    cooldown_days: int
    requires_new_research: bool


@dataclass
class EdgePolicy:
    """Structured view of edge_policy.yaml + evaluation methods."""

    paper: PaperGates
    live: DegradationTriggers
    retirement: RetirementPolicy
    alert_severity: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Promotion gate
    # ------------------------------------------------------------------

    def evaluate_promotion(self, metrics: StrategyMetrics) -> PromotionDecision:
        """Gate paper-mode strategies against ALL promotion criteria.

        All gates must pass. Returns a PromotionDecision with structured
        reasons so the caller can surface them in dashboards / runbooks.
        """
        reasons: list[str] = []
        failed: list[str] = []

        if metrics.paper_days < self.paper.min_days_before_live:
            failed.append("min_days_before_live")
            reasons.append(
                f"paper_days={metrics.paper_days} < required {self.paper.min_days_before_live}"
            )
        else:
            reasons.append(f"paper_days={metrics.paper_days} ≥ {self.paper.min_days_before_live} ✓")

        if metrics.null_max_p > self.paper.null_p_max:
            failed.append("null_p_max")
            reasons.append(f"null p={metrics.null_max_p:.3f} > max {self.paper.null_p_max}")
        else:
            reasons.append(f"null p={metrics.null_max_p:.3f} ≤ {self.paper.null_p_max} ✓")

        if metrics.reality_check_p > self.paper.reality_check_p_max:
            failed.append("reality_check_p_max")
            reasons.append(
                f"reality_check p={metrics.reality_check_p:.3f} "
                f"> max {self.paper.reality_check_p_max}"
            )
        else:
            reasons.append(
                f"reality_check p={metrics.reality_check_p:.3f} "
                f"≤ {self.paper.reality_check_p_max} ✓"
            )

        if metrics.edge_concentration > self.paper.edge_concentration_max:
            failed.append("edge_concentration_max")
            reasons.append(
                f"edge_concentration={metrics.edge_concentration:.2f} "
                f"> max {self.paper.edge_concentration_max}"
            )
        else:
            reasons.append(
                f"edge_concentration={metrics.edge_concentration:.2f} "
                f"≤ {self.paper.edge_concentration_max} ✓"
            )

        verdict = PromotionVerdict.APPROVED if not failed else PromotionVerdict.REJECTED
        return PromotionDecision(verdict=verdict, reasons=reasons, failed_gates=failed)

    # ------------------------------------------------------------------
    # Live degradation triggers
    # ------------------------------------------------------------------

    def evaluate_live(self, snapshot: LiveSnapshot) -> LiveAction:
        """Map a live snapshot to the most-severe applicable action.

        Order of severity (most → least): RETIRE > HALT > REDUCE > REVIEW > CONTINUE.
        We return the first applicable action in that order, so a strategy
        that triggers both decay-retire AND z-halt retires.
        """
        if snapshot.decay_tau is not None and snapshot.decay_tau <= self.live.decay_tau_retire:
            return LiveAction.RETIRE_STRATEGY

        if snapshot.sharpe_z <= self.live.sharpe_z_halt:
            return LiveAction.HALT_STRATEGY

        if snapshot.live_mean_return < self.live.live_mean_return_floor:
            # Negative mean return forces at least size-down regardless of z.
            if snapshot.sharpe_z <= self.live.sharpe_z_reduce_size_50pct:
                return LiveAction.HALT_STRATEGY
            return LiveAction.REDUCE_SIZE_50PCT

        if snapshot.sharpe_z <= self.live.sharpe_z_reduce_size_50pct:
            return LiveAction.REDUCE_SIZE_50PCT

        if snapshot.sharpe_z <= self.live.sharpe_z_review:
            return LiveAction.REVIEW

        return LiveAction.CONTINUE

    # ------------------------------------------------------------------
    # Retirement cooldown
    # ------------------------------------------------------------------

    def is_retirement_eligible(
        self,
        retired_at: datetime,
        today: datetime,
    ) -> bool:
        """Returns True iff the cooldown window has elapsed.

        NOTE: this checks ONLY the time-based cooldown. If
        `requires_new_research` is True (the default), the caller must
        additionally verify a new-research artifact exists before re-promoting.
        """
        elapsed = today - retired_at
        return elapsed >= timedelta(days=self.retirement.cooldown_days)

    def alert_severity_for(self, action: LiveAction) -> str:
        """Map an action to the Alertmanager severity tag from the policy."""
        return self.alert_severity.get(action.value, "warning")


# =============================================================================
# Loading
# =============================================================================


def load_edge_policy(path: Path | str = DEFAULT_POLICY_PATH) -> EdgePolicy:
    """Parse edge_policy.yaml and validate schema.

    Raises:
        FileNotFoundError if the file is missing.
        ValueError if required keys are absent or malformed.
    """
    p = Path(path)
    if not p.exists():
        msg = f"edge policy not found at {p}"
        raise FileNotFoundError(msg)

    with p.open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        msg = f"edge policy at {p} did not parse to a dict"
        raise ValueError(msg)

    paper_cfg = _require(raw, "paper_trading", dict)
    paper = PaperGates(
        min_days_before_live=int(_require(paper_cfg, "min_days_before_live", int)),
        null_p_max=float(
            _require(_require(paper_cfg, "required_tests", dict), "null_p_max", (int, float)),
        ),
        reality_check_p_max=float(
            _require(
                _require(paper_cfg, "required_tests", dict), "reality_check_p_max", (int, float)
            ),
        ),
        edge_concentration_max=float(
            _require(
                _require(paper_cfg, "required_tests", dict), "edge_concentration_max", (int, float)
            ),
        ),
    )

    live_cfg = _require(raw, "live_trading", dict)
    triggers_cfg = _require(live_cfg, "degradation_triggers", dict)
    live = DegradationTriggers(
        sharpe_z_review=float(_require(triggers_cfg, "sharpe_z_review", (int, float))),
        sharpe_z_reduce_size_50pct=float(
            _require(triggers_cfg, "sharpe_z_reduce_size_50pct", (int, float)),
        ),
        sharpe_z_halt=float(_require(triggers_cfg, "sharpe_z_halt", (int, float))),
        decay_tau_retire=float(_require(triggers_cfg, "decay_tau_retire", (int, float))),
        live_mean_return_floor=float(
            _require(triggers_cfg, "live_mean_return_floor", (int, float)),
        ),
    )

    retirement_cfg = _require(raw, "retirement", dict)
    retirement = RetirementPolicy(
        cooldown_days=int(_require(retirement_cfg, "cooldown_days", int)),
        requires_new_research=bool(
            _require(retirement_cfg, "requires_new_research", bool),
        ),
    )

    alert_severity = raw.get("alerts", {})
    if not isinstance(alert_severity, dict):
        alert_severity = {}

    policy = EdgePolicy(
        paper=paper,
        live=live,
        retirement=retirement,
        alert_severity=alert_severity,
    )
    logger.info(
        "Loaded edge policy from %s (paper days=%d, sharpe halt=%.2f, cooldown=%d days)",
        p,
        policy.paper.min_days_before_live,
        policy.live.sharpe_z_halt,
        policy.retirement.cooldown_days,
    )
    return policy


def _require(d: dict[str, Any], key: str, expected_type: Any) -> Any:
    """Defensive accessor: KeyError + clear message when a required key is missing."""
    if key not in d:
        msg = f"required key '{key}' missing from edge policy section"
        raise ValueError(msg)
    val = d[key]
    if not isinstance(val, expected_type):
        msg = f"key '{key}' has wrong type: expected {expected_type}, got {type(val).__name__}"
        raise ValueError(msg)
    return val
