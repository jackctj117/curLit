"""Feature edge attribution (G5 / CL-m0h) — which features actually contribute?

The G1 null hypothesis framework asks "is this strategy better than random
overall?" — but a strategy that passes G1 may still be carrying overfit
features that contribute nothing real, with the edge actually concentrated
in just one or two features. G5 isolates per-feature contributions via
ablation and compares them against a random-feature noise floor.

Approach:

    score_fn(features) → Sharpe         user-provided callable that runs the
                                        strategy/backtest given a feature dict
                                        and returns a Sharpe.
    baseline = score_fn(all_features)   reference Sharpe.
    For each feature f:
        ablated = score_fn(features - {f})
        contribution = baseline - ablated
    For each random feature draw (n_random_tests times):
        with_random = score_fn(features + {random_f})
        random_contribution = with_random - baseline
    noise_floor = 2 × std(random_contributions)
    feature is "above noise floor" iff |contribution| > noise_floor.

The reference (reference/14_edge_testing.md "Feature Edge Attribution")
couples to a strategy class with disable_feature/add_feature methods. That
constrains the strategy API; instead we accept a score_fn callable that
the strategy author supplies. Cleaner separation between attribution
mechanics and strategy/backtest internals.

Reference: reference/14_edge_testing.md "Feature Edge Attribution".
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default number of random-feature tests for the noise floor estimate.
# 10 is the reference value; CV(noise_floor) ~ 1/sqrt(2*n) ≈ 22% at n=10
# is acceptable for "above-floor or below-floor" decisions.
_DEFAULT_N_RANDOM_TESTS: int = 10

# Multiplier on the random-contribution stdev to define the noise floor.
# 2 × std ≈ 95% confidence band; matches reference.
_NOISE_FLOOR_SIGMA_MULTIPLIER: float = 2.0


@dataclass
class FeatureContribution:
    """One feature's ablation result."""

    feature: str
    baseline_sharpe: float
    without_feature_sharpe: float
    contribution: float
    pct_contribution: float
    above_noise_floor: bool
    noise_floor_2sigma: float
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "baseline_sharpe": self.baseline_sharpe,
            "without_feature_sharpe": self.without_feature_sharpe,
            "contribution": self.contribution,
            "pct_contribution": self.pct_contribution,
            "above_noise_floor": self.above_noise_floor,
            "noise_floor_2sigma": self.noise_floor_2sigma,
            "warning": self.warning,
        }


@dataclass
class AttributionReport:
    """Aggregated feature attribution result."""

    baseline_sharpe: float
    noise_floor_2sigma: float
    n_features: int
    n_random_tests: int
    contributions: dict[str, FeatureContribution] = field(default_factory=dict)
    random_contributions: list[float] = field(default_factory=list)

    @property
    def features_above_floor(self) -> list[str]:
        return [
            name for name, c in self.contributions.items()
            if c.above_noise_floor
        ]

    @property
    def features_below_floor(self) -> list[str]:
        return [
            name for name, c in self.contributions.items()
            if not c.above_noise_floor
        ]

    @property
    def edge_concentration(self) -> float:
        """Fraction of total positive contribution from the single biggest feature.

        High concentration (>0.6) means the strategy's edge rests on one
        feature — fragile under regime change. Used by the G9 promotion gate
        (`edge_concentration_max` config).
        """
        positive = [c.contribution for c in self.contributions.values() if c.contribution > 0]
        if not positive:
            return 0.0
        total = sum(positive)
        if total <= 0:
            return 0.0
        return float(max(positive) / total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_sharpe": self.baseline_sharpe,
            "noise_floor_2sigma": self.noise_floor_2sigma,
            "n_features": self.n_features,
            "n_random_tests": self.n_random_tests,
            "edge_concentration": self.edge_concentration,
            "features_above_floor": list(self.features_above_floor),
            "features_below_floor": list(self.features_below_floor),
            "contributions": {
                name: c.to_dict() for name, c in self.contributions.items()
            },
        }


# =============================================================================
# Attributor
# =============================================================================


# Type alias: score_fn takes a feature dict and returns a Sharpe.
ScoreFn = Callable[[Mapping[str, pd.Series]], float]


class FeatureEdgeAttributor:
    """Ablation + random-noise-floor estimation for per-feature edge attribution.

    Stateless. Construct once per analysis; reuse across strategies if
    score_fn dispatches by strategy.
    """

    def __init__(
        self,
        n_random_tests: int = _DEFAULT_N_RANDOM_TESTS,
        random_feature_length: int | None = None,
        seed: int | None = None,
    ) -> None:
        assert n_random_tests >= 2, (
            f"n_random_tests must be >= 2 for std estimate, got {n_random_tests}"
        )
        self.n_random_tests = n_random_tests
        self.random_feature_length = random_feature_length
        self._rng = np.random.default_rng(seed)

    def attribute(
        self,
        score_fn: ScoreFn,
        baseline_features: Mapping[str, pd.Series],
    ) -> AttributionReport:
        """Run ablation + noise-floor and produce an AttributionReport.

        Args:
            score_fn: callable mapping a feature dict → Sharpe. Should be
                deterministic given the same inputs (otherwise the noise
                floor will be inflated by the score_fn's own variance).
            baseline_features: the strategy's full feature set as
                {name: Series}. Each Series must share an index.

        Returns AttributionReport with per-feature contributions, the noise
        floor, and edge_concentration (used by the G9 promotion gate).
        """
        feature_names = list(baseline_features.keys())
        assert feature_names, "baseline_features must be non-empty"

        # Determine the random-feature length. Default to whatever the first
        # feature has — assumes all features share an index.
        first_series = next(iter(baseline_features.values()))
        rand_len = self.random_feature_length or len(first_series)

        baseline_sharpe = float(score_fn(baseline_features))
        logger.info("Feature attribution baseline Sharpe = %.3f", baseline_sharpe)

        # ---- Random-feature noise floor ----
        random_contributions: list[float] = []
        for _ in range(self.n_random_tests):
            rand_series = pd.Series(
                self._rng.standard_normal(rand_len),
                index=first_series.index[:rand_len] if hasattr(first_series, "index") else None,
            )
            augmented = dict(baseline_features)
            augmented["_random_test"] = rand_series
            try:
                with_random = float(score_fn(augmented))
            except Exception:
                logger.exception("score_fn failed during random-feature draw")
                continue
            random_contributions.append(with_random - baseline_sharpe)

        if len(random_contributions) >= 2:
            noise_floor = float(
                _NOISE_FLOOR_SIGMA_MULTIPLIER
                * np.std(random_contributions, ddof=1),
            )
        else:
            # Degenerate case: not enough random tests succeeded — fall back
            # to a conservative 0 floor so every feature is "above floor".
            noise_floor = 0.0

        # ---- Per-feature ablation ----
        contributions: dict[str, FeatureContribution] = {}
        for feat in feature_names:
            ablated_features = {
                k: v for k, v in baseline_features.items() if k != feat
            }
            try:
                ablated_sharpe = float(score_fn(ablated_features))
            except Exception:
                logger.exception("score_fn failed during ablation of %s", feat)
                continue

            contribution = baseline_sharpe - ablated_sharpe
            pct = contribution / baseline_sharpe if baseline_sharpe > 0 else 0.0
            above_floor = abs(contribution) > noise_floor
            warning = (
                f"contribution {contribution:.3f} within 2σ of noise "
                f"floor {noise_floor:.3f} — likely overfit"
                if not above_floor
                else ""
            )
            contributions[feat] = FeatureContribution(
                feature=feat,
                baseline_sharpe=baseline_sharpe,
                without_feature_sharpe=ablated_sharpe,
                contribution=contribution,
                pct_contribution=pct,
                above_noise_floor=above_floor,
                noise_floor_2sigma=noise_floor,
                warning=warning,
            )

        return AttributionReport(
            baseline_sharpe=baseline_sharpe,
            noise_floor_2sigma=noise_floor,
            n_features=len(feature_names),
            n_random_tests=len(random_contributions),
            contributions=contributions,
            random_contributions=random_contributions,
        )


__all__ = [
    "AttributionReport",
    "FeatureContribution",
    "FeatureEdgeAttributor",
    "ScoreFn",
]
