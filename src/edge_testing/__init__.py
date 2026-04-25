"""Edge-testing framework — statistical rigor checks for strategy P&L.

Modules:
    null_hypothesis    — Test strategy Sharpe vs 6 random/naive baselines (G1).
    multiple_testing   — Bonferroni / Benjamini-Hochberg / White's Reality Check (G2).
"""

from src.edge_testing.multiple_testing import (
    MultipleTestingCorrection,
    MultipleTestingReport,
    RealityCheckResult,
    benjamini_hochberg,
    bonferroni,
    whites_reality_check,
)
from src.edge_testing.null_hypothesis import (
    NullHypothesisFramework,
    NullHypothesisReport,
    NullResult,
)

__all__ = [
    "MultipleTestingCorrection",
    "MultipleTestingReport",
    "NullHypothesisFramework",
    "NullHypothesisReport",
    "NullResult",
    "RealityCheckResult",
    "benjamini_hochberg",
    "bonferroni",
    "whites_reality_check",
]
