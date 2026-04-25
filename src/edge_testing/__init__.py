"""Edge-testing framework — statistical rigor checks for strategy P&L.

Modules:
    null_hypothesis    — Test strategy Sharpe vs 6 random/naive baselines (G1).
    multiple_testing   — Bonferroni / Benjamini-Hochberg / White's Reality Check (G2).
    live_tracker       — Live-vs-backtest divergence monitor with consequences (G3).
"""

from src.edge_testing.live_tracker import (
    BacktestExpectations,
    LiveEdgeAssessment,
    LiveEdgeTracker,
    Recommendation,
    Severity,
)
from src.edge_testing.paper_live_divergence import (
    DivergenceReport,
    MatchedPair,
    PaperLiveDivergence,
)
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
    "BacktestExpectations",
    "DivergenceReport",
    "LiveEdgeAssessment",
    "LiveEdgeTracker",
    "MatchedPair",
    "MultipleTestingCorrection",
    "MultipleTestingReport",
    "NullHypothesisFramework",
    "NullHypothesisReport",
    "NullResult",
    "PaperLiveDivergence",
    "RealityCheckResult",
    "Recommendation",
    "Severity",
    "benjamini_hochberg",
    "bonferroni",
    "whites_reality_check",
]
