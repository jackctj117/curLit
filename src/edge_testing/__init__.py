"""Edge-testing framework — statistical rigor checks for strategy P&L.

Modules:
    null_hypothesis  — Test strategy Sharpe vs 6 random/naive baselines (G1).
"""

from src.edge_testing.null_hypothesis import (
    NullHypothesisFramework,
    NullHypothesisReport,
    NullResult,
)

__all__ = [
    "NullHypothesisFramework",
    "NullHypothesisReport",
    "NullResult",
]
