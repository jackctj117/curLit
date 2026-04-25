"""Portfolio coordination — combines strategy intents, manages allocations, applies constraints.

Sits above strategies, below the OMS. Wired into LiveEngine via D7 (CL-amf).
"""

from src.portfolio.coordinator import (
    PortfolioConstraints,
    PortfolioCoordinator,
    PortfolioStateProtocol,
    StrategyAllocation,
)

__all__ = [
    "PortfolioConstraints",
    "PortfolioCoordinator",
    "PortfolioStateProtocol",
    "StrategyAllocation",
]
