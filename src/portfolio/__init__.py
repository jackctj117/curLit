"""Portfolio coordination — combines strategy intents, manages allocations, applies constraints.

Sits above strategies, below the OMS. Wired into LiveEngine via D7 (CL-amf).
"""

from src.portfolio.coordinator import (
    PortfolioConstraints,
    PortfolioCoordinator,
    PortfolioStateProtocol,
    StrategyAllocation,
)
from src.portfolio.pretrade import (
    PreTradeValidator,
    RejectionEvent,
    RejectionReason,
    TradabilityChecker,
)
from src.portfolio.reconciler import (
    PositionReconciler,
    ReconciliationEntry,
    ReconciliationPolicy,
    ReconciliationReport,
    ReconciliationStatus,
)
from src.portfolio.risk_parity import (
    diagnose_allocation,
    risk_parity_weights,
    rolling_risk_parity_weights,
)
from src.portfolio.state_store import PortfolioStateStore

__all__ = [
    "PortfolioConstraints",
    "PortfolioCoordinator",
    "PortfolioStateProtocol",
    "PortfolioStateStore",
    "PositionReconciler",
    "PreTradeValidator",
    "ReconciliationEntry",
    "ReconciliationPolicy",
    "ReconciliationReport",
    "ReconciliationStatus",
    "RejectionEvent",
    "RejectionReason",
    "StrategyAllocation",
    "TradabilityChecker",
    "diagnose_allocation",
    "risk_parity_weights",
    "rolling_risk_parity_weights",
]
