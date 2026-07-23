"""Capital allocation policy for new strategies (CL-15r4).

Governs the **promotion** decision: does a strategy that has been in
paper mode long enough actually get capital, and how much?

The policy answers four questions:

  1. Has the strategy spent enough time in paper mode? (default 90d)
  2. Is its paper-track-record actually positive enough to promote?
     (configurable Sharpe floor or min-return)
  3. What initial weight does it get on day-one of live? (default 5%)
  4. How fast does that weight ramp up? (+5% per profitable month,
     capped at max-per-strategy)

Pairs with src/portfolio/coordinator.py promote_strategy_to_live(),
which calls evaluate() at promotion time and ramp_weight() on each
month-end review. CL-6vv covers the lifecycle plumbing; this module
is the pure policy layer with no broker / DB dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default policy values — sourced from reference/06_portfolio.md and
# the CL-15r4 acceptance criteria. Each constant here represents a
# governance decision; override per-strategy via PolicyConfig if a
# strategy needs special treatment.

# 90 days minimum paper track. Three months captures multiple weekly
# data releases, at least one Fed meeting, and quarterly seasonality
# in FX flows. Below 90d, sample is too small to distinguish edge
# from luck. (CL-15r4 acceptance criterion.)
_DEFAULT_MIN_PAPER_DAYS: int = 90

# 5% initial allocation. Small enough to absorb a modeling error
# without blowing up the book; large enough that broker costs (1bp
# round-trip) don't dominate the signal at typical strategy Sharpe.
_DEFAULT_INITIAL_PCT: float = 0.05

# +5% per profitable month. Compounding 5%/month from a 5% base
# reaches the 30% cap in 5 monthly increments — fast enough to
# scale a working strategy, slow enough to catch a regime break.
_DEFAULT_RAMP_PER_MONTH: float = 0.05

# 30% per-strategy cap. Above this, the portfolio becomes
# concentration-risk dominated. Risk parity (CL-6tu) uses 40% as its
# bound; allocation policy stays below to leave headroom for the
# parity optimizer to maneuver.
_DEFAULT_MAX_PCT: float = 0.30

# 0.5 Sharpe floor for promotion. A paper Sharpe of 0.5 is roughly
# 1σ above zero on a 90-day track (n=63 daily returns); the noise
# floor for "this beats holding cash". Conservative — net of costs
# in live should still be positive.
_DEFAULT_MIN_PAPER_SHARPE: float = 0.5

# 252 trading days per year for annualization, matching the rest
# of the codebase (CL-6h6 RV20, walk-forward analytics, kill_switches).
_TRADING_DAYS_PER_YEAR: int = 252


class PromotionVerdict(Enum):
    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


@dataclass
class PromotionDecision:
    verdict: PromotionVerdict
    initial_weight: float = 0.0
    reason: str = ""
    # The metrics used to make the decision, for logs/audits.
    paper_days: int = 0
    paper_sharpe: float | None = None


@dataclass
class PolicyConfig:
    """Per-strategy or global allocation policy parameters.

    Default values come from reference/06_portfolio.md and CL-15r4.
    Override at construction time for a strategy with special
    governance needs (e.g. flagship strategy with track record from
    a previous deployment may skip the min_paper_days requirement).
    """

    min_paper_days: int = _DEFAULT_MIN_PAPER_DAYS
    initial_pct: float = _DEFAULT_INITIAL_PCT
    ramp_per_month: float = _DEFAULT_RAMP_PER_MONTH
    max_pct: float = _DEFAULT_MAX_PCT
    min_paper_sharpe: float = _DEFAULT_MIN_PAPER_SHARPE
    # Per-strategy overrides — keyed by strategy_id, fall back to globals.
    overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    def for_strategy(self, strategy_id: str) -> PolicyConfig:
        """Return a config with per-strategy overrides applied."""
        ov = self.overrides.get(strategy_id, {})
        if not ov:
            return self
        return PolicyConfig(
            min_paper_days=int(ov.get("min_paper_days", self.min_paper_days)),
            initial_pct=float(ov.get("initial_pct", self.initial_pct)),
            ramp_per_month=float(ov.get("ramp_per_month", self.ramp_per_month)),
            max_pct=float(ov.get("max_pct", self.max_pct)),
            min_paper_sharpe=float(
                ov.get(
                    "min_paper_sharpe",
                    self.min_paper_sharpe,
                )
            ),
        )


class AllocationPolicy:
    """Promotion-time allocation rules + monthly ramp."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    # -- Promotion decision -------------------------------------------

    def evaluate(
        self,
        strategy_id: str,
        paper_start: datetime,
        paper_returns: pd.Series,
        now: datetime | None = None,
    ) -> PromotionDecision:
        """Decide whether ``strategy_id`` should be promoted to live."""
        now = now or datetime.now(UTC)
        cfg = self.config.for_strategy(strategy_id)

        paper_days = (now - paper_start).days
        if paper_days < cfg.min_paper_days:
            return PromotionDecision(
                verdict=PromotionVerdict.HOLD,
                reason=(f"Paper days {paper_days} < required {cfg.min_paper_days}"),
                paper_days=paper_days,
            )

        sharpe = self._sharpe(paper_returns)
        if sharpe is None:
            return PromotionDecision(
                verdict=PromotionVerdict.HOLD,
                reason="Insufficient paper returns to compute Sharpe",
                paper_days=paper_days,
            )
        if sharpe < cfg.min_paper_sharpe:
            return PromotionDecision(
                verdict=PromotionVerdict.REJECT,
                reason=(f"Paper Sharpe {sharpe:.2f} < threshold {cfg.min_paper_sharpe}"),
                paper_days=paper_days,
                paper_sharpe=sharpe,
            )

        return PromotionDecision(
            verdict=PromotionVerdict.PROMOTE,
            initial_weight=cfg.initial_pct,
            reason=(f"Paper Sharpe {sharpe:.2f} ≥ {cfg.min_paper_sharpe} over {paper_days}d"),
            paper_days=paper_days,
            paper_sharpe=sharpe,
        )

    # -- Monthly ramp -------------------------------------------------

    def ramp_weight(
        self,
        strategy_id: str,
        current_weight: float,
        monthly_returns: list[float],
        n_months_live: int,
    ) -> float:
        """Apply ramp logic. Returns the new target weight (caller pushes
        through PortfolioCoordinator's reweight broadcast).

        Logic:
          - Each profitable month adds ramp_per_month to target weight.
          - Loss month resets the ramp (no compounding through losses).
          - Hard cap at max_pct regardless of streak length.
        """
        cfg = self.config.for_strategy(strategy_id)
        if not monthly_returns:
            return current_weight

        # Count consecutive profitable months from the most recent end.
        streak = 0
        for r in reversed(monthly_returns):
            if r > 0:
                streak += 1
            else:
                break
        if streak == 0:
            # Reset to initial weight on a losing streak — preserves the
            # strategy in the book but at minimum size while it works
            # itself out.
            return cfg.initial_pct

        # Initial weight + ramp_per_month * (streak - 1) so that month-1
        # of being live keeps the initial allocation; month-2 picks up the
        # first ramp increment.
        target = cfg.initial_pct + cfg.ramp_per_month * (streak - 1)
        return min(target, cfg.max_pct)

    # -- Helpers ------------------------------------------------------

    @staticmethod
    def _sharpe(returns: pd.Series) -> float | None:
        if returns is None or len(returns) < 2:
            return None
        sd = float(returns.std(ddof=1))
        if sd == 0 or not np.isfinite(sd):
            return None
        return float(returns.mean() / sd * np.sqrt(_TRADING_DAYS_PER_YEAR))


# Convenience: standard "30 days" default for legacy CL-6vv add_strategy
# call sites; production should call policy.evaluate() at the 90-day mark.
DEFAULT_PAPER_PERIOD: timedelta = timedelta(days=_DEFAULT_MIN_PAPER_DAYS)
