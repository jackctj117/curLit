"""Portfolio-level monitoring metric emission (CL-5lq).

Thin helper that wraps the gauges in src/monitoring/metrics.py. Called
by PortfolioCoordinator on each rebalance cycle. Failure is non-fatal —
metrics are observability, not control.

Inputs:
  - allocations: target allocation weight per strategy (sums to ~1.0)
  - exposure_multipliers: coordinator's runtime override per strategy
  - positions: current broker positions (for leverage)
  - equity: current account equity
  - attributed_pnls: dict[strategy_id, StrategyPnL] from PnLAttributor
  - daily_returns: optional dict[strategy_id, pd.Series] for rolling Sharpe
  - conflicts: count of opposite-sign-on-same-symbol intent collisions
  - n_intents: total intents processed in this cycle (denominator)
  - max_pair_corr: max pairwise correlation across strategy returns
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.metrics import (
    portfolio_conflicts_rate,
    portfolio_correlation_max,
    portfolio_gross_leverage,
    portfolio_net_leverage,
    strategy_allocation,
    strategy_attributed_pnl_usd,
    strategy_attributed_sharpe,
    strategy_exposure_mult,
)

logger = logging.getLogger(__name__)


# 252 trading days/year — annualization for the rolling Sharpe gauge.
# Same convention used by realized vol (CL-6h6) and walk-forward
# analytics, so the unit on Grafana stays consistent.
_TRADING_DAYS_PER_YEAR: int = 252


def _safe_set(label_metric: Any, value: float, **labels: str) -> None:
    """Best-effort gauge set — never raise into the caller."""
    try:
        if labels:
            label_metric.labels(**labels).set(float(value))
        else:
            label_metric.set(float(value))
    except Exception:
        logger.debug("Failed to set gauge", exc_info=True)


def _annualized_sharpe(returns: pd.Series) -> float:
    if returns is None or len(returns) < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(returns.mean() / sd * np.sqrt(_TRADING_DAYS_PER_YEAR))


def emit_portfolio_metrics(
    allocations: dict[str, float],
    exposure_multipliers: dict[str, float],
    positions: list[Any],
    equity: float,
    attributed_pnls: dict[str, Any] | None = None,
    daily_returns: dict[str, pd.Series] | None = None,
    conflicts: int = 0,
    n_intents: int = 0,
    max_pair_corr: float | None = None,
) -> None:
    """Emit all portfolio-level gauges from one rebalance snapshot."""
    # Leverage: sum of |notional| over equity. Position notional is
    # quantity × avg_price (stored on Position); we fall back to
    # quantity alone if avg_price is missing (shouldn't happen, but
    # broker drift is a real failure mode and we want a number not
    # a stack trace on the dashboard).
    gross = 0.0
    net = 0.0
    for p in positions:
        qty = float(getattr(p, "quantity", 0.0))
        px = float(getattr(p, "avg_price", 0.0) or 0.0)
        notional = qty * (px if px else 1.0)
        gross += abs(notional)
        net += notional
    if equity > 0:
        _safe_set(portfolio_gross_leverage, gross / equity)
        _safe_set(portfolio_net_leverage, net / equity)

    for sid, w in allocations.items():
        _safe_set(strategy_allocation, w, strategy=sid)
    for sid, m in exposure_multipliers.items():
        _safe_set(strategy_exposure_mult, m, strategy=sid)

    if attributed_pnls:
        for sid, pnl in attributed_pnls.items():
            total = float(getattr(pnl, "realized", 0)) + float(
                getattr(pnl, "unrealized", 0),
            )
            _safe_set(strategy_attributed_pnl_usd, total, strategy=sid)

    if daily_returns:
        for sid, series in daily_returns.items():
            _safe_set(
                strategy_attributed_sharpe,
                _annualized_sharpe(series),
                strategy=sid,
            )

    if n_intents > 0:
        _safe_set(portfolio_conflicts_rate, conflicts / n_intents)
    elif conflicts == 0:
        _safe_set(portfolio_conflicts_rate, 0.0)

    if max_pair_corr is not None:
        _safe_set(portfolio_correlation_max, max_pair_corr)
