"""Risk parity weight computation — shared module used by PortfolioCoordinator + backtest.

Inverse-volatility risk parity solves for weights where each strategy contributes
equal absolute risk to the portfolio. Implementation:

    minimize    Σ_i (rc_i - mean(rc))²
    subject to  Σ w_i = 1
                bound[0] ≤ w_i ≤ bound[1]
    where       rc_i = w_i × (Σ w)_i / sqrt(w' Σ w)
                Σ    = annualized covariance matrix

Two output modes:
    target_total_vol=None  — weights sum to 1 (allocation only)
    target_total_vol=X     — weights scaled so portfolio annualized vol = X
                             (sum may exceed 1, representing leverage)

Reference: reference/06_portfolio.md.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _solve_risk_parity_unbounded(
    cov: np.ndarray[Any, Any],
    max_iter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray[Any, Any]:
    """Damped Spinu/Maillard-Roncalli iteration for equal-risk-contribution weights.

    At the risk-parity solution, w_i × (Σw)_i is constant across i, giving the
    fixed point w_i ∝ 1 / (Σw)_i. The undamped iteration can oscillate when the
    covariance has negative off-diagonal terms (which is common for diversified
    strategy returns); a damping factor of 0.5 stabilizes convergence at the
    cost of more iterations.

    Returns weights summing to 1, no bound enforcement (handled by caller).
    """
    # Initialize at inverse-vol — already close to the solution for diagonal-
    # dominant covariance, reducing iterations dramatically.
    diag_vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    w = (1.0 / diag_vols) / float((1.0 / diag_vols).sum())

    # Damping factor — 0.5 trades convergence speed for stability under mixed-
    # sign covariances. Too high (e.g. 1.0) oscillates; too low (e.g. 0.1) is
    # slow. 0.5 is the standard choice in Spinu (2013).
    damp = 0.5

    for _ in range(max_iter):
        marginal = cov @ w
        # Floor at a small positive value: directions where w' Σ has driven
        # marginal toward zero or negative are pushed back to positive territory.
        marginal = np.where(marginal > 1e-12, marginal, 1e-12)
        w_target = 1.0 / marginal
        w_target = w_target / float(w_target.sum())
        # Damped update.
        w_new = (1.0 - damp) * w + damp * w_target
        # Renormalize against floating-point drift in the convex combination.
        w_new = w_new / float(w_new.sum())
        if np.max(np.abs(w_new - w)) < tol:
            return np.asarray(w_new, dtype=float)
        w = w_new
    logger.debug("Risk parity iteration hit max_iter=%d without convergence", max_iter)
    return np.asarray(w, dtype=float)


def _project_to_bounds(
    w: np.ndarray[Any, Any],
    bounds: tuple[float, float],
) -> np.ndarray[Any, Any]:
    """Project w onto {x : sum(x) = 1, lower ≤ x_i ≤ upper}.

    Active-set method: at each iteration, redistribute the remaining budget
    proportionally among free weights, then fix the SINGLE most-violating
    weight at its bound. Fixing one at a time avoids the failure mode where
    several simultaneous fixes leave the budget unbalanced.

    Converges in at most n iterations. Caller must ensure feasibility
    (n*upper >= 1 and n*lower <= 1) — asserted in risk_parity_weights.
    """
    lower, upper = bounds
    n = len(w)
    out = np.asarray(w, dtype=float).copy()
    fixed = np.zeros(n, dtype=bool)

    for _ in range(n):
        free = ~fixed
        if not free.any():
            break

        # Distribute remaining budget proportionally to current free weights.
        budget = 1.0 - float(out[fixed].sum())
        free_sum = float(out[free].sum())
        if free_sum > 0:
            out[free] = out[free] * (budget / free_sum)
        else:
            out[free] = budget / float(free.sum())

        # Identify worst violator among free weights.
        free_idx = np.where(free)[0]
        upper_excess = out[free_idx] - upper
        lower_deficit = lower - out[free_idx]
        max_upper = float(upper_excess.max()) if len(upper_excess) > 0 else -np.inf
        max_lower = float(lower_deficit.max()) if len(lower_deficit) > 0 else -np.inf

        # No remaining bound violations among free weights — done.
        if max_upper <= 1e-12 and max_lower <= 1e-12:
            break

        if max_upper >= max_lower:
            # Fix the most-over-cap weight at upper.
            i_in_free = int(np.argmax(upper_excess))
            i = int(free_idx[i_in_free])
            out[i] = upper
        else:
            # Fix the most-under-floor weight at lower.
            i_in_free = int(np.argmax(lower_deficit))
            i = int(free_idx[i_in_free])
            out[i] = lower
        fixed[i] = True

    return out


# Trading days per year — used to annualize daily covariance.
_TRADING_DAYS_PER_YEAR: int = 252

# Default per-strategy bounds. Match coordinator defaults.
_DEFAULT_BOUNDS: tuple[float, float] = (0.05, 0.40)


def risk_parity_weights(
    returns: pd.DataFrame,
    cov_matrix: np.ndarray[Any, Any] | None = None,
    bounds: tuple[float, float] = _DEFAULT_BOUNDS,
    target_total_vol: float | None = None,
) -> pd.Series:
    """Solve for inverse-volatility risk parity weights.

    Args:
        returns: DataFrame with strategies as columns, time-series of daily
                 fractional returns as rows.
        cov_matrix: Optional pre-computed annualized covariance matrix. If
                    None, computes from `returns`.
        bounds: (min, max) per-strategy weight bounds applied to the optimizer.
        target_total_vol: If None (default), return weights normalized to sum
                          to 1. If a float, scale weights so that the resulting
                          portfolio's annualized vol equals this target — sum
                          may then exceed 1, representing leverage.

    Returns:
        pd.Series of weights indexed by strategy id.
    """
    assert not returns.empty, "returns must be non-empty"
    assert returns.shape[1] >= 1, "need at least one strategy column"
    assert bounds[0] >= 0, f"lower bound must be non-negative, got {bounds[0]}"
    assert bounds[1] > bounds[0], f"upper bound {bounds[1]} must exceed lower bound {bounds[0]}"
    if target_total_vol is not None:
        assert target_total_vol > 0, "target_total_vol must be positive"

    if cov_matrix is None:
        cov_matrix = returns.cov().values * _TRADING_DAYS_PER_YEAR

    n = len(cov_matrix)
    strategy_ids = returns.columns.tolist()
    assert n == len(strategy_ids), (
        f"covariance shape {n} doesn't match strategy count {len(strategy_ids)}"
    )

    # Single-strategy edge case — bypass optimizer + bound checks (trivially 1.0).
    if n == 1:
        weight = 1.0
        if target_total_vol is not None:
            single_vol = float(np.sqrt(cov_matrix[0, 0]))
            if single_vol > 0:
                weight = target_total_vol / single_vol
        return pd.Series([weight], index=strategy_ids)

    # Feasibility: sum-to-1 requires bounds to span 1.0 collectively.
    assert n * bounds[1] >= 1.0, (
        f"upper bound {bounds[1]} × {n} strategies = {n * bounds[1]:.2f} < 1.0; sum-to-1 infeasible"
    )
    assert n * bounds[0] <= 1.0, (
        f"lower bound {bounds[0]} × {n} strategies = {n * bounds[0]:.2f} > 1.0; sum-to-1 infeasible"
    )

    # Solve unbounded risk parity via robust iterative algorithm, then project
    # onto bounds. Iterative method handles the convex landscape better than
    # SLSQP, which fails on dense covariance matrices.
    weights_unbounded = _solve_risk_parity_unbounded(cov_matrix)
    weights_arr = _project_to_bounds(weights_unbounded, bounds)

    # Final normalization (defensive — projection should already give sum=1).
    total = float(weights_arr.sum())
    assert total > 0, "risk parity returned non-positive weight sum"
    weights_arr = weights_arr / total

    if target_total_vol is not None:
        portfolio_vol = float(np.sqrt(weights_arr @ cov_matrix @ weights_arr))
        if portfolio_vol > 0:
            scale = target_total_vol / portfolio_vol
            weights_arr = weights_arr * scale

    return pd.Series(weights_arr, index=strategy_ids)


def diagnose_allocation(
    weights: pd.Series,
    cov_matrix: np.ndarray[Any, Any],
    returns: pd.DataFrame,
) -> dict[str, Any]:
    """Compute (and log) per-strategy risk contribution diagnostics.

    Returns a dict with portfolio_vol, contributions (as Series), and the
    correlation matrix — useful for tests and dashboards.
    """
    w = weights.values.astype(float)
    portfolio_vol = float(np.sqrt(w @ cov_matrix @ w))

    if portfolio_vol == 0:
        contributions = pd.Series(np.zeros(len(weights)), index=weights.index)
    else:
        marginal = cov_matrix @ w
        contributions = pd.Series(
            w * marginal / portfolio_vol,
            index=weights.index,
        )

    correlation = returns.corr()

    logger.info(
        "Allocation diagnose: portfolio_vol=%.4f weights=%s contribs=%s",
        portfolio_vol,
        weights.round(4).to_dict(),
        contributions.round(6).to_dict(),
    )

    return {
        "portfolio_vol": portfolio_vol,
        "contributions": contributions,
        "correlation": correlation,
    }


def rolling_risk_parity_weights(
    returns: pd.DataFrame,
    window_days: int = _TRADING_DAYS_PER_YEAR,
    halflife_days: int = 60,
    target_vol: float | None = 0.10,
    refit_freq_days: int = 21,
    bounds: tuple[float, float] = _DEFAULT_BOUNDS,
) -> pd.DataFrame:
    """Compute time-varying risk parity weights using EWMA covariance.

    Re-fits at intervals of `refit_freq_days`. EWMA halflife controls how
    aggressively recent returns dominate the covariance estimate.

    Args:
        returns: DataFrame indexed by date, strategies as columns.
        window_days: Lookback window in trading days for each refit.
        halflife_days: EWMA halflife in trading days.
        target_vol: If set, scale weights to target portfolio vol. If None,
                    weights at each refit sum to 1.
        refit_freq_days: How often to refit (e.g. 21 = monthly).
        bounds: per-strategy weight bounds passed to the optimizer.

    Returns:
        DataFrame indexed by refit date, columns are strategy ids.
    """
    assert window_days > 0, "window_days must be positive"
    assert halflife_days > 0, "halflife_days must be positive"
    assert refit_freq_days > 0, "refit_freq_days must be positive"
    if target_vol is not None:
        assert target_vol > 0, "target_vol must be positive"

    if len(returns) <= window_days:
        logger.warning(
            "rolling_risk_parity_weights: insufficient history (%d <= %d)",
            len(returns),
            window_days,
        )
        return pd.DataFrame(columns=returns.columns)

    decay = float(np.log(2) / halflife_days)
    weights_history: list[dict[str, Any]] = []
    refit_indices = list(range(window_days, len(returns), refit_freq_days))

    for end_idx in refit_indices:
        start_idx = max(0, end_idx - window_days)
        window_returns = returns.iloc[start_idx:end_idx]

        # EWMA weights: most recent observation gets the largest weight.
        ages = np.arange(len(window_returns))[::-1]
        ewma = np.exp(-decay * ages)
        ewma /= ewma.sum()

        mean = (window_returns.values * ewma[:, None]).sum(axis=0)
        centered = window_returns.values - mean
        cov = (centered * ewma[:, None]).T @ centered * _TRADING_DAYS_PER_YEAR

        weights = risk_parity_weights(
            window_returns,
            cov_matrix=cov,
            bounds=bounds,
            target_total_vol=target_vol,
        )

        weights_history.append(
            {
                "date": returns.index[end_idx - 1],
                **weights.to_dict(),
            }
        )

    if not weights_history:
        return pd.DataFrame(columns=returns.columns)

    return pd.DataFrame(weights_history).set_index("date")
