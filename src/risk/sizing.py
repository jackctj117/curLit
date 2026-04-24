"""Position sizing — volatility-targeted, Kelly fraction, risk-parity."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionSizer:
    @staticmethod
    def fixed_fractional(
        capital: float, risk_pct: float, stop_distance: float, price: float,
    ) -> float:
        """Size a position risking fixed % of capital on a stop-loss.
        
        Args:
            capital: total account equity in currency units
            risk_pct: fraction of capital to risk (e.g. 0.01 = 1%)
            stop_distance: absolute distance from entry to stop in price units
            price: current entry price
            
        Returns:
            number of units to trade
        """
        assert capital > 0, f"capital must be positive, got {capital}"
        assert 0 < risk_pct <= 1, f"risk_pct must be in (0, 1], got {risk_pct}"
        assert stop_distance > 0, f"stop_distance must be positive, got {stop_distance}"
        assert price > 0, f"price must be positive, got {price}"

        risk_amount = capital * risk_pct
        size = risk_amount / stop_distance
        # cap at 100% of capital for sanity
        max_size = capital / price
        return min(size, max_size)

    @staticmethod
    def volatility_target(
        capital: float, target_vol: float, realized_vol: float, price: float,
    ) -> float:
        """Size a position to contribute target_vol annualized vol to portfolio.
        
        Uses the relationship: position_vol = notional_vol / equity
        where notional_vol = realized_vol (annualized) of the pair.
        
        target_vol = 0.10 (10% annualized) is typical for a single strategy leg
        per Architecture doc Section 5.1. Max 20% per position per RiskManager.
        """
        assert capital > 0, f"capital must be positive, got {capital}"
        assert 0 < target_vol <= 0.50, f"target_vol implausible: {target_vol}"
        assert price > 0, f"price must be positive, got {price}"

        if realized_vol <= 0:
            logger.warning("volatility_target: realized_vol=%.4f, returning 0", realized_vol)
            return 0.0

        notional = capital * target_vol / realized_vol
        size = notional / price

        logger.debug("vol_target: equity=%.0f target_vol=%.2f rv=%.2f price=%.4f -> size=%.0f",
                      capital, target_vol, realized_vol, price, size)
        return size

    @staticmethod
    def kelly(edge: float, odds: float, kelly_fraction: float = 0.25) -> float:
        """Kelly-inspired position sizing fraction.
        
        Full Kelly: f* = edge - (1 - edge) / odds
        Half Kelly produces 75% of growth rate with 25% of drawdown risk (Thorp 1997).
        We default to 0.25 (quarter Kelly) for survival: 
        higher retention of growth with dramatically lower ruin probability.
        """
        assert 0 <= edge <= 1, f"edge must be in [0, 1], got {edge}"
        assert odds >= 0, f"odds must be non-negative, got {odds}"
        assert 0 <= kelly_fraction <= 1, f"fraction must be in [0, 1], got {kelly_fraction}"

        if odds <= 0:
            logger.warning("kelly: odds=%.3f, returning 0", odds)
            return 0.0

        full_kelly = edge - (1.0 - edge) / odds
        result = max(0.0, full_kelly * kelly_fraction)

        logger.debug("kelly: edge=%.3f odds=%.2f full=%.3f frac=%.3f", 
                      edge, odds, full_kelly, result)
        assert 0.0 <= result <= 1.0, f"kelly result {result} out of [0,1]"
        return result

    @staticmethod
    def risk_parity_weights(cov_matrix: pd.DataFrame) -> pd.Series:
        """Compute equal risk contribution weights via SLSQP optimization.
        
        Each asset contributes 1/n of total portfolio risk.
        Adapted from Maillard, Roncalli, Teiletche (2010) 'On the Property
        of Equally Weighted Risk Contribution Portfolios'.
        """
        assert len(cov_matrix) >= 2, "need at least 2 assets for risk parity"
        assert all(d > 0 for d in np.diag(cov_matrix.values)), "variances must be positive"

        from scipy.optimize import minimize

        n = len(cov_matrix)
        cov_values = cov_matrix.values

        def _risk_contribution(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
            portfolio_var = w @ cov @ w
            assert portfolio_var > 0, f"zero portfolio variance with weights {w}"
            marginal = cov @ w
            return w * marginal / np.sqrt(portfolio_var)

        def _objective(w: np.ndarray) -> float:
            rc = _risk_contribution(w, cov_values)
            target = 1.0 / n
            return float(((rc - target) ** 2).sum())

        result = minimize(
            _objective,
            x0=np.ones(n) / n,
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
        )

        assert result.success, f"risk parity optimization failed: {result.message}"

        weights = pd.Series(result.x, index=cov_matrix.index)
        logger.info("Risk parity: %d assets, loss=%.6f", n, _objective(result.x))

        # Verify equal contribution property
        rc = _risk_contribution(result.x, cov_values)
        max_diff = float(np.abs(rc - rc.mean()).max())
        logger.debug("Risk parity max contribution diff: %.6f", max_diff)

        return weights
