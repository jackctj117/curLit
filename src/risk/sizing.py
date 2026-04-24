"""Position sizing — volatility-targeted, Kelly fraction, risk-parity."""

import numpy as np
import pandas as pd


class PositionSizer:
    @staticmethod
    def fixed_fractional(
        capital: float, risk_pct: float, stop_distance: float, price: float,
    ) -> float:
        risk_amount = capital * risk_pct
        return risk_amount / stop_distance

    @staticmethod
    def volatility_target(
        capital: float, target_vol: float, realized_vol: float, price: float,
    ) -> float:
        if realized_vol <= 0:
            return 0.0
        notional = capital * target_vol / realized_vol
        return notional / price

    @staticmethod
    def kelly(edge: float, odds: float, kelly_fraction: float = 0.25) -> float:
        full_kelly = edge - (1 - edge) / odds if odds > 0 else 0.0
        return max(0.0, full_kelly * kelly_fraction)

    @staticmethod
    def risk_parity_weights(cov_matrix: pd.DataFrame) -> pd.Series:
        from scipy.optimize import minimize

        n = len(cov_matrix)
        cov_values = cov_matrix.values

        def risk_contribution(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
            portfolio_var = w @ cov @ w
            marginal = cov @ w
            return w * marginal / np.sqrt(portfolio_var)

        def objective(w: np.ndarray) -> float:
            rc = risk_contribution(w, cov_values)
            target = 1.0 / n
            return float(((rc - target) ** 2).sum())

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
        )
        return pd.Series(result.x, index=cov_matrix.index)
