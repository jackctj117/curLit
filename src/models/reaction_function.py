"""Reaction function models — Taylor-rule style for major central banks."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FedReactionFunction:
    def __init__(
        self,
        pce_target: float = 2.0,
        u_star: float = 4.2,
        r_star: float = 0.5,
    ) -> None:
        self.pce_target = pce_target
        self.u_star = u_star
        self.r_star = r_star
        self.w_core_pce = 0.6
        self.w_unemployment = 0.3
        self.w_fci = 0.1

    def implied_rate(self, core_pce: float, unemployment: float, fci: float = 0.0) -> float:
        inflation_gap = core_pce - self.pce_target
        unemployment_gap = self.u_star - unemployment
        return max(
            0.0,
            self.r_star + core_pce
            + self.w_core_pce * 1.5 * inflation_gap
            + self.w_unemployment * unemployment_gap
            + self.w_fci * (0.0 - fci),
        )

    def fit(self, historical: dict[str, list[float]]) -> None:
        from scipy.optimize import minimize

        df = historical
        actual = np.array(df["fed_funds"])

        def loss(params: np.ndarray) -> float:
            self.w_core_pce, self.w_unemployment, self.w_fci = params
            pred = np.array([
                self.implied_rate(pce, u, fci)
                for pce, u, fci in zip(df["core_pce"], df["unemployment"], df["fci"], strict=False)
            ])
            return float(((pred - actual) ** 2).sum() / len(actual))

        result = minimize(loss, x0=[0.6, 0.3, 0.1], bounds=[(0, 2), (0, 2), (0, 1)])
        self.w_core_pce, self.w_unemployment, self.w_fci = result.x
        logger.info("Fed RF calibrated: w_pce=%.3f w_u=%.3f w_fci=%.3f",
                      self.w_core_pce, self.w_unemployment, self.w_fci)
