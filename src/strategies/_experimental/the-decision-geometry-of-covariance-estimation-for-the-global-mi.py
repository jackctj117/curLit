"""Long-only GMVP over a G10 FX basket with constant-correlation shrinkage.

Implements hypothesis
"the-decision-geometry-of-covariance-estimation-for-the-global-mi":
a monthly-rebalanced Global Minimum-Variance Portfolio (long-only in
*currency* space, weights sum to 1) over seven G10 currencies vs USD.
The covariance matrix of daily USD-numeraire log returns is estimated by
shrinking the sample covariance toward a constant-correlation target
(Ledoit and Wolf, 2004, "Honey, I Shrunk the Sample Covariance Matrix"),
with the shrinkage intensity computed analytically (no extra free
parameter). Setting use_shrinkage=False reproduces the pre-registered
sample-covariance baseline for the abandon-condition comparison.

Quoting-convention handling (per the brief): for pairs quoted XXXUSD
(EURUSD, GBPUSD, AUDUSD, NZDUSD) the price *is* the USD value of one
unit of foreign currency; for pairs quoted USDXXX (USDCAD, USDJPY,
USDCHF) the USD value of one unit of foreign currency is 1 / price.
Covariance is estimated on log returns of these USD values. A
currency-space long weight w on CAD/JPY/CHF is mapped to a pair-space
position of -w on USDCAD/USDJPY/USDCHF (exact replication in log-return
space, first-order exact in simple returns).

Joint multi-asset mode (CL-40n2 v2): generate_signals returns a
DataFrame whose columns are the seven tradeable pairs and whose values
are per-bar position weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Pairs where USD is the QUOTE currency: price = USD per 1 unit foreign.
_USD_QUOTE_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD")
# Pairs where USD is the BASE currency: price = foreign units per 1 USD.
_USD_BASE_PAIRS = ("USDCAD", "USDJPY", "USDCHF")

_ALL_PAIRS = _USD_QUOTE_PAIRS + _USD_BASE_PAIRS
_N_ASSETS = len(_ALL_PAIRS)

# Sign mapping from currency-space long weights to pair-space positions:
# +1 for XXXUSD pairs (long pair == long foreign currency), -1 for
# USDXXX pairs (short pair == long foreign currency).
_PAIR_SIGN = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

# Fixed numerical constants (documented, NOT tunable hyperparameters):
_MIN_HISTORY_DAYS = 60  # below this, fall back to equal currency weights
_RIDGE = 1e-10  # diagonal loading for numerical stability


@dataclass
class GMVPShrinkageConfig:
    """Configuration for the shrinkage-GMVP strategy.

    Free parameters (4 total):

    lookback_days : int, default 252
        Trailing window of daily returns used to estimate the covariance
        matrix at each monthly rebalance. Acceptable range: [126, 504].
    use_shrinkage : bool, default True
        If True, shrink the sample covariance toward the
        constant-correlation target (the hypothesis strategy). If False,
        use the raw sample covariance (the pre-registered baseline).
    shrinkage_intensity : Optional[float], default None
        If None, the shrinkage intensity is estimated analytically via
        the Ledoit-Wolf (2004) formula at every rebalance. If a float is
        supplied it is used as a fixed intensity. Acceptable range:
        [0.0, 1.0]. Ignored when use_shrinkage is False.
    max_weight : float, default 0.60
        Per-currency cap on the long-only GMVP weight. Acceptable range:
        [1/7 (~0.143), 1.0]; must satisfy max_weight * 7 >= 1 so the
        fully-invested constraint stays feasible.
    """

    lookback_days: int = 252
    use_shrinkage: bool = True
    shrinkage_intensity: float | None = None
    max_weight: float = 0.60


class Strategy:
    """Monthly-rebalanced long-only GMVP over G10 currencies vs USD."""

    id = "the-decision-geometry-of-covariance-estimation-for-the-global-mi"

    # Class-level list of full pair codes (harness reads this off the
    # class without instantiating).
    symbols = [
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
        "NZDUSD",
        "USDCAD",
        "USDJPY",
        "USDCHF",
    ]

    # Joint multi-asset strategy: generate_signals returns a DataFrame
    # over all pairs; execution_symbol set for protocol completeness.
    execution_symbol = "EURUSD"

    def __init__(self, config: GMVPShrinkageConfig | None = None):
        self.config = config if config is not None else GMVPShrinkageConfig()
        cfg = self.config
        if not 20 <= cfg.lookback_days <= 2000:
            raise ValueError("lookback_days out of plausible range [20, 2000]")
        if cfg.shrinkage_intensity is not None and not (0.0 <= cfg.shrinkage_intensity <= 1.0):
            raise ValueError("shrinkage_intensity must be in [0, 1]")
        if not 1.0 / _N_ASSETS <= cfg.max_weight <= 1.0:
            raise ValueError("max_weight must be in [1/7, 1.0]")
        # Tail of in-sample USD-value log returns, stored by fit() so the
        # first out-of-sample rebalance has a full covariance window.
        self._hist_rets: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.DataFrame) -> None:
        """Cache the tail of in-sample returns for warm-starting.

        There are no fitted statistical parameters: covariance is
        re-estimated on a trailing window at each rebalance, and the
        Ledoit-Wolf intensity is analytic. fit() only (a) validates the
        input columns and (b) stores the last lookback_days rows of
        in-sample USD-value log returns so the first out-of-sample
        rebalance does not fall back to equal weights. Only rows with
        timestamps strictly before the test window are ever reused
        (re-enforced in generate_signals), so there is no look-ahead.
        """
        self._validate_columns(train_data)
        rets = self._usd_value_log_returns(train_data).dropna(how="any")
        self._hist_rets = rets.tail(self.config.lookback_days + 5)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return per-bar position weights for the seven pairs.

        At the first bar of each calendar month, the long-only GMVP is
        re-solved on the trailing lookback_days of daily USD-value log
        returns using information *strictly before* that bar
        (shift-by-one), then held constant until the next month. Output
        values are pair-space weights: currency-space long weights
        multiplied by the quoting-convention sign (+1 for XXXUSD, -1 for
        USDXXX). Gross exposure is 1 on every bar, so signals are
        non-zero on 100 percent of bars.
        """
        self._validate_columns(data)
        cfg = self.config
        idx = data.index

        rets = self._usd_value_log_returns(data)

        # Prepend in-sample history strictly before the test window so
        # early rebalances have a full estimation window. No look-ahead:
        # every prepended timestamp precedes idx[0].
        if self._hist_rets is not None and len(idx) > 0:
            prior = self._hist_rets[self._hist_rets.index < idx[0]]
            full_rets = pd.concat([prior.tail(cfg.lookback_days), rets])
        else:
            full_rets = rets
        offset = len(full_rets) - len(rets)
        ret_values = full_rets.to_numpy(dtype=float)

        # Rebalance at the first bar of each calendar month.
        months = idx.to_period("M")
        is_rebal = np.ones(len(idx), dtype=bool)
        if len(idx) > 1:
            is_rebal[1:] = months[1:] != months[:-1]

        rows = []
        stamps = []
        for t in np.flatnonzero(is_rebal):
            # History strictly before bar t (shift-by-one: the weight at
            # t uses returns through t-1 only).
            hist = ret_values[: offset + t]
            hist = hist[np.isfinite(hist).all(axis=1)]
            if len(hist) > cfg.lookback_days:
                hist = hist[-cfg.lookback_days :]
            if len(hist) >= _MIN_HISTORY_DAYS:
                sigma = self._estimate_cov(hist)
                if np.isfinite(sigma).all():
                    w_ccy = self._solve_long_only_gmvp(sigma)
                else:
                    w_ccy = self._equal_weights()
            else:
                w_ccy = self._equal_weights()
            rows.append(w_ccy * _PAIR_SIGN)
            stamps.append(idx[t])

        weights = pd.DataFrame(rows, index=stamps, columns=list(_ALL_PAIRS))
        weights = weights.reindex(idx).ffill()
        return weights

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(data: pd.DataFrame) -> None:
        missing = [p for p in _ALL_PAIRS if p not in data.columns]
        if missing:
            raise ValueError("missing required symbol columns: " + str(missing))

    @staticmethod
    def _usd_value_log_returns(data: pd.DataFrame) -> pd.DataFrame:
        """Daily log returns of the USD value of each foreign currency."""
        px = data.loc[:, list(_ALL_PAIRS)].astype(float)
        usd_val = px.copy()
        for pair in _USD_BASE_PAIRS:
            usd_val[pair] = 1.0 / px[pair]
        usd_val = usd_val.where(usd_val > 0)
        return np.log(usd_val).diff()

    @staticmethod
    def _equal_weights() -> np.ndarray:
        return np.full(_N_ASSETS, 1.0 / _N_ASSETS)

    def _estimate_cov(self, hist: np.ndarray) -> np.ndarray:
        """Sample covariance, optionally shrunk to constant correlation."""
        x = hist - hist.mean(axis=0, keepdims=True)
        t_obs = len(x)
        sample = x.T @ x / t_obs  # T-normalized, per Ledoit-Wolf (2004)
        if not self.config.use_shrinkage:
            return sample
        target, rbar = self._constant_corr_target(sample)
        if self.config.shrinkage_intensity is not None:
            delta = float(self.config.shrinkage_intensity)
        else:
            delta = self._ledoit_wolf_intensity(x, sample, target, rbar)
        return delta * target + (1.0 - delta) * sample

    @staticmethod
    def _constant_corr_target(sample: np.ndarray):
        """Constant-correlation target F and mean off-diagonal corr."""
        d = np.sqrt(np.clip(np.diag(sample), 1e-16, None))
        denom = np.outer(d, d)
        corr = sample / denom
        n = sample.shape[0]
        rbar = (corr.sum() - n) / (n * (n - 1))
        target = rbar * denom
        np.fill_diagonal(target, np.diag(sample))
        return target, rbar

    @staticmethod
    def _ledoit_wolf_intensity(
        x: np.ndarray, sample: np.ndarray, target: np.ndarray, rbar: float
    ) -> float:
        """Analytic optimal shrinkage intensity, Ledoit-Wolf (2004)."""
        t_obs, n = x.shape
        # dev[t, i, j] = x_it * x_jt - s_ij   (T x n x n; small: n = 7)
        prod = x[:, :, None] * x[:, None, :]
        dev = prod - sample[None, :, :]
        pi_mat = (dev**2).mean(axis=0)
        pi_hat = pi_mat.sum()
        # theta_ii,ij and theta_jj,ij (dev is symmetric in i, j).
        diag_dev = dev[:, np.arange(n), np.arange(n)]  # x_it^2 - s_ii
        theta_i = np.einsum("ti,tij->ij", diag_dev, dev) / t_obs
        theta_j = theta_i.T
        d = np.sqrt(np.clip(np.diag(sample), 1e-16, None))
        ratio = np.outer(1.0 / d, d)  # [i, j] = sqrt(s_jj / s_ii)
        off = ~np.eye(n, dtype=bool)
        rho_hat = np.trace(pi_mat) + (rbar / 2.0) * (
            (ratio * theta_i + ratio.T * theta_j)[off].sum()
        )
        gamma_hat = ((sample - target) ** 2).sum()
        if gamma_hat < 1e-20:
            return 1.0  # sample == target; intensity is irrelevant
        kappa = (pi_hat - rho_hat) / gamma_hat
        return float(np.clip(kappa / t_obs, 0.0, 1.0))

    def _solve_long_only_gmvp(self, sigma: np.ndarray) -> np.ndarray:
        """Long-only, fully-invested minimum-variance weights."""
        n = sigma.shape[0]
        sigma = 0.5 * (sigma + sigma.T) + _RIDGE * np.eye(n)
        ub = self.config.max_weight
        w0 = self._equal_weights()

        def objective(w):
            return float(w @ sigma @ w)

        def gradient(w):
            return 2.0 * (sigma @ w)

        def budget(w):
            return float(w.sum() - 1.0)

        def budget_jac(w):
            return np.ones(n)

        constraint = dict(type="eq", fun=budget, jac=budget_jac)
        result = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, ub)] * n,
            constraints=[constraint],
            options=dict(maxiter=200, ftol=1e-12),
        )
        if result.success and np.isfinite(result.x).all():
            w = np.clip(result.x, 0.0, ub)
        else:
            # Fallback: clipped closed-form GMVP, then renormalize.
            try:
                raw = np.linalg.solve(sigma, np.ones(n))
            except np.linalg.LinAlgError:
                raw = np.ones(n)
            w = np.clip(raw, 0.0, None)
            if w.sum() <= 0.0 or not np.isfinite(w).all():
                w = np.ones(n)
        w = w / w.sum()
        # Enforce the cap after normalization by redistributing excess.
        for _ in range(20):
            over = w > ub + 1e-12
            if not over.any():
                break
            excess = float((w[over] - ub).sum())
            w[over] = ub
            under = ~over
            under_sum = float(w[under].sum())
            if under_sum > 0.0:
                w[under] += excess * w[under] / under_sum
            else:
                w[under] += excess / max(int(under.sum()), 1)
        return w / w.sum()
