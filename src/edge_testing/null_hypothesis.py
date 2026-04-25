"""Null hypothesis framework (G1 / CL-0eo) — test strategy Sharpe vs 6 baselines.

Every strategy must beat ALL six null hypotheses at p<0.05 before it gets a live
allocation. The nulls are intentionally cheap-to-implement, well-understood
baselines that any "edgy" strategy should comfortably outperform:

    1. RANDOM_LONGSHORT       — random ±1 signal matching the strategy's turnover
    2. RANDOM_AUTOCORR        — random signal with the strategy's lag-1 autocorr
    3. BUY_AND_HOLD           — bootstrap of the underlying asset's Sharpe
    4. EQUAL_WEIGHT_BASKET    — bootstrap of equal-weight basket (optional inputs)
    5. SIMPLE_MOMENTUM        — sign of trailing 12-1 month return
    6. SIMPLE_CARRY           — highest yielding asset (optional yields input)

For each null we either generate a randomized null distribution (#1, #2) or
bootstrap a deterministic baseline (#3-#6). The strategy's Sharpe is compared
to the null distribution one-sided; p_value = P(null_sharpe >= strategy_sharpe).

A strategy fails the framework if ANY evaluated null returns p >= alpha.
Unevaluated nulls (missing optional inputs) are noted but don't gate edge_exists.

Reference: reference/14_edge_testing.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Type alias used internally — numpy ndarray with arbitrary dtype/shape. Mypy
# requires explicit type args on np.ndarray; this keeps the signatures readable.
_NDArray = np.ndarray[Any, Any]


# Significance level for the per-null edge test. 0.05 is the conventional
# cutoff; the multiple-testing-correction issue (CL-311 / G2) will replace
# this with Bonferroni / BH adjustment once it lands.
_DEFAULT_ALPHA: float = 0.05

# Default number of simulations for randomized + bootstrap nulls. 10000 is a
# good balance between distribution stability (CV(p) ~ 1/sqrt(n)) and runtime.
_DEFAULT_N_SIMULATIONS: int = 10_000

# Trading days per year for annualized Sharpe. Convention.
_TRADING_DAYS_PER_YEAR: int = 252

# Block mean length for stationary bootstrap. ~20 days = 1 month captures
# typical regime persistence in FX returns; aggressive enough to include
# stress events, mild enough to avoid IID bootstrap pitfalls.
_BOOTSTRAP_BLOCK_MEAN_LEN: int = 20

# Momentum: 12-1 month standard convention. 12-month total return excluding
# the most recent month, which has well-known reversal effects.
_MOMENTUM_LOOKBACK_MONTHS: int = 12
_MOMENTUM_SKIP_MONTHS: int = 1
_BUSINESS_DAYS_PER_MONTH: int = 21

# Minimum non-degenerate sample size for any bootstrap. Below this the Sharpe
# estimate is too noisy to compare against — we skip with a note.
_MIN_BOOTSTRAP_SAMPLE: int = 60


@dataclass
class NullResult:
    """One null hypothesis evaluation result."""

    null_name: str
    evaluated: bool
    strategy_sharpe: float
    p_value: float | None
    null_mean: float | None
    null_std: float | None
    n_simulations: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "null_name": self.null_name,
            "evaluated": self.evaluated,
            "strategy_sharpe": self.strategy_sharpe,
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "n_simulations": self.n_simulations,
            "note": self.note,
        }


@dataclass
class NullHypothesisReport:
    """Aggregated outcome across all 6 nulls plus the gate decision."""

    edge_exists: bool
    alpha: float
    strategy_sharpe: float
    results: dict[str, NullResult] = field(default_factory=dict)

    def passed_nulls(self) -> list[str]:
        return [
            name for name, r in self.results.items()
            if r.evaluated and r.p_value is not None and r.p_value < self.alpha
        ]

    def failed_nulls(self) -> list[str]:
        return [
            name for name, r in self.results.items()
            if r.evaluated and r.p_value is not None and r.p_value >= self.alpha
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_exists": self.edge_exists,
            "alpha": self.alpha,
            "strategy_sharpe": self.strategy_sharpe,
            "passed": self.passed_nulls(),
            "failed": self.failed_nulls(),
            "results": {n: r.to_dict() for n, r in self.results.items()},
        }


# =============================================================================
# Sharpe + helpers
# =============================================================================


def _sharpe(returns: _NDArray, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sharpe (zero risk-free assumed). Returns 0 on degenerate input."""
    if len(returns) == 0:
        return 0.0
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(returns) / sd * np.sqrt(periods_per_year))


def _signal_turnover(signal: _NDArray) -> float:
    """Fraction of bars where the signal changes sign or magnitude."""
    if len(signal) < 2:
        return 0.0
    diffs = np.diff(signal)
    return float(np.mean(diffs != 0))


def _signal_ar1(signal: _NDArray) -> float:
    """Lag-1 autocorrelation of the signal (returns 0 on degenerate input)."""
    if len(signal) < 3:
        return 0.0
    if np.std(signal) == 0:
        return 0.0
    return float(np.corrcoef(signal[:-1], signal[1:])[0, 1])


def _stationary_bootstrap_sharpes(
    returns: _NDArray,
    n_simulations: int,
    block_mean_len: int,
    rng: np.random.Generator,
) -> _NDArray:
    """Stationary block bootstrap of Sharpes. Reuses logic from src/backtest/bootstrap.py
    but uses an explicit RNG for reproducibility."""
    n = len(returns)
    if n < _MIN_BOOTSTRAP_SAMPLE:
        return np.full(n_simulations, np.nan)
    p = 1.0 / block_mean_len
    out = np.zeros(n_simulations)
    for b in range(n_simulations):
        indices = np.empty(n, dtype=np.int64)
        i = int(rng.integers(0, n))
        for k in range(n):
            indices[k] = i
            i = int(rng.integers(0, n)) if rng.random() < p else (i + 1) % n
        sample = returns[indices]
        out[b] = _sharpe(sample)
    return out


# =============================================================================
# Null generators
# =============================================================================


def _null_random_longshort(
    asset_returns: _NDArray,
    target_turnover: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> _NDArray:
    """Random ±1 signals at the target turnover. Returns a Sharpe per simulation."""
    n = len(asset_returns)
    out = np.zeros(n_simulations)
    target_turnover = max(0.0, min(1.0, target_turnover))
    for b in range(n_simulations):
        signal = np.empty(n, dtype=np.int8)
        # Initial direction is random.
        cur = 1 if rng.random() < 0.5 else -1
        signal[0] = cur
        for k in range(1, n):
            if rng.random() < target_turnover:
                cur = -cur
            signal[k] = cur
        strat_returns = signal * asset_returns
        out[b] = _sharpe(strat_returns)
    return out


def _null_random_autocorr(
    asset_returns: _NDArray,
    target_ar1: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> _NDArray:
    """AR(1) latent process thresholded to ±1, matching target lag-1 autocorr."""
    n = len(asset_returns)
    target_ar1 = max(-0.99, min(0.99, target_ar1))
    sigma_eps = float(np.sqrt(max(1.0 - target_ar1**2, 1e-6)))
    out = np.zeros(n_simulations)
    for b in range(n_simulations):
        x = np.empty(n)
        x[0] = float(rng.standard_normal())
        for k in range(1, n):
            x[k] = target_ar1 * x[k - 1] + sigma_eps * float(rng.standard_normal())
        signal = np.where(x >= 0, 1, -1)
        strat_returns = signal * asset_returns
        out[b] = _sharpe(strat_returns)
    return out


def _null_buy_and_hold(
    asset_returns: _NDArray,
    n_simulations: int,
    rng: np.random.Generator,
) -> _NDArray:
    return _stationary_bootstrap_sharpes(
        asset_returns, n_simulations, _BOOTSTRAP_BLOCK_MEAN_LEN, rng,
    )


def _null_equal_weight_basket(
    basket_returns: pd.DataFrame,
    n_simulations: int,
    rng: np.random.Generator,
) -> _NDArray:
    """Equal-weight portfolio of basket_returns columns, then bootstrap."""
    if basket_returns.empty or basket_returns.shape[1] < 2:
        return np.full(n_simulations, np.nan)
    portfolio = basket_returns.mean(axis=1).values
    return _stationary_bootstrap_sharpes(
        portfolio, n_simulations, _BOOTSTRAP_BLOCK_MEAN_LEN, rng,
    )


def _null_simple_momentum(
    asset_returns: pd.Series,
    n_simulations: int,
    rng: np.random.Generator,
) -> _NDArray:
    """Sign of trailing 12-1 month return drives ±1 position; bootstrap Sharpe."""
    lookback_days = _MOMENTUM_LOOKBACK_MONTHS * _BUSINESS_DAYS_PER_MONTH
    skip_days = _MOMENTUM_SKIP_MONTHS * _BUSINESS_DAYS_PER_MONTH
    if len(asset_returns) < lookback_days + skip_days + _MIN_BOOTSTRAP_SAMPLE:
        return np.full(n_simulations, np.nan)
    # Trailing return of the lookback window excluding the most recent skip days.
    cum = (1.0 + asset_returns).cumprod()
    trailing_return = cum.shift(skip_days) / cum.shift(lookback_days + skip_days) - 1
    signal = np.sign(trailing_return.values)
    # Strategy returns = signal × next-period asset return.
    strat_returns = (
        pd.Series(signal, index=asset_returns.index).shift(1)
        * asset_returns
    ).dropna().values
    return _stationary_bootstrap_sharpes(
        strat_returns, n_simulations, _BOOTSTRAP_BLOCK_MEAN_LEN, rng,
    )


def _null_simple_carry(
    asset_returns: _NDArray,
    yields: pd.DataFrame | None,
    n_simulations: int,
    rng: np.random.Generator,
) -> tuple[_NDArray, str]:
    """Carry baseline. With yields, signal = sign of the asset's yield differential.

    Without yields, fallback: highest realized 12-month return ⇒ +1 signal.
    Returns (sharpes, note).
    """
    if yields is None or yields.empty or yields.shape[1] < 2:
        # Fallback: trailing-12-month winner is +1, otherwise -1 (long the
        # higher-yielding side). Without true yield data this is more momentum-
        # adjacent than carry, but it's a sane proxy.
        n = len(asset_returns)
        lookback = 12 * _BUSINESS_DAYS_PER_MONTH
        if n < lookback + _MIN_BOOTSTRAP_SAMPLE:
            return np.full(n_simulations, np.nan), "fallback skipped — not enough history"
        rolling_mean = pd.Series(asset_returns).rolling(lookback).mean()
        signal = np.where(rolling_mean.values > 0, 1, -1)
        strat_returns = (signal * asset_returns)[lookback:]
        return _stationary_bootstrap_sharpes(
            strat_returns, n_simulations, _BOOTSTRAP_BLOCK_MEAN_LEN, rng,
        ), "fallback (no yield data; used 12m return sign)"
    # With yields: asset (column 0) yield minus quote (column 1) yield is the
    # carry score. If positive, hold long; if negative, short.
    if "base" in yields.columns and "quote" in yields.columns:
        diff = (yields["base"] - yields["quote"]).values
    else:
        diff = (yields.iloc[:, 0] - yields.iloc[:, 1]).values
    signal = np.where(diff > 0, 1, -1)
    strat_returns = signal * asset_returns
    return _stationary_bootstrap_sharpes(
        strat_returns, n_simulations, _BOOTSTRAP_BLOCK_MEAN_LEN, rng,
    ), "carry from yield differential"


# =============================================================================
# Framework
# =============================================================================


class NullHypothesisFramework:
    """Run all 6 null hypotheses against a strategy's realized returns."""

    def __init__(
        self,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        alpha: float = _DEFAULT_ALPHA,
        seed: int | None = None,
    ) -> None:
        assert n_simulations >= 100, (
            f"n_simulations must be >= 100, got {n_simulations}"
        )
        assert 0 < alpha < 1, f"alpha must be in (0, 1), got {alpha}"
        self.n_simulations = n_simulations
        self.alpha = alpha
        self._rng = np.random.default_rng(seed)

    def test_strategy(
        self,
        strategy_returns: pd.Series,
        asset_returns: pd.Series,
        signal: pd.Series | None = None,
        basket_returns: pd.DataFrame | None = None,
        yields: pd.DataFrame | None = None,
    ) -> NullHypothesisReport:
        """Test a strategy's Sharpe against the 6 nulls.

        Args:
            strategy_returns: realized strategy returns (daily fractional).
            asset_returns: the underlying asset's returns over the same window.
            signal: optional ±1 (or float) position signal series. If absent,
                    derived as sign(strategy_returns / asset_returns) so we can
                    estimate turnover and autocorrelation from observed flips.
            basket_returns: optional DataFrame for null #4 (equal-weight basket).
            yields: optional DataFrame with base/quote yield columns for null #6.
        """
        assert len(strategy_returns) > 0, "strategy_returns must be non-empty"
        assert len(strategy_returns) == len(asset_returns), (
            "strategy_returns and asset_returns must align in length"
        )

        strat_arr = strategy_returns.dropna().to_numpy()
        asset_arr = asset_returns.reindex(strategy_returns.index).fillna(0.0).to_numpy()
        strategy_sharpe = _sharpe(strat_arr)

        if signal is None:
            # Heuristic: when asset_returns ≠ 0, the sign of strat/asset is the
            # implied position (this is exact when strat = signal × asset).
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(asset_arr) > 1e-12, strat_arr / asset_arr, 0.0)
            signal_arr = np.sign(ratio)
        else:
            signal_arr = signal.reindex(strategy_returns.index).fillna(0.0).to_numpy()
            signal_arr = np.sign(signal_arr)

        results: dict[str, NullResult] = {}

        # Null 1: random long/short with same turnover.
        turnover = _signal_turnover(signal_arr)
        n1 = _null_random_longshort(asset_arr, turnover, self.n_simulations, self._rng)
        results["random_longshort"] = self._compile_result(
            "random_longshort", strategy_sharpe, n1,
            note=f"turnover={turnover:.3f}",
        )

        # Null 2: random with same lag-1 autocorr.
        ar1 = _signal_ar1(signal_arr)
        n2 = _null_random_autocorr(asset_arr, ar1, self.n_simulations, self._rng)
        results["random_autocorr"] = self._compile_result(
            "random_autocorr", strategy_sharpe, n2,
            note=f"ar1={ar1:.3f}",
        )

        # Null 3: buy-and-hold the asset.
        n3 = _null_buy_and_hold(asset_arr, self.n_simulations, self._rng)
        results["buy_and_hold"] = self._compile_result(
            "buy_and_hold", strategy_sharpe, n3, note="bootstrap of asset returns",
        )

        # Null 4: equal-weight basket (optional input).
        if basket_returns is not None and not basket_returns.empty:
            n4 = _null_equal_weight_basket(basket_returns, self.n_simulations, self._rng)
            results["equal_weight_basket"] = self._compile_result(
                "equal_weight_basket", strategy_sharpe, n4,
                note=f"basket size={basket_returns.shape[1]}",
            )
        else:
            results["equal_weight_basket"] = NullResult(
                null_name="equal_weight_basket",
                evaluated=False,
                strategy_sharpe=strategy_sharpe,
                p_value=None,
                null_mean=None,
                null_std=None,
                n_simulations=0,
                note="no basket_returns provided",
            )

        # Null 5: simple momentum (12-1).
        n5 = _null_simple_momentum(asset_returns, self.n_simulations, self._rng)
        results["simple_momentum"] = self._compile_result(
            "simple_momentum", strategy_sharpe, n5,
            note=f"{_MOMENTUM_LOOKBACK_MONTHS}-{_MOMENTUM_SKIP_MONTHS} month",
        )

        # Null 6: simple carry (uses yields if provided).
        n6, note6 = _null_simple_carry(
            asset_arr, yields, self.n_simulations, self._rng,
        )
        results["simple_carry"] = self._compile_result(
            "simple_carry", strategy_sharpe, n6, note=note6,
        )

        # Edge exists iff strategy beats every EVALUATED null at p < alpha.
        evaluated_p_values: list[float] = [
            r.p_value for r in results.values()
            if r.evaluated and r.p_value is not None
        ]
        edge_exists = bool(evaluated_p_values) and all(
            p < self.alpha for p in evaluated_p_values
        )

        return NullHypothesisReport(
            edge_exists=edge_exists,
            alpha=self.alpha,
            strategy_sharpe=strategy_sharpe,
            results=results,
        )

    def _compile_result(
        self,
        null_name: str,
        strategy_sharpe: float,
        null_distribution: _NDArray,
        note: str = "",
    ) -> NullResult:
        # Drop NaNs from degenerate sims (insufficient history, etc).
        valid = null_distribution[~np.isnan(null_distribution)]
        if len(valid) == 0:
            return NullResult(
                null_name=null_name,
                evaluated=False,
                strategy_sharpe=strategy_sharpe,
                p_value=None,
                null_mean=None,
                null_std=None,
                n_simulations=0,
                note=note + " (insufficient sims to compute p-value)" if note else
                "insufficient sims to compute p-value",
            )
        # One-sided p-value: probability under H0 of observing a Sharpe at
        # least as extreme as the strategy's. Add 1 to numerator + denominator
        # for the conservative "plus-one" continuity correction.
        p_value = float((np.sum(valid >= strategy_sharpe) + 1) / (len(valid) + 1))
        return NullResult(
            null_name=null_name,
            evaluated=True,
            strategy_sharpe=strategy_sharpe,
            p_value=p_value,
            null_mean=float(np.mean(valid)),
            null_std=float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            n_simulations=len(valid),
            note=note,
        )
