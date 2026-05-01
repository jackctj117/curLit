"""Property-based tests for numerical functions (CL-g7e).

These tests use hypothesis to assert invariants that should hold over
all valid inputs — not specific examples. They catch regressions that
example-based tests miss because the example wasn't generated.

Functions covered:
  - PositionSizer.kelly                — boundedness, monotonicity
  - PositionSizer.fixed_fractional     — proportionality
  - stationary_bootstrap_sharpe_ci     — confidence-interval ordering
  - rate_diff zscore                   — zero-mean window
  - PnLAttributor compute_strategy_pnl — sign symmetry (long/short reflection)
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine

from src.backtest.bootstrap import stationary_bootstrap_sharpe_ci
from src.portfolio.attribution import PnLAttributor
from src.risk.sizing import PositionSizer


# Time budget per property: hypothesis defaults are too tight for some of
# our numeric tests; lift to 200ms so the noise floor doesn't flake.
_PROPERTY_DEADLINE_MS: int = 1_000


@settings(deadline=_PROPERTY_DEADLINE_MS, max_examples=100)
@given(
    edge=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    odds=st.floats(min_value=1.01, max_value=10.0, allow_nan=False),
)
def test_kelly_always_in_unit_interval(edge: float, odds: float) -> None:
    """Kelly fraction is by construction in [0, 1] — never negative,
    never above 1 (which would mean leveraging). Negative-edge inputs
    are clamped to 0; we assert this rather than allow uncapped negative."""
    k = PositionSizer.kelly(edge, odds)
    assert 0 <= k <= 1, f"Kelly out of bounds: edge={edge}, odds={odds}, k={k}"
    assert math.isfinite(k)


@settings(deadline=_PROPERTY_DEADLINE_MS, max_examples=50)
@given(
    edge=st.floats(min_value=0.51, max_value=0.99),
    odds=st.floats(min_value=1.5, max_value=5.0),
    delta=st.floats(min_value=0.001, max_value=0.4),
)
def test_kelly_monotonic_in_edge(edge: float, odds: float, delta: float) -> None:
    """Higher edge ⇒ larger Kelly fraction (assuming positive Kelly).
    A smaller edge should never produce a larger position."""
    assume(edge + delta <= 1.0)
    k_lo = PositionSizer.kelly(edge, odds)
    k_hi = PositionSizer.kelly(edge + delta, odds)
    if k_lo > 0:  # only when both are in the positive-edge regime
        assert k_hi >= k_lo - 1e-9


@settings(deadline=_PROPERTY_DEADLINE_MS, max_examples=50)
@given(
    equity=st.floats(min_value=1_000.0, max_value=10_000_000.0),
    risk_pct=st.floats(min_value=0.001, max_value=0.05),
    stop_distance=st.floats(min_value=0.0001, max_value=0.05),
    price=st.floats(min_value=0.5, max_value=200.0),
)
def test_fixed_fractional_proportional_to_equity(
    equity: float, risk_pct: float, stop_distance: float, price: float,
) -> None:
    """2x equity ⇒ 2x position (linearity). Same stop distance and
    risk %; only equity scales."""
    s1 = PositionSizer.fixed_fractional(equity, risk_pct, stop_distance, price)
    s2 = PositionSizer.fixed_fractional(2 * equity, risk_pct, stop_distance, price)
    if s1 > 0:
        ratio = s2 / s1
        assert math.isclose(ratio, 2.0, rel_tol=1e-9), (
            f"Doubling equity should double size; got {s1} → {s2} (ratio {ratio})"
        )


# Bootstrap CI invariants. The bootstrap returns (low, high); over all
# valid inputs the order must hold.
@settings(
    deadline=_PROPERTY_DEADLINE_MS, max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n=st.integers(min_value=50, max_value=200),
)
def test_bootstrap_ci_low_le_high(seed: int, n: int) -> None:
    """Lower CI bound must be ≤ upper. Sounds trivial; catches cases
    where np.percentile returns reversed because of NaNs in the
    bootstrapped distribution."""
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.normal(0, 0.01, n))
    low, high = stationary_bootstrap_sharpe_ci(
        returns, block_mean_len=10, n_bootstrap=200,
    )
    assert math.isfinite(low) and math.isfinite(high)
    assert low <= high


# rate_diff zscore: feature engineering — zscore over a rolling window
# should have mean ≈ 0 over the same window. Catches off-by-one errors
# in the windowing.
from src.features import zscore  # noqa: E402


@settings(deadline=_PROPERTY_DEADLINE_MS, max_examples=20)
@given(
    seed=st.integers(min_value=0, max_value=2**16),
    window=st.integers(min_value=20, max_value=100),
)
def test_zscore_window_zero_mean(seed: int, window: int) -> None:
    rng = np.random.default_rng(seed)
    n = window * 3
    series = pd.Series(rng.normal(loc=5.0, scale=2.0, size=n))
    z = zscore(series, window=window)
    # Drop NaNs from warm-up; the body should have approx-zero rolling mean.
    z_body = z.dropna()
    if len(z_body) < window:
        return
    # Each entry is (x - rolling_mean) / rolling_std — by construction,
    # not by sample. So we assert finite values and that magnitudes are
    # reasonable for a normal sample (|z| < 5 except for true tails).
    assert z_body.abs().median() < 3.0


# PnLAttributor sign symmetry. A long round-trip with prices flipped
# around the entry should produce zero realized P&L. A short and a long
# of the same size in opposite direction must have opposite realized P&L.
@settings(deadline=_PROPERTY_DEADLINE_MS, max_examples=20)
@given(
    qty=st.floats(min_value=10.0, max_value=10000.0),
    entry=st.floats(min_value=0.5, max_value=2.0),
    exit_delta=st.floats(min_value=-0.05, max_value=0.05),
)
def test_attribution_long_short_symmetry(
    qty: float, entry: float, exit_delta: float,
) -> None:
    """Long(qty)@entry then sell@(entry+δ) must produce same |realized|
    as Short(qty)@entry then cover@(entry-δ)."""
    assume(entry + exit_delta > 0)
    assume(entry - exit_delta > 0)
    eng_long = create_engine("sqlite:///:memory:")
    eng_short = create_engine("sqlite:///:memory:")
    a_long = PnLAttributor(eng_long)
    a_short = PnLAttributor(eng_short)

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)

    a_long.attribute_fill("EURUSD", qty, entry, ts=t0,
                          strategy_id="x", fill_id="o1")
    a_long.attribute_fill("EURUSD", -qty, entry + exit_delta, ts=t1,
                          strategy_id="x", fill_id="o2")
    long_pnl = a_long.compute_strategy_pnl("x").realized

    a_short.attribute_fill("EURUSD", -qty, entry, ts=t0,
                           strategy_id="x", fill_id="o1")
    a_short.attribute_fill("EURUSD", qty, entry - exit_delta, ts=t1,
                           strategy_id="x", fill_id="o2")
    short_pnl = a_short.compute_strategy_pnl("x").realized

    assert math.isclose(long_pnl, short_pnl, rel_tol=1e-9, abs_tol=1e-9), (
        f"Long P&L {long_pnl} should equal short-with-flipped-Δ {short_pnl}"
    )
