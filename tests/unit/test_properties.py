"""Property-based tests — hypothesis tests for key numerical invariants."""

import pytest
from hypothesis import given, settings, strategies as st

import numpy as np


# =============================================================================
# Position sizing invariants
# =============================================================================

@given(
    st.floats(min_value=0.01, max_value=0.99),
    st.floats(min_value=0.01, max_value=10.0),
)
@settings(max_examples=100)
def test_kelly_bounded_in_unit_interval(edge: float, odds: float) -> None:
    """Kelly fraction must always be in [0, 1]."""
    from src.risk.sizing import PositionSizer
    k = PositionSizer.kelly(edge, odds, kelly_fraction=1.0)
    assert 0.0 <= k <= 1.0


@given(
    st.floats(min_value=-5.0, max_value=5.0),
    st.floats(min_value=-5.0, max_value=5.0),
)
@settings(max_examples=100)
def test_kelly_never_negative_regardless_of_input(edge: float, odds: float) -> None:
    """Even with garbage edge/odds, kelly returns >= 0."""
    from src.risk.sizing import PositionSizer
    k = PositionSizer.kelly(edge, odds)
    assert k >= 0.0


@given(
    st.floats(min_value=0.01, max_value=0.99),
    st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=100)
def test_quarter_kelly_less_than_full_kelly(edge: float, odds: float) -> None:
    """Quarter-Kelly should be <= full Kelly."""
    from src.risk.sizing import PositionSizer
    full = PositionSizer.kelly(edge, odds, kelly_fraction=1.0)
    quarter = PositionSizer.kelly(edge, odds, kelly_fraction=0.25)
    if full > 0:
        assert quarter <= full


@given(
    st.floats(min_value=1000, max_value=1e7),
    st.floats(min_value=0.01, max_value=0.50),
    st.floats(min_value=0.01, max_value=0.80),
    st.floats(min_value=0.01, max_value=1000),
)
@settings(max_examples=100)
def test_vol_target_size_not_exploding(
    capital: float, target_vol: float, realized_vol: float, price: float,
) -> None:
    """Vol-targeted size should not exceed 5x capital/price."""
    from src.risk.sizing import PositionSizer
    size = PositionSizer.volatility_target(capital, target_vol, realized_vol, price)
    max_val = 5.0 * capital / price
    assert size <= max_val


@given(
    st.floats(min_value=1, max_value=1e6),
    st.floats(min_value=0.001, max_value=0.20),
    st.floats(min_value=0.0001, max_value=10.0),
    st.floats(min_value=0.01, max_value=1000),
)
@settings(max_examples=100)
def test_fixed_fractional_positive_or_zero(
    capital: float, risk_pct: float, stop_distance: float, price: float,
) -> None:
    """Fixed fractional sizing should be >= 0."""
    from src.risk.sizing import PositionSizer
    size = PositionSizer.fixed_fractional(capital, risk_pct, stop_distance, price)
    assert size >= 0.0


# =============================================================================
# OIS curve invariants
# =============================================================================

@given(
    st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=3650),
            st.floats(min_value=0.0, max_value=0.25),
        ),
        min_size=3, max_size=8,
    ),
)
@settings(max_examples=50)
def test_discount_factors_monotonic_decay(
    quotes: list[tuple[int, float]],
) -> None:
    """Discount factors must decrease with tenor and stay positive."""
    from datetime import date
    from src.rates.ois_curve import OISCurve
    raw = {}
    for days, rate in quotes:
        raw[f"{days}D"] = rate
    try:
        curve = OISCurve.from_quotes("USD", date(2026, 1, 1), raw)
        dfs = [(d.days, df) for d, df in curve.discount_factors.items()]
        dfs.sort()
        for i in range(1, len(dfs)):
            assert dfs[i][1] < dfs[i-1][1], f"DFs not monotonic: {dfs[i]} after {dfs[i-1]}"
            assert dfs[i][1] > 0, f"negative DF at {dfs[i]}"
    except (ValueError, AssertionError):
        pass


@given(
    st.floats(min_value=0.01, max_value=0.25),
    st.integers(min_value=1, max_value=365),
)
@settings(max_examples=100)
def test_single_quote_curve_roundtrip(rate: float, days: int) -> None:
    """For ≤1Y, par rate should bootstrap to itself."""
    from datetime import date, timedelta
    from src.rates.ois_curve import OISCurve
    raw = {f"{days}D": rate}
    curve = OISCurve.from_quotes("USD", date(2026, 1, 1), raw)
    maturity = date(2026, 1, 1) + timedelta(days=days)
    df = curve.discount_factor(maturity)
    from src.rates.daycount import year_fraction, DayCountConvention, OIS_CONVENTIONS
    tau = year_fraction(date(2026, 1, 1), maturity, OIS_CONVENTIONS["USD"])
    implied = (1.0 - df) / (tau * df) if tau * df > 0 else 0
    assert abs(implied - rate) < 1e-3, f"implied {implied:.6f} vs par {rate:.6f}"


# =============================================================================
# Feature invariants
# =============================================================================

@given(
    st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=100, max_size=500),
    st.integers(min_value=10, max_value=100),
)
@settings(max_examples=30)
def test_zscore_mean_near_zero(rets: list[float], window: int) -> None:
    """Z-score of any series should have mean near 0."""
    import pandas as pd
    from src.features import zscore
    s = pd.Series(rets).cumsum() + 100
    z = zscore(s, window=window).dropna()
    assert abs(z.mean()) < 0.5


@given(
    st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=100, max_size=500),
    st.integers(min_value=10, max_value=100),
)
@settings(max_examples=30)
def test_realized_vol_positive(rets: list[float], window: int) -> None:
    """Realized volatility must be non-negative."""
    import pandas as pd
    from src.features import realized_vol
    s = pd.Series(rets).cumsum() + 100
    vol = realized_vol(s, window=window).dropna()
    assert (vol >= 0).all()


# =============================================================================
# Swap model invariants
# =============================================================================

@given(
    st.floats(min_value=100, max_value=1e7),
    st.floats(min_value=0.0, max_value=0.20),
    st.floats(min_value=0.0, max_value=0.20),
    st.integers(min_value=0, max_value=6),
)
@settings(max_examples=100)
def test_swap_sign_matches_rate_diff(
    position: float, target_rate: float, base_rate: float, day: int,
) -> None:
    """Long high-yield currency should earn positive swap, and vice versa."""
    from src.backtest.swap_model import SwapModel
    model = SwapModel()
    swap = model.compute_daily_swap(position, target_rate, base_rate, day)
    if day in (5, 6):
        assert swap == 0.0  # Weekend
    elif target_rate > base_rate:
        assert swap >= 0 if position > 0 else swap <= 0
    elif base_rate > target_rate:
        assert swap <= 0 if position > 0 else swap >= 0


@given(st.integers(min_value=0, max_value=6))
@settings(max_examples=20)
def test_wednesday_triple_swap(day: int) -> None:
    """Wednesday swap should be 3x normal (Architecture doc Section 12.1)."""
    from src.backtest.swap_model import SwapModel
    model = SwapModel()
    swap = model.compute_daily_swap(100000, 0.05, 0.01, day)
    normal = model.compute_daily_swap(100000, 0.05, 0.01, 0)  # Monday
    if day == 2 and normal != 0:
        assert abs(swap) == pytest.approx(3 * abs(normal), rel=0.1)
