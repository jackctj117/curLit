"""Property-based tests — hypothesis tests for key numerical invariants.

Modules under test are imported at module scope, never inside a property body:
Hypothesis times every example against its deadline (default 200 ms), so a
cold first import (``src.rates.ois_curve`` pulls in scipy.interpolate, ~0.3-0.45 s)
inside the body makes the first example blow the deadline and the run
FlakyFailure in isolation (CL-twwp).
"""

from datetime import date, timedelta

import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.backtest.swap_model import SwapModel
from src.features import realized_vol, zscore
from src.rates.daycount import OIS_CONVENTIONS, year_fraction
from src.rates.ois_curve import OISCurve, OISQuote
from src.risk.sizing import PositionSizer

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
    k = PositionSizer.kelly(edge, odds, kelly_fraction=1.0)
    assert 0.0 <= k <= 1.0


@given(
    st.floats(min_value=-5.0, max_value=5.0),
    st.floats(min_value=-5.0, max_value=5.0),
)
@settings(max_examples=100)
def test_kelly_never_negative_regardless_of_input(edge: float, odds: float) -> None:
    """Even with garbage edge/odds, kelly returns >= 0."""
    k = PositionSizer.kelly(edge, odds)
    assert k >= 0.0


@given(
    st.floats(min_value=0.01, max_value=0.99),
    st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=100)
def test_quarter_kelly_less_than_full_kelly(edge: float, odds: float) -> None:
    """Quarter-Kelly should be <= full Kelly."""
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
    capital: float,
    target_vol: float,
    realized_vol: float,
    price: float,
) -> None:
    """Vol-targeted size stays under 5×capital/price for realistic vol ratios.

    The function deliberately doesn't clamp size (clamping is RiskManager's
    layer per docstring); for size <= 5*capital/price to hold, we need
    realized_vol / target_vol >= 1/5. assume() filters the regime where
    realized_vol is so low relative to target that the unclamped output
    legitimately exceeds the bound.
    """
    assume(realized_vol >= target_vol / 5.0)
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
    capital: float,
    risk_pct: float,
    stop_distance: float,
    price: float,
) -> None:
    """Fixed fractional sizing should be >= 0."""
    size = PositionSizer.fixed_fractional(capital, risk_pct, stop_distance, price)
    assert size >= 0.0


# =============================================================================
# OIS curve invariants (CL-twwp)
# =============================================================================
#
# Independent analytical oracle. USD OIS (SOFR) accrues on an Actual/360 basis
# (ISDA 2006 Definitions §4.16(e); ARRC SOFR OIS conventions). A single-period
# OIS pays one fixed coupon S*tau at maturity T, so pricing it at par gives the
# money-market relation  1 = DF(T) * (1 + S*tau)  =>  DF(T) = 1 / (1 + S*tau)
# (Hull, "Options, Futures, and Other Derivatives", OIS chapter). OISCurve
# documents that it applies this single-coupon convention to every quote with
# tenor < 730 days (src/rates/ois_curve.py, _solve_df_for_quote). The market
# convention for SOFR OIS is single-payment only up to 1Y, so for 1Y < T < 2Y
# this oracle checks the code's DOCUMENTED convention, not the market one.
#
# The oracle deliberately hard-codes the day basis and formula instead of
# calling year_fraction / OIS_CONVENTIONS, so a defect in either is caught.
# For a non-negative rate, 1/(1 + S*tau) is positive and non-increasing in
# tau, which is the valid domain of the monotonic-decay property.

_OIS_VALUATION_DATE = date(2026, 1, 1)
# Actual/360: accrual = calendar days / 360 (ISDA 2006 §4.16(e)).
_USD_OIS_DAY_BASIS = 360.0
# OISCurve's documented single-coupon regime is tenor_days < 730.
_SINGLE_COUPON_MAX_DAYS = 729
# Modified-following moves a date by at most a weekend plus one adjacent
# holiday: +3 (Sat -> Tue when Monday is a holiday) or -3 at a month end.
# 4 calendar days bounds every US federal holiday placement with a margin.
_MAX_BUSINESS_DAY_ADJUSTMENT = 4
# Rates span zero to 25% — wider than any observed USD policy-rate regime.
_MAX_OIS_RATE = 0.25
# The curve's DFs and the oracle compute the same closed form with a possibly
# different floating-point operation order; 1e-12 relative covers rounding
# while catching any economically meaningful deviation (the mutation test
# below perturbs by 1e-6).
_ORACLE_REL_TOL = 1e-12


def _analytic_single_coupon_df(par_rate: float, accrual_days: int) -> float:
    """Independent oracle: DF(T) = 1 / (1 + S * days/360) for one coupon at T."""
    return 1.0 / (1.0 + par_rate * (accrual_days / _USD_OIS_DAY_BASIS))


def _assert_monotonic_decay(discount_factors: dict[date, float], valuation: date) -> None:
    """Oracle: DF(valuation) == 1, every DF > 0, DFs non-increasing in tenor.

    Ties are allowed: distinct quotes can adjust onto the same business day,
    and a zero rate gives a flat DF of 1.
    """
    points = sorted(((d - valuation).days, df) for d, df in discount_factors.items())
    assert points[0] == (0, 1.0), f"curve must start at DF(valuation)=1, got {points[0]}"
    for prev, cur in zip(points, points[1:], strict=False):
        assert cur[1] > 0.0, f"non-positive DF at {cur}"
        assert cur[1] <= prev[1], f"DFs not monotonic: {cur} after {prev}"


def _assert_matches_analytic(curve: OISCurve, raw_quotes: dict[str, float]) -> None:
    """Every bootstrapped single-coupon DF equals the analytical oracle value."""
    assert len(curve.quotes) == len(raw_quotes)
    for quote in curve.quotes:
        assert quote.par_rate == raw_quotes[quote.tenor_label]
        unadjusted = curve.valuation_date + timedelta(days=int(quote.tenor_label[:-1]))
        shift = (quote.maturity_date - unadjusted).days
        assert abs(shift) <= _MAX_BUSINESS_DAY_ADJUSTMENT, f"{quote} moved {shift} days"
        assert quote.maturity_date.weekday() < 5, f"{quote} matures on a weekend"
        accrual_days = (quote.maturity_date - curve.valuation_date).days
        expected = _analytic_single_coupon_df(quote.par_rate, accrual_days)
        actual = curve.discount_factors[quote.maturity_date]
        assert actual == pytest.approx(expected, rel=_ORACLE_REL_TOL, abs=0.0), (
            f"{quote.tenor_label}: DF {actual!r} != analytic {expected!r}"
        )


@st.composite
def _upward_sloping_quotes(draw: st.DrawFn) -> dict[str, float]:
    """Single-coupon quotes with par rates non-decreasing in tenor.

    Tenors are spaced >= 10 days apart (> 2 x the maximum business-day
    adjustment) so no two quotes collapse onto one maturity date.
    """
    first = draw(st.integers(min_value=1, max_value=100))
    gaps = draw(st.lists(st.integers(min_value=10, max_value=100), min_size=2, max_size=6))
    tenors = [first]
    for gap in gaps:
        tenors.append(tenors[-1] + gap)
    assert tenors[-1] <= _SINGLE_COUPON_MAX_DAYS
    rates = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=_MAX_OIS_RATE),
                min_size=len(tenors),
                max_size=len(tenors),
            )
        )
    )
    return {f"{days}D": rate for days, rate in zip(tenors, rates, strict=True)}


@given(
    st.floats(min_value=0.0, max_value=_MAX_OIS_RATE),
    st.lists(st.integers(min_value=1, max_value=_SINGLE_COUPON_MAX_DAYS), min_size=3, max_size=8),
)
@settings(max_examples=50)
def test_discount_factors_monotonic_decay(rate: float, tenors: list[int]) -> None:
    """Constant non-negative-rate curve: DFs decay monotonically and equal the
    analytical Actual/360 single-coupon values.

    No exception is caught: every input in this domain is valid, so any
    ValueError from construction or AssertionError from the oracle fails the
    test (the pre-CL-twwp version swallowed both).
    """
    raw = {f"{days}D": rate for days in tenors}
    curve = OISCurve.from_quotes("USD", _OIS_VALUATION_DATE, raw)
    _assert_monotonic_decay(curve.discount_factors, curve.valuation_date)
    _assert_matches_analytic(curve, raw)


@given(_upward_sloping_quotes())
@settings(max_examples=50)
def test_discount_factors_monotonic_for_upward_sloping_par_curve(
    raw: dict[str, float],
) -> None:
    """Non-decreasing non-negative par rates over increasing tenors give
    non-decreasing S*tau, hence non-increasing analytical DFs."""
    curve = OISCurve.from_quotes("USD", _OIS_VALUATION_DATE, raw)
    _assert_monotonic_decay(curve.discount_factors, curve.valuation_date)
    _assert_matches_analytic(curve, raw)


@given(
    st.floats(min_value=0.0, max_value=_MAX_OIS_RATE),
    st.lists(st.integers(min_value=1, max_value=3650), min_size=3, max_size=8),
)
@settings(max_examples=50)
def test_long_tenor_curves_reject_or_decay(rate: float, tenors: list[int]) -> None:
    """Multi-coupon (>= 730D) tenors without a covering annual schedule are
    an explicitly documented invalid input ("OIS bootstrap gap" ValueError).

    Only that construction call is guarded, and only for that message; any
    curve that does build must satisfy the monotonic-decay oracle unguarded.
    """
    raw = {f"{days}D": rate for days in tenors}
    try:
        curve = OISCurve.from_quotes("USD", _OIS_VALUATION_DATE, raw)
    except ValueError as exc:
        assert max(tenors) > _SINGLE_COUPON_MAX_DAYS, f"valid short curve rejected: {exc}"
        assert str(exc).startswith("OIS bootstrap gap"), f"unexpected rejection: {exc}"
        return
    _assert_monotonic_decay(curve.discount_factors, curve.valuation_date)


def test_long_tenor_without_annual_schedule_is_rejected() -> None:
    """The documented invalid-input failure is a ValueError, not a bad curve."""
    with pytest.raises(ValueError, match="OIS bootstrap gap"):
        OISCurve.from_quotes("USD", _OIS_VALUATION_DATE, {"30D": 0.03, "800D": 0.03})


def test_analytic_oracle_known_values() -> None:
    """Hand-computed Actual/360 vectors pin the oracle itself.

    5% for 180 days: tau = 0.5, DF = 1/1.025. 4% for 90 days: tau = 0.25,
    DF = 1/1.01. Zero rate: DF = 1.
    """
    assert _analytic_single_coupon_df(0.05, 180) == pytest.approx(1.0 / 1.025, rel=1e-15)
    assert _analytic_single_coupon_df(0.04, 90) == pytest.approx(1.0 / 1.01, rel=1e-15)
    assert _analytic_single_coupon_df(0.0, 365) == 1.0


def test_monotonic_oracle_rejects_non_monotonic_sequence() -> None:
    """The oracle itself raises on rising, non-positive, or mis-anchored DFs."""
    val = _OIS_VALUATION_DATE
    with pytest.raises(AssertionError, match="not monotonic"):
        _assert_monotonic_decay(
            {val: 1.0, val + timedelta(days=30): 0.99, val + timedelta(days=60): 0.995}, val
        )
    with pytest.raises(AssertionError, match="non-positive"):
        _assert_monotonic_decay({val: 1.0, val + timedelta(days=30): 0.0}, val)
    with pytest.raises(AssertionError, match="DF\\(valuation\\)=1"):
        _assert_monotonic_decay({val: 0.9, val + timedelta(days=30): 0.8}, val)


def test_non_monotonic_curve_mutation_fails_property(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: DFs that RISE above 1 with tenor must fail the property.

    The pre-CL-twwp test wrapped its oracle in ``except AssertionError: pass``
    and passed under exactly this mutation.
    """

    def _rising_df(self: OISCurve, quote: OISQuote) -> float:
        return 1.0 + quote.tenor_days * 1e-4

    monkeypatch.setattr(OISCurve, "_solve_df_for_quote", _rising_df)
    with pytest.raises(AssertionError, match="not monotonic"):
        test_discount_factors_monotonic_decay()


def test_off_analytic_curve_mutation_fails_property(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: a monotone curve scaled 1e-6 off the analytic value must fail."""
    original = OISCurve._solve_df_for_quote

    def _scaled_df(self: OISCurve, quote: OISQuote) -> float:
        return original(self, quote) * (1.0 - 1e-6)

    monkeypatch.setattr(OISCurve, "_solve_df_for_quote", _scaled_df)
    with pytest.raises(AssertionError, match="!= analytic"):
        test_discount_factors_monotonic_decay()


@given(
    st.floats(min_value=0.01, max_value=0.25),
    st.integers(min_value=1, max_value=365),
)
@settings(max_examples=100)
def test_single_quote_curve_roundtrip(rate: float, days: int) -> None:
    """For ≤1Y, par rate should bootstrap to itself."""
    raw = {f"{days}D": rate}
    curve = OISCurve.from_quotes("USD", date(2026, 1, 1), raw)
    maturity = date(2026, 1, 1) + timedelta(days=days)
    df = curve.discount_factor(maturity)
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
    """Z-score of any series should have mean near 0.

    Filters degenerate inputs where the rolling-window std is zero (constant
    series) — z-score is mathematically undefined there (NaN), so the mean
    assertion can't hold. Our zscore() doesn't guard against zero std because
    constant features are caller-side data-quality issues, not numerical
    invariants of the function.
    """
    # Stationary input — z-score's mean-zero invariant holds for stationary
    # series but not for cumulative (random-walk) series, where rolling
    # z-scores carry the trend bias. Testing the function's correctness, not
    # whether arbitrary non-stationary inputs are mean-centered.
    s = pd.Series(rets) + 100
    z = zscore(s, window=window).dropna()
    # Need a full window of valid z-scores; constant rolling subseries → NaN.
    assume(len(z) >= window)
    assert abs(z.mean()) < 0.5


@given(
    st.lists(st.floats(min_value=-0.5, max_value=0.5), min_size=100, max_size=500),
    st.integers(min_value=10, max_value=100),
)
@settings(max_examples=30)
def test_realized_vol_positive(rets: list[float], window: int) -> None:
    """Realized volatility must be non-negative."""
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
    position: float,
    target_rate: float,
    base_rate: float,
    day: int,
) -> None:
    """Long high-yield currency should earn positive swap, and vice versa.

    Filters cases where |rate_diff| <= broker markup — there the markup
    floor swallows the carry and the net-swap sign no longer tracks the
    rate-diff sign, which is correct behavior but not what this test verifies.
    """
    model = SwapModel()
    if day not in (5, 6):
        markup_rate = model.config.broker_markup_pct / 100.0
        assume(abs(target_rate - base_rate) > markup_rate)
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
    model = SwapModel()
    swap = model.compute_daily_swap(100000, 0.05, 0.01, day)
    normal = model.compute_daily_swap(100000, 0.05, 0.01, 0)  # Monday
    if day == 2 and normal != 0:
        assert abs(swap) == pytest.approx(3 * abs(normal), rel=0.1)
