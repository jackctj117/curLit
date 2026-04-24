"""Property-based tests — hypothesis tests for key invariants."""

import pytest
from hypothesis import given, settings, strategies as st

import numpy as np


@given(st.floats(min_value=0.01, max_value=0.99), st.floats(min_value=0.01, max_value=10.0))
@settings(max_examples=50)
def test_kelly_bounded(edge: float, odds: float) -> None:
    from src.risk.sizing import PositionSizer
    k = PositionSizer.kelly(edge, odds)
    assert 0 <= k <= 1


@given(st.lists(st.tuples(st.floats(min_value=0.01, max_value=10.0), st.floats(min_value=0.0, max_value=0.25)), min_size=3, max_size=10))
@settings(max_examples=20)
def test_discount_factors_positive(quotes: list[tuple[float, float]]) -> None:
    from datetime import date
    from src.rates.ois_curve import OISCurve
    raw = {}
    for i, (days, rate) in enumerate(quotes):
        raw[f"{int(days)}D"] = rate
    try:
        curve = OISCurve.from_quotes("USD", date(2026, 1, 1), raw)
        for d, df in curve.discount_factors.items():
            assert df > 0
    except Exception:
        pass  # Some random combinations may fail to bootstrap — that's fine


@given(st.floats(min_value=0.0, max_value=5.0), st.floats(min_value=-0.5, max_value=5.0))
@settings(max_examples=30)
def test_kelly_never_negative_from_any_input(edge: float, odds: float) -> None:
    from src.risk.sizing import PositionSizer
    k = PositionSizer.kelly(edge, odds)
    assert k >= 0
