"""Independent cash conservation for an explicitly named allocation policy (CL-sweu)."""

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.execution.alpaca_historical_allocation import allocate_cash


def test_pro_rata_cash_is_not_a_fabricated_per_lot_fill_price():
    result = allocate_cash(Decimal("1970.27"), {"a": Decimal(50), "b": Decimal(51)})
    assert sum(result.values()) == Decimal("1970.27")
    assert result["a"] == Decimal("975.38118811")
    assert result["b"] == Decimal("994.88881189")


@given(
    total=st.decimals(min_value="-100000", max_value="100000", places=8),
    a=st.decimals(min_value="0.000001", max_value="10000", places=6),
    b=st.decimals(min_value="0.000001", max_value="10000", places=6),
)
def test_cash_conservation_order_independence_and_sign(total, a, b):
    first = allocate_cash(total, {"a": a, "b": b})
    assert first == allocate_cash(total, {"b": b, "a": a})
    assert sum(first.values()) == total
    assert all(v * total >= 0 for v in first.values())
    assert all(abs(v) <= abs(total) for v in first.values())


@pytest.mark.parametrize(
    "weights", [{}, {"a": Decimal(0)}, {"a": Decimal(-1)}, {"a": Decimal("NaN")}]
)
def test_invalid_weights_refuse_attribution(weights):
    with pytest.raises(ValueError):
        allocate_cash(Decimal(1), weights)
