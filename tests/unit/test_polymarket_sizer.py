"""Tests for the Polymarket Kelly sizer (CL-poly-2)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.risk.polymarket_sizer import PolySizeDecision, size_polymarket_order


_BANKROLL: Decimal = Decimal("10000")


class TestEdgeFloor:
    def test_below_min_edge_returns_none(self) -> None:
        # market = 0.50, model = 0.51 → 1pp edge < default 3pp.
        out = size_polymarket_order(
            bankroll=_BANKROLL,
            market_price=Decimal("0.50"),
            model_prob=Decimal("0.51"),
            side="YES",
        )
        assert out is None

    def test_at_min_edge_passes(self) -> None:
        # 3pp edge passes the default min_edge of 3pp.
        out = size_polymarket_order(
            bankroll=_BANKROLL,
            market_price=Decimal("0.50"),
            model_prob=Decimal("0.53"),
            side="YES",
        )
        assert out is not None
        assert out.shares > 0
        assert out.cost_usdc > 0

    def test_wrong_side_edge_returns_none(self) -> None:
        # YES side, but model says probability is LOWER than market price.
        out = size_polymarket_order(
            bankroll=_BANKROLL,
            market_price=Decimal("0.60"),
            model_prob=Decimal("0.40"),
            side="YES",
        )
        assert out is None

    def test_no_side_with_overpriced_market(self) -> None:
        # market 0.70 (implied 70% YES); model thinks YES is 50%.
        # NO edge = 0.70 - 0.50 = 0.20 → strong NO bet.
        out = size_polymarket_order(
            bankroll=_BANKROLL,
            market_price=Decimal("0.70"),
            model_prob=Decimal("0.50"),
            side="NO",
        )
        assert out is not None
        assert out.cost_usdc > 0


class TestKellyCap:
    def test_size_capped_by_max_market_fraction(self) -> None:
        # Huge edge: market 0.50, model 0.95 → f* = 0.45/0.50 = 0.90.
        # With kelly_fraction=0.25 → 0.225. But max_market_fraction=0.05.
        out = size_polymarket_order(
            bankroll=_BANKROLL,
            market_price=Decimal("0.50"),
            model_prob=Decimal("0.95"),
            side="YES",
            max_market_fraction=Decimal("0.05"),
        )
        assert out is not None
        assert out.fraction_of_bankroll == Decimal("0.05")

    def test_kelly_fraction_scales_size(self) -> None:
        # Same edge, two different kelly_fractions → larger fraction
        # ⇒ larger position (modulo cap).
        kwargs = {
            "bankroll": _BANKROLL,
            "market_price": Decimal("0.40"),
            "model_prob": Decimal("0.50"),
            "side": "YES",
            "max_market_fraction": Decimal("0.99"),  # disable cap
        }
        small = size_polymarket_order(**kwargs, kelly_fraction=Decimal("0.10"))
        large = size_polymarket_order(**kwargs, kelly_fraction=Decimal("0.50"))
        assert small is not None and large is not None
        assert large.cost_usdc > small.cost_usdc


class TestInputValidation:
    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError, match="side must be"):
            size_polymarket_order(
                bankroll=_BANKROLL,
                market_price=Decimal("0.50"),
                model_prob=Decimal("0.60"),
                side="MAYBE",
            )

    def test_price_outside_range_raises(self) -> None:
        with pytest.raises(ValueError, match="market_price"):
            size_polymarket_order(
                bankroll=_BANKROLL,
                market_price=Decimal("0"),
                model_prob=Decimal("0.50"),
                side="YES",
            )
        with pytest.raises(ValueError, match="market_price"):
            size_polymarket_order(
                bankroll=_BANKROLL,
                market_price=Decimal("1.0"),
                model_prob=Decimal("0.50"),
                side="YES",
            )

    def test_model_prob_outside_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="model_prob"):
            size_polymarket_order(
                bankroll=_BANKROLL,
                market_price=Decimal("0.50"),
                model_prob=Decimal("1.5"),
                side="YES",
            )

    def test_zero_bankroll_raises(self) -> None:
        with pytest.raises(ValueError, match="bankroll"):
            size_polymarket_order(
                bankroll=Decimal("0"),
                market_price=Decimal("0.50"),
                model_prob=Decimal("0.60"),
                side="YES",
            )

    def test_kelly_fraction_validation(self) -> None:
        with pytest.raises(ValueError, match="kelly_fraction"):
            size_polymarket_order(
                bankroll=_BANKROLL,
                market_price=Decimal("0.50"),
                model_prob=Decimal("0.60"),
                side="YES",
                kelly_fraction=Decimal("0"),
            )


# Property: any valid input that returns a decision must respect the
# max_market_fraction cap.
@settings(deadline=None, max_examples=50)
@given(
    bankroll=st.decimals(
        min_value=Decimal("100"), max_value=Decimal("1000000"),
        allow_nan=False, allow_infinity=False, places=2,
    ),
    p=st.decimals(min_value=Decimal("0.05"), max_value=Decimal("0.95"),
                  allow_nan=False, allow_infinity=False, places=2),
    edge=st.decimals(min_value=Decimal("0.04"), max_value=Decimal("0.40"),
                     allow_nan=False, allow_infinity=False, places=2),
    cap=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("0.20"),
                    allow_nan=False, allow_infinity=False, places=2),
)
def test_decision_respects_cap(
    bankroll: Decimal, p: Decimal, edge: Decimal, cap: Decimal,
) -> None:
    q = p + edge
    if q >= 1:
        return
    out = size_polymarket_order(
        bankroll=bankroll,
        market_price=p,
        model_prob=q,
        side="YES",
        max_market_fraction=cap,
        min_edge=Decimal("0.03"),
    )
    if out is None:
        return  # min_edge filter is its own concern
    assert isinstance(out, PolySizeDecision)
    assert out.fraction_of_bankroll <= cap + Decimal("0.0001")
    # Cost should be non-negative and bounded by bankroll * cap.
    assert out.cost_usdc <= bankroll * cap + Decimal("0.01")
