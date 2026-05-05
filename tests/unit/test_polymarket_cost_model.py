"""Tests for PolymarketCostModel (CL-poly-2)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.polymarket_cost_model import PolymarketCostModel


class TestZeroFeeBaseline:
    def test_default_fees_yields_only_gas(self) -> None:
        # Polymarket headline fees are 0; only gas remains.
        m = PolymarketCostModel()
        cost = m.estimate(
            side="BUY", role="taker",
            price=Decimal("0.50"), size=Decimal("100"),
        )
        assert cost == m.gas_usdc_estimate


class TestFeeApplication:
    def test_maker_fee_applies(self) -> None:
        m = PolymarketCostModel(maker_fee_bps=Decimal("10"))
        cost = m.estimate(
            side="BUY", role="maker",
            price=Decimal("0.50"), size=Decimal("100"),
        )
        # notional = 50, 10bps = 0.005 * 50 = 0.05; plus gas 0.01 = 0.06.
        assert cost == Decimal("0.06")

    def test_taker_fee_applies_for_taker(self) -> None:
        m = PolymarketCostModel(taker_fee_bps=Decimal("20"))
        cost = m.estimate(
            side="BUY", role="taker",
            price=Decimal("0.50"), size=Decimal("100"),
        )
        # 20bps of 50 = 0.10; plus gas 0.01 = 0.11.
        assert cost == Decimal("0.11")

    def test_maker_fee_does_not_apply_to_taker_fill(self) -> None:
        m = PolymarketCostModel(
            maker_fee_bps=Decimal("100"),  # huge maker fee
            taker_fee_bps=Decimal("0"),
        )
        cost = m.estimate(
            side="BUY", role="taker",
            price=Decimal("0.50"), size=Decimal("100"),
        )
        # Taker fee is 0; maker_fee_bps must NOT apply.
        assert cost == m.gas_usdc_estimate


class TestSlippageProxy:
    def test_taker_above_book_threshold_adds_slippage(self) -> None:
        m = PolymarketCostModel(
            slippage_bps_on_deep_taker=Decimal("100"),  # 1%
        )
        # Order size 100, book depth 100 → ratio 1.0 > 0.5 → triggers.
        cost = m.estimate(
            side="BUY", role="taker",
            price=Decimal("0.50"), size=Decimal("100"),
            book_depth_at_price=Decimal("100"),
        )
        # notional = 50, slippage = 50 * 0.01 = 0.50; plus gas 0.01.
        assert cost == Decimal("0.51")

    def test_taker_below_threshold_no_slippage(self) -> None:
        m = PolymarketCostModel()
        # Order size 10, book depth 100 → ratio 0.1 < 0.5 → no slippage.
        cost = m.estimate(
            side="BUY", role="taker",
            price=Decimal("0.50"), size=Decimal("10"),
            book_depth_at_price=Decimal("100"),
        )
        assert cost == m.gas_usdc_estimate

    def test_maker_does_not_pay_slippage(self) -> None:
        m = PolymarketCostModel(
            slippage_bps_on_deep_taker=Decimal("100"),
        )
        # Even with deep size, MAKER pays no slippage.
        cost = m.estimate(
            side="BUY", role="maker",
            price=Decimal("0.50"), size=Decimal("100"),
            book_depth_at_price=Decimal("100"),
        )
        assert cost == m.gas_usdc_estimate


class TestNeverNegative:
    def test_returns_zero_floor(self) -> None:
        # Hand-construct a model with negative fees (rebates aren't
        # supported in v1); the floor protects strategies from being
        # told a fill makes them money before it's settled.
        m = PolymarketCostModel(
            maker_fee_bps=Decimal("-1000"),
            gas_usdc_estimate=Decimal("0"),
        )
        cost = m.estimate(
            side="BUY", role="maker",
            price=Decimal("0.50"), size=Decimal("100"),
        )
        assert cost >= 0
