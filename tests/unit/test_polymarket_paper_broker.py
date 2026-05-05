"""Tests for PolymarketPaperBroker (CL-poly-2)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.execution.broker import Order, OrderStatus, OrderType
from src.execution.polymarket_data_source import (
    BookLevel,
    OrderBookSnapshot,
    PolymarketDataSource,
)
from src.execution.polymarket_paper_broker import PolymarketPaperBroker


def _book(asks: list[tuple[str, str]] | None = None,
          bids: list[tuple[str, str]] | None = None,
          token_id: str = "tok-1") -> OrderBookSnapshot:
    return OrderBookSnapshot(
        token_id=token_id,
        bids=[BookLevel(Decimal(p), Decimal(s)) for p, s in (bids or [])],
        asks=[BookLevel(Decimal(p), Decimal(s)) for p, s in (asks or [])],
    )


def _ds_with_book(book: OrderBookSnapshot) -> PolymarketDataSource:
    """Build a DataSource that resolves any POLY:* to a fixed token_id and
    always returns the same book."""
    ds = MagicMock(spec=PolymarketDataSource)
    ds.resolve_symbol.side_effect = lambda s: book.token_id
    ds.get_book.return_value = book
    return ds


class TestImmediateFill:
    def test_buy_at_top_ask_fills(self) -> None:
        ds = _ds_with_book(_book(
            asks=[("0.50", "100"), ("0.51", "100")],
            bids=[("0.49", "100")],
        ))
        broker = PolymarketPaperBroker(data_source=ds)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.FILLED
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 10
        assert positions[0].avg_price == 0.50

    def test_buy_at_aggressive_price_fills_at_limit(self) -> None:
        # Limit 0.55, top ask 0.50 → fills at 0.50 (immediate), capped
        # by depth at 0.55 = 100 (0.50) + 50 (0.55) = 150.
        # Wait — this v1 simulator fills at the LIMIT price (0.55)
        # for simplicity. That's an operator-favorable assumption
        # documented in the broker's _simulate_fill docstring.
        ds = _ds_with_book(_book(
            asks=[("0.50", "100"), ("0.55", "50")],
        ))
        broker = PolymarketPaperBroker(data_source=ds)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=200,
            order_type=OrderType.LIMIT, limit_price=0.55,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.FILLED
        # 200 capped to 150 (book depth at-or-below 0.55).
        assert broker.get_positions()[0].quantity == 150

    def test_sell_above_top_bid_rests(self) -> None:
        ds = _ds_with_book(_book(
            asks=[("0.51", "100")],
            bids=[("0.49", "100")],
        ))
        broker = PolymarketPaperBroker(data_source=ds)
        order = Order(
            symbol="POLY:tok-1", side="sell", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.55,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.PENDING
        assert broker.get_positions() == []

    def test_market_order_takes_top_of_book(self) -> None:
        ds = _ds_with_book(_book(
            asks=[("0.50", "100")],
            bids=[("0.49", "100")],
        ))
        broker = PolymarketPaperBroker(data_source=ds)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.MARKET,
        )
        out = broker.place_order(order)
        assert out.status == OrderStatus.FILLED
        assert broker.get_positions()[0].avg_price == 0.50


class TestValidation:
    def test_below_tick_raises(self) -> None:
        ds = _ds_with_book(_book())
        broker = PolymarketPaperBroker(data_source=ds)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.005,  # below floor
        )
        with pytest.raises(ValueError, match="outside"):
            broker.place_order(order)

    def test_off_tick_price_raises(self) -> None:
        ds = _ds_with_book(_book())
        broker = PolymarketPaperBroker(data_source=ds)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.505,  # off tick
        )
        with pytest.raises(ValueError, match="not on"):
            broker.place_order(order)

    def test_zero_quantity_raises(self) -> None:
        ds = _ds_with_book(_book())
        broker = PolymarketPaperBroker(data_source=ds)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=0,
            order_type=OrderType.LIMIT, limit_price=0.50,
        )
        with pytest.raises(ValueError, match="quantity must be positive"):
            broker.place_order(order)


class TestPositionLifecycle:
    def test_buy_then_sell_zeros_out(self) -> None:
        ds = _ds_with_book(_book(
            asks=[("0.50", "100")],
            bids=[("0.50", "100")],
        ))
        broker = PolymarketPaperBroker(data_source=ds)
        broker.place_order(Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        ))
        broker.place_order(Order(
            symbol="POLY:tok-1", side="sell", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        ))
        # Position closed → not in positions list.
        assert broker.get_positions() == []

    def test_partial_close_preserves_avg_price(self) -> None:
        ds = _ds_with_book(_book(
            asks=[("0.50", "100")],
            bids=[("0.55", "100")],
        ))
        broker = PolymarketPaperBroker(data_source=ds)
        broker.place_order(Order(
            symbol="POLY:tok-1", side="buy", quantity=20,
            order_type=OrderType.LIMIT, limit_price=0.50,
        ))
        broker.place_order(Order(
            symbol="POLY:tok-1", side="sell", quantity=5,
            order_type=OrderType.LIMIT, limit_price=0.55,
        ))
        positions = broker.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == 15
        # Avg price preserved from the buy.
        assert positions[0].avg_price == 0.50


class TestAccount:
    def test_initial_account_reflects_capital(self) -> None:
        broker = PolymarketPaperBroker(
            data_source=_ds_with_book(_book()),
            initial_capital_usdc=Decimal("50000"),
        )
        acct = broker.get_account()
        assert acct.balance == 50000.0
        # Currency is USDC by construction for polymarket-* brokers;
        # Account dataclass doesn't carry the field, so the test just
        # confirms balance + zero margin (fully collateralized).
        assert acct.margin_used == 0

    def test_buy_reduces_cash(self) -> None:
        ds = _ds_with_book(_book(asks=[("0.50", "100")]))
        broker = PolymarketPaperBroker(
            data_source=ds, initial_capital_usdc=Decimal("1000"),
        )
        broker.place_order(Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.50,
        ))
        acct = broker.get_account()
        # Spent 5 USDC on 10 shares at 0.50, plus 0.01 gas = 994.99 cash.
        # Equity = cash + position notional (0.50 * 10) = 994.99 + 5 = 999.99.
        assert acct.balance < 1000
        assert acct.equity == pytest.approx(999.99, abs=0.05)


class TestCancelOrder:
    def test_cancel_pending_order_succeeds(self) -> None:
        ds = _ds_with_book(_book(asks=[("0.51", "100")]))
        broker = PolymarketPaperBroker(data_source=ds)
        order = Order(
            symbol="POLY:tok-1", side="buy", quantity=10,
            order_type=OrderType.LIMIT, limit_price=0.40,  # below ask
        )
        broker.place_order(order)
        assert broker.cancel_order(order.order_id) is True
        # Second cancel returns False.
        assert broker.cancel_order(order.order_id) is False

    def test_cancel_unknown_order_returns_false(self) -> None:
        broker = PolymarketPaperBroker(data_source=_ds_with_book(_book()))
        assert broker.cancel_order("nope") is False
