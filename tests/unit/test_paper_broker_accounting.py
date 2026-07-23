"""PaperBroker average-entry / realized-P&L accounting (CL-qyav P2).

The old formula blended EVERY fill into avg_price — reducing a long
corrupted the basis (e.g. 100 @ 1.10 then sell 50 @ 1.20 gave avg 3.40)
instead of realizing P&L, and a flip through zero inherited a nonsensical
basis. Rules under test:

    add same direction → volume-weighted average basis
    partial reduce     → basis unchanged, P&L realized on the closed qty
    full close         → basis unchanged, all P&L realized, qty 0
    flip through zero  → realize on the whole old position, basis resets to
                         the fill price for the residual quantity
"""

from __future__ import annotations

import pytest

from src.execution.broker import Order, OrderStatus, OrderType
from src.execution.paper_broker import PaperBroker


def _order(side: str, qty: float, symbol: str = "EURUSD") -> Order:
    return Order(symbol=symbol, side=side, quantity=qty, order_type=OrderType.MARKET)


def _pos(broker: PaperBroker, symbol: str = "EURUSD"):
    return next(p for p in broker.get_positions() if p.symbol == symbol)


class TestMissingPriceFailsClosed:
    """CL-n3pt (P1): place_order must fail CLOSED on a symbol with no
    set_price — not fabricate a ~1.10 fill (nonsense for metals/commodities)."""

    def test_rejects_order_with_no_price(self) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("EURUSD", 1.0999, 1.1001)
        order = b.place_order(_order("buy", 1000, symbol="XAU_USD"))  # never priced
        assert order.status == OrderStatus.REJECTED
        assert "NO_PRICE" in (order.reject_reason or "")
        # No fabricated fill, no position opened at ~1.10.
        assert not any(p.symbol == "XAU_USD" for p in b.get_positions())

    def test_priced_symbol_still_fills(self) -> None:
        b = PaperBroker(initial_capital=100_000.0)
        b.set_price("XAU_USD", 1999.0, 2001.0)  # a realistic metal quote
        order = b.place_order(_order("buy", 10, symbol="XAU_USD"))
        assert order.status == OrderStatus.FILLED
        pos = _pos(b, symbol="XAU_USD")
        assert pos.avg_price == pytest.approx(2001.0)  # filled at the ask


class TestAveragePriceAccounting:
    def test_add_same_direction_volume_weights_basis(self) -> None:
        b = PaperBroker()
        b.set_price("EURUSD", 1.0999, 1.1000)
        b.place_order(_order("buy", 100))  # 100 @ 1.1000
        b.set_price("EURUSD", 1.1999, 1.2000)
        b.place_order(_order("buy", 100))  # 100 @ 1.2000
        pos = _pos(b)
        assert pos.quantity == pytest.approx(200)
        assert pos.avg_price == pytest.approx(1.15)
        assert pos.realized_pnl == pytest.approx(0.0)

    def test_partial_reduce_keeps_basis_and_realizes(self) -> None:
        b = PaperBroker()
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 100))  # 100 @ 1.1000 (ask)
        b.set_price("EURUSD", 1.2000, 1.2002)
        b.place_order(_order("sell", 50))  # close 50 @ 1.2000 (bid)
        pos = _pos(b)
        assert pos.quantity == pytest.approx(50)
        assert pos.avg_price == pytest.approx(1.1000)  # basis UNCHANGED
        assert pos.realized_pnl == pytest.approx(50 * (1.2000 - 1.1000))

    def test_full_close_realizes_everything(self) -> None:
        b = PaperBroker()
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 100))
        b.set_price("EURUSD", 1.2000, 1.2002)
        b.place_order(_order("sell", 100))
        pos = _pos(b)
        assert pos.quantity == pytest.approx(0.0)
        assert pos.avg_price == pytest.approx(1.1000)  # basis not fabricated
        assert pos.realized_pnl == pytest.approx(100 * (1.2000 - 1.1000))

    def test_flip_long_to_short_resets_basis_to_fill(self) -> None:
        b = PaperBroker()
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 100))  # long 100 @ 1.1000
        b.set_price("EURUSD", 1.2000, 1.2002)
        b.place_order(_order("sell", 150))  # fill @ 1.2000 (bid)
        pos = _pos(b)
        assert pos.quantity == pytest.approx(-50)
        assert pos.avg_price == pytest.approx(1.2000)  # residual basis = fill
        assert pos.realized_pnl == pytest.approx(100 * (1.2000 - 1.1000))

    def test_flip_short_to_long_resets_basis_to_fill(self) -> None:
        b = PaperBroker()
        b.set_price("EURUSD", 1.2000, 1.2002)
        b.place_order(_order("sell", 100))  # short 100 @ 1.2000 (bid)
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 150))  # fill @ 1.1000 (ask)
        pos = _pos(b)
        assert pos.quantity == pytest.approx(50)
        assert pos.avg_price == pytest.approx(1.1000)
        # Short profit: sold at 1.2000, covered at 1.1000, on 100 units.
        assert pos.realized_pnl == pytest.approx(100 * (1.2000 - 1.1000))

    def test_short_reduce_keeps_basis(self) -> None:
        b = PaperBroker()
        b.set_price("EURUSD", 1.2000, 1.2002)
        b.place_order(_order("sell", 100))  # short 100 @ 1.2000
        b.set_price("EURUSD", 1.2498, 1.2500)
        b.place_order(_order("buy", 40))  # cover 40 @ 1.2500 (ask)
        pos = _pos(b)
        assert pos.quantity == pytest.approx(-60)
        assert pos.avg_price == pytest.approx(1.2000)
        # Covered above entry → realized LOSS on the short.
        assert pos.realized_pnl == pytest.approx(-40 * (1.2500 - 1.2000))

    def test_reopen_after_full_close_starts_fresh_basis(self) -> None:
        b = PaperBroker()
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 100))
        b.set_price("EURUSD", 1.2000, 1.2002)
        b.place_order(_order("sell", 100))  # flat
        b.set_price("EURUSD", 1.2998, 1.3000)
        b.place_order(_order("buy", 10))  # fresh entry @ 1.3000
        pos = _pos(b)
        assert pos.quantity == pytest.approx(10)
        assert pos.avg_price == pytest.approx(1.3000)
        # Prior realized P&L is preserved on the record.
        assert pos.realized_pnl == pytest.approx(100 * (1.2000 - 1.1000))

    def test_realized_pnl_flows_to_equity(self) -> None:
        b = PaperBroker(initial_capital=10_000.0)
        b.set_price("EURUSD", 1.0998, 1.1000)
        b.place_order(_order("buy", 100))
        b.set_price("EURUSD", 1.2000, 1.2002)
        b.place_order(_order("sell", 100))
        costs = 100 * 1.1000 * 0.0001 + 100 * 1.2000 * 0.0001
        expected = 10_000.0 + 100 * (1.2000 - 1.1000) - costs
        assert b.equity == pytest.approx(expected)

    def test_all_orders_still_fill(self) -> None:
        # Accounting changes must not alter fill semantics.
        b = PaperBroker()
        b.set_price("EURUSD", 1.0998, 1.1000)
        assert b.place_order(_order("buy", 100)).status == OrderStatus.FILLED
        assert b.place_order(_order("sell", 250)).status == OrderStatus.FILLED
