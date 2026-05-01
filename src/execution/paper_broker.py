"""Paper broker — simulated execution with cost model, full P&L tracking."""

import logging
from datetime import UTC, datetime

from .broker import Account, Broker, Order, OrderStatus, Position

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self._capital = initial_capital
        self._equity = initial_capital
        self._peak_equity = initial_capital
        self._positions: dict[str, Position] = {}
        self._trade_log: list[dict] = []
        self._prices: dict[str, tuple[float, float]] = {}

    # -- Broker ABC ----------------------------------------------------

    def place_order(self, order: Order) -> Order:
        bid, ask = self._prices.get(order.symbol, (1.1000, 1.1002))
        fill_price = ask if order.side == "buy" else bid
        cost = abs(order.quantity) * fill_price * 0.0001  # 1 bp round-trip

        current = self._positions.get(order.symbol)
        old_qty = current.quantity if current else 0.0
        new_qty = old_qty + (order.quantity if order.side == "buy" else -order.quantity)
        avg_price = (
            fill_price if new_qty == 0 or old_qty == 0
            else (old_qty * (current.avg_price if current else 0) + order.quantity * fill_price) / new_qty
        )

        self._positions[order.symbol] = Position(
            symbol=order.symbol,
            quantity=new_qty,
            avg_price=avg_price,
        )
        self._equity -= cost
        order.status = OrderStatus.FILLED
        self._trade_log.append({
            "ts": datetime.now(UTC),
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "fill_price": fill_price,
            "cost": cost,
        })
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_order(self, order_id: str) -> Order:
        raise NotImplementedError

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> Account:
        return Account(
            balance=self._capital,
            equity=self._equity,
            margin_used=abs(sum(p.quantity * p.avg_price for p in self._positions.values())) * 0.02,
        )

    def get_price(self, symbol: str) -> tuple[float, float]:
        return self._prices.get(symbol, (1.1000, 1.1002))

    async def stream_prices(self, symbols: list[str]):
        import asyncio
        while True:
            for sym in symbols:
                yield {"symbol": sym, "bid": 1.1000, "ask": 1.1002, "ts": datetime.now(UTC).isoformat()}
            await asyncio.sleep(1.0)

    # -- Test helpers --------------------------------------------------

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        self._prices[symbol] = (bid, ask)

    @property
    def equity(self) -> float:
        return self._equity
