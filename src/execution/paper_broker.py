"""Paper broker — simulated execution with cost model, full P&L tracking."""

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from .broker import Account, Broker, Order, OrderStatus, Position

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self._capital = initial_capital
        self._equity = initial_capital
        self._peak_equity = initial_capital
        self._positions: dict[str, Position] = {}
        self._trade_log: list[dict[str, Any]] = []
        self._prices: dict[str, tuple[float, float]] = {}

    # -- Broker ABC ----------------------------------------------------

    def place_order(self, order: Order) -> Order:
        bid, ask = self._prices.get(order.symbol, (1.1000, 1.1002))
        fill_price = ask if order.side == "buy" else bid

        # Slippage enforcement (CL-qyav) — simulated equivalent of OANDA's
        # FOK priceBound: reference is the current mid; a fill beyond
        # mid*(1±bps/1e4) is REJECTED (buy bound above, sell bound below).
        # The OMS raises REJECTED through BrokerRejectedOrderError, same as
        # a venue reject.
        if order.max_slippage_bps is not None and order.max_slippage_bps > 0:
            mid = (bid + ask) / 2.0
            frac = order.max_slippage_bps / 10_000.0
            bound = mid * (1 + frac) if order.side == "buy" else mid * (1 - frac)
            beyond = (
                fill_price > bound if order.side == "buy" else fill_price < bound
            )
            if beyond:
                order.status = OrderStatus.REJECTED
                order.reject_reason = (
                    f"SLIPPAGE_EXCEEDED: {order.side} fill {fill_price:.6f} "
                    f"beyond bound {bound:.6f} (mid {mid:.6f}, "
                    f"max {order.max_slippage_bps} bps)"
                )
                logger.warning(
                    "PaperBroker REJECTED %s %s x%s: %s",
                    order.symbol, order.side, order.quantity,
                    order.reject_reason,
                )
                return order

        cost = abs(order.quantity) * fill_price * 0.0001  # 1 bp round-trip

        current = self._positions.get(order.symbol)
        old_qty = current.quantity if current else 0.0
        old_avg = current.avg_price if current else 0.0
        realized = current.realized_pnl if current else 0.0
        delta = order.quantity if order.side == "buy" else -order.quantity
        new_qty = old_qty + delta

        # Average-entry accounting (CL-qyav P2 fix): the old formula blended
        # the fill into the basis on EVERY order, so a reduce corrupted
        # avg_price instead of realizing P&L, and a flip inherited a
        # nonsensical basis. Rules:
        #   open / add same-direction → volume-weighted average basis
        #   reduce / full close      → basis UNCHANGED, P&L realized on the
        #                              closed quantity
        #   flip through zero        → realize P&L on the whole old position,
        #                              basis resets to the fill price for the
        #                              residual quantity
        direction = 1.0 if old_qty > 0 else -1.0
        if old_qty == 0.0:
            avg_price = fill_price
        elif (old_qty > 0) == (delta > 0):
            # Adding to an existing position (delta == 0 degenerates to
            # one of the branches below with a no-op result).
            avg_price = (
                abs(old_qty) * old_avg + abs(delta) * fill_price
            ) / abs(new_qty)
        elif abs(delta) <= abs(old_qty):
            # Partial reduce or full close: realize on the closed quantity.
            realized += abs(delta) * (fill_price - old_avg) * direction
            avg_price = old_avg
        else:
            # Flip through zero: close the whole old position, residual
            # opens at the fill price.
            realized += abs(old_qty) * (fill_price - old_avg) * direction
            avg_price = fill_price

        realized_delta = realized - (current.realized_pnl if current else 0.0)
        self._positions[order.symbol] = Position(
            symbol=order.symbol,
            quantity=new_qty,
            avg_price=avg_price,
            realized_pnl=realized,
        )
        self._equity += realized_delta - cost
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
        """Bid/ask for a configured symbol. RAISES on unknown symbols
        (ultrareview follow-up): the old silent (1.1000, 1.1002) default was
        the same fabricated-price fail-open as the coordinator's mid=1.0 —
        it priced gold/indices as if they were EURUSD in paper soaks and
        masked missing set_price wiring. Callers that can tolerate a missing
        price (coordinator._get_price) catch and fail closed."""
        try:
            return self._prices[symbol]
        except KeyError:
            msg = (
                f"PaperBroker has no price for {symbol!r} — call "
                f"set_price('{symbol}', bid, ask) first (known: "
                f"{sorted(self._prices)})"
            )
            raise KeyError(msg) from None

    async def stream_prices(
        self, symbols: list[str],
    ) -> AsyncIterator[dict[str, Any]]:
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
