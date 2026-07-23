"""Paper broker — simulated execution with cost model, full P&L tracking."""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from .broker import Account, Broker, Order, OrderStatus, Position

logger = logging.getLogger(__name__)

#: Half-spread applied around the set_price mid by stream_prices, in basis
#: points (CL-8lv6): a deterministic ±0.5 bp synthetic book.
STREAM_HALF_SPREAD_BPS = 0.5


class PaperBroker(Broker):
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        stream_interval_sec: float = 1.0,
    ) -> None:
        self._capital = initial_capital
        self._stream_interval_sec = stream_interval_sec
        self._equity = initial_capital
        self._peak_equity = initial_capital
        self._positions: dict[str, Position] = {}
        self._trade_log: list[dict[str, Any]] = []
        self._prices: dict[str, tuple[float, float]] = {}

    # -- Broker ABC ----------------------------------------------------

    def place_order(self, order: Order) -> Order:
        quote = self._prices.get(order.symbol)
        if quote is None:
            # CL-n3pt (P1): fail CLOSED, like get_price/stream. A symbol with
            # no set_price must NOT fill at a fabricated ~1.10 book — metals
            # and commodities trade in the 100s-1000s, so a 1.10 fill produces
            # nonsense fills, PnL and stops in a soak. REJECT so it flows
            # through the OMS's BrokerRejectedOrderError path exactly like a
            # venue reject, rather than silently filling at a made-up price.
            order.status = OrderStatus.REJECTED
            order.reject_reason = (
                f"NO_PRICE: PaperBroker has no price for {order.symbol!r} "
                "(set_price never called)"
            )
            logger.warning(
                "PaperBroker REJECTED %s %s x%s — no price set",
                order.symbol, order.side, order.quantity,
            )
            return order
        bid, ask = quote
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
        """Synthetic ticks for PRICED symbols only — fail closed (CL-8lv6 P0).

        The old implementation fabricated bid=1.1000/ask=1.1002 for EVERY
        requested symbol, ignoring set_price: the live engine's paper mode
        fills _last_prices from this stream, so all instruments were marked
        ~1.10 and paper-soak sizing/stops/PnL were meaningless. Same policy
        as get_price now: a symbol without a set_price value is ABSENT from
        the stream — never fabricated. Priced symbols tick around the
        CURRENT set_price mid with a deterministic ±STREAM_HALF_SPREAD_BPS
        synthetic spread, so set_price updates are reflected on the next
        pass. Tick shape matches OandaBroker.stream_prices:
        {symbol, bid, ask, ts}.
        """
        while True:
            for sym in symbols:
                quote = self._prices.get(sym)
                if quote is None:
                    continue  # fail closed: no set_price → no tick
                bid, ask = quote
                mid = (bid + ask) / 2.0
                half = mid * (STREAM_HALF_SPREAD_BPS / 10_000.0)
                yield {
                    "symbol": sym,
                    "bid": mid - half,
                    "ask": mid + half,
                    "ts": datetime.now(UTC).isoformat(),
                }
            await asyncio.sleep(self._stream_interval_sec)

    # -- Test helpers --------------------------------------------------

    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        self._prices[symbol] = (bid, ask)

    @property
    def equity(self) -> float:
        return self._equity
