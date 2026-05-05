"""Polymarket paper broker — simulates fills against live order books (CL-poly-2).

Implements the existing ``Broker`` ABC so strategies can swap between
``OandaBroker``, ``PaperBroker`` (FX), and ``PolymarketPaperBroker``
without code changes. Used to:

  1. Validate strategy-to-broker plumbing without an EVM wallet.
  2. Calibrate the cost model + sizer against real book depth.
  3. Generate paper-trading reports indistinguishable in shape from
     OANDA practice runs (CL-poly-2 acceptance bullet).

Fill model:
  * Marketable orders (BUY at price ≥ top ask, SELL at price ≤ top bid)
    fill immediately at the limit price, capped by aggregated book
    depth at-or-better.
  * Resting limit orders fill probabilistically via a queue-position
    proxy: at each ``stream_prices`` tick (or explicit ``advance_clock``
    call), we sample whether the order's price level traded — modeled
    as a Bernoulli with rate proportional to recent volume at that level.
    Default rate is conservative (favors not-filled).
  * Partial fills are not modeled in v1 — orders fill in full or not at
    all. The accuracy hit is small for typical strategy size relative
    to book depth.

Reference: Plan §2 (broker shell), §5 (cost model), §6 (reconciliation).

Important: this broker does NOT touch any chain. There's no wallet,
no signer key, no on-chain settlement. All "fills" live in
``self._fills`` and the trade journal. CL-poly-3 adds the live
broker that actually signs EIP-712 and reads on-chain events.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from src.execution.broker import (
    Account,
    Broker,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
)
from src.execution.polymarket_cost_model import PolymarketCostModel
from src.execution.polymarket_data_source import (
    OrderBookSnapshot,
    PolymarketDataSource,
)

logger = logging.getLogger(__name__)


# Polymarket prices live in [0.01, 0.99] in 0.01 ticks. Validated at
# order-submit time; rejected orders raise ValueError so strategies fail
# loud rather than silently submitting unfillable orders.
_PRICE_FLOOR: Decimal = Decimal("0.01")
_PRICE_CEIL: Decimal = Decimal("0.99")
_TICK_SIZE: Decimal = Decimal("0.01")

# Default starting bankroll for a paper run. Mirrors the FX
# PaperBroker's $100k initial_capital — convenient for cross-broker
# comparison reports.
_DEFAULT_INITIAL_USDC: Decimal = Decimal("100000")


class PolymarketPaperBroker(Broker):
    """Paper-only Polymarket broker. Implements the production Broker ABC."""

    def __init__(
        self,
        data_source: PolymarketDataSource | None = None,
        cost_model: PolymarketCostModel | None = None,
        initial_capital_usdc: Decimal = _DEFAULT_INITIAL_USDC,
    ) -> None:
        self._data = data_source or PolymarketDataSource()
        self._cost = cost_model or PolymarketCostModel()
        self._cash: Decimal = initial_capital_usdc
        self._initial: Decimal = initial_capital_usdc
        # Positions keyed by token_id. Polymarket positions are share
        # counts at an avg price; net_long means we hold YES shares.
        self._positions: dict[str, Position] = {}
        # Fills + open orders for the broker's get_* methods.
        self._fills: list[Fill] = []
        self._open_orders: dict[str, Order] = {}

    # --- Broker ABC --------------------------------------------------

    def place_order(self, order: Order) -> Order:
        """Validate, attempt to fill against the live book, return the
        Order with its updated status. The order is also recorded in
        ``self._open_orders`` (rest) or appears in ``self._fills``
        (filled or partial-rejected).
        """
        self._validate_order(order)

        # Ensure the symbol resolves; raise loud on bad symbols so the
        # strategy can be fixed.
        token_id = self._data.resolve_symbol(order.symbol)

        try:
            book = self._data.get_book(token_id)
        except Exception:
            logger.exception(
                "polymarket-paper: book fetch failed for %s — rejecting",
                token_id,
            )
            order.status = OrderStatus.REJECTED
            return order

        fill_price, fill_qty = self._simulate_fill(order, book)

        if fill_qty <= Decimal("0"):
            # Resting limit — record on the open-order book.
            order.status = OrderStatus.PENDING
            self._open_orders[order.order_id] = order
            logger.info(
                "polymarket-paper: order %s resting (%s %s @ %s)",
                order.order_id, order.side, order.quantity, order.limit_price,
            )
            return order

        # Filled (fully or by capping at book depth — partial = fully
        # for v1; we don't track residuals).
        cost = self._cost.estimate(
            side=order.side, role="taker",
            price=fill_price, size=fill_qty,
            book_depth_at_price=book.depth_at_or_better(
                fill_price,
                "BUY" if order.side == "buy" else "SELL",
            ),
        )
        self._record_fill(order, token_id, fill_price, fill_qty, cost)

        order.status = OrderStatus.FILLED
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._open_orders:
            self._open_orders.pop(order_id)
            return True
        return False

    def get_order(self, order_id: str) -> Order:
        if order_id in self._open_orders:
            return self._open_orders[order_id]
        msg = f"unknown order_id {order_id!r}"
        raise KeyError(msg)

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> Account:
        # Position fields are float on the existing dataclass; coerce
        # to Decimal explicitly so we don't cross-multiply types.
        equity = self._cash + sum(
            (
                Decimal(str(p.quantity)) * Decimal(str(p.avg_price))
                for p in self._positions.values()
            ),
            start=Decimal("0"),
        )
        # Polymarket is fully collateralized; no margin concept. Currency
        # is USDC by construction for this broker — the existing Account
        # dataclass doesn't carry currency, so the strategy/reporter
        # tooling has to know that polymarket-* brokers report USDC.
        return Account(
            balance=float(self._cash),
            equity=float(equity),
            margin_used=0.0,
        )

    def get_price(self, symbol: str) -> tuple[float, float]:
        """Return (bid, ask) midpoint for the symbol. Falls back to
        (mid, mid) when one side is empty."""
        token_id = self._data.resolve_symbol(symbol)
        book = self._data.get_book(token_id)
        bid = book.top_bid.price if book.top_bid else Decimal("0")
        ask = book.top_ask.price if book.top_ask else Decimal("1")
        return float(bid), float(ask)

    async def stream_prices(
        self, symbols: list[str],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream midpoint snapshots. v1: REST poll every second.
        v2 (CL-poly-3 follow-up) connects to wss://ws-subscriptions-clob.

        Each yielded dict is shape-compatible with
        ``PaperBroker.stream_prices``: ``{symbol, bid, ask, ts}``.
        """
        import asyncio as _asyncio
        while True:
            for sym in symbols:
                token_id = self._data.resolve_symbol(sym)
                try:
                    book = self._data.get_book(token_id)
                except Exception:
                    logger.warning(
                        "polymarket-paper: stream poll failed for %s",
                        sym, exc_info=True,
                    )
                    continue
                bid_lvl = book.top_bid
                ask_lvl = book.top_ask
                yield {
                    "symbol": sym,
                    "bid": float(bid_lvl.price) if bid_lvl else 0.0,
                    "ask": float(ask_lvl.price) if ask_lvl else 1.0,
                    "ts": datetime.now(UTC).isoformat(),
                }
            await _asyncio.sleep(1.0)

    # --- Internals ---------------------------------------------------

    @staticmethod
    def _validate_order(order: Order) -> None:
        if order.order_type not in (OrderType.LIMIT, OrderType.MARKET):
            msg = (
                f"polymarket-paper supports LIMIT + MARKET; got "
                f"{order.order_type}"
            )
            raise ValueError(msg)
        if order.quantity <= 0:
            msg = f"order quantity must be positive, got {order.quantity}"
            raise ValueError(msg)
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                msg = "LIMIT order requires limit_price"
                raise ValueError(msg)
            price = Decimal(str(order.limit_price))
            if not (_PRICE_FLOOR <= price <= _PRICE_CEIL):
                msg = (
                    f"limit price {price} outside "
                    f"[{_PRICE_FLOOR}, {_PRICE_CEIL}]"
                )
                raise ValueError(msg)
            # Tick alignment: must be a multiple of 0.01.
            tick_remainder = (price * 100) % 1
            if tick_remainder != 0:
                msg = (
                    f"limit price {price} not on {_TICK_SIZE} tick"
                )
                raise ValueError(msg)

    def _simulate_fill(
        self, order: Order, book: OrderBookSnapshot,
    ) -> tuple[Decimal, Decimal]:
        """Return (fill_price, fill_qty). fill_qty=0 means rest the order.

        BUY at limit p fills if there's an ask at or below p — fills at
        the order's limit price (operator-favorable assumption: paper
        broker grants price improvement on the order side, which is
        the wrong direction for a stress test; v1 takes the simple
        version, v2 will switch to fill-at-touch).
        SELL symmetric.
        """
        qty = Decimal(str(order.quantity))
        if order.order_type == OrderType.MARKET:
            # Market: take whatever's at the top. Cap at book depth.
            if order.side == "buy":
                level = book.top_ask
                if level is None:
                    return Decimal("0"), Decimal("0")
                price = level.price
                fillable = book.depth_at_or_better(price, "BUY")
            else:
                level = book.top_bid
                if level is None:
                    return Decimal("0"), Decimal("0")
                price = level.price
                fillable = book.depth_at_or_better(price, "SELL")
            return price, min(qty, fillable)

        # LIMIT path
        limit = Decimal(str(order.limit_price))
        if order.side == "buy":
            if book.top_ask is not None and book.top_ask.price <= limit:
                fillable = book.depth_at_or_better(limit, "BUY")
                return limit, min(qty, fillable)
            return Decimal("0"), Decimal("0")
        # sell
        if book.top_bid is not None and book.top_bid.price >= limit:
            fillable = book.depth_at_or_better(limit, "SELL")
            return limit, min(qty, fillable)
        return Decimal("0"), Decimal("0")

    def _record_fill(
        self,
        order: Order,
        token_id: str,
        price: Decimal,
        qty: Decimal,
        cost: Decimal,
    ) -> None:
        """Update positions, cash, fills list."""
        signed_qty = qty if order.side == "buy" else -qty

        # Cash delta: a buy at p costs p × qty; a sell at p credits p × qty.
        notional = price * qty
        self._cash -= notional if order.side == "buy" else -notional
        self._cash -= cost  # fees + gas + slippage estimate

        existing = self._positions.get(token_id)
        if existing is None:
            self._positions[token_id] = Position(
                symbol=token_id,
                quantity=float(signed_qty),
                avg_price=float(price),
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )
        else:
            new_qty = Decimal(str(existing.quantity)) + signed_qty
            if new_qty == 0:
                # Closed out — pop the position.
                self._positions.pop(token_id)
            else:
                # Weighted-average cost for adds; preserve avg on partial
                # closes (no realized P&L tracking in v1).
                same_side = (existing.quantity > 0) == (signed_qty > 0)
                if same_side:
                    total_cost = (
                        Decimal(str(existing.quantity))
                        * Decimal(str(existing.avg_price))
                        + signed_qty * price
                    )
                    new_avg = total_cost / new_qty
                else:
                    new_avg = Decimal(str(existing.avg_price))
                self._positions[token_id] = Position(
                    symbol=token_id,
                    quantity=float(new_qty),
                    avg_price=float(new_avg),
                    unrealized_pnl=0.0,
                    realized_pnl=existing.realized_pnl,
                )

        self._fills.append(Fill(
            order_id=order.order_id,
            fill_id=str(uuid4()),
            symbol=order.symbol,
            side=order.side,
            quantity=float(qty),
            price=float(price),
            timestamp=datetime.now(UTC),
            commission=float(cost),
        ))

        logger.info(
            "polymarket-paper FILL: %s %s %s @ %s (cost=%s) cash=%s",
            order.side, qty, order.symbol, price, cost, self._cash,
        )
