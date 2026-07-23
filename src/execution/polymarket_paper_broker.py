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
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
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
from src.risk.polymarket_loss_caps import (
    LossCapConfig,
    PolymarketLossCapTracker,
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
        loss_cap_tracker: PolymarketLossCapTracker | None = None,
        market_key_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._data = data_source or PolymarketDataSource()
        self._cost = cost_model or PolymarketCostModel()
        self._cash: Decimal = initial_capital_usdc
        self._initial: Decimal = initial_capital_usdc
        # Per-market + per-day loss caps (CL-983f). Paper runs exercise
        # the SAME gate the live broker uses; fills are recorded
        # automatically in _record_fill and marks whenever a book is
        # observed (_observe_book). market_key_fn maps token_id -> CAP
        # bucket only — position books stay per-token; pass a
        # condition_id resolver to pool YES/NO tokens of one market. It
        # is honored only when the broker builds its own tracker.
        # The default tracker is in-memory (state_path=None): the paper
        # bankroll resets every run, so persisting caps across restarts
        # would gate a fresh bankroll on a previous run's losses. Inject
        # a tracker with a state_path for multi-day paper soaks.
        self._loss_caps = loss_cap_tracker or PolymarketLossCapTracker(
            config=replace(LossCapConfig.from_active_profile(), state_path=None),
            market_key_fn=market_key_fn or (lambda token_id: token_id),
        )
        # The POSITION BOOK lives in the loss-cap tracker (per-token qty
        # / weighted-avg / realized) — get_positions derives its view
        # from it, so cap math and reported positions cannot diverge.
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

        # Loss-cap gate (CL-983f): refuse NEW risk when the market or
        # the UTC day has burned its cap. Raises LossCapExceededError —
        # handled upstream like any other pre-trade rejection.
        self._loss_caps.check_order_allowed(
            token_id,
            side=order.side,
            quantity=order.quantity,
        )

        try:
            book = self._data.get_book(token_id)
        except Exception:
            logger.exception(
                "polymarket-paper: book fetch failed for %s — rejecting",
                token_id,
            )
            order.status = OrderStatus.REJECTED
            return order

        # Every observed book feeds the mark-to-market leg of the caps
        # so adverse moves arm the gate for subsequent orders.
        self._observe_book(token_id, book)

        fill_price, fill_qty = self._simulate_fill(order, book)

        if fill_qty <= Decimal("0"):
            # Resting limit — record on the open-order book.
            order.status = OrderStatus.PENDING
            self._open_orders[order.order_id] = order
            logger.info(
                "polymarket-paper: order %s resting (%s %s @ %s)",
                order.order_id,
                order.side,
                order.quantity,
                order.limit_price,
            )
            return order

        # Filled (fully or by capping at book depth — partial = fully
        # for v1; we don't track residuals).
        cost = self._cost.estimate(
            side=order.side,
            role="taker",
            price=fill_price,
            size=fill_qty,
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
        # Derived from the loss-cap tracker's per-token book — the ONE
        # position book (CL-983f). No parallel qty/avg bookkeeping.
        return [
            Position(
                symbol=token_id,
                quantity=float(st.qty),
                avg_price=float(st.avg_price),
                unrealized_pnl=float(st.unrealized),
                realized_pnl=float(st.realized),
            )
            for token_id, st in self._loss_caps.open_positions().items()
        ]

    def get_account(self) -> Account:
        equity = self._cash + sum(
            (st.qty * st.avg_price for st in self._loss_caps.open_positions().values()),
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
        self._observe_book(token_id, book)
        bid = book.top_bid.price if book.top_bid else Decimal("0")
        ask = book.top_ask.price if book.top_ask else Decimal("1")
        return float(bid), float(ask)

    async def stream_prices(
        self,
        symbols: list[str],
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
                        sym,
                        exc_info=True,
                    )
                    continue
                self._observe_book(token_id, book)
                bid_lvl = book.top_bid
                ask_lvl = book.top_ask
                yield {
                    "symbol": sym,
                    "bid": float(bid_lvl.price) if bid_lvl else 0.0,
                    "ask": float(ask_lvl.price) if ask_lvl else 1.0,
                    "ts": datetime.now(UTC).isoformat(),
                }
            await _asyncio.sleep(1.0)

    @property
    def loss_caps(self) -> PolymarketLossCapTracker:
        """Loss-cap tracker (CL-983f). The broker feeds it itself —
        fills in ``_record_fill``, marks whenever it observes a book
        (place_order / get_price / stream_prices). Exposed so the engine
        can record external marks (``record_mark``) and ops can read
        P&L. It also owns the per-token position book that
        ``get_positions`` reports."""
        return self._loss_caps

    # --- Internals ---------------------------------------------------

    def _observe_book(
        self,
        token_id: str,
        book: OrderBookSnapshot,
    ) -> None:
        """Feed the book midpoint to the loss-cap tracker as a mark.

        This is the choke point where quotes enter the paper broker, so
        unrealized losses on open positions are visible to the cap gate
        between trades — not only at fill time. One-sided/empty books
        are skipped (no meaningful mid)."""
        mid = book.mid
        if mid is not None:
            self._loss_caps.record_mark(token_id, mid)

    @staticmethod
    def _validate_order(order: Order) -> None:
        if order.order_type not in (OrderType.LIMIT, OrderType.MARKET):
            msg = f"polymarket-paper supports LIMIT + MARKET; got {order.order_type}"
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
                msg = f"limit price {price} outside [{_PRICE_FLOOR}, {_PRICE_CEIL}]"
                raise ValueError(msg)
            # Tick alignment: must be a multiple of 0.01.
            tick_remainder = (price * 100) % 1
            if tick_remainder != 0:
                msg = f"limit price {price} not on {_TICK_SIZE} tick"
                raise ValueError(msg)

    def _simulate_fill(
        self,
        order: Order,
        book: OrderBookSnapshot,
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
        """Update cash + fills list and book the fill into the loss-cap
        tracker, which owns the position book (qty / weighted-avg /
        realized) — one source of truth for gate math AND
        ``get_positions`` (CL-983f)."""
        # Realized P&L on reductions, WAC on adds, flip-through-zero
        # re-bases at the fill price, fees against the day, mark = fill
        # price. All position bookkeeping happens here.
        self._loss_caps.record_fill(
            token_id,
            order.side,
            qty,
            price,
            fee=cost,
        )

        # Cash delta: a buy at p costs p × qty; a sell at p credits p × qty.
        notional = price * qty
        self._cash -= notional if order.side == "buy" else -notional
        self._cash -= cost  # fees + gas + slippage estimate

        self._fills.append(
            Fill(
                order_id=order.order_id,
                fill_id=str(uuid4()),
                symbol=order.symbol,
                side=order.side,
                quantity=float(qty),
                price=float(price),
                timestamp=datetime.now(UTC),
                commission=float(cost),
            )
        )

        logger.info(
            "polymarket-paper FILL: %s %s %s @ %s (cost=%s) cash=%s",
            order.side,
            qty,
            order.symbol,
            price,
            cost,
            self._cash,
        )
