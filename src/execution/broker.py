"""Broker interface — abstract base class and dataclasses for orders, positions, accounts."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "GTC"
    order_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class Fill:
    order_id: str
    fill_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0


def canonical_symbol(symbol: str) -> str:
    """Canonical instrument key for POSITION MATCHING (CL-qqra).

    The repo speaks two symbol dialects: OANDA-underscore (``EUR_USD``,
    ``USD_NOK`` — event legs, instrument maps) and compact (``EURUSD`` —
    broker positions after ``_from_oanda``, rate-diff strategies). Comparing
    them raw silently fails: the OMS looked up ``current_positions.get(
    "USD_CAD")`` against broker keys like ``"USDCAD"``, always got 0, and so
    computed exit deltas of 0 — EVENT POSITIONS NEVER CLOSED AT THE BROKER
    (observed live: the reconciler's restart flatten, not the strategy's time
    stop, is what closed the first two event legs). Entries likewise saw
    "flat" and could stack.

    This is the matching key ONLY — strip separators, upper-case. Order
    ROUTING keeps the original symbol (OandaBroker._to_oanda is idempotent
    for both dialects). The full canonical-instrument-type refactor is the
    structural epic; every position lookup must go through this until then.
    """
    return str(symbol).replace("_", "").replace("/", "").replace("-", "").upper()


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Account:
    balance: float
    equity: float
    margin_used: float = 0.0
    margin_available: float = 0.0


class Broker(ABC):
    """Abstract broker interface for order placement and account management."""

    @abstractmethod
    def place_order(self, order: Order) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_order(self, order_id: str) -> Order: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_price(self, symbol: str) -> tuple[float, float]:  # (bid, ask)
        ...

    @abstractmethod
    def stream_prices(
        self, symbols: list[str],
    ) -> AsyncIterator[dict[str, Any]]:
        # Implementations are async generators (`async def ... yield`).
        # ABC uses non-async `def` so the declared return type is
        # AsyncIterator directly, not Coroutine[..., AsyncIterator].
        ...
