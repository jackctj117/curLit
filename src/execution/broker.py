"""Broker interface — abstract base class and dataclasses for orders, positions, accounts."""

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


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
    #: Venue-supplied reason when status is REJECTED (CL-h4as follow-up).
    reject_reason: str | None = None
    #: Max tolerated slippage vs the reference price, in basis points
    #: (CL-qyav). Set by the OMS from OrderIntent.max_slippage_bps so brokers
    #: can ENFORCE it at execution (OANDA: FOK `priceBound`; PaperBroker:
    #: simulated bound check) instead of merely journaling it. None disables
    #: enforcement (ad-hoc/direct Order construction stays unconstrained).
    max_slippage_bps: float | None = None
    #: True for risk-REDUCING emergency orders (kill-switch flatten/reduce —
    #: OMS sets it from bypass_halt, CL-8lv6). Brokers fail OPEN on ancillary
    #: failures for emergency orders (e.g. place unbound when the pricing
    #: fetch for priceBound dies: getting flat beats slippage protection) and
    #: fail CLOSED for everything else.
    emergency: bool = False


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


class BrokerRejectedOrderError(RuntimeError):
    """A broker returned a REJECTED order status (no exception was raised).

    Raised by the OMS so venue rejects flow through the SAME RejectionHandler
    policy path (classification by message text, halved retries, strategy
    halt) as transport exceptions — ultrareview #2: before this, a REJECTED
    status silently entered _pending forever and was journaled ORDER_PLACED.
    """


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


def currency_pair(symbol: str) -> tuple[str, str] | None:
    """(base, quote) for an FX pair in ANY symbol dialect, or None.

    Raw slicing mis-parses OANDA-form symbols (``'EUR_USD'[3:6] == '_US'``)
    and fabricates currency legs for indices (``SPX500_USD`` → ``'X50'``) —
    CL-rybp. Canonicalize first; only a 6-letter result is a currency pair
    (metals count: ``XAU``/``XAG`` are ISO currency codes). Non-pairs return
    None so callers skip them instead of mis-attributing risk.
    """
    s = canonical_symbol(symbol)
    if len(s) == 6 and s.isalpha():
        return s[:3], s[3:6]
    return None


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


class BrokerCredentialsError(RuntimeError):
    """An OANDA broker mode was requested but credentials are missing.

    Raised by build_fx_broker so a mis-provisioned host FAILS FAST instead of
    silently trading against an in-process PaperBroker while every log line,
    dashboard, and operator believes it is talking to OANDA (CL-qyav P1).
    """


#: Explicit opt-in for the paper fallback when OANDA creds are missing.
#: Anything other than "1" means fail fast.
ALLOW_PAPER_FALLBACK_ENV = "ALLOW_PAPER_FALLBACK"


def build_fx_broker(mode: str, *, initial_paper_capital: float = 100_000.0) -> Broker:
    """Construct the FX broker for ``mode`` — fail-fast credentials policy.

    Modes: ``paper`` | ``oanda-practice`` | ``oanda-live`` (the engine's
    polymarket-* modes are wired separately in run_engine).

    Policy (CL-qyav P1): requesting an OANDA mode without OANDA_API_KEY +
    OANDA_ACCOUNT_ID raises BrokerCredentialsError. The old behavior —
    logging an error and silently returning a PaperBroker — made the system
    lie about its broker mode. The fallback now requires the explicit env
    opt-in ``ALLOW_PAPER_FALLBACK=1`` and logs at CRITICAL when it triggers;
    callers MUST report the effective mode via effective_broker_mode(), not
    the requested one.
    """
    # Local import: paper_broker imports from this module, so a top-level
    # import would be circular.
    from .paper_broker import PaperBroker

    if mode == "paper":
        return PaperBroker(initial_capital=initial_paper_capital)
    if mode in ("oanda-practice", "oanda-live"):
        oanda_key = os.environ.get("OANDA_API_KEY", "")
        oanda_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        if oanda_key and oanda_id:
            from .oanda_broker import OandaBroker
            return OandaBroker(
                oanda_key, oanda_id,
                practice=(mode == "oanda-practice"),
            )
        if os.environ.get(ALLOW_PAPER_FALLBACK_ENV) == "1":
            logger.critical(
                "OANDA mode %r requested but OANDA_API_KEY/OANDA_ACCOUNT_ID "
                "are missing — %s=1 is set, so falling back to an in-process "
                "PaperBroker. THE SYSTEM IS NOT CONNECTED TO OANDA. All "
                "mode/status reporting must reflect 'paper' "
                "(effective_broker_mode).",
                mode, ALLOW_PAPER_FALLBACK_ENV,
            )
            return PaperBroker(initial_capital=initial_paper_capital)
        msg = (
            f"broker mode {mode!r} requires OANDA_API_KEY and "
            f"OANDA_ACCOUNT_ID in the environment (missing: "
            f"{'OANDA_API_KEY ' if not oanda_key else ''}"
            f"{'OANDA_ACCOUNT_ID' if not oanda_id else ''}".rstrip() + "). "
            f"Refusing to silently fall back to PaperBroker — set the "
            f"credentials, or set {ALLOW_PAPER_FALLBACK_ENV}=1 to explicitly "
            f"accept a paper fallback."
        )
        raise BrokerCredentialsError(msg)
    msg = f"unknown FX broker mode: {mode!r}"
    raise ValueError(msg)


def effective_broker_mode(requested_mode: str, broker: Broker) -> str:
    """The mode the system is ACTUALLY running in, for status/log reporting.

    When the ALLOW_PAPER_FALLBACK opt-in downgraded an OANDA request to a
    PaperBroker, every report path must say "paper" — a dashboard claiming
    "oanda-live" over a simulated book is the exact lie the fail-fast policy
    exists to prevent.
    """
    from .paper_broker import PaperBroker

    if isinstance(broker, PaperBroker):
        return "paper"
    return requested_mode
