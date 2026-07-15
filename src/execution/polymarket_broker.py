"""Polymarket live broker — EIP-712 signing via py-clob-client (CL-poly-3 scaffold).

⚠️  HARD-GATED on mainnet. Live mainnet trading is REJECTED at construction
unless ALL acceptance gates in CL-poly-3 are satisfied. See
``polymarket_preflight.py`` for the gate list.

Implements the existing ``Broker`` ABC. py-clob-client owns the EIP-712
typed-data construction, signing, order submission, and CLOB REST
endpoints. We don't reimplement signing — that surface area is
security-sensitive and the SDK is the canonical reference.

Reference: Plan §2.

To install py-clob-client: ``pip install py-clob-client``. The dep is
in the ``polymarket`` extras (``pip install '.[polymarket]'``) so the
base curLit install doesn't pull EVM tooling.

The class structure mirrors OandaBroker so swap-in is mechanical:
strategies see the same place_order / get_positions / get_account /
stream_prices ABC and don't know which broker they're talking to.

This module is a SCAFFOLD. The acceptance gates in CL-poly-3 are:
  * py-clob-client installed in production
  * Vault secrets loaded on production server
  * Funder wallet funded with USDC.e + MATIC
  * CTFExchange allowance set to working-cap + buffer (NOT MaxUint256)
  * Amoy testnet end-to-end run clean
  * Safety review by a second engineer
  * Kill-switch tested
  * Per-market + per-day loss caps wired
  * Resolution-monitor process running
  * One week mainnet at $100 cap with clean reconciliation

Until those land, ``--broker polymarket-mainnet`` raises in run_engine.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from typing import Any

from src.execution.broker import (
    Account,
    Broker,
    Order,
    OrderStatus,
    OrderType,
    Position,
)
from src.execution.polymarket_secrets import load_polymarket_creds
from src.risk.polymarket_loss_caps import (
    LossCapConfig,
    PolymarketLossCapTracker,
)

logger = logging.getLogger(__name__)


# Polymarket CLOB hosts. Amoy host changes when the testnet is
# redeployed; verify before each new test run via Polymarket docs.
_HOSTS: dict[str, str] = {
    "mainnet": "https://clob.polymarket.com",
    "amoy":    "https://clob-amoy.polymarket.com",
}

# Polymarket signature_type values:
#   0 = direct EOA. Simplest custody. Recommended (Plan §9 q1).
#   1 = EOA proxy.
#   2 = magic-link funder.
# Default to 0; production operator may override.
_DEFAULT_SIGNATURE_TYPE: int = 0


class PolymarketBroker(Broker):
    """Live Polymarket broker via py-clob-client.

    Construction order:
      1. Load creds from vault.
      2. Instantiate ClobClient with signer key + funder address.
      3. Hydrate API credentials (or derive on first run).
      4. Caller is expected to have already run preflight (the
         run_engine wiring does this).
    """

    def __init__(
        self,
        env: str = "mainnet",
        signature_type: int = _DEFAULT_SIGNATURE_TYPE,
        loss_cap_tracker: PolymarketLossCapTracker | None = None,
        market_key_fn: Callable[[str], str] | None = None,
    ) -> None:
        if env not in _HOSTS:
            msg = f"unknown polymarket env: {env}"
            raise ValueError(msg)
        # Lazy import — keeps the base curLit install free of py-clob-client.
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.constants import AMOY, POLYGON
        except ImportError as exc:
            msg = (
                "py-clob-client not installed. Install via "
                "`pip install py-clob-client` (polymarket extras)."
            )
            raise ImportError(msg) from exc

        creds = load_polymarket_creds(env)
        chain_id = POLYGON if env == "mainnet" else AMOY

        self._client = ClobClient(
            host=_HOSTS[env],
            key=creds.signer_pk,
            chain_id=chain_id,
            funder=creds.funder_address,
            signature_type=signature_type,
        )
        # Derive (or load) the L2 API creds the CLOB authenticates with.
        # The SDK caches these on disk by default; we don't fight it
        # here, but the production preflight asserts the cache lives in
        # a secure path (CL-poly-3 acceptance).
        self._client.set_api_creds(self._client.create_or_derive_api_creds())
        self._env = env
        # Per-market + per-day loss caps (CL-983f acceptance gate).
        # place_order refuses new risk once a cap is burned. Fills are
        # async on the CLOB, so the fill/reconciliation loop feeds the
        # tracker via ``self.loss_caps.record_fill`` / ``record_mark``.
        # market_key_fn maps token_id -> market key (default identity;
        # pass a condition_id resolver to pool YES/NO tokens).
        self._loss_caps = loss_cap_tracker or PolymarketLossCapTracker(
            config=LossCapConfig.from_active_profile(),
        )
        self._market_key_fn = market_key_fn or (lambda token_id: token_id)

    # --- Broker ABC --------------------------------------------------

    def place_order(self, order: Order) -> Order:
        """Sign + submit a limit order to the CLOB.

        ``order.symbol`` must be a CLOB token_id (resolved upstream by
        ``PolymarketDataSource.resolve_symbol``).
        ``order.side`` is "buy" / "sell".
        ``order.limit_price`` must be in [0.01, 0.99] in 0.01 ticks.
        """
        # Lazy import — confined to the live path.
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.clob_types import OrderType as ClobOrderType

        self._validate_order(order)

        # Loss-cap gate (CL-983f): refuse NEW risk when this market or
        # the UTC day has burned its cap. Raises LossCapExceededError BEFORE
        # anything is signed — handled upstream like any other
        # pre-trade rejection (RejectionHandler -> ABORT, no retry).
        self._loss_caps.check_order_allowed(
            self._market_key_fn(order.symbol), side=order.side,
        )

        args = OrderArgs(
            token_id=order.symbol,
            price=float(order.limit_price or 0),
            size=float(order.quantity),
            side="BUY" if order.side == "buy" else "SELL",
        )
        signed = self._client.create_order(args)
        # GTC = resting limit. FOK / FAK / GTD also supported by the SDK.
        resp = self._client.post_order(signed, ClobOrderType.GTC)
        if not resp.get("success"):
            order.status = OrderStatus.REJECTED
            logger.error("polymarket order rejected: %s", resp)
            return order

        order.order_id = str(resp["orderID"])
        order.status = OrderStatus.PENDING  # CLOB orders are async-fill
        logger.info(
            "polymarket order placed: id=%s side=%s qty=%s @ %s",
            order.order_id, order.side, order.quantity, order.limit_price,
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        resp = self._client.cancel(order_id)
        return bool(resp.get("canceled", False))

    def get_order(self, order_id: str) -> Order:
        raise NotImplementedError(
            "polymarket get_order — fetch via /orders/{id} when needed",
        )

    def get_positions(self) -> list[Position]:
        # CLOB doesn't expose positions directly; the data API does.
        # py-clob-client wraps it.
        raw = self._client.get_positions()
        return [self._raw_to_position(p) for p in raw]

    def get_account(self) -> Account:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        usdc = self._client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
        )
        # USDC has 6 decimals on-chain; convert to human float.
        balance = float(Decimal(usdc["balance"]) / Decimal(10**6))
        return Account(
            balance=balance,
            equity=balance,         # No mark-to-market on the broker side
            margin_used=0.0,        # Polymarket is fully collateralized
        )

    def get_price(self, symbol: str) -> tuple[float, float]:
        book = self._client.get_order_book(symbol)
        bids = book.bids or []
        asks = book.asks or []
        bid = float(bids[0].price) if bids else 0.0
        ask = float(asks[0].price) if asks else 1.0
        return bid, ask

    async def stream_prices(
        self, symbols: list[str],
    ) -> AsyncIterator[dict[str, Any]]:
        """Polymarket WS at wss://ws-subscriptions-clob.polymarket.com/ws/market.

        Implemented as a placeholder REST-poll in v1; the WS subscription
        is filed as a CL-poly-3 follow-up (the WS message format is
        reasonably stable but adds dep complexity we don't need for
        the initial mainnet-tiny-cap run).
        """
        import asyncio as _asyncio
        from datetime import UTC as _UTC
        from datetime import datetime as _dt
        while True:
            for sym in symbols:
                bid, ask = self.get_price(sym)
                yield {
                    "symbol": sym, "bid": bid, "ask": ask,
                    "ts": _dt.now(_UTC).isoformat(),
                }
            await _asyncio.sleep(2.0)

    @property
    def loss_caps(self) -> PolymarketLossCapTracker:
        """Loss-cap tracker (CL-983f). CLOB fills are async, so the
        fill/reconciliation loop records them here (``record_fill``)
        and streams marks (``record_mark``); place_order reads it."""
        return self._loss_caps

    # --- Internals ---------------------------------------------------

    @staticmethod
    def _validate_order(order: Order) -> None:
        if order.order_type != OrderType.LIMIT:
            msg = (
                f"polymarket live: only LIMIT orders supported in v1, "
                f"got {order.order_type}"
            )
            raise ValueError(msg)
        if order.limit_price is None:
            msg = "polymarket live: LIMIT order requires limit_price"
            raise ValueError(msg)
        price = Decimal(str(order.limit_price))
        if not (Decimal("0.01") <= price <= Decimal("0.99")):
            msg = f"polymarket price {price} outside [0.01, 0.99]"
            raise ValueError(msg)
        if (price * 100) % 1 != 0:
            msg = f"polymarket price {price} not on 0.01 tick"
            raise ValueError(msg)
        if order.quantity <= 0:
            msg = f"polymarket size must be positive, got {order.quantity}"
            raise ValueError(msg)

    @staticmethod
    def _raw_to_position(p: dict[str, Any]) -> Position:
        return Position(
            symbol=str(p["asset"]),
            quantity=float(p["size"]),
            avg_price=float(p["avgPrice"]),
            unrealized_pnl=float(p.get("cashPnl", 0)),
            realized_pnl=0.0,
        )
