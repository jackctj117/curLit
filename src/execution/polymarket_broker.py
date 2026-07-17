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
from collections.abc import AsyncIterator, Callable, Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

from src.execution.broker import (
    Account,
    Broker,
    Order,
    OrderStatus,
    OrderType,
    Position,
)
from src.execution.polymarket_reconciler import OnchainFill
from src.execution.polymarket_secrets import load_polymarket_creds
from src.risk.polymarket_loss_caps import (
    LossCapConfig,
    PolymarketLossCapTracker,
)

logger = logging.getLogger(__name__)

# USDC.e (collateral) has asset id 0 in CTFExchange OrderFilled events;
# both USDC and CTF outcome tokens carry 6 decimals on Polygon.
_COLLATERAL_ASSET_ID: int = 0
_TOKEN_DECIMALS: Decimal = Decimal(10**6)

# Unfed-tracker warning cadence: warn on the first offending order and
# then every Nth so a runaway loop doesn't spam WARNING per order.
_UNFED_WARN_EVERY: int = 25


def _norm_hash(h: str) -> str:
    """Case-fold a tx/order hash and strip any 0x prefix so keys from
    the CLOB REST response and web3 logs compare equal."""
    return h.lower().removeprefix("0x")


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
        self._funder = creds.funder_address
        # Per-market + per-day loss caps (CL-983f acceptance gate).
        # place_order refuses new risk once a cap is burned. The tracker
        # is fed from three sources (see the runtime contract on the
        # ``loss_caps`` property):
        #   1. Immediate CLOB matches recorded by place_order itself.
        #   2. ``ingest_reconciled_fills`` — the runtime loop pushes
        #      on-chain OrderFilled events (idempotent, safe to replay).
        #   3. Marks: place_order records the current book mid before
        #      gating; the runtime price loop should also stream
        #      ``loss_caps.record_mark`` between orders.
        # market_key_fn maps token_id -> CAP bucket only (position books
        # stay per-token); pass a condition_id resolver to pool YES/NO
        # tokens of one market. It is honored only when the broker builds
        # its own tracker — an injected tracker brings its own key fn.
        self._loss_caps = loss_cap_tracker or PolymarketLossCapTracker(
            config=LossCapConfig.from_active_profile(),
            market_key_fn=market_key_fn or (lambda token_id: token_id),
        )
        # Orders placed while the tracker had no fills/marks despite
        # existing exposure — see _warn_if_tracker_unfed.
        self._unfed_order_count = 0

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

        # Refresh the mark from the live book BEFORE gating so the gate
        # sees current unrealized P&L, then fail loud if we're trading
        # with an unfed tracker (CL-983f).
        self._record_book_mark(order.symbol)
        self._warn_if_tracker_unfed()

        # Loss-cap gate (CL-983f): refuse NEW risk when this market or
        # the UTC day has burned its cap. Raises LossCapExceededError BEFORE
        # anything is signed — handled upstream like any other
        # pre-trade rejection (RejectionHandler -> ABORT, no retry).
        self._loss_caps.check_order_allowed(
            order.symbol, side=order.side, quantity=order.quantity,
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
        # Immediate matches are booked into the tracker right here;
        # anything that fills later arrives via ingest_reconciled_fills.
        self._record_immediate_fill(order, resp)
        logger.info(
            "polymarket order placed: id=%s side=%s qty=%s @ %s status=%s",
            order.order_id, order.side, order.quantity, order.limit_price,
            order.status,
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
        """Loss-cap tracker (CL-983f).

        RUNTIME CONTRACT — what the engine loop must call:
          * ``broker.ingest_reconciled_fills(fills)`` with the
            ``OnchainFill`` list from ``polymarket_reconciler.
            fetch_onchain_fills`` on every reconcile pass. Idempotent —
            overlapping block ranges / re-runs cannot double-count.
          * ``broker.loss_caps.record_mark(token_id, price)`` from its
            price feed for every open-position token, so unrealized
            losses trip the caps BETWEEN orders (place_order refreshes
            the mark for its own token, but only when an order is sent).
        The broker records immediate CLOB matches itself. Tracker state
        persists across restarts (see ``LossCapConfig.state_path``).
        Until the loop is wired, place_order logs a WARNING (rate
        limited) whenever it runs with an unfed tracker despite open
        orders/positions existing — the cap gate cannot see losses it
        was never told about.
        """
        return self._loss_caps

    def ingest_reconciled_fills(self, fills: Iterable[OnchainFill]) -> int:
        """Book on-chain ``OrderFilled`` events into the loss-cap tracker.

        The runtime reconcile loop calls this with the output of
        ``polymarket_reconciler.fetch_onchain_fills``. Each fill is
        keyed by ``tx_hash:order_hash`` for idempotency (persisted), so
        re-reconciling an overlapping block range is safe; fills already
        booked as immediate matches by place_order are skipped too.
        Returns the number of fills newly booked.
        """
        funder = self._funder.lower()
        booked = 0
        for f in fills:
            order_hash = _norm_hash(f.order_hash)
            key = f"{_norm_hash(f.tx_hash)}:{order_hash}"
            if self._loss_caps.has_recorded_fill(key) or (
                self._loss_caps.has_recorded_fill(f"order:{order_hash}")
            ):
                continue
            parsed = self._parse_onchain_fill(f, funder)
            if parsed is None:
                logger.warning(
                    "polymarket loss caps: cannot interpret onchain fill "
                    "tx=%s order=%s (maker=%s taker=%s assets=%s/%s) — "
                    "NOT booked",
                    f.tx_hash, f.order_hash, f.maker, f.taker,
                    f.maker_asset_id, f.taker_asset_id,
                )
                continue
            token_id, side, qty, price, fee = parsed
            if self._loss_caps.record_fill(
                token_id, side, qty, price, fee=fee, fill_id=key,
            ):
                booked += 1
        if booked:
            logger.info(
                "polymarket loss caps: booked %d reconciled fill(s)", booked,
            )
        return booked

    # --- Internals ---------------------------------------------------

    @staticmethod
    def _parse_onchain_fill(
        f: OnchainFill, funder_lower: str,
    ) -> tuple[str, str, Decimal, Decimal, Decimal] | None:
        """Map an OrderFilled event to (token_id, side, qty, price, fee)
        from the funder's perspective. Returns None when the fill does
        not involve the funder or isn't a token<->collateral trade.

        The named maker gives makerAsset / receives takerAsset; the
        named taker the mirror. Asset id 0 is USDC.e collateral; both
        legs carry 6 decimals on Polygon.
        """
        if f.maker.lower() == funder_lower:
            gave = (f.maker_asset_id, f.maker_amount_filled)
            got = (f.taker_asset_id, f.taker_amount_filled)
        elif f.taker.lower() == funder_lower:
            gave = (f.taker_asset_id, f.taker_amount_filled)
            got = (f.maker_asset_id, f.maker_amount_filled)
        else:
            return None

        if gave[0] == _COLLATERAL_ASSET_ID and got[0] != _COLLATERAL_ASSET_ID:
            # Paid USDC, received outcome tokens — a buy.
            side, token_id = "buy", str(got[0])
            qty = Decimal(got[1]) / _TOKEN_DECIMALS
            notional = Decimal(gave[1]) / _TOKEN_DECIMALS
        elif got[0] == _COLLATERAL_ASSET_ID and gave[0] != _COLLATERAL_ASSET_ID:
            # Gave outcome tokens, received USDC — a sell.
            side, token_id = "sell", str(gave[0])
            qty = Decimal(gave[1]) / _TOKEN_DECIMALS
            notional = Decimal(got[1]) / _TOKEN_DECIMALS
        else:
            return None
        if qty <= 0:
            return None
        return (
            token_id,
            side,
            qty,
            notional / qty,
            Decimal(f.fee) / _TOKEN_DECIMALS,
        )

    def _record_immediate_fill(
        self, order: Order, resp: dict[str, Any],
    ) -> None:
        """Book a fill reported as matched by the post_order response.

        Sets order.status: matched -> FILLED (or PARTIAL when the CLOB
        reports a smaller matched size), otherwise PENDING (async fill —
        reconciliation will book it). Dedup keys are aligned with the
        reconciler's ``tx_hash:order_hash`` scheme via the response's
        ``transactionsHashes`` so the same match cannot be booked twice.
        """
        status = str(resp.get("status") or "").lower()
        if status != "matched":
            order.status = OrderStatus.PENDING  # CLOB orders are async-fill
            return

        qty = Decimal(str(order.quantity))
        price = Decimal(str(order.limit_price))
        try:
            making = Decimal(str(resp["makingAmount"]))
            taking = Decimal(str(resp["takingAmount"]))
            m_qty, m_notional = (
                (taking, making) if order.side == "buy" else (making, taking)
            )
            if m_qty > 0 and m_notional > 0:
                qty, price = m_qty, m_notional / m_qty
        except (KeyError, InvalidOperation, TypeError):
            # No/garbled size info — assume the full order matched at
            # the limit price; reconciliation trues it up on-chain.
            logger.debug(
                "polymarket: matched response without parsable amounts "
                "(%s) — booking full order size", resp,
            )

        order_hash = _norm_hash(order.order_id)
        txs = [t for t in (resp.get("transactionsHashes") or []) if t]
        keys = [f"{_norm_hash(tx)}:{order_hash}" for tx in txs] or [
            f"order:{order_hash}",
        ]
        self._loss_caps.record_fill(
            order.symbol, order.side, qty, price, fill_id=keys[0],
        )
        for extra in keys[1:]:
            self._loss_caps.register_processed_fill(extra)
        order.status = (
            OrderStatus.FILLED
            if qty >= Decimal(str(order.quantity))
            else OrderStatus.PARTIAL
        )
        logger.info(
            "polymarket immediate fill: %s %s %s @ %s (order %s)",
            order.side, qty, order.symbol, price, order.order_id,
        )

    def _record_book_mark(self, token_id: str) -> None:
        """Best-effort mark from the current order book so the cap gate
        prices unrealized P&L off fresh data. Skips one-sided/empty
        books (0.0 bid / 1.0 ask are get_price's empty sentinels)."""
        try:
            bid, ask = self.get_price(token_id)
        except Exception:
            logger.debug(
                "polymarket: book-mark fetch failed for %s", token_id,
                exc_info=True,
            )
            return
        if not (bid > 0.0 and 0.0 < ask < 1.0):
            return
        mid = (Decimal(str(bid)) + Decimal(str(ask))) / 2
        self._loss_caps.record_mark(token_id, mid)

    def _warn_if_tracker_unfed(self) -> None:
        """Fail-loud on the known wiring gap (CL-983f): placing orders
        while the tracker has NEVER seen a fill or mark even though open
        orders/positions exist means the loss-cap gate is flying blind.
        Logged at WARNING with a counter (first offense + every
        ``_UNFED_WARN_EVERY``th), not per-order spam."""
        if self._loss_caps.has_activity:
            return
        try:
            exposed = bool(self._client.get_orders()) or bool(
                self._client.get_positions(),
            )
        except Exception:
            logger.debug(
                "polymarket: unfed-tracker exposure probe failed",
                exc_info=True,
            )
            return
        if not exposed:
            return
        self._unfed_order_count += 1
        if (
            self._unfed_order_count == 1
            or self._unfed_order_count % _UNFED_WARN_EVERY == 0
        ):
            logger.warning(
                "polymarket loss caps: %d order(s) placed while the "
                "tracker has received no fills or marks despite existing "
                "open orders/positions — the loss-cap gate cannot see "
                "prior losses. Wire the runtime loop to "
                "ingest_reconciled_fills() and loss_caps.record_mark() "
                "(see PolymarketBroker.loss_caps).",
                self._unfed_order_count,
            )

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
