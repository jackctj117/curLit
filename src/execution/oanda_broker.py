"""OANDA broker — REST + streaming API v20."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any

import httpx

from .broker import Account, Broker, Order, OrderStatus, Position

logger = logging.getLogger(__name__)


class SlippageRefUnavailableError(RuntimeError):
    """The /pricing fetch backing a slippage priceBound failed (CL-8lv6).

    Raised by _compute_price_bound for a NON-emergency order that carries
    max_slippage_bps: rather than silently placing the order UNBOUND (the old
    fail-open), place_order catches this and REJECTS the order with
    reject_reason="SLIPPAGE_REF_UNAVAILABLE" so it flows through the normal
    RejectionHandler path. Emergency (risk-reducing kill-switch) orders do
    NOT raise this — they place unbound with a CRITICAL log.
    """


def _price_bound_str(
    side: str, reference_price: str, max_slippage_bps: float,
) -> str:
    """Direction-aware FOK price bound as an OANDA PriceValue string (CL-qyav).

    buy  → bound ABOVE the reference ask:  ref * (1 + bps/1e4)
    sell → bound BELOW the reference bid:  ref * (1 - bps/1e4)

    Precision is taken from the venue's own quote string (``reference_price``
    as returned by /pricing) so we never exceed the instrument's allowed
    precision (MARKET_ORDER_PRICE_BOUND_PRECISION_EXCEEDED). Rounding is
    toward the reference (floor for buys, ceiling for sells) so quantization
    can only TIGHTEN the tolerance, never widen it.
    """
    ref = Decimal(reference_price)
    frac = Decimal(str(max_slippage_bps)) / Decimal(10_000)
    raw = ref * (1 + frac) if side == "buy" else ref * (1 - frac)
    exponent = ref.as_tuple().exponent
    quantum = Decimal(1).scaleb(int(exponent)) if int(exponent) < 0 else Decimal(1)
    rounding = ROUND_FLOOR if side == "buy" else ROUND_CEILING
    return str(raw.quantize(quantum, rounding=rounding))


class OandaBroker(Broker):
    LIVE_URL = "https://api-fxtrade.oanda.com"
    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    STREAM_LIVE = "https://stream-fxtrade.oanda.com"
    STREAM_PRACTICE = "https://stream-fxpractice.oanda.com"

    def __init__(self, api_key: str, account_id: str, practice: bool = True) -> None:
        self.api_key = api_key
        self.account_id = account_id
        base = self.PRACTICE_URL if practice else self.LIVE_URL
        stream = self.STREAM_PRACTICE if practice else self.STREAM_LIVE
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # OANDA's edge 307-redirects some requests back to the same
        # path. Blanket follow_redirects is NOT safe here: on 301/302/303
        # httpx re-issues a POST as a GET, which for the orders endpoint
        # silently turns "place order" into "list orders". GETs may
        # follow freely; writes go through _send_following_307 so only
        # method-preserving redirects (307/308) are retried and anything
        # else fails closed.
        #
        # Thread-safety (CL-8cw1): these clients are hit concurrently from
        # asyncio.to_thread workers (OMS submit, coordinator, health tick).
        # That is safe: httpx documents Client as shareable between threads
        # (httpx.Client docstring, "It can be shared between threads";
        # the connection pool is internally locked). self.headers is built
        # once here and never mutated afterwards — per-request state goes
        # through local params=/json= arguments only.
        self.client = httpx.Client(
            base_url=base, headers=self.headers, timeout=10.0,
            follow_redirects=True,
        )
        self.write_client = httpx.Client(
            base_url=base, headers=self.headers, timeout=10.0,
            follow_redirects=False,
        )
        self.stream_client = httpx.AsyncClient(
            base_url=stream, headers=self.headers, timeout=None,
            follow_redirects=True,
        )
        # Last streamed quote per OANDA-symbol, for the slippage reference
        # (CL-7vn9). Populated by stream_prices as ticks flow; read by
        # _compute_price_bound to skip a redundant /pricing GET when the
        # streamed mid is fresh. Each entry is a single dict replaced
        # atomically (one dict.__setitem__), so the worker-thread reader in
        # place_order sees either the old or the new entry, never a torn one.
        self._last_stream_price: dict[str, dict[str, Any]] = {}

    _MAX_307_HOPS = 3

    #: Max age of a streamed quote still usable as a slippage reference
    #: (CL-7vn9). Beyond this the streamed mid is considered stale and
    #: place_order falls back to a fresh /pricing GET. OANDA streams FX ticks
    #: sub-second in an active session, so 2 s comfortably covers a healthy
    #: stream while rejecting a mid left over from a stalled/reconnecting one.
    _REF_PRICE_MAX_AGE_SEC = 2.0

    def _send_following_307(
        self, method: str, url: str, json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Write request that follows only method-preserving redirects
        (307/308). Any other 3xx raises instead of letting httpx downgrade
        the write to a GET — for order placement/cancel a silent method
        change is worse than a hard failure. (Generalized from POST-only per
        ultrareview #4 so cancel PUTs get the same protection.)
        """
        for _ in range(self._MAX_307_HOPS):
            resp = self.write_client.request(method, url, json=json_body)
            if resp.status_code in (307, 308) and resp.headers.get("location"):
                url = resp.headers["location"]
                continue
            resp.raise_for_status()
            return resp
        msg = f"OANDA {method} {url}: exceeded {self._MAX_307_HOPS} redirect hops"
        raise httpx.TooManyRedirects(msg)

    def _post_following_307(self, url: str, json_body: dict[str, Any]) -> httpx.Response:
        return self._send_following_307("POST", url, json_body)

    def place_order(self, order: Order) -> Order:
        body = {
            "order": {
                "type": "MARKET",
                "instrument": self._to_oanda(order.symbol),
                "units": str(int(order.quantity) if order.side == "buy" else -int(order.quantity)),
                "timeInForce": "FOK",
            }
        }
        # Slippage enforcement (CL-qyav): the intent's max_slippage_bps was
        # journaled but never enforced. priceBound makes the venue reject a
        # FOK market order that could only fill beyond tolerance (parsed
        # below as orderRejectTransaction/orderCancelTransaction → REJECTED,
        # which the OMS raises through the RejectionHandler path).
        try:
            price_bound = self._compute_price_bound(order)
        except SlippageRefUnavailableError as exc:
            # Fail CLOSED (CL-8lv6): a capped, non-emergency order must not
            # go out unbound just because /pricing hiccuped. Same terminal
            # REJECTED shape as a venue reject.
            order.status = OrderStatus.REJECTED
            order.reject_reason = "SLIPPAGE_REF_UNAVAILABLE"
            logger.warning(
                "OANDA REJECTED order %s %s x%s pre-flight: %s",
                order.symbol, order.side, order.quantity, exc,
            )
            return order
        if price_bound is not None:
            body["order"]["priceBound"] = price_bound
        resp = self._post_following_307(f"/v3/accounts/{self.account_id}/orders", body)
        data = resp.json()
        # Rejects are HTTP 201 with orderRejectTransaction / orderCancel
        # Transaction bodies (CL-h4as) — before this check they parsed as an
        # empty txn → PENDING, and the OMS journaled ORDER_PLACED for an
        # order OANDA had refused (ghost fills). A FOK market order that
        # didn't fill is equally terminal.
        reject = data.get("orderRejectTransaction")
        cancel = data.get("orderCancelTransaction")
        if reject is not None:
            order.status = OrderStatus.REJECTED
            order.order_id = str(reject.get("id", ""))
            reason = reject.get("rejectReason") or reject.get("reason") or "?"
            order.reject_reason = str(reason)
            logger.warning(
                "OANDA REJECTED order %s %s x%s: %s",
                order.symbol, order.side, order.quantity, reason,
            )
            return order
        if "orderFillTransaction" in data:
            txn = data["orderFillTransaction"]
            order.order_id = str(txn.get("id", ""))
            order.status = OrderStatus.FILLED
            return order
        if cancel is not None:
            # Created but immediately cancelled (FOK couldn't fill).
            order.status = OrderStatus.REJECTED
            order.order_id = str(cancel.get("id", ""))
            order.reject_reason = str(cancel.get("reason", "venue cancel"))
            logger.warning(
                "OANDA order %s %s x%s cancelled by venue: %s",
                order.symbol, order.side, order.quantity,
                cancel.get("reason", "?"),
            )
            return order
        txn = data.get("orderCreateTransaction", {})
        order.order_id = str(txn.get("id", ""))
        order.status = OrderStatus.PENDING if order.order_id else OrderStatus.REJECTED
        if not order.order_id:
            order.reject_reason = "no fill/reject/create transaction in response"
            logger.warning(
                "OANDA response had no fill/reject/create txn for %s %s x%s: %s",
                order.symbol, order.side, order.quantity, str(data)[:200],
            )
        return order

    def _fresh_stream_ref(self, order: Order) -> str | None:
        """Fill-side reference from the last STREAMED quote, if fresh enough
        (CL-7vn9). buy → ask, sell → bid, as the raw venue string so the
        derived bound keeps the instrument's quote precision.

        Returns None when there is no cached streamed quote for the symbol or
        the newest one is older than ``_REF_PRICE_MAX_AGE_SEC`` — the caller
        then falls back to a fresh /pricing GET. Freshness uses
        ``time.monotonic`` so a wall-clock adjustment can't make a stale quote
        look fresh (or vice-versa).
        """
        cache = getattr(self, "_last_stream_price", None)
        if not cache:
            return None
        entry = cache.get(self._to_oanda(order.symbol))
        if entry is None:
            return None
        age = time.monotonic() - entry["mono"]
        if age > self._REF_PRICE_MAX_AGE_SEC:
            return None
        return str(entry["ask"] if order.side == "buy" else entry["bid"])

    def _compute_price_bound(self, order: Order) -> str | None:
        """Reference-price → FOK priceBound for an outgoing market order.

        Reference is the venue's live quote on the fill side (buy → ask,
        sell → bid), as the raw string so the bound inherits the instrument's
        quote precision. Returns None (no bound) when the order carries no
        slippage cap.

        Reference source (CL-7vn9): prefer the engine's last STREAMED quote
        when it's fresh (≤ _REF_PRICE_MAX_AGE_SEC old) — the stream already
        carries a sub-second mid, so a separate /pricing round-trip per order
        was redundant latency + quota. Fall back to a fresh /pricing GET when
        no fresh streamed quote exists (stale stream, cold start, symbol never
        streamed).

        Fetch-failure posture (CL-8lv6 — was fail-open for everything):

        - normal capped order → raise SlippageRefUnavailableError; the order
          is REJECTED with reject_reason="SLIPPAGE_REF_UNAVAILABLE" instead
          of going out UNBOUND.
        - ``order.emergency`` (risk-reducing kill-switch flatten/reduce) →
          place unbound with a CRITICAL log: getting flat beats slippage
          protection.
        """
        bps = order.max_slippage_bps
        if bps is None or bps <= 0:
            return None
        # Fresh streamed mid wins — no /pricing call needed (CL-7vn9).
        stream_ref = self._fresh_stream_ref(order)
        if stream_ref is not None:
            return _price_bound_str(order.side, stream_ref, bps)
        try:
            resp = self.client.get(
                f"/v3/accounts/{self.account_id}/pricing",
                params={"instruments": self._to_oanda(order.symbol)},
            )
            resp.raise_for_status()
            p = resp.json()["prices"][0]
            ref_str = (
                str(p["asks"][0]["price"]) if order.side == "buy"
                else str(p["bids"][0]["price"])
            )
        except Exception as exc:
            if order.emergency:
                logger.critical(
                    "emergency order placed without slippage bound: "
                    "reference-price fetch failed for %s — %s x%s goes out "
                    "UNBOUND (max %.2f bps unenforced this order; "
                    "risk-reducing order takes precedence)",
                    order.symbol, order.side, order.quantity, bps,
                    exc_info=True,
                )
                return None
            msg = (
                f"reference price unavailable for {order.symbol} "
                f"({order.side} x{order.quantity}, max {bps} bps)"
            )
            raise SlippageRefUnavailableError(msg) from exc
        return _price_bound_str(order.side, ref_str, bps)

    def cancel_order(self, order_id: str) -> bool:
        """Real cancel via the v20 API (CL-h4as — was an unconditional
        ``return True`` that never touched the venue, leaving working orders
        live while the OMS believed them cancelled)."""
        try:
            resp = self._send_following_307(
                "PUT", f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel",
            )
            if resp.status_code == 200:
                return True
            logger.warning(
                "OANDA cancel %s failed: HTTP %s %s",
                order_id, resp.status_code, resp.text[:150],
            )
            return False
        except Exception as exc:
            logger.warning("OANDA cancel %s errored: %s", order_id, str(exc)[:150])
            return False

    def get_order(self, order_id: str) -> Order:
        raise NotImplementedError

    def get_positions(self) -> list[Position]:
        resp = self.client.get(f"/v3/accounts/{self.account_id}/positions")
        resp.raise_for_status()
        positions = []
        for p in resp.json().get("positions", []):
            lq = float(p["long"]["units"])
            sq = float(p["short"]["units"])
            net = lq + sq
            if net == 0:
                continue
            avg_price = float(p["long"]["averagePrice"]) if net > 0 else float(p["short"]["averagePrice"])
            positions.append(Position(
                symbol=self._from_oanda(p["instrument"]),
                quantity=net,
                avg_price=avg_price,
            ))
        return positions

    def get_account(self) -> Account:
        resp = self.client.get(f"/v3/accounts/{self.account_id}/summary")
        resp.raise_for_status()
        a = resp.json()["account"]
        return Account(
            balance=float(a["balance"]),
            equity=float(a["NAV"]),
            margin_used=float(a.get("marginUsed", 0)),
        )

    def get_price(self, symbol: str) -> tuple[float, float]:
        resp = self.client.get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={"instruments": self._to_oanda(symbol)},
        )
        resp.raise_for_status()
        p = resp.json()["prices"][0]
        return float(p["bids"][0]["price"]), float(p["asks"][0]["price"])

    #: Reconnect backoff bounds for the price stream (CL-vff9).
    _STREAM_BACKOFF_START = 1.0
    _STREAM_BACKOFF_MAX = 30.0

    async def stream_prices(
        self, symbols: list[str],
    ) -> AsyncIterator[dict[str, Any]]:
        """Self-healing OANDA price stream (CL-vff9).

        The stream is a long-lived HTTP connection that dies on any network
        blip (WiFi roam, DNS hiccup, laptop wake) — before this it died hard
        and only the engine's outer 5 s retry re-established it. Now the
        connection + read loop live inside a reconnect supervisor: transient
        errors (connect/read/protocol/OS) are caught, logged, and retried
        with exponential backoff (1 s → 30 s), rebuilding the client each
        time; a clean server-side close also reconnects. So a blip
        self-heals in seconds instead of relying on the caller. CANCELLED
        (shutdown) always propagates. A 4xx (bad credentials / bad request)
        is PERMANENT — logged CRITICAL and raised, since retrying can't fix
        it and hammering OANDA with bad auth is pointless.
        """
        oanda_syms = ",".join(self._to_oanda(s) for s in symbols)
        base_url = (
            self.STREAM_PRACTICE if "practice" in str(self.client.base_url)
            else self.STREAM_LIVE
        )
        # Lazily ensure the streamed-quote cache exists (CL-7vn9) — the engine
        # builds the broker via __init__, but some tests construct it with
        # __new__ and never run __init__.
        if not hasattr(self, "_last_stream_price"):
            self._last_stream_price = {}
        backoff = self._STREAM_BACKOFF_START
        while True:
            try:
                async with (
                    httpx.AsyncClient(
                        base_url=base_url, headers=self.headers, timeout=None,
                    ) as client,
                    client.stream(
                        "GET",
                        f"/v3/accounts/{self.account_id}/pricing/stream",
                        params={"instruments": oanda_syms},
                    ) as resp,
                ):
                    if resp.status_code >= 400:
                        if resp.status_code < 500:
                            logger.critical(
                                "OANDA price stream rejected with %d "
                                "(bad credentials / request) — NOT retrying",
                                resp.status_code,
                            )
                            resp.raise_for_status()
                        logger.warning(
                            "OANDA price stream HTTP %d — reconnecting in "
                            "%.0fs", resp.status_code, backoff,
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, self._STREAM_BACKOFF_MAX)
                        continue
                    backoff = self._STREAM_BACKOFF_START  # connected — reset
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            msg = json.loads(line)
                        except ValueError:
                            continue  # heartbeat / partial line — skip
                        if msg.get("type") == "PRICE":
                            # Cache the RAW venue quote strings + a monotonic
                            # capture time (CL-7vn9) so _compute_price_bound
                            # can reuse this streamed quote as its slippage
                            # reference — at the venue's own precision, with a
                            # clock-adjustment-immune freshness check — instead
                            # of issuing a fresh /pricing GET. Single atomic
                            # dict assignment; see _last_stream_price note.
                            self._last_stream_price[msg["instrument"]] = {
                                "bid": str(msg["bids"][0]["price"]),
                                "ask": str(msg["asks"][0]["price"]),
                                "mono": time.monotonic(),
                            }
                            yield {
                                "symbol": self._from_oanda(msg["instrument"]),
                                "bid": float(msg["bids"][0]["price"]),
                                "ask": float(msg["asks"][0]["price"]),
                                "ts": msg["time"],
                            }
                    # aiter_lines ended without error → server closed the
                    # stream cleanly; reconnect (short pause).
                    logger.warning(
                        "OANDA price stream closed by server — reconnecting",
                    )
                    await asyncio.sleep(self._STREAM_BACKOFF_START)
            except asyncio.CancelledError:
                raise  # shutdown — never swallow
            except httpx.HTTPStatusError:
                # A 4xx we deliberately surfaced above (bad creds/request) is
                # PERMANENT — must escape the reconnect loop, not retry
                # forever. Listed before the transport-error catch below
                # (HTTPStatusError is itself an httpx.HTTPError).
                raise
            except (httpx.HTTPError, OSError) as exc:
                logger.warning(
                    "OANDA price stream error (%s: %s) — reconnecting in "
                    "%.0fs", type(exc).__name__, str(exc)[:120], backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._STREAM_BACKOFF_MAX)

    @staticmethod
    def _to_oanda(sym: str) -> str:
        # Idempotent (CL-03f5): event legs already arrive OANDA-formatted
        # (BCO_USD, XAU_USD, EUR_USD) and must NOT be re-split — the old
        # f"{sym[:3]}_{sym[3:]}" turned EUR_USD into EUR__USD and NATGAS_USD
        # into NAT_GAS_USD, which OANDA 400-rejects as malformed, so NO event
        # order ever placed. Only a plain 6-char FX pair (EURUSD) needs the
        # underscore inserted.
        if "_" in sym:
            return sym
        if len(sym) == 6:
            return f"{sym[:3]}_{sym[3:]}"
        return sym

    @staticmethod
    def _from_oanda(sym: str) -> str:
        return sym.replace("_", "")
