"""OANDA broker — REST + streaming API v20."""

import logging
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

    _MAX_307_HOPS = 3

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

    def _compute_price_bound(self, order: Order) -> str | None:
        """Reference-price → FOK priceBound for an outgoing market order.

        Reference is the venue's live quote on the fill side (buy → ask,
        sell → bid), fetched as the raw string so the bound inherits the
        instrument's quote precision. Returns None (no bound) when the order
        carries no slippage cap.

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

    async def stream_prices(
        self, symbols: list[str],
    ) -> AsyncIterator[dict[str, Any]]:
        oanda_syms = ",".join(self._to_oanda(s) for s in symbols)
        base_url = self.STREAM_PRACTICE if "practice" in str(self.client.base_url) else self.STREAM_LIVE
        async with (
            httpx.AsyncClient(base_url=base_url, headers=self.headers, timeout=None) as client,
            client.stream(
                "GET", f"/v3/accounts/{self.account_id}/pricing/stream",
                params={"instruments": oanda_syms},
            ) as resp,
        ):
                async for line in resp.aiter_lines():
                    import json
                    if not line.strip():
                        continue
                    msg = json.loads(line)
                    if msg.get("type") == "PRICE":
                        yield {
                            "symbol": self._from_oanda(msg["instrument"]),
                            "bid": float(msg["bids"][0]["price"]),
                            "ask": float(msg["asks"][0]["price"]),
                            "ts": msg["time"],
                        }

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
