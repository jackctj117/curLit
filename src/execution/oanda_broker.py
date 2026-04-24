"""OANDA broker — REST + streaming API v20."""

import asyncio
import logging

import httpx

from .broker import Account, Broker, Order, OrderStatus, OrderType, Position

logger = logging.getLogger(__name__)


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
        self.client = httpx.Client(base_url=base, headers=self.headers, timeout=10.0)
        self.stream_client = httpx.AsyncClient(base_url=stream, headers=self.headers, timeout=None)

    def place_order(self, order: Order) -> Order:
        body = {
            "order": {
                "type": "MARKET",
                "instrument": self._to_oanda(order.symbol),
                "units": str(int(order.quantity) if order.side == "buy" else -int(order.quantity)),
                "timeInForce": "FOK",
            }
        }
        resp = self.client.post(f"/v3/accounts/{self.account_id}/orders", json=body)
        resp.raise_for_status()
        data = resp.json()
        txn = data.get("orderFillTransaction") or data.get("orderCreateTransaction", {})
        order.order_id = str(txn.get("id", ""))
        order.status = OrderStatus.FILLED if "orderFillTransaction" in data else OrderStatus.PENDING
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

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

    async def stream_prices(self, symbols: list[str]):
        oanda_syms = ",".join(self._to_oanda(s) for s in symbols)
        async with httpx.AsyncClient(base_url=self.STREAM_PRACTICE if "practice" in str(self.client.base_url) else self.STREAM_LIVE,
                                      headers=self.headers, timeout=None) as client:
            async with client.stream(
                "GET", f"/v3/accounts/{self.account_id}/pricing/stream",
                params={"instruments": oanda_syms},
            ) as resp:
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
        return f"{sym[:3]}_{sym[3:]}"

    @staticmethod
    def _from_oanda(sym: str) -> str:
        return sym.replace("_", "")
