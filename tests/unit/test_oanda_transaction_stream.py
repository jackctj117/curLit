"""OANDA transaction stream → OMS fill confirmation (CL-vj74).

The transactions stream carries ORDER_FILL events with orderID, echoed
clientExtensions.id, units and price. stream_transactions normalizes them and
uses the same self-healing reconnect supervisor as stream_prices.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src.execution.oanda_broker import OandaBroker

_FILL = (
    '{"type":"ORDER_FILL","id":"1001","orderID":"1000",'
    '"instrument":"EUR_USD","units":"1000","price":"1.0841",'
    '"clientExtensions":{"id":"intent-abc"},"time":"2026-07-23T14:00:00Z"}'
)
_HEARTBEAT = '{"type":"HEARTBEAT","time":"2026-07-23T14:00:01Z"}'
_CREATE = '{"type":"ORDER_CREATE","id":"999","time":"2026-07-23T14:00:00Z"}'


def _broker() -> OandaBroker:
    b = OandaBroker.__new__(OandaBroker)  # skip network __init__
    b.account_id = "ACC"
    b.headers = {}
    b.client = SimpleNamespace(base_url="https://api-fxpractice.oanda.com")
    return b


class _FakeResp:
    def __init__(self, lines: list[str], status: int = 200) -> None:
        self.status_code = status
        self._lines = lines

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("GET", "http://x"),
                response=httpx.Response(self.status_code),
            )

    async def aiter_lines(self):  # noqa: ANN202
        for ln in self._lines:
            yield ln


class _StreamCtx:
    def __init__(self, item: object) -> None:
        self._item = item

    async def __aenter__(self):  # noqa: ANN202
        if isinstance(self._item, BaseException):
            raise self._item
        return self._item

    async def __aexit__(self, *a: object) -> bool:
        return False


class _FakeClient:
    def __init__(self, script: list, state: dict) -> None:
        self._script = script
        self._state = state

    async def __aenter__(self):  # noqa: ANN202
        return self

    async def __aexit__(self, *a: object) -> bool:
        return False

    def stream(self, *a: object, **k: object) -> _StreamCtx:
        i = self._state["i"]
        self._state["i"] += 1
        return _StreamCtx(self._script[min(i, len(self._script) - 1)])


def _patch(monkeypatch, script: list) -> None:
    state = {"i": 0}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _FakeClient(script, state))

    async def _no_sleep(*_a: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


def _first(monkeypatch, script: list) -> dict:
    _patch(monkeypatch, script)

    async def run():  # noqa: ANN202
        gen = _broker().stream_transactions()
        ev = await gen.__anext__()
        await gen.aclose()
        return ev

    return asyncio.run(run())


def test_yields_normalized_order_fill(monkeypatch):
    ev = _first(monkeypatch, [_FakeResp([_FILL])])
    assert ev["type"] == "ORDER_FILL"
    assert ev["transaction_id"] == "1001"
    assert ev["order_id"] == "1000"
    assert ev["client_order_id"] == "intent-abc"
    assert ev["instrument"] == "EURUSD"  # compact dialect
    assert ev["units"] == 1000.0 and ev["price"] == 1.0841


def test_skips_heartbeat_and_non_fill_transactions(monkeypatch):
    # HEARTBEAT + ORDER_CREATE precede the fill and must be skipped.
    ev = _first(monkeypatch, [_FakeResp([_HEARTBEAT, _CREATE, _FILL])])
    assert ev["transaction_id"] == "1001"  # first yielded is the fill


def test_reconnects_past_transient_error(monkeypatch):
    ev = _first(
        monkeypatch,
        [
            httpx.ConnectError("nodename nor servname provided"),
            _FakeResp([_FILL]),
        ],
    )
    assert ev["order_id"] == "1000"


def test_4xx_is_permanent(monkeypatch):
    _patch(monkeypatch, [_FakeResp([], status=401)])

    async def run():  # noqa: ANN202
        gen = _broker().stream_transactions()
        await gen.__anext__()

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())


def test_missing_client_extensions_yields_none(monkeypatch):
    fill = (
        '{"type":"ORDER_FILL","id":"2001","orderID":"2000",'
        '"instrument":"XAU_USD","units":"-5","price":"1999.5"}'
    )
    ev = _first(monkeypatch, [_FakeResp([fill])])
    assert ev["client_order_id"] is None
    assert ev["units"] == -5.0
