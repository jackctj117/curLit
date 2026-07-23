"""OANDA price-stream auto-reconnect (CL-vff9).

Live incident 2026-07-23: a network/DNS blip killed the stream and it did
not self-heal. stream_prices now reconnects with backoff instead of dying
on a transient error; cancellation (shutdown) still propagates; a 4xx
(bad creds) is permanent.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src.execution.oanda_broker import OandaBroker

_PRICE_LINE = (
    '{"type":"PRICE","instrument":"EUR_USD",'
    '"bids":[{"price":"1.0840"}],"asks":[{"price":"1.0842"}],'
    '"time":"2026-07-23T14:00:00Z"}'
)


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
                "err", request=httpx.Request("GET", "http://x"),
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
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **k: _FakeClient(script, state))

    async def _no_sleep(*_a: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


def test_reconnects_past_transient_error(monkeypatch):
    # First connect raises ConnectError (the DNS blip); reconnect yields.
    _patch(monkeypatch, [
        httpx.ConnectError("nodename nor servname provided"),
        _FakeResp([_PRICE_LINE]),
    ])

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        tick = await gen.__anext__()
        await gen.aclose()
        return tick

    tick = asyncio.run(run())
    assert tick["symbol"] == "EURUSD"
    assert tick["bid"] == 1.0840 and tick["ask"] == 1.0842


def test_reconnects_after_clean_server_close(monkeypatch):
    # First stream ends with no lines (clean close) → reconnect → price.
    _patch(monkeypatch, [_FakeResp([]), _FakeResp([_PRICE_LINE])])

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        tick = await gen.__anext__()
        await gen.aclose()
        return tick

    assert asyncio.run(run())["symbol"] == "EURUSD"


def test_cancellation_propagates(monkeypatch):
    _patch(monkeypatch, [asyncio.CancelledError()])

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        await gen.__anext__()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())


def test_4xx_is_permanent(monkeypatch):
    # A 401/403 must NOT retry forever — it raises.
    _patch(monkeypatch, [_FakeResp([], status=401)])

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        await gen.__anext__()

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())
