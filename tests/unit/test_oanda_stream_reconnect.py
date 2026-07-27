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


def test_reconnects_past_transient_error(monkeypatch):
    # First connect raises ConnectError (the DNS blip); reconnect yields.
    _patch(
        monkeypatch,
        [
            httpx.ConnectError("nodename nor servname provided"),
            _FakeResp([_PRICE_LINE]),
        ],
    )

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


def test_4xx_is_permanent_after_bounded_retries(monkeypatch):
    # A 401/403 must NOT retry forever — but CL-j2y9 gives it a few attempts
    # first (OANDA emits spurious 401s on rapid reconnects). Exhaust the
    # budget → raise.
    n = OandaBroker._STREAM_MAX_4XX_RETRIES + 1
    _patch(monkeypatch, [_FakeResp([], status=401) for _ in range(n)])

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        await gen.__anext__()

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())


def test_spurious_4xx_recovers(monkeypatch):
    """CL-j2y9: a SINGLE spurious 401 (rapid-reconnect artifact) used to kill
    the stream permanently while the same credentials kept working — the
    engine then halted on stale prices until a manual restart."""
    _patch(monkeypatch, [_FakeResp([], status=401), _FakeResp([_PRICE_LINE])])

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        tick = await gen.__anext__()
        await gen.aclose()
        return tick

    assert asyncio.run(run())["symbol"] == "EURUSD"


def test_4xx_budget_resets_after_a_good_connect(monkeypatch):
    # 401s that are separated by a SUCCESSFUL connect must not accumulate
    # toward the permanent-failure budget.
    script = []
    for _ in range(OandaBroker._STREAM_MAX_4XX_RETRIES):
        script += [_FakeResp([], status=401), _FakeResp([_PRICE_LINE])]
    _patch(monkeypatch, script)

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        ticks = [await gen.__anext__() for _ in range(OandaBroker._STREAM_MAX_4XX_RETRIES)]
        await gen.aclose()
        return ticks

    ticks = asyncio.run(run())  # never raises despite N total 401s
    assert len(ticks) == OandaBroker._STREAM_MAX_4XX_RETRIES


def test_silent_stall_reconnects(monkeypatch):
    """CL-j2y9 root cause: the client used timeout=None, so a half-open
    connection (socket up, zero bytes, not even a heartbeat) blocked in
    aiter_lines() FOREVER — no exception, so the reconnect supervisor never
    fired. Prices silently stopped and stale_prices halted trading for 56h
    until a manual restart. A read timeout makes the stall retryable."""
    _patch(monkeypatch, [httpx.ReadTimeout("stalled"), _FakeResp([_PRICE_LINE])])

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        tick = await gen.__anext__()
        await gen.aclose()
        return tick

    assert asyncio.run(run())["symbol"] == "EURUSD"


def test_stream_client_has_bounded_read_timeout():
    # The regression guard for the actual defect: timeout=None is what let a
    # dead-but-open connection hang forever.
    t = OandaBroker._stream_timeout()
    assert t.read == OandaBroker._STREAM_READ_TIMEOUT
    assert t.read is not None and t.read > 0
    assert t.connect is not None and t.connect > 0


def test_streams_pass_a_real_timeout_to_the_client(monkeypatch):
    """THE regression guard for CL-j2y9. Injecting a ReadTimeout only proves
    the handler works — it would pass on the buggy code too, because
    timeout=None simply meant the timeout never FIRED. Assert the client is
    actually constructed with a bounded read timeout."""
    seen: list[object] = []
    state = {"i": 0}

    def _capture(**kwargs: object):  # noqa: ANN202
        seen.append(kwargs.get("timeout"))
        return _FakeClient([_FakeResp([_PRICE_LINE])], state)

    monkeypatch.setattr(httpx, "AsyncClient", _capture)

    async def _no_sleep(*_a: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def run():  # noqa: ANN202
        gen = _broker().stream_prices(["EUR_USD"])
        await gen.__anext__()
        await gen.aclose()

    asyncio.run(run())
    assert seen, "AsyncClient was never constructed"
    timeout = seen[0]
    assert timeout is not None, "timeout=None lets a stalled stream hang forever"
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == OandaBroker._STREAM_READ_TIMEOUT


def test_stream_populates_slippage_reference_cache(monkeypatch):
    # CL-7vn9: each streamed PRICE caches the RAW venue quote strings +
    # a monotonic capture time, so _compute_price_bound can reuse it as the
    # slippage reference instead of issuing a fresh /pricing GET.
    import time

    _patch(monkeypatch, [_FakeResp([_PRICE_LINE])])
    b = _broker()

    async def run():  # noqa: ANN202
        gen = b.stream_prices(["EUR_USD"])
        await gen.__anext__()
        await gen.aclose()

    before = time.monotonic()
    asyncio.run(run())
    entry = b._last_stream_price["EUR_USD"]
    # Raw strings (venue precision preserved), not floats.
    assert entry["bid"] == "1.0840" and entry["ask"] == "1.0842"
    assert isinstance(entry["bid"], str) and isinstance(entry["ask"], str)
    # Monotonic stamp captured at yield time.
    assert entry["mono"] >= before
