"""OANDA 401 diagnostics + read-only single retry (CL-wrsa).

Sporadic practice 401s on GET accounts/<id>/summary and /positions (and on
the price stream) were logged as bare tracebacks — no body, no OANDA
RequestID, no server clock — so the cause could not be established. These
tests pin the new contract at the HTTP transport boundary (real
``httpx.Client`` over ``httpx.MockTransport``, so ``raise_for_status`` and
header handling are httpx's own, not a fake):

* a 401 on the READ-ONLY summary/positions endpoints is logged with
  request_id / server_date / body and retried EXACTLY once;
* a second 401 propagates as ``httpx.HTTPStatusError`` (no retry loop);
* order placement, cancel, and the order-path /pricing GET are NEVER retried
  on 401 (exactly one request reaches the transport);
* the API token never appears in any log line.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

import httpx
import pytest

from src.execution import oanda_broker as ob
from src.execution.broker import Order, OrderStatus, OrderType
from src.execution.oanda_broker import OandaBroker

# Fake credential for the test broker. Deliberately distinctive so a leak
# into any log record is unambiguous.
# Assembled at import time (not one literal) so secret scanners do not flag
# this obviously fake value; the redaction assertions use the joined string.
_TOKEN = "-".join(("tok", "CLwrsa", "0123456789abcdef", "SECRET"))
_ACC = "101-001-0000000-001"
_REQ_ID = "88051327329381234"
_DATE = "Wed, 01 Oct 2026 16:00:20 GMT"
_ERR_BODY = '{"errorMessage":"Insufficient authorization to perform request."}'

_SUMMARY = {
    "account": {"balance": "100000.0", "NAV": "100123.5", "marginUsed": "250.0"},
}
_POSITIONS = {
    "positions": [
        {
            "instrument": "USD_CAD",
            "long": {"units": "0", "averagePrice": "0"},
            "short": {"units": "-8916", "averagePrice": "1.3650"},
        }
    ]
}

Handler = Callable[[httpx.Request], httpx.Response]


def _r401(echo_token: bool = False) -> httpx.Response:
    body = _ERR_BODY
    if echo_token:
        # Worst case: an upstream/proxy echoes the Authorization header back.
        body = json.dumps({"errorMessage": "denied", "echo": f"Bearer {_TOKEN}"})
    return httpx.Response(
        401,
        headers={"RequestID": _REQ_ID, "Date": _DATE, "Content-Type": "application/json"},
        content=body.encode(),
    )


def _scripted(responses: list[httpx.Response], seen: list[httpx.Request]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if not responses:
            raise AssertionError(f"unexpected extra request {request.method} {request.url}")
        return responses.pop(0)

    return handler


def _broker(handler: Handler, monkeypatch: pytest.MonkeyPatch) -> OandaBroker:
    b = OandaBroker(_TOKEN, _ACC, practice=True)
    transport = httpx.MockTransport(handler)
    b.client = httpx.Client(
        base_url=b.PRACTICE_URL,
        headers=b.headers,
        transport=transport,
        follow_redirects=True,
    )
    b.write_client = httpx.Client(
        base_url=b.PRACTICE_URL,
        headers=b.headers,
        transport=transport,
        follow_redirects=False,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(ob.time, "sleep", lambda s: sleeps.append(s))
    b._test_sleeps = sleeps  # type: ignore[attr-defined]
    return b


def _all_log_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Read-only endpoints: one retry
# ---------------------------------------------------------------------------


def test_summary_401_then_200_recovers_with_one_retry(monkeypatch, caplog):
    seen: list[httpx.Request] = []
    b = _broker(_scripted([_r401(), httpx.Response(200, json=_SUMMARY)], seen), monkeypatch)
    with caplog.at_level(logging.WARNING, logger="src.execution.oanda_broker"):
        acct = b.get_account()
    assert acct.balance == 100000.0 and acct.equity == 100123.5 and acct.margin_used == 250.0
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", f"/v3/accounts/{_ACC}/summary"),
        ("GET", f"/v3/accounts/{_ACC}/summary"),
    ]
    assert b._test_sleeps == [OandaBroker._READ_401_RETRY_DELAY_SEC]
    text = _all_log_text(caplog)
    assert f"request_id={_REQ_ID}" in text
    assert f"server_date={_DATE}" in text
    assert "Insufficient authorization" in text
    assert "retrying ONCE" in text
    assert "recovered on the single 401 retry" in text
    assert _TOKEN not in text


def test_positions_401_then_200_recovers_with_one_retry(monkeypatch, caplog):
    seen: list[httpx.Request] = []
    b = _broker(_scripted([_r401(), httpx.Response(200, json=_POSITIONS)], seen), monkeypatch)
    with caplog.at_level(logging.WARNING, logger="src.execution.oanda_broker"):
        positions = b.get_positions()
    assert len(seen) == 2
    assert all(r.method == "GET" and r.url.path.endswith("/positions") for r in seen)
    assert len(positions) == 1
    assert positions[0].symbol == "USDCAD" and positions[0].quantity == -8916.0
    assert positions[0].avg_price == 1.3650
    assert f"request_id={_REQ_ID}" in _all_log_text(caplog)


@pytest.mark.parametrize("method_name", ["get_account", "get_positions"])
def test_read_401_twice_raises_after_exactly_one_retry(monkeypatch, caplog, method_name):
    seen: list[httpx.Request] = []
    b = _broker(_scripted([_r401(), _r401()], seen), monkeypatch)
    with (
        caplog.at_level(logging.WARNING, logger="src.execution.oanda_broker"),
        pytest.raises(httpx.HTTPStatusError) as ei,
    ):
        getattr(b, method_name)()
    assert ei.value.response.status_code == 401
    assert len(seen) == 2  # original + ONE retry, never more
    assert b._test_sleeps == [OandaBroker._READ_401_RETRY_DELAY_SEC]
    text = _all_log_text(caplog)
    assert "retry ALSO rejected" in text
    assert text.count(f"request_id={_REQ_ID}") == 2  # both attempts diagnosed


@pytest.mark.parametrize("second_status", [304, 403, 503])
def test_read_401_then_other_error_is_not_reported_as_recovery(monkeypatch, caplog, second_status):
    """Review round 1: a 401 followed by a non-401 failure used to log
    "recovered" and drop the second response's evidence before raising."""
    seen: list[httpx.Request] = []
    second = httpx.Response(
        second_status,
        headers={"RequestID": "second-req-77", "Date": _DATE},
        content=b'{"errorMessage":"second failure"}',
    )
    b = _broker(_scripted([_r401(), second], seen), monkeypatch)
    with (
        caplog.at_level(logging.WARNING, logger="src.execution.oanda_broker"),
        pytest.raises(httpx.HTTPStatusError) as ei,
    ):
        b.get_positions()
    assert ei.value.response.status_code == second_status
    assert len(seen) == 2
    text = _all_log_text(caplog)
    assert "recovered" not in text
    assert "second failure" in text  # the retry's own body is diagnosed
    if second_status == 403:
        assert "request_id=second-req-77" in text


@pytest.mark.parametrize("status", [403, 404, 500, 503])
def test_read_non_401_errors_are_not_retried(monkeypatch, status):
    seen: list[httpx.Request] = []
    b = _broker(_scripted([httpx.Response(status, text="x")], seen), monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        b.get_account()
    assert len(seen) == 1
    assert b._test_sleeps == []


def test_read_200_makes_a_single_request(monkeypatch, caplog):
    seen: list[httpx.Request] = []
    b = _broker(_scripted([httpx.Response(200, json=_SUMMARY)], seen), monkeypatch)
    with caplog.at_level(logging.WARNING, logger="src.execution.oanda_broker"):
        b.get_account()
    assert len(seen) == 1 and b._test_sleeps == []
    assert caplog.records == []


def test_retry_helper_refuses_non_allowlisted_endpoints(monkeypatch):
    seen: list[httpx.Request] = []
    b = _broker(_scripted([], seen), monkeypatch)
    for suffix in ("orders", "trades", "positions/USD_CAD/close", "pricing"):
        with pytest.raises(AssertionError):
            b._get_account_read(suffix)
    assert seen == []


# ---------------------------------------------------------------------------
# Order / write endpoints: NEVER retried
# ---------------------------------------------------------------------------


def _order(**kw: object) -> Order:
    return Order(
        symbol="USD_CAD",
        side="sell",
        quantity=8916,
        order_type=OrderType.MARKET,
        **kw,  # type: ignore[arg-type]
    )


def test_place_order_401_is_never_retried(monkeypatch, caplog):
    seen: list[httpx.Request] = []
    # A second (successful) response is queued: if the code retried, the
    # order would "fill" and len(seen) would be 2.
    fill = httpx.Response(201, json={"orderFillTransaction": {"id": "2"}})
    b = _broker(_scripted([_r401(), fill], seen), monkeypatch)
    with (
        caplog.at_level(logging.WARNING, logger="src.execution.oanda_broker"),
        pytest.raises(httpx.HTTPStatusError),
    ):
        b.place_order(_order())
    assert [(r.method, r.url.path) for r in seen] == [("POST", f"/v3/accounts/{_ACC}/orders")]
    assert b._test_sleeps == []
    text = _all_log_text(caplog)
    assert "NOT retried" in text and f"request_id={_REQ_ID}" in text
    assert _TOKEN not in text


def test_cancel_order_401_is_never_retried(monkeypatch):
    seen: list[httpx.Request] = []
    ok = httpx.Response(200, json={})
    b = _broker(_scripted([_r401(), ok], seen), monkeypatch)
    assert b.cancel_order("123") is False
    assert [(r.method, r.url.path) for r in seen] == [
        ("PUT", f"/v3/accounts/{_ACC}/orders/123/cancel"),
    ]
    assert b._test_sleeps == []


def test_order_path_pricing_401_is_never_retried(monkeypatch):
    """The /pricing GET backing a capped order's priceBound is order-path:
    a 401 rejects the order (SLIPPAGE_REF_UNAVAILABLE) with no retry and
    no order POST."""
    seen: list[httpx.Request] = []
    price = httpx.Response(
        200, json={"prices": [{"bids": [{"price": "1.3650"}], "asks": [{"price": "1.3652"}]}]}
    )
    b = _broker(_scripted([_r401(), price], seen), monkeypatch)
    out = b.place_order(_order(max_slippage_bps=5.0))
    assert out.status == OrderStatus.REJECTED
    assert out.reject_reason == "SLIPPAGE_REF_UNAVAILABLE"
    assert [(r.method, r.url.path) for r in seen] == [("GET", f"/v3/accounts/{_ACC}/pricing")]


def test_get_price_401_is_not_retried(monkeypatch):
    seen: list[httpx.Request] = []
    b = _broker(_scripted([_r401()], seen), monkeypatch)
    with pytest.raises(httpx.HTTPStatusError):
        b.get_price("USDCAD")
    assert len(seen) == 1 and b._test_sleeps == []


# ---------------------------------------------------------------------------
# Redaction + clock fields
# ---------------------------------------------------------------------------


def test_token_echoed_in_401_body_is_redacted(monkeypatch, caplog):
    seen: list[httpx.Request] = []
    b = _broker(_scripted([_r401(echo_token=True), _r401(echo_token=True)], seen), monkeypatch)
    with (
        caplog.at_level(logging.DEBUG, logger="src.execution.oanda_broker"),
        pytest.raises(httpx.HTTPStatusError),
    ):
        b.get_positions()
    text = _all_log_text(caplog)
    assert _TOKEN not in text
    assert "Bearer <redacted>" in text
    # The request itself did carry the token — the redaction is real.
    assert seen[0].headers["Authorization"] == f"Bearer {_TOKEN}"


def test_redact_handles_bare_token_and_bearer_form():
    assert ob._redact(f"x {_TOKEN} y", _TOKEN) == "x <redacted> y"
    assert ob._redact("authorization: bearer abc.DEF-123", None) == (
        "authorization: Bearer <redacted>"
    )


def test_diag_fields_redact_token_in_every_header_field():
    fields = ob._auth_diag_fields(
        {"RequestID": f"id {_TOKEN}", "Date": f"Bearer {_TOKEN}"}, "", _TOKEN
    )
    assert _TOKEN not in "".join(fields.values())


def test_diag_fields_clock_skew_and_truncation():
    fields = ob._auth_diag_fields({"Date": "garbage"}, "b" * 2000, None)
    assert fields["clock_skew_s"] == "unparseable"
    assert len(fields["body"]) == ob._DIAG_BODY_MAX_CHARS
    fields = ob._auth_diag_fields({}, "", None)
    assert fields["request_id"] == "-" and fields["server_date"] == "-"
    assert fields["clock_skew_s"] == "-"
    fields = ob._auth_diag_fields({"Date": _DATE}, "", None)
    # Signed float seconds (local now − server Date); the fixed past date
    # yields a large positive skew — the exact value is wall-clock dependent.
    assert fields["clock_skew_s"].startswith(("+", "-"))
    float(fields["clock_skew_s"])


# ---------------------------------------------------------------------------
# Price stream 401: same diagnostics, existing bounded-retry policy unchanged
# ---------------------------------------------------------------------------

_PRICE_LINE = (
    '{"type":"PRICE","instrument":"EUR_USD",'
    '"bids":[{"price":"1.0840"}],"asks":[{"price":"1.0842"}],'
    '"time":"2026-09-30T23:00:41Z"}'
)


class _StreamResp:
    def __init__(self, status: int, body: bytes = b"", lines: list[str] | None = None) -> None:
        self.status_code = status
        self.headers = httpx.Headers({"RequestID": _REQ_ID, "Date": _DATE}) if status >= 400 else {}
        self._body = body
        self._lines = lines or []

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("GET", "http://x"),
                response=httpx.Response(self.status_code),
            )

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self):  # noqa: ANN201
        for ln in self._lines:
            yield ln


class _Ctx:
    def __init__(self, item: _StreamResp) -> None:
        self._item = item

    async def __aenter__(self) -> _StreamResp:
        return self._item

    async def __aexit__(self, *a: object) -> bool:
        return False


class _Client:
    def __init__(self, script: list[_StreamResp]) -> None:
        self._script = script

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *a: object) -> bool:
        return False

    def stream(self, *a: object, **k: object) -> _Ctx:
        return _Ctx(self._script.pop(0))


def test_price_stream_401_logs_redacted_diagnostics(monkeypatch, caplog):
    body = json.dumps({"errorMessage": "denied", "echo": f"Bearer {_TOKEN}"}).encode()
    script = [_StreamResp(401, body=body), _StreamResp(200, lines=[_PRICE_LINE])]
    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Client(script))

    async def _no_sleep(*_a: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    b = OandaBroker(_TOKEN, _ACC, practice=True)

    async def run() -> dict[str, object]:
        gen = b.stream_prices(["EUR_USD"])
        tick = await gen.__anext__()
        await gen.aclose()
        return tick

    with caplog.at_level(logging.WARNING, logger="src.execution.oanda_broker"):
        tick = asyncio.run(run())
    assert tick["symbol"] == "EURUSD"
    text = _all_log_text(caplog)
    assert "price stream rejected with 401 (attempt 1/3)" in text
    assert f"request_id={_REQ_ID}" in text and f"server_date={_DATE}" in text
    assert _TOKEN not in text and "Bearer <redacted>" in text
