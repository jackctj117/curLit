"""OANDA reject/cancel handling regressions (CL-h4as).

Rejected orders used to parse as an empty txn -> PENDING (OMS journaled
ORDER_PLACED for orders the venue refused), and cancel_order returned True
without touching the API.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.execution.broker import Order, OrderStatus, OrderType
from src.execution.oanda_broker import OandaBroker


def _broker() -> OandaBroker:
    b = OandaBroker.__new__(OandaBroker)  # no network in __init__ path
    b.account_id = "ACC"
    return b


def _order() -> Order:
    return Order(symbol="USD_CAD", side="sell", quantity=8916, order_type=OrderType.MARKET)


def _resp(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(json=lambda: payload)


def test_reject_transaction_marks_rejected(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_post_following_307",
        lambda url, body: _resp(
            {
                "orderRejectTransaction": {"id": "55", "rejectReason": "INSTRUMENT_NOT_TRADEABLE"},
            }
        ),
    )
    out = b.place_order(_order())
    assert out.status == OrderStatus.REJECTED
    assert out.order_id == "55"


def test_fok_cancel_marks_rejected(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_post_following_307",
        lambda url, body: _resp(
            {
                "orderCreateTransaction": {"id": "77"},
                "orderCancelTransaction": {"id": "78", "reason": "INSUFFICIENT_LIQUIDITY"},
            }
        ),
    )
    out = b.place_order(_order())
    assert out.status == OrderStatus.REJECTED


def test_fill_still_fills(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_post_following_307",
        lambda url, body: _resp(
            {
                "orderCreateTransaction": {"id": "1"},
                "orderFillTransaction": {"id": "2"},
            }
        ),
    )
    out = b.place_order(_order())
    assert out.status == OrderStatus.FILLED and out.order_id == "2"


def test_empty_txn_no_longer_pending_ghost(monkeypatch):
    b = _broker()
    monkeypatch.setattr(b, "_post_following_307", lambda url, body: _resp({}))
    out = b.place_order(_order())
    assert out.status == OrderStatus.REJECTED  # was PENDING with empty id


def test_cancel_hits_api(monkeypatch):
    b = _broker()
    calls = []

    def fake_request(method, url, json=None):
        calls.append((method, url))
        return SimpleNamespace(status_code=200, text="", headers={}, raise_for_status=lambda: None)

    b.write_client = SimpleNamespace(request=fake_request)
    assert b.cancel_order("123") is True
    assert calls == [("PUT", "/v3/accounts/ACC/orders/123/cancel")]


def test_cancel_failure_returns_false(monkeypatch):
    import httpx

    def fake_request(method, url, json=None):
        def _raise():
            raise httpx.HTTPStatusError("404", request=None, response=None)

        return SimpleNamespace(status_code=404, text="nope", headers={}, raise_for_status=_raise)

    b = _broker()
    b.write_client = SimpleNamespace(request=fake_request)
    assert b.cancel_order("123") is False


def test_reject_carries_reason(monkeypatch):
    b = _broker()
    monkeypatch.setattr(
        b,
        "_post_following_307",
        lambda url, body: _resp(
            {
                "orderRejectTransaction": {"id": "9", "rejectReason": "INSUFFICIENT_MARGIN"},
            }
        ),
    )
    out = b.place_order(_order())
    assert out.reject_reason == "INSUFFICIENT_MARGIN"
