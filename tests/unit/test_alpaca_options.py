"""Tests for Alpaca paper options execution (CL-ldd2).

Injected transport — no live Alpaca. Covers moneyness/DTE parsing, target
strike/expiry, request construction, and nearest-contract resolution.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.execution.alpaca_options import (
    AlpacaOptionsClient,
    ContractSelectionConfig,
    contract_target,
    parse_dte_days,
    parse_moneyness,
    resolve_contract,
)

TODAY = date(2026, 7, 21)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def test_parse_moneyness():
    assert parse_moneyness("slightly OTM (~5% above spot)", 0.04) == 0.05
    assert parse_moneyness("calls, 3-6 weeks", 0.04) == 0.04  # no % -> default
    assert parse_moneyness(None, 0.03) == 0.03


def test_parse_dte():
    assert parse_dte_days("3-5 weeks to expiry", 28) == 28  # mid 4 weeks
    assert parse_dte_days("3 weeks", 28) == 21
    assert parse_dte_days("21 days out", 28) == 21
    assert parse_dte_days("calls", 28) == 28  # default


def test_contract_target_call_is_otm_above():
    right, strike, exp = contract_target(
        {"action": "buy_calls", "preferred_instrument": "~5% OTM, 4 weeks"},
        underlying_price=100.0,
        today=TODAY,
        cfg=ContractSelectionConfig(),
    )
    assert right == "call"
    assert strike == 105.0
    assert (exp - TODAY).days == 28


def test_contract_target_put_is_otm_below():
    right, strike, _ = contract_target(
        {"action": "buy_puts", "preferred_instrument": "~10% OTM puts"},
        underlying_price=200.0,
        today=TODAY,
        cfg=ContractSelectionConfig(),
    )
    assert right == "put"
    assert strike == 180.0


# --------------------------------------------------------------------------- #
# client request construction
# --------------------------------------------------------------------------- #


class _FakeAPI:
    def __init__(self, **responses: Any) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(self, method, url, headers, params, json_body):
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "params": params, "json": json_body}
        )
        if "/v2/account" in url:
            return self.responses.get("account", {})
        if "/v2/options/contracts" in url:
            return self.responses.get("contracts", {"option_contracts": []})
        if "/v2/orders" in url:
            return self.responses.get("order", {"id": "o1", "status": "accepted"})
        if "/v2/positions" in url:
            return self.responses.get("positions", [])
        return {}


def test_client_uses_paper_base_and_auth():
    api = _FakeAPI(account={"buying_power": "100000"})
    c = AlpacaOptionsClient("KEY", "SECRET", paper=True, request_fn=api)
    acct = c.get_account()
    assert acct["buying_power"] == "100000"
    call = api.calls[0]
    assert call["url"].startswith("https://paper-api.alpaca.markets")
    assert call["headers"]["APCA-API-KEY-ID"] == "KEY"
    assert call["headers"]["APCA-API-SECRET-KEY"] == "SECRET"


def test_find_contracts_builds_query():
    api = _FakeAPI()
    c = AlpacaOptionsClient("K", "S", request_fn=api)
    c.find_contracts("aapl", "call", date(2026, 8, 8), date(2026, 8, 28), 100.0, 120.0)
    p = api.calls[0]["params"]
    assert p["underlying_symbols"] == "AAPL"
    assert p["type"] == "call"
    assert p["expiration_date_gte"] == "2026-08-08"
    assert p["strike_price_gte"] == "100.00"
    assert p["status"] == "active"


def test_submit_order_body():
    api = _FakeAPI(order={"id": "abc", "status": "accepted"})
    c = AlpacaOptionsClient("K", "S", request_fn=api)
    out = c.submit_option_order("AAPL260821C00105000", 2)
    assert out["id"] == "abc"
    body = api.calls[0]["json"]
    assert body == {
        "symbol": "AAPL260821C00105000",
        "qty": "2",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }


def test_list_option_positions_filters_asset_class():
    api = _FakeAPI(
        positions=[
            {"symbol": "AAPL", "qty": "1", "asset_class": "us_equity"},
            {"symbol": "AAPL260821C00105000", "qty": "1", "asset_class": "us_option"},
        ]
    )
    c = AlpacaOptionsClient("K", "S", request_fn=api)
    pos = c.list_option_positions()
    assert len(pos) == 1 and pos[0]["asset_class"] == "us_option"


# --------------------------------------------------------------------------- #
# resolve_contract
# --------------------------------------------------------------------------- #


class _FakeClient:
    def __init__(self, contracts: list[dict[str, Any]], raises: bool = False) -> None:
        self._contracts = contracts
        self.raises = raises

    def find_contracts(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
        if self.raises:
            raise RuntimeError("alpaca down")
        return self._contracts


def _idea() -> dict[str, Any]:
    return {
        "ticker": "RTX",
        "action": "buy_calls",
        "preferred_instrument": "slightly OTM (~5%), 4 weeks",
    }


def test_resolve_picks_nearest_expiry_then_strike():
    # target: spot 100 * 1.05 = 105 strike, ~2026-08-18 expiry.
    contracts = [
        {"symbol": "RTX-A", "strike_price": "110", "expiration_date": "2026-08-18"},
        {"symbol": "RTX-B", "strike_price": "105", "expiration_date": "2026-08-18"},
        {"symbol": "RTX-C", "strike_price": "105", "expiration_date": "2026-09-19"},
    ]
    best = resolve_contract(_FakeClient(contracts), _idea(), 100.0, TODAY)
    assert best["symbol"] == "RTX-B"  # same (nearest) expiry, closest strike


def test_resolve_none_when_no_contracts():
    assert resolve_contract(_FakeClient([]), _idea(), 100.0, TODAY) is None


def test_resolve_fail_soft_on_error():
    assert resolve_contract(_FakeClient([], raises=True), _idea(), 100.0, TODAY) is None


def test_resolve_none_on_bad_price():
    assert resolve_contract(_FakeClient([{"symbol": "x"}]), _idea(), 0.0, TODAY) is None
