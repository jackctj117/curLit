"""Exposure interlock oracles (CL-0deu.1.1).

Broker JSON fixtures exercise the real clients. Fraction arithmetic is an
independent oracle for decimal parsing; hand-counted inventory vectors and
sign/monotonicity properties cover the executor aggregation paths.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.execution.alpaca_equity import AlpacaEquityClient
from src.execution.alpaca_equity_executor import EquityExecConfig, _held_ticker_qty
from src.execution.alpaca_exposure import (
    ExposureUnavailableError,
    option_underlying,
    position_quantity,
)
from src.execution.alpaca_options import AlpacaOptionsClient
from src.execution.alpaca_options_executor import OptionsExecConfig, _held_contract_counts


@pytest.mark.parametrize(
    "client_type,method",
    [
        (AlpacaOptionsClient, "list_option_positions"),
        (AlpacaEquityClient, "list_equity_positions"),
    ],
)
@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        "",
        False,
        [None],
        [{"symbol": "RTX", "qty": "1"}],
        [{"symbol": "RTX", "qty": "1", "asset_class": "us_equiy"}],
        [{"symbol": "", "qty": "1", "asset_class": "us_equity"}],
        [{"symbol": "RTX", "asset_class": "us_equity"}],
        [{"symbol": "RTX", "qty": "NaN", "asset_class": "us_equity"}],
        [{"symbol": "NOT_AN_OCC", "qty": "1", "asset_class": "us_option"}],
        [{"symbol": "RTX260821C00105000", "qty": "0.5", "asset_class": "us_option"}],
        [{"symbol": "RTX", "qty": "1", "asset_class": "us_equity"}] * 2,
    ],
)
def test_clients_do_not_convert_bad_responses_to_flat(
    client_type: type, method: str, payload: object
) -> None:
    calls: list[str] = []

    def request(
        verb: str,
        url: str,
        headers: dict[str, str],
        params: dict[str, str] | None,
        body: dict[str, Any] | None,
    ) -> object:
        calls.append(verb)
        assert url.endswith("/v2/positions")
        return payload

    client = client_type("fixture-key", "fixture-secret", request_fn=request)
    with pytest.raises(ExposureUnavailableError):
        getattr(client, method)()
    assert calls == ["GET"]


@pytest.mark.parametrize(
    "client_type,method,expected",
    [
        (AlpacaOptionsClient, "list_option_positions", "RTX260821C00105000"),
        (AlpacaEquityClient, "list_equity_positions", "RTX"),
    ],
)
def test_clients_preserve_valid_position_fields(
    client_type: type, method: str, expected: str
) -> None:
    rows = [
        {"symbol": "RTX", "qty": "-0.25", "asset_class": "us_equity", "current_price": "95.0"},
        {
            "symbol": "RTX260821C00105000",
            "qty": "2",
            "asset_class": "us_option",
            "current_price": "1.0",
        },
        {"symbol": "BTCUSD", "qty": "0.01", "asset_class": "crypto"},
    ]
    client = client_type("K", "S", request_fn=lambda *_: rows)
    assert getattr(client, method)() == [row for row in rows if row["symbol"] == expected]
    empty = client_type("K", "S", request_fn=lambda *_: [])
    assert getattr(empty, method)() == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "",
        "not-a-number",
        "NaN",
        "sNaN",
        "Infinity",
        "-Infinity",
        float("nan"),
        float("inf"),
        {},
        [],
    ],
)
def test_invalid_quantity_is_unavailable(value: object) -> None:
    with pytest.raises(ExposureUnavailableError):
        position_quantity(value)


@given(st.integers(min_value=-1_000_000, max_value=1_000_000))
def test_quantity_preserves_fraction_and_sign(units: int) -> None:
    # Three decimal places provide a fixed rational oracle without floats.
    raw = f"{'-' if units < 0 else ''}{abs(units) // 1000}.{abs(units) % 1000:03d}"
    result = position_quantity(raw)
    assert Fraction(result) == Fraction(units, 1000)
    assert position_quantity(str(result)) == result
    assert position_quantity(str(-result)) == -result


@given(st.integers(min_value=-1_000_000, max_value=1_000_000))
def test_equity_exposure_keeps_fractional_inventory(units: int) -> None:
    qty = Decimal(units) / 1000
    client = SimpleNamespace(
        list_equity_positions=lambda: [
            {"symbol": "RTX", "qty": str(qty), "asset_class": "us_equity"}
        ]
    )
    actual = _held_ticker_qty(client, "RTX")
    assert Fraction(actual) == Fraction(abs(units), 1000)


@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
def test_option_exposure_is_absolute_and_monotonic(first: int, additional: int) -> None:
    first_row = {"symbol": "RTX260821C00105000", "qty": str(-first), "asset_class": "us_option"}
    second_row = {
        "symbol": "RTX260821P00110000",
        "qty": str(additional),
        "asset_class": "us_option",
    }
    before = SimpleNamespace(list_option_positions=lambda: [first_row])
    after = SimpleNamespace(list_option_positions=lambda: [first_row, second_row])
    assert _held_contract_counts(before, "RTX260821C00105000", "RTX") == (first, first)
    same, total = _held_contract_counts(after, "RTX260821C00105000", "RTX")
    assert same == first
    assert total == first + additional
    assert total >= first


@pytest.mark.parametrize(
    "symbol,root",
    [
        ("ASC260821C00017500", "ASC"),
        ("ASTL260821P00004000", "ASTL"),
        ("RTX1260821C00105000", "RTX1"),
        ("BRK.B260821C00105000", "BRK.B"),
    ],
)
def test_option_root_matches_full_symbol(symbol: str, root: str) -> None:
    assert option_underlying(symbol) == root


def test_adjusted_option_root_cannot_evade_underlying_cap() -> None:
    client = SimpleNamespace(
        list_option_positions=lambda: [
            {
                "symbol": "RTX1260821C00105000",
                "qty": "1",
                "asset_class": "us_option",
            }
        ]
    )
    with pytest.raises(ExposureUnavailableError, match="adjusted_option_underlying_unavailable"):
        _held_contract_counts(client, "RTX260821C00105000", "RTX")


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "1", float("nan")])
def test_invalid_proposed_quantity_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="qty"):
        OptionsExecConfig(qty=value)


@pytest.mark.parametrize(
    "field",
    ["max_contracts_per_symbol", "max_contracts_per_underlying", "max_per_day", "max_per_hour"],
)
@pytest.mark.parametrize("value", [-1, 1.5, True, "1", float("inf")])
def test_invalid_option_caps_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        OptionsExecConfig(**{field: value})


@pytest.mark.parametrize("field", ["max_positions_per_ticker", "max_per_day", "max_per_hour"])
@pytest.mark.parametrize("value", [-1, 1.5, True, "1", float("inf")])
def test_invalid_equity_caps_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        EquityExecConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1, True, "1000", float("nan"), float("inf")])
def test_invalid_equity_notional_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="notional_usd"):
        EquityExecConfig(notional_usd=value)
