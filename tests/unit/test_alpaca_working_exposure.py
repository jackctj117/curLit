"""Independent remaining-inventory oracles for CL-0deu.1 working orders."""

from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.execution.alpaca_equity import AlpacaEquityClient
from src.execution.alpaca_exposure import ExposureUnavailableError, working_order_quantities
from src.execution.alpaca_options import AlpacaOptionsClient


def order(**changes: Any) -> dict[str, Any]:
    return {
        "id": "external-1",
        "symbol": "RTX",
        "asset_class": "us_equity",
        "qty": "10",
        "filled_qty": "4",
        "side": "buy",
        "status": "partially_filled",
        **changes,
    }


@pytest.mark.parametrize("client_type", [AlpacaEquityClient, AlpacaOptionsClient])
def test_clients_request_full_account_open_orders(client_type: type) -> None:
    calls = []

    def request(*args: Any) -> list[dict[str, Any]]:
        calls.append(args)
        return [order()]

    client = client_type("fixture", "fixture", request_fn=request)
    assert client.list_open_orders() == [order()]
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/v2/orders")
    assert calls[0][3] == {"status": "open", "limit": "500", "nested": "false"}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        [None],
        [order()] * 500,
        [order(), order()],
        [order(status="unknown")],
        [order(status="filled")],
        [order(side="unknown")],
        [order(qty=None)],
        [order(filled_qty=None)],
        [order(qty="NaN")],
        [order(qty="-1")],
        [order(filled_qty="-1")],
        [order(filled_qty="11")],
        [order(asset_class="us_equiy")],
        [order(symbol="")],
        [order(id="")],
        [order(order_class="bracket")],
        [order(legs=[order()])],
        [order(symbol="RTX1260821C00105000", asset_class="us_option")],
        [order(symbol="RTX260821C00105000", asset_class="us_option", qty="1.5")],
    ],
)
def test_incomplete_or_malformed_account_snapshot_is_unknown(payload: object) -> None:
    with pytest.raises(ExposureUnavailableError):
        working_order_quantities(payload, asset_class="us_equity")


def test_remaining_quantities_never_subtract_pending_closes() -> None:
    payload = [order(), order(id="external-2", qty="7", filled_qty="2", side="sell")]
    # Hand oracle: six unfilled buys + five unfilled sells = eleven gross shares.
    assert working_order_quantities(payload, asset_class="us_equity") == {"RTX": Decimal(11)}


@given(st.integers(1, 100000), st.integers(0, 100000))
def test_remaining_fractional_quantity_matches_rational_oracle(total: int, filled: int) -> None:
    filled = min(filled, total)
    payload = [order(qty=str(Decimal(total) / 100), filled_qty=str(Decimal(filled) / 100))]
    observed = working_order_quantities(payload, asset_class="us_equity")["RTX"]
    assert Fraction(observed) == Fraction(total - filled, 100)
    assert 0 <= observed <= Decimal(total) / 100
