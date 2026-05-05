"""Tests for PolymarketDataSource (CL-poly-1).

All tests use injected fake http_get_json — no live network. The live
endpoints are exercised by the integration suite under
tests/integration/test_live_external_endpoints.py (gated on
CURLIT_RUN_NETWORK_TESTS).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from src.execution.polymarket_data_source import (
    BookLevel,
    OrderBookSnapshot,
    PolymarketDataSource,
    _parse_book,
)


def _fake_http(responses: dict[str, dict[str, Any]]):  # type: ignore[no-untyped-def]
    """Build a fake http_get_json that returns a canned response per URL."""
    def _shim(url: str, params: dict[str, str]) -> dict[str, Any]:
        if url in responses:
            return responses[url]
        for prefix, body in responses.items():
            if url.startswith(prefix):
                return body
        raise KeyError(f"no fake response for {url}")
    return _shim


class TestBookParse:
    def test_sorts_bids_desc_asks_asc(self) -> None:
        # Intentionally pass unsorted; parser should canonicalize.
        data = {
            "bids": [
                {"price": "0.40", "size": "50"},
                {"price": "0.45", "size": "30"},
            ],
            "asks": [
                {"price": "0.55", "size": "20"},
                {"price": "0.50", "size": "40"},
            ],
        }
        book = _parse_book("tok-1", data)
        assert book.bids[0].price == Decimal("0.45")
        assert book.bids[-1].price == Decimal("0.40")
        assert book.asks[0].price == Decimal("0.50")
        assert book.asks[-1].price == Decimal("0.55")

    def test_top_bid_ask_and_mid(self) -> None:
        data = {
            "bids": [{"price": "0.45", "size": "30"}],
            "asks": [{"price": "0.50", "size": "40"}],
        }
        book = _parse_book("tok-1", data)
        assert book.top_bid is not None and book.top_bid.price == Decimal("0.45")
        assert book.top_ask is not None and book.top_ask.price == Decimal("0.50")
        assert book.mid == Decimal("0.475")

    def test_mid_none_when_one_side_empty(self) -> None:
        data = {"bids": [], "asks": [{"price": "0.50", "size": "10"}]}
        book = _parse_book("tok-1", data)
        assert book.mid is None


class TestDepthAggregation:
    def test_buy_depth_aggregates_at_or_below_price(self) -> None:
        book = OrderBookSnapshot(
            token_id="tok-1",
            bids=[],
            asks=[
                BookLevel(Decimal("0.50"), Decimal("10")),
                BookLevel(Decimal("0.51"), Decimal("20")),
                BookLevel(Decimal("0.55"), Decimal("100")),
            ],
        )
        # Buying up to 0.51: should fill 10 (at 0.50) + 20 (at 0.51) = 30.
        assert book.depth_at_or_better(Decimal("0.51"), "BUY") == Decimal("30")
        # Buying up to 0.49: nothing crosses → 0.
        assert book.depth_at_or_better(Decimal("0.49"), "BUY") == Decimal("0")

    def test_sell_depth_aggregates_at_or_above_price(self) -> None:
        book = OrderBookSnapshot(
            token_id="tok-1",
            bids=[
                BookLevel(Decimal("0.45"), Decimal("30")),
                BookLevel(Decimal("0.44"), Decimal("70")),
                BookLevel(Decimal("0.40"), Decimal("100")),
            ],
            asks=[],
        )
        # Selling at 0.44 or better: hits 0.45 (30) + 0.44 (70) = 100.
        assert book.depth_at_or_better(Decimal("0.44"), "SELL") == Decimal("100")

    def test_unknown_side_raises(self) -> None:
        book = OrderBookSnapshot(token_id="t", bids=[], asks=[])
        with pytest.raises(ValueError, match="side must be"):
            book.depth_at_or_better(Decimal("0.5"), "MAYBE")


class TestSymbolResolution:
    def test_token_id_passthrough(self) -> None:
        ds = PolymarketDataSource(http_get_json=_fake_http({}))
        assert ds.resolve_symbol("POLY:0xdeadbeef") == "0xdeadbeef"

    def test_condition_id_outcome_yes_resolves(self) -> None:
        # Gamma returns clobTokenIds = [yes_id, no_id].
        gamma = {
            "https://gamma-api.polymarket.com/markets/0xcond": {
                "clobTokenIds": '["123_yes", "456_no"]',
            },
        }
        ds = PolymarketDataSource(http_get_json=_fake_http(gamma))
        assert ds.resolve_symbol("POLY:0xcond:0") == "123_yes"
        assert ds.resolve_symbol("POLY:0xcond:1") == "456_no"

    def test_condition_id_outcome_native_list(self) -> None:
        # Some Gamma responses return clobTokenIds as a native list, not
        # JSON-encoded string. Resolver must handle both.
        gamma = {
            "https://gamma-api.polymarket.com/markets/0xcond": {
                "clobTokenIds": ["123_yes", "456_no"],
            },
        }
        ds = PolymarketDataSource(http_get_json=_fake_http(gamma))
        assert ds.resolve_symbol("POLY:0xcond:0") == "123_yes"

    def test_outside_outcome_index_raises(self) -> None:
        gamma = {
            "https://gamma-api.polymarket.com/markets/0xcond": {
                "clobTokenIds": ["a", "b"],
            },
        }
        ds = PolymarketDataSource(http_get_json=_fake_http(gamma))
        with pytest.raises(ValueError, match="no token at outcome"):
            ds.resolve_symbol("POLY:0xcond:5")

    def test_non_poly_symbol_raises(self) -> None:
        ds = PolymarketDataSource(http_get_json=_fake_http({}))
        with pytest.raises(ValueError, match="not a POLY symbol"):
            ds.resolve_symbol("EURUSD")

    def test_non_int_outcome_raises(self) -> None:
        ds = PolymarketDataSource(http_get_json=_fake_http({}))
        with pytest.raises(ValueError, match="outcome index must be int"):
            ds.resolve_symbol("POLY:0xcond:foo")
