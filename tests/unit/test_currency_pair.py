"""Tests for dialect-safe currency-pair parsing (CL-rybp).

Raw symbol[:3]/[3:6] slicing fabricated '_US' quote legs for OANDA-form
symbols and 'X50' legs for indices; currency_pair() canonicalizes first
and refuses non-pairs.
"""

from __future__ import annotations

from src.execution.broker import currency_pair
from src.portfolio.pretrade import PreTradeValidator


def test_compact_pair():
    assert currency_pair("EURUSD") == ("EUR", "USD")


def test_oanda_underscore_pair():
    assert currency_pair("EUR_USD") == ("EUR", "USD")
    assert currency_pair("USD_CHF") == ("USD", "CHF")


def test_metals_are_pairs():
    assert currency_pair("XAU_USD") == ("XAU", "USD")


def test_indices_and_junk_are_not_pairs():
    assert currency_pair("SPX500_USD") is None
    assert currency_pair("NAS100USD") is None
    assert currency_pair("EUR") is None
    assert currency_pair("") is None
    assert currency_pair("EUR_USD_X") is None


def test_pretrade_currency_from_symbol_handles_both_dialects():
    assert PreTradeValidator._currency_from_symbol("EUR_USD") == "EUR"
    assert PreTradeValidator._currency_from_symbol("EURUSD") == "EUR"
    assert PreTradeValidator._currency_from_symbol("SPX500_USD") is None
