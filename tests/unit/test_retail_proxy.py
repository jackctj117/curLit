"""Unit tests for the Robinhood retail-execution proxy layer (CL-vowz).

Covers the shipped config (loads/validates, every ETF ``[A-Z]{1,5}``),
the :func:`retail_proxy` resolver across the four instrument classes
(commodity, index-with-inverse, FX pair, plain equity, unknown), the
compact/full renderers including the short-direction inverse surfacing
and escaping, and both wired render sites (digest idea sub-line, the
``idea <id>`` Robinhood section).
"""

from __future__ import annotations

import textwrap

import pytest

from src.events.retail_proxy import (
    DEFAULT_PROXIES_PATH,
    RetailProxy,
    compact_label,
    full_detail,
    load_retail_proxies,
    retail_proxy,
)

# --------------------------------------------------------------------- #
# Config load + validation
# --------------------------------------------------------------------- #


class TestConfigLoads:
    def test_shipped_config_loads(self) -> None:
        table = load_retail_proxies()
        assert table  # non-empty
        # Spot-check the required mappings from the bead.
        assert table["XAU_USD"].proxies == ("GLD", "IAU")
        assert table["BCO_USD"].proxies == ("BNO",)
        assert table["WTICO_USD"].proxies == ("USO",)
        assert table["NATGAS_USD"].proxies == ("UNG",)
        assert table["NATGAS_USD"].leveraged_warning is True
        assert table["XCU_USD"].proxies == ("CPER", "COPX")
        assert table["XAG_USD"].proxies == ("SLV",)
        assert table["XPT_USD"].proxies == ("PPLT",)
        assert table["XPD_USD"].proxies == ("PALL",)
        assert table["WHEAT_USD"].proxies == ("WEAT",)
        assert table["CORN_USD"].proxies == ("CORN",)

    def test_index_inverse_pairs(self) -> None:
        table = load_retail_proxies()
        assert table["SPX500_USD"].proxies == ("SPY",)
        assert table["SPX500_USD"].inverse == ("SH", "SDS")
        assert table["NAS100_USD"].proxies == ("QQQ",)
        assert table["NAS100_USD"].inverse == ("PSQ", "SQQQ")

    def test_fx_pairs_flagged_not_tradable(self) -> None:
        table = load_retail_proxies()
        for pair in (
            "USD_JPY", "USD_CAD", "USD_NOK", "USD_ZAR", "EUR_USD",
            "GBP_USD", "AUD_USD", "NZD_USD", "USD_CHF", "USD_CNH",
        ):
            assert pair in table, pair
            assert table[pair].tradable is False, pair
            assert table[pair].proxies == (), pair
            assert "not available on Robinhood" in table[pair].note.lower() \
                or "not available on robinhood" in table[pair].note.lower()

    def test_every_etf_is_valid_ticker(self) -> None:
        import re  # noqa: PLC0415

        rgx = re.compile(r"^[A-Z]{1,5}$")
        table = load_retail_proxies()
        for name, proxy in table.items():
            for etf in (*proxy.proxies, *proxy.inverse):
                assert rgx.match(etf), f"{name}: bad ETF ticker {etf!r}"

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_retail_proxies(tmp_path / "nope.yaml")

    def test_invalid_ticker_rejected(self, tmp_path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            textwrap.dedent(
                """
                instruments:
                  XAU_USD:
                    proxies: [gld123]
                """
            )
        )
        with pytest.raises(ValueError, match="invalid ETF ticker"):
            load_retail_proxies(bad)

    def test_tradable_entry_needs_proxies(self, tmp_path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            textwrap.dedent(
                """
                instruments:
                  XAU_USD:
                    note: no proxies here
                """
            )
        )
        with pytest.raises(ValueError, match="no 'proxies'"):
            load_retail_proxies(bad)

    def test_empty_instruments_raises(self, tmp_path) -> None:
        bad = tmp_path / "empty.yaml"
        bad.write_text("instruments: {}\n")
        with pytest.raises(ValueError, match="missing or empty"):
            load_retail_proxies(bad)

    def test_default_path_points_at_config(self) -> None:
        assert DEFAULT_PROXIES_PATH.name == "retail_proxies.yaml"


# --------------------------------------------------------------------- #
# retail_proxy resolver
# --------------------------------------------------------------------- #


@pytest.fixture
def table() -> dict[str, RetailProxy]:
    return load_retail_proxies()


class TestRetailProxy:
    def test_commodity_long(self, table) -> None:
        p = retail_proxy("XAU_USD", "long", proxies=table)
        assert p is not None
        assert p.proxies == ("GLD", "IAU")
        assert p.active == ("GLD", "IAU")
        assert p.tradable is True
        assert p.trades_directly is False
        assert p.is_bearish is False

    def test_index_short_surfaces_inverse(self, table) -> None:
        p = retail_proxy("SPX500_USD", "short", proxies=table)
        assert p is not None
        assert p.direction == "short"
        assert p.is_bearish is True
        # active is the INVERSE side for a bearish call.
        assert p.active == ("SH", "SDS")
        assert p.proxies == ("SPY",)

    def test_index_long_surfaces_proxies(self, table) -> None:
        p = retail_proxy("NAS100_USD", "long", proxies=table)
        assert p is not None
        assert p.active == ("QQQ",)

    def test_bearish_action_alias(self, table) -> None:
        # 'buy_puts' / 'sell' / 'bearish' all fold to short.
        for token in ("buy_puts", "sell", "bearish", "SHORT"):
            p = retail_proxy("SPX500_USD", token, proxies=table)
            assert p is not None and p.is_bearish, token

    def test_fx_pair_not_tradable(self, table) -> None:
        p = retail_proxy("USD_JPY", "short", proxies=table)
        assert p is not None
        assert p.tradable is False
        assert p.proxies == ()

    def test_plain_equity_trades_directly(self, table) -> None:
        for ticker in ("DHT", "TEVA", "TSM"):
            p = retail_proxy(ticker, "long", proxies=table)
            assert p is not None, ticker
            assert p.trades_directly is True, ticker
            assert p.tradable is True, ticker
            assert p.active == (ticker,), ticker
            assert "directly" in p.note

    def test_unknown_oanda_shaped_is_none(self, table) -> None:
        # An OANDA-shaped instrument we did not map → None-safe.
        assert retail_proxy("USD_SEK", "long", proxies=table) is None
        assert retail_proxy("XYZ_ABC", None, proxies=table) is None

    def test_empty_and_junk_none_safe(self, table) -> None:
        assert retail_proxy("", "long", proxies=table) is None
        assert retail_proxy(None, None, proxies=table) is None  # type: ignore[arg-type]
        # Too-long / lowercase junk that is neither a ticker nor OANDA.
        assert retail_proxy("toolongsymbol", None, proxies=table) is None

    def test_no_direction_defaults_to_proxies(self, table) -> None:
        p = retail_proxy("SPX500_USD", None, proxies=table)
        assert p is not None
        assert p.direction == ""
        assert p.active == ("SPY",)  # long side when direction unknown


# --------------------------------------------------------------------- #
# compact_label
# --------------------------------------------------------------------- #


class TestCompactLabel:
    def test_commodity(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        assert compact_label("XAU_USD", "long") == "GLD/IAU"

    def test_index_short(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        assert compact_label("SPX500_USD", "short") == "SPY→short via SH/SDS"

    def test_index_long(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        assert compact_label("NAS100_USD", "long") == "QQQ"

    def test_commodity_short_no_inverse(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        # Brent has no inverse fund → puts instruction.
        assert compact_label("BCO_USD", "short") == "BNO puts (or short)"

    def test_fx(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        assert compact_label("USD_JPY", "short") == "FX — n/a"

    def test_equity(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        assert compact_label("DHT", "long") == "trades directly"

    def test_unknown(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        assert compact_label("USD_SEK", "long") == "n/a"


# --------------------------------------------------------------------- #
# full_detail
# --------------------------------------------------------------------- #


class TestFullDetail:
    def test_commodity_long(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        lines = full_detail("XAU_USD", "long")
        body = "\n".join(lines)
        assert "Robinhood proxies: GLD, IAU" in body
        assert "LONG XAU_USD → buy GLD or IAU." in body
        assert "track imperfectly" in body
        assert "not financial advice" in body.lower()

    def test_index_short_inverse_and_puts(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        lines = full_detail("SPX500_USD", "short")
        body = "\n".join(lines)
        assert "Inverse (for shorts): SH, SDS" in body
        assert "SHORT SPX500_USD → buy SH/SDS or SPY puts." in body
        # leveraged decay caveat present for an index with inverse funds.
        assert "DECAY" in body
        assert "broker approval" in body.lower()

    def test_fx_pair(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        lines = full_detail("USD_JPY", "short")
        body = "\n".join(lines)
        assert "is FX — not tradable on Robinhood" in body
        assert "Skip the FX leg" in body
        assert "not financial advice" in body.lower()

    def test_equity_direct(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        lines = full_detail("TEVA", "long")
        body = "\n".join(lines)
        assert "TEVA trades directly" in body
        assert "buy TEVA shares or calls" in body

    def test_equity_short(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        body = "\n".join(full_detail("DHT", "short"))
        assert "short shares (margin) or buy DHT puts" in body

    def test_unknown_returns_empty(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        assert full_detail("USD_SEK", "long") == []

    def test_commodity_short_no_inverse(self, monkeypatch, table) -> None:
        _patch_table(monkeypatch, table)
        body = "\n".join(full_detail("BCO_USD", "short"))
        assert "SHORT BCO_USD → buy BNO puts" in body


def _patch_table(monkeypatch, table) -> None:
    """Point the module cache at a test table (no disk / no real config
    dependency for the render tests)."""
    monkeypatch.setattr(
        "src.events.retail_proxy._cached_proxies", lambda: table,
    )
