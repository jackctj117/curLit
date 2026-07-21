"""Tests for the US-listed symbol universe (CL-tzug).

Mocked file bodies only — no live network. Covers: parsing both NASDAQ
Trader file formats (header + 'File Creation Time' trailer skip, test-issue
exclusion, ETF flag, exchange-code mapping), upsert idempotence, exists/get/
is_etf/robinhood_tradeable, resolve_name ranking + limit, and single-file
fetch-failure tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.data.symbols import (
    NASDAQ_LISTED_URL,
    OTHER_LISTED_URL,
    SymbolUniverse,
    parse_nasdaq_listed,
    parse_other_listed,
)

# --------------------------------------------------------------------------- #
# mocked file bodies
# --------------------------------------------------------------------------- #

NASDAQ_BODY = "\n".join([
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
    "Round Lot Size|ETF|NextShares",
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N",
    "TEVA|Teva Pharmaceutical Industries Limited - ADS|Q|N|N|100|N|N",
    "TSTQ|Nasdaq Test Ticker - Common Stock|Q|Y|N|100|N|N",  # test issue
    "QQQ|Invesco QQQ Trust, Series 1|Q|N|N|100|Y|N",  # ETF
    "File Creation Time: 0720202601:23|||||||",  # trailer
])

OTHER_BODY = "\n".join([
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
    "Test Issue|NASDAQ Symbol",
    "DHT|DHT Holdings, Inc. Common Stock|N|DHT|N|100|N|DHT",  # NYSE
    "FRO|Frontline Ltd. Ordinary Shares|N|FRO|N|100|N|FRO",  # NYSE
    "GLD|SPDR Gold Trust|P|GLD|Y|100|N|GLD",  # ARCA ETF
    "IMO|Imperial Oil Limited|A|IMO|N|100|N|IMO",  # AMEX
    "TESTX|NYSE Test Security|N|TESTX|N|100|Y|TESTX",  # test issue
    "WEIRD|Weird Exchange Co|X|WEIRD|N|100|N|WEIRD",  # unknown -> OTHER
    "File Creation Time: 0720202601:23||||||||",  # trailer
])


def _fake_http(bodies: dict[str, str]):
    """Return an http_get shim that serves canned bodies, raising for URLs
    mapped to None (simulates a down file)."""
    def _get(url: str) -> str:
        body = bodies.get(url)
        if body is None:
            raise RuntimeError(f"simulated fetch failure for {url}")
        return body
    return _get


BOTH_OK = {NASDAQ_LISTED_URL: NASDAQ_BODY, OTHER_LISTED_URL: OTHER_BODY}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine(tmp_path: Path):
    """Sqlite engine with the symbols migration applied (shimmed for sqlite)."""
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'symbols.db'}")
    sql = _strip_sql_comments(
        Path("migrations/010_symbols.sql").read_text(),
    )
    sql = sql.replace("TIMESTAMPTZ", "TEXT").replace("BIGSERIAL", "INTEGER")
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def test_parse_nasdaq_listed_skips_header_trailer_and_test_issue():
    rows = parse_nasdaq_listed(NASDAQ_BODY)
    symbols = {r["symbol"] for r in rows}
    assert symbols == {"AAPL", "TEVA", "QQQ"}  # TSTQ (test issue) excluded
    assert all(r["exchange"] == "NASDAQ" for r in rows)
    assert all(r["source"] == "nasdaqlisted" for r in rows)


def test_parse_nasdaq_etf_flag():
    rows = {r["symbol"]: r for r in parse_nasdaq_listed(NASDAQ_BODY)}
    assert rows["QQQ"]["is_etf"] is True
    assert rows["AAPL"]["is_etf"] is False


def test_parse_other_listed_exchange_mapping_and_test_issue():
    rows = {r["symbol"]: r for r in parse_other_listed(OTHER_BODY)}
    assert "TESTX" not in rows  # test issue excluded
    assert rows["DHT"]["exchange"] == "NYSE"  # N
    assert rows["FRO"]["exchange"] == "NYSE"  # N
    assert rows["GLD"]["exchange"] == "ARCA"  # P
    assert rows["IMO"]["exchange"] == "AMEX"  # A
    assert rows["WEIRD"]["exchange"] == "OTHER"  # unknown code
    assert all(r["source"] == "otherlisted" for r in rows.values())


def test_parse_other_etf_flag():
    rows = {r["symbol"]: r for r in parse_other_listed(OTHER_BODY)}
    assert rows["GLD"]["is_etf"] is True
    assert rows["DHT"]["is_etf"] is False


def test_parse_ignores_blank_lines():
    body = NASDAQ_BODY.replace("AAPL", "\n\nAAPL")
    rows = parse_nasdaq_listed(body)
    assert any(r["symbol"] == "AAPL" for r in rows)


# --------------------------------------------------------------------------- #
# refresh / upsert
# --------------------------------------------------------------------------- #


def test_refresh_counts_and_populates(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    counts = uni.refresh()
    # AAPL, TEVA, QQQ (nasdaq) + DHT, FRO, GLD, IMO, WEIRD (other) = 8
    assert counts["inserted"] == 8
    assert counts["updated"] == 0
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM symbols")).scalar()
    assert total == 8


def test_refresh_idempotent(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    counts2 = uni.refresh()
    assert counts2["inserted"] == 0
    assert counts2["updated"] == 8  # refreshed in place, not duplicated
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM symbols")).scalar()
    assert total == 8  # no duplication


def test_refresh_updates_in_place_on_name_change(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    changed = {
        NASDAQ_LISTED_URL: NASDAQ_BODY.replace(
            "Apple Inc. - Common Stock", "Apple Inc. NEW NAME",
        ),
        OTHER_LISTED_URL: OTHER_BODY,
    }
    uni2 = SymbolUniverse(engine, http_get=_fake_http(changed))
    uni2.refresh()
    assert uni2.get("AAPL")["security_name"] == "Apple Inc. NEW NAME"


# --------------------------------------------------------------------------- #
# lookups
# --------------------------------------------------------------------------- #


def test_exists(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    assert uni.exists("AAPL") is True
    assert uni.exists("aapl") is True  # case-insensitive
    assert uni.exists("  teva  ") is True  # trimmed + case-insensitive
    assert uni.exists("ZZZZFAKE") is False
    assert uni.exists("") is False
    assert uni.exists("TSTQ") is False  # test issue never ingested


def test_get(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    row = uni.get("GLD")
    assert row == {
        "symbol": "GLD",
        "security_name": "SPDR Gold Trust",
        "exchange": "ARCA",
        "is_etf": True,
    }
    assert uni.get("NOPE") is None


def test_is_etf(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    assert uni.is_etf("GLD") is True
    assert uni.is_etf("QQQ") is True
    assert uni.is_etf("TEVA") is False
    assert uni.is_etf("NOPE") is False


def test_robinhood_tradeable(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    assert uni.robinhood_tradeable("DHT") is True  # NYSE common
    assert uni.robinhood_tradeable("GLD") is True  # ARCA ETF
    assert uni.robinhood_tradeable("ZZZZFAKE") is False


# --------------------------------------------------------------------------- #
# name resolution
# --------------------------------------------------------------------------- #


def test_resolve_name_substring(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    hits = uni.resolve_name("Teva")
    assert any(h["symbol"] == "TEVA" for h in hits)

    hits = uni.resolve_name("Frontline")
    assert hits[0]["symbol"] == "FRO"


def test_resolve_name_case_insensitive(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    assert any(h["symbol"] == "TEVA" for h in uni.resolve_name("teva"))


def test_resolve_name_ranking_prefix_beats_substring(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    # "Apple" is a whole word / prefix of "Apple Inc." -> AAPL ranks first.
    hits = uni.resolve_name("Apple")
    assert hits[0]["symbol"] == "AAPL"


def test_resolve_name_limit(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    # "Inc" appears in several security names; limit caps the result count.
    hits = uni.resolve_name("Inc", limit=2)
    assert len(hits) <= 2


def test_resolve_name_empty_query(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    assert uni.resolve_name("") == []


# --------------------------------------------------------------------------- #
# fetch-failure tolerance
# --------------------------------------------------------------------------- #


def test_refresh_tolerates_nasdaq_file_down(engine):
    bodies = {NASDAQ_LISTED_URL: None, OTHER_LISTED_URL: OTHER_BODY}
    uni = SymbolUniverse(engine, http_get=_fake_http(bodies))
    counts = uni.refresh()
    # Only otherlisted ingests: DHT, FRO, GLD, IMO, WEIRD = 5
    assert counts["inserted"] == 5
    assert uni.exists("DHT") is True
    assert uni.exists("AAPL") is False  # nasdaq file was down


def test_refresh_tolerates_other_file_down(engine):
    bodies = {NASDAQ_LISTED_URL: NASDAQ_BODY, OTHER_LISTED_URL: None}
    uni = SymbolUniverse(engine, http_get=_fake_http(bodies))
    counts = uni.refresh()
    assert counts["inserted"] == 3  # AAPL, TEVA, QQQ
    assert uni.exists("AAPL") is True
    assert uni.exists("DHT") is False


def test_refresh_both_files_down_returns_zero(engine):
    bodies = {NASDAQ_LISTED_URL: None, OTHER_LISTED_URL: None}
    uni = SymbolUniverse(engine, http_get=_fake_http(bodies))
    counts = uni.refresh()
    assert counts == {"inserted": 0, "updated": 0, "skipped": 0}
