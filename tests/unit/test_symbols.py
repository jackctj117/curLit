"""Tests for the US-listed symbol universe (CL-tzug).

Mocked file bodies only — no live network. Covers: parsing both NASDAQ
Trader file formats (header + 'File Creation Time' trailer skip, test-issue
exclusion, ETF flag, exchange-code mapping), upsert idempotence, exists/get/
is_etf/robinhood_tradeable, resolve_name ranking + limit, and single-file
fetch-failure tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.data.symbols import (
    NASDAQ_LISTED_URL,
    OTHER_LISTED_URL,
    SEC_COMPANY_TICKERS_URL,
    SymbolUniverse,
    parse_nasdaq_listed,
    parse_other_listed,
    parse_sec_company_tickers,
)

# --------------------------------------------------------------------------- #
# mocked file bodies
# --------------------------------------------------------------------------- #

NASDAQ_BODY = "\n".join(
    [
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
        "Round Lot Size|ETF|NextShares",
        "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N",
        "TEVA|Teva Pharmaceutical Industries Limited - ADS|Q|N|N|100|N|N",
        "TSTQ|Nasdaq Test Ticker - Common Stock|Q|Y|N|100|N|N",  # test issue
        "QQQ|Invesco QQQ Trust, Series 1|Q|N|N|100|Y|N",  # ETF
        "File Creation Time: 0720202601:23|||||||",  # trailer
    ]
)

OTHER_BODY = "\n".join(
    [
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
        "DHT|DHT Holdings, Inc. Common Stock|N|DHT|N|100|N|DHT",  # NYSE
        "FRO|Frontline Ltd. Ordinary Shares|N|FRO|N|100|N|FRO",  # NYSE
        "GLD|SPDR Gold Trust|P|GLD|Y|100|N|GLD",  # ARCA ETF
        "IMO|Imperial Oil Limited|A|IMO|N|100|N|IMO",  # AMEX
        "TESTX|NYSE Test Security|N|TESTX|N|100|Y|TESTX",  # test issue
        "WEIRD|Weird Exchange Co|X|WEIRD|N|100|N|WEIRD",  # unknown -> OTHER
        "File Creation Time: 0720202601:23||||||||",  # trailer
    ]
)


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

# SEC company_tickers.json shape (CL-9xha): object keyed by index string,
# each {cik_str, ticker, title}. AAPL/TEVA/WEIRD are in the mocked universe;
# ZUEXTRA is not (→ unmatched, never inserted); the blank-ticker row is
# dropped at parse. WEIRD's SEC title deliberately shares NO word with its
# NASDAQ "Weird Exchange Co" name, so a "Wonderful" query can only resolve via
# the SEC name — proving resolve_name searches it.
SEC_BODY = json.dumps(
    {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 12345, "ticker": "TEVA", "title": "Teva Pharmaceutical Industries Ltd"},
        "2": {"cik_str": 99999, "ticker": "WEIRD", "title": "Wonderful Alphabet Holdings Corp"},
        "3": {"cik_str": 55555, "ticker": "ZUEXTRA", "title": "Ghost Co Not In Universe"},
        "4": {"cik_str": 0, "ticker": "", "title": "Blank Ticker Row"},
    }
)


def _fake_sec(body: str):
    """SEC http_get shim serving a canned JSON body (asserts the SEC URL)."""

    def _get(url: str) -> str:
        assert url == SEC_COMPANY_TICKERS_URL
        return body

    return _get


def _raising_sec(url: str) -> str:
    raise RuntimeError("simulated SEC fetch failure")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _apply_migration(eng, path: str) -> None:
    """Apply a migration .sql to a sqlite engine, shimming Postgres-isms."""
    from migrations.run import _strip_sql_comments

    sql = _strip_sql_comments(Path(path).read_text())
    sql = (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("BIGSERIAL", "INTEGER")
        # sqlite ADD COLUMN has no IF NOT EXISTS (fresh table per test anyway).
        .replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN")
    )
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))


@pytest.fixture
def engine_no_sec(tmp_path: Path):
    """Sqlite engine with ONLY migration 010 — no SEC columns (pre-011 schema).

    Exercises the graceful-degradation path where SEC enrichment no-ops.
    """
    eng = create_engine(f"sqlite:///{tmp_path / 'symbols.db'}")
    _apply_migration(eng, "migrations/010_symbols.sql")
    return eng


@pytest.fixture
def engine(tmp_path: Path):
    """Sqlite engine with migrations 010 + 011 applied (shimmed for sqlite)."""
    eng = create_engine(f"sqlite:///{tmp_path / 'symbols.db'}")
    _apply_migration(eng, "migrations/010_symbols.sql")
    _apply_migration(eng, "migrations/011_symbols_sec.sql")
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
            "Apple Inc. - Common Stock",
            "Apple Inc. NEW NAME",
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


# --------------------------------------------------------------------------- #
# company-name enrichment for operator notifications (CL-ikz2)
# --------------------------------------------------------------------------- #


# A verbose "- Class A Common Stock" tail to prove the suffix trim, plus a
# bare-ticker FX-style underscore symbol that must never resolve to a name.
VG_BODY = "\n".join(
    [
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol",
        "VG|Venture Global, Inc. Class A Common Stock|N|VG|N|100|N|VG",  # verbose
        "File Creation Time: 0720202601:23||||||||",
    ]
)
VG_ONLY = {NASDAQ_LISTED_URL: "", OTHER_LISTED_URL: VG_BODY}


def test_company_name_prefers_clean_sec_name(engine):
    """SEC name ("Venture Global, Inc.") wins over the verbose NASDAQ name."""
    sec = json.dumps(
        {
            "0": {"cik_str": 42, "ticker": "VG", "title": "Venture Global, Inc."},
        }
    )
    uni = SymbolUniverse(
        engine,
        http_get=_fake_http(VG_ONLY),
        sec_http_get=_fake_sec(sec),
    )
    uni.refresh()
    uni.refresh_sec_names()
    assert uni.company_name("VG") == "Venture Global, Inc."
    assert uni.company_name("vg") == "Venture Global, Inc."  # case-insensitive


def test_company_name_trims_verbose_suffix_without_sec(engine):
    """No SEC name → the verbose class-share suffix is trimmed off."""
    uni = SymbolUniverse(engine, http_get=_fake_http(VG_ONLY))
    uni.refresh()  # no refresh_sec_names → sec_name stays NULL
    assert uni.company_name("VG") == "Venture Global, Inc."


def test_company_name_trims_common_stock_suffix(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    # "Apple Inc. - Common Stock" → "Apple Inc." (", Inc." kept).
    assert uni.company_name("AAPL") == "Apple Inc."
    # "DHT Holdings, Inc. Common Stock" → "DHT Holdings, Inc." (no separator).
    assert uni.company_name("DHT") == "DHT Holdings, Inc."


def test_company_name_names_etf(engine):
    """ETFs still get a (fund) name — helpful colour, not skipped."""
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    assert uni.company_name("GLD") == "SPDR Gold Trust"


def test_company_name_none_for_fx_underscore_and_unknown(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    assert uni.company_name("BCO_USD") is None  # FX/CFD — underscore skip
    assert uni.company_name("USD_JPY") is None
    assert uni.company_name("ZZZZFAKE") is None  # unknown ticker
    assert uni.company_name("") is None
    assert uni.company_name("TSTQ") is None  # test issue never ingested


def test_company_names_batch_only_resolvable(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    got = uni.company_names(["AAPL", "GLD", "BCO_USD", "ZZZZFAKE", "DHT"])
    assert got == {
        "AAPL": "Apple Inc.",
        "GLD": "SPDR Gold Trust",
        "DHT": "DHT Holdings, Inc.",
    }  # underscore-FX + unknown absent → caller renders those bare


def test_company_names_preserves_input_key_case(engine):
    uni = SymbolUniverse(engine, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    got = uni.company_names(["aapl"])
    assert got == {"aapl": "Apple Inc."}  # keyed by the string passed in


# --------------------------------------------------------------------------- #
# SEC EDGAR name enrichment (CL-9xha)
# --------------------------------------------------------------------------- #


def test_parse_sec_company_tickers():
    rows = {r["symbol"]: r for r in parse_sec_company_tickers(SEC_BODY)}
    # Blank-ticker row dropped; the other four kept.
    assert set(rows) == {"AAPL", "TEVA", "WEIRD", "ZUEXTRA"}
    assert rows["AAPL"]["sec_name"] == "Apple Inc."
    assert rows["AAPL"]["cik"] == 320193


def test_parse_sec_company_tickers_bad_json():
    assert parse_sec_company_tickers("not json at all") == []
    assert parse_sec_company_tickers("") == []


def test_parse_sec_company_tickers_tolerates_missing_fields():
    body = json.dumps(
        {
            "0": {"ticker": "NOCIK", "title": "No Cik Co"},  # cik_str absent
            "1": {"cik_str": 7, "ticker": "NONAME"},  # title absent
        }
    )
    rows = {r["symbol"]: r for r in parse_sec_company_tickers(body)}
    assert rows["NOCIK"]["cik"] is None
    assert rows["NONAME"]["sec_name"] is None


def test_refresh_sec_names_enriches_existing(engine):
    uni = SymbolUniverse(
        engine,
        http_get=_fake_http(BOTH_OK),
        sec_http_get=_fake_sec(SEC_BODY),
    )
    uni.refresh()
    counts = uni.refresh_sec_names()
    # AAPL, TEVA, WEIRD are in the universe; ZUEXTRA is not.
    assert counts == {"matched": 3, "unmatched": 1, "skipped": 0}
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT sec_name, cik FROM symbols WHERE symbol = 'AAPL'",
            )
        ).one()
    assert row[0] == "Apple Inc."
    assert row[1] == 320193


def test_get_cik_after_enrichment(engine):
    uni = SymbolUniverse(
        engine,
        http_get=_fake_http(BOTH_OK),
        sec_http_get=_fake_sec(SEC_BODY),
    )
    uni.refresh()
    assert uni.get_cik("AAPL") is None  # not yet enriched
    uni.refresh_sec_names()
    assert uni.get_cik("AAPL") == 320193
    assert uni.get_cik("aapl") == 320193  # case-insensitive
    assert uni.get_cik("ZZZZFAKE") is None
    assert uni.get_cik("") is None


def test_get_cik_none_without_columns(engine_no_sec):
    uni = SymbolUniverse(engine_no_sec, http_get=_fake_http(BOTH_OK))
    uni.refresh()
    assert uni.get_cik("AAPL") is None  # no cik column → graceful None


def test_refresh_sec_names_does_not_insert_unknown(engine):
    uni = SymbolUniverse(
        engine,
        http_get=_fake_http(BOTH_OK),
        sec_http_get=_fake_sec(SEC_BODY),
    )
    uni.refresh()
    uni.refresh_sec_names()
    # SEC-only ticker must never be added to the US-listed universe.
    assert uni.exists("ZUEXTRA") is False
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM symbols")).scalar()
    assert total == 8  # unchanged from the NASDAQ ingest


def test_refresh_sec_names_preserves_nasdaq_display_name(engine):
    uni = SymbolUniverse(
        engine,
        http_get=_fake_http(BOTH_OK),
        sec_http_get=_fake_sec(SEC_BODY),
    )
    uni.refresh()
    uni.refresh_sec_names()
    # get() keeps returning the NASDAQ security name + its stable 4-key shape.
    assert uni.get("AAPL") == {
        "symbol": "AAPL",
        "security_name": "Apple Inc. - Common Stock",
        "exchange": "NASDAQ",
        "is_etf": False,
    }


def test_resolve_name_matches_via_sec_name(engine):
    uni = SymbolUniverse(
        engine,
        http_get=_fake_http(BOTH_OK),
        sec_http_get=_fake_sec(SEC_BODY),
    )
    uni.refresh()
    # Before enrichment: "Wonderful" matches nothing (NASDAQ name is
    # "Weird Exchange Co").
    assert uni.resolve_name("Wonderful") == []
    uni.refresh_sec_names()
    # After: it resolves to WEIRD purely via the SEC official name.
    hits = uni.resolve_name("Wonderful")
    assert [h["symbol"] for h in hits] == ["WEIRD"]


def test_refresh_sec_names_tolerates_fetch_failure(engine):
    uni = SymbolUniverse(
        engine,
        http_get=_fake_http(BOTH_OK),
        sec_http_get=_raising_sec,
    )
    uni.refresh()
    counts = uni.refresh_sec_names()
    assert counts == {"matched": 0, "unmatched": 0, "skipped": 0}
    # Existing NASDAQ data untouched and still usable.
    assert uni.exists("AAPL") is True
    assert uni.get("AAPL")["security_name"] == "Apple Inc. - Common Stock"


def test_refresh_sec_names_tolerates_bad_json(engine):
    uni = SymbolUniverse(
        engine,
        http_get=_fake_http(BOTH_OK),
        sec_http_get=_fake_sec("garbage"),
    )
    uni.refresh()
    assert uni.refresh_sec_names() == {
        "matched": 0,
        "unmatched": 0,
        "skipped": 0,
    }


def test_refresh_sec_names_noop_without_columns(engine_no_sec):
    """On a pre-011 schema, enrichment no-ops and core lookups still work."""
    uni = SymbolUniverse(
        engine_no_sec,
        http_get=_fake_http(BOTH_OK),
        sec_http_get=_fake_sec(SEC_BODY),
    )
    uni.refresh()
    counts = uni.refresh_sec_names()
    assert counts == {"matched": 0, "unmatched": 0, "skipped": 0}
    # Core NASDAQ-backed lookups are unaffected by the missing SEC columns.
    assert uni.exists("AAPL") is True
    assert any(h["symbol"] == "TEVA" for h in uni.resolve_name("Teva"))
