"""Tests for the OANDA daily-candle vol backfill (CL-lb03).

Covers: candle parse (complete-only, nanosecond time, malformed), batched
fetch via injected transport, refresh upsert into prices + fail-soft per
instrument, the unmapped-instrument filter, and end-to-end that
get_realized_vol then returns a value for a previously-vol-less instrument.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.data.oanda_candles import (
    fetch_daily_candles,
    parse_daily_candles,
    refresh_daily_candles,
    unmapped_tradables,
)
from src.data.provider import DataProvider


def _candles_payload(closes: list[float]) -> dict[str, Any]:
    candles = []
    for i, close in enumerate(closes):
        candles.append(
            {
                "time": f"2026-06-{i + 1:02d}T21:00:00.000000000Z",
                "mid": {"o": str(close), "h": str(close + 1), "l": str(close - 1), "c": str(close)},
                "volume": 1000 + i,
                "complete": True,
            }
        )
    # A still-forming (incomplete) current bar that must be excluded.
    candles.append(
        {
            "time": "2026-06-30T21:00:00.000000000Z",
            "mid": {"o": "9", "h": "9", "l": "9", "c": "9"},
            "volume": 1,
            "complete": False,
        }
    )
    return {"candles": candles}


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #


def test_parse_keeps_complete_only_and_tz():
    rows = parse_daily_candles(_candles_payload([100.0, 101.0, 102.0]))
    assert len(rows) == 3  # incomplete bar dropped
    assert rows[0]["close"] == 100.0
    assert rows[0]["high"] == 101.0
    assert rows[0]["volume"] == 1000
    # Nanosecond time parsed to an aware datetime.
    assert rows[0]["ts"].tzinfo is not None
    assert rows[0]["ts"].year == 2026


def test_parse_empty_and_malformed():
    assert parse_daily_candles({}) == []
    assert parse_daily_candles({"candles": [None, 5]}) == []
    assert (
        parse_daily_candles(
            {
                "candles": [
                    {"complete": True, "time": "2026-06-01T21:00:00Z", "mid": {"o": "x"}},
                ]
            }
        )
        == []
    )  # bad price / missing keys


# --------------------------------------------------------------------------- #
# fetch (injected transport)
# --------------------------------------------------------------------------- #


def test_fetch_builds_request():
    seen: dict[str, Any] = {}

    def fake_get(url: str, headers: dict, params: dict) -> dict:
        seen.update(url=url, headers=headers, params=params)
        return _candles_payload([100.0, 101.0])

    out = fetch_daily_candles("NATGAS_USD", "KEY", "ACC", count=60, http_get=fake_get)
    assert len(out) == 2
    assert seen["url"].endswith("/accounts/ACC/instruments/NATGAS_USD/candles")
    assert seen["params"] == {"granularity": "D", "count": "60", "price": "M"}
    assert seen["headers"]["Authorization"] == "Bearer KEY"


# --------------------------------------------------------------------------- #
# refresh into prices
# --------------------------------------------------------------------------- #


@pytest.fixture
def prices_engine(tmp_path):  # type: ignore[no-untyped-def]
    engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE prices (
                ts TIMESTAMP NOT NULL, symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT,
                PRIMARY KEY (ts, symbol, source)
            )
        """)
        )
    return engine


def test_refresh_inserts_bars(prices_engine):
    closes = [100.0 + i for i in range(25)]
    counts = refresh_daily_candles(
        prices_engine,
        ["NATGAS_USD"],
        "K",
        "A",
        http_get=lambda u, h, p: _candles_payload(closes),
    )
    assert counts == {"instruments": 1, "bars": 25}
    with prices_engine.connect() as conn:
        n = conn.execute(
            text(
                "SELECT COUNT(*) FROM prices WHERE symbol='NATGAS_USD' AND source='oanda_daily'",
            )
        ).scalar()
    assert n == 25


def test_refresh_fail_soft_per_instrument(prices_engine):
    def flaky(url, headers, params):
        if "BROKEN" in url:
            raise RuntimeError("oanda 400")
        return _candles_payload([100.0, 101.0, 102.0])

    counts = refresh_daily_candles(
        prices_engine,
        ["BROKEN", "NATGAS_USD"],
        "K",
        "A",
        http_get=flaky,
    )
    # BROKEN skipped, NATGAS_USD still ingested.
    assert counts["instruments"] == 1
    assert counts["bars"] == 3


def test_refresh_enables_realized_vol(prices_engine):
    # A wiggling series so realized vol is > 0.
    closes = [100.0 + (1.0 if i % 2 else -1.0) for i in range(30)]
    refresh_daily_candles(
        prices_engine,
        ["NATGAS_USD"],
        "K",
        "A",
        http_get=lambda u, h, p: _candles_payload(closes),
    )
    prov = DataProvider(prices_engine)
    vol = prov.get_realized_vol(
        "NATGAS_USD",
        window=20,
        as_of=datetime(2026, 7, 1, tzinfo=UTC),
    )
    assert vol is not None and vol > 0  # previously would have been None


# --------------------------------------------------------------------------- #
# unmapped filter
# --------------------------------------------------------------------------- #


def test_unmapped_tradables_filters_mapped():
    got = unmapped_tradables(
        ["XAU_USD", "BCO_USD", "EUR_USD", "NATGAS_USD", "NAS100_USD", "XPT_USD"],
    )
    # Mapped (have a daily alias) excluded; unmapped kept.
    assert "XAU_USD" not in got and "BCO_USD" not in got and "EUR_USD" not in got
    assert set(got) == {"NATGAS_USD", "NAS100_USD", "XPT_USD"}


# ---------------------------------------------------------------------------
# Intraday M5 backfill (CL-b425)
# ---------------------------------------------------------------------------


def _mba_candle(
    time_iso: str,
    mid: float,
    bid: float | None = None,
    ask: float | None = None,
    complete: bool = True,
) -> dict:
    c: dict = {"time": time_iso, "complete": complete, "mid": {"c": str(mid)}}
    if bid is not None:
        c["bid"] = {"c": str(bid)}
    if ask is not None:
        c["ask"] = {"c": str(ask)}
    return c


class TestParseMbaCandles:
    def test_end_stamping_no_lookahead(self):
        from src.data.oanda_candles import parse_mba_candles

        payload = {"candles": [_mba_candle("2026-07-21T12:00:00.000000000Z", 80.0, 79.9, 80.1)]}
        rows = parse_mba_candles(payload, 5)
        assert len(rows) == 1
        # A candle STARTING 12:00 closes at 12:05 — its close is only known
        # then, so the row must be stamped 12:05 (the no-lookahead guarantee).
        assert rows[0]["ts"] == datetime(2026, 7, 21, 12, 5, tzinfo=UTC)
        assert rows[0]["mid"] == pytest.approx(80.0)
        assert rows[0]["bid"] == pytest.approx(79.9)
        assert rows[0]["ask"] == pytest.approx(80.1)

    def test_incomplete_and_malformed_skipped(self):
        from src.data.oanda_candles import parse_mba_candles

        payload = {
            "candles": [
                _mba_candle("2026-07-21T12:00:00Z", 80.0, complete=False),
                {"time": "2026-07-21T12:05:00Z", "complete": True, "mid": {"c": "junk"}},
                {"complete": True, "mid": {"c": "1.0"}},  # no time
                _mba_candle("2026-07-21T12:10:00Z", 81.0),  # good, mid-only
            ]
        }
        rows = parse_mba_candles(payload, 5)
        assert len(rows) == 1
        assert rows[0]["mid"] == pytest.approx(81.0)
        assert rows[0]["bid"] is None and rows[0]["ask"] is None


class TestFetchIntradayHistory:
    def test_paginates_and_clips_to_window(self):
        from src.data.oanda_candles import fetch_intraday_history

        start = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)
        end = datetime(2026, 7, 21, 12, 20, tzinfo=UTC)
        calls: list[dict] = []

        def _fake(url, headers, params):  # noqa: ANN001, ANN202
            calls.append(dict(params))
            if len(calls) == 1:
                return {
                    "candles": [
                        _mba_candle("2026-07-21T12:00:00Z", 80.0),
                        _mba_candle("2026-07-21T12:05:00Z", 80.1),
                    ]
                }
            return {
                "candles": [
                    _mba_candle("2026-07-21T12:10:00Z", 80.2),  # ends 12:15 (in)
                    _mba_candle("2026-07-21T12:20:00Z", 80.3),  # ends 12:25 (OUT)
                ]
            }

        rows = fetch_intraday_history("BCO_USD", "k", "a", start=start, end=end, http_get=_fake)
        # 12:05, 12:10 from page 1; 12:15 from page 2; 12:25 clipped (> end).
        assert [r["ts"].minute for r in rows] == [5, 10, 15]
        assert calls[0]["price"] == "MBA"
        assert calls[0]["granularity"] == "M5"

    def test_unsupported_granularity_raises(self):
        from src.data.oanda_candles import fetch_intraday_history

        with pytest.raises(ValueError, match="granularity"):
            fetch_intraday_history(
                "BCO_USD",
                "k",
                "a",
                start=datetime(2026, 7, 21, tzinfo=UTC),
                end=datetime(2026, 7, 22, tzinfo=UTC),
                granularity="S5",
            )


class TestBackfillIntradayQuotes:
    @pytest.fixture
    def quotes_engine(self, tmp_path):  # noqa: ANN001, ANN202
        from migrations.run import _strip_sql_comments

        eng = create_engine(f"sqlite:///{tmp_path / 'q.db'}")
        sql = _strip_sql_comments(Path("migrations/012_intraday_quotes.sql").read_text())
        sql = sql.replace("TIMESTAMPTZ", "TEXT").replace("NUMERIC", "FLOAT")
        with eng.begin() as conn:
            for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                low = stmt.lower()
                if "create_hypertable" in low or "create index" in low:
                    continue  # timescale/index DDL — not needed on sqlite
                conn.execute(text(stmt))
        return eng

    def test_upserts_idempotently_under_backfill_source(self, quotes_engine):
        from src.data.oanda_candles import backfill_intraday_quotes

        def _fake(url, headers, params):  # noqa: ANN001, ANN202
            return {"candles": [_mba_candle("2026-07-21T12:00:00Z", 80.0, 79.9, 80.1)]}

        for _ in range(2):  # run twice — idempotent
            counts = backfill_intraday_quotes(
                quotes_engine,
                ["BCO_USD"],
                "k",
                "a",
                start=datetime(2026, 7, 21, 11, 0, tzinfo=UTC),
                end=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
                http_get=_fake,
            )
            assert counts == {"instruments": 1, "rows": 1}
        with quotes_engine.connect() as conn:
            rows = conn.execute(text("SELECT symbol, source, mid FROM intraday_quotes")).fetchall()
        assert len(rows) == 1  # upsert, not duplicate
        assert rows[0][1] == "oanda_m5_backfill"

    def test_one_bad_instrument_fails_soft(self, quotes_engine):
        from src.data.oanda_candles import backfill_intraday_quotes

        def _fake(url, headers, params):  # noqa: ANN001, ANN202
            if "BAD_USD" in url:
                raise RuntimeError("rejected by OANDA")
            return {"candles": [_mba_candle("2026-07-21T12:00:00Z", 80.0)]}

        counts = backfill_intraday_quotes(
            quotes_engine,
            ["BAD_USD", "BCO_USD"],
            "k",
            "a",
            start=datetime(2026, 7, 21, 11, 0, tzinfo=UTC),
            end=datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            http_get=_fake,
        )
        assert counts == {"instruments": 1, "rows": 1}
