"""Tests for the intraday OANDA quote feed (CL-dz71).

Covers: OANDA pricing parse (tradeable filter, malformed rows), batched
fetch via an injected transport, IntradayPricer insert/prune/fail-soft,
DataProvider.get_intraday_value (nearest-at-or-before + staleness bound +
missing table), and the confluence behavior change — an intraday move now
CONFIRMS an event that a flat daily close would have EXPIRED.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.data.intraday_pricer import (
    IntradayPricer,
    fetch_oanda_pricing,
    parse_oanda_pricing,
)
from src.data.provider import DataProvider
from src.events.confluence import ConfluenceConfig, EventConfluence


def _pricing_payload() -> dict[str, Any]:
    return {
        "prices": [
            {
                "instrument": "EUR_USD",
                "tradeable": True,
                "bids": [{"price": "1.0800"}],
                "asks": [{"price": "1.0802"}],
            },
            {
                "instrument": "XAU_USD",
                "tradeable": True,
                "bids": [{"price": "2400.0"}],
                "asks": [{"price": "2400.4"}],
            },
            {
                "instrument": "USD_JPY",
                "tradeable": False,  # halted → skipped
                "bids": [{"price": "150.0"}],
                "asks": [{"price": "150.02"}],
            },
            {"instrument": "BAD", "tradeable": True, "bids": [], "asks": []},  # no book
        ]
    }


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #


def test_parse_happy_and_filters():
    rows = {r["symbol"]: r for r in parse_oanda_pricing(_pricing_payload())}
    assert set(rows) == {"EUR_USD", "XAU_USD"}  # non-tradeable + no-book dropped
    assert rows["EUR_USD"]["mid"] == pytest.approx(1.0801)
    assert rows["XAU_USD"]["bid"] == 2400.0
    assert rows["XAU_USD"]["ask"] == 2400.4


def test_parse_empty_and_garbage():
    assert parse_oanda_pricing({}) == []
    assert parse_oanda_pricing({"prices": [None, 3, "x"]}) == []
    assert (
        parse_oanda_pricing(
            {
                "prices": [
                    {"instrument": "EUR_USD", "bids": [{"price": "x"}], "asks": [{"price": "1"}]},
                ]
            }
        )
        == []
    )  # unparseable price


# --------------------------------------------------------------------------- #
# fetch (injected transport)
# --------------------------------------------------------------------------- #


def test_fetch_builds_request_and_parses():
    seen: dict[str, Any] = {}

    def fake_get(url: str, headers: dict, params: dict) -> dict:
        seen["url"] = url
        seen["headers"] = headers
        seen["params"] = params
        return _pricing_payload()

    out = fetch_oanda_pricing(
        ["EUR_USD", "XAU_USD"],
        "KEY",
        "ACC",
        practice=True,
        http_get=fake_get,
    )
    assert {r["symbol"] for r in out} == {"EUR_USD", "XAU_USD"}
    assert seen["url"].endswith("/v3/accounts/ACC/pricing")
    assert seen["headers"]["Authorization"] == "Bearer KEY"
    assert seen["params"]["instruments"] == "EUR_USD,XAU_USD"
    assert "fxpractice" in seen["url"]


def test_fetch_empty_instruments_no_call():
    called = {"n": 0}

    def fake_get(url: str, headers: dict, params: dict) -> dict:
        called["n"] += 1
        return {}

    assert fetch_oanda_pricing([], "K", "A", http_get=fake_get) == []
    assert called["n"] == 0


# --------------------------------------------------------------------------- #
# IntradayPricer + intraday_quotes table
# --------------------------------------------------------------------------- #


@pytest.fixture
def intraday_engine(tmp_path):  # type: ignore[no-untyped-def]
    """sqlite with the intraday_quotes table (no Timescale hypertable)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'iq.db'}")
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE intraday_quotes (
                ts TIMESTAMP NOT NULL,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'oanda',
                bid FLOAT, ask FLOAT, mid FLOAT NOT NULL,
                PRIMARY KEY (ts, symbol, source)
            )
        """)
        )
    return engine


def _fixed_clock(ts: datetime):
    return lambda: ts


def test_poll_once_writes_quotes(intraday_engine):
    ts = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    pricer = IntradayPricer(
        intraday_engine,
        ["EUR_USD", "XAU_USD"],
        "K",
        "A",
        http_get=lambda u, h, p: _pricing_payload(),
        clock=_fixed_clock(ts),
    )
    counts = pricer.poll_once()
    assert counts["written"] == 2
    with intraday_engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM intraday_quotes")).scalar()
    assert n == 2


def test_poll_once_prunes_old(intraday_engine):
    now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    old = now - timedelta(hours=48)
    with intraday_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO intraday_quotes (ts, symbol, source, mid) "
                "VALUES (:ts, 'EUR_USD', 'oanda', 1.0)",
            ),
            {"ts": old},
        )
    pricer = IntradayPricer(
        intraday_engine,
        ["EUR_USD"],
        "K",
        "A",
        retention_hours=24,
        http_get=lambda u, h, p: _pricing_payload(),
        clock=_fixed_clock(now),
    )
    counts = pricer.poll_once()
    assert counts["pruned"] == 1  # the 48h-old row is gone
    with intraday_engine.connect() as conn:
        remaining = conn.execute(
            text(
                "SELECT COUNT(*) FROM intraday_quotes WHERE ts = :ts",
            ),
            {"ts": old},
        ).scalar()
    assert remaining == 0


def test_poll_once_fail_soft(intraday_engine):
    def boom(url, headers, params):
        raise RuntimeError("oanda down")

    pricer = IntradayPricer(
        intraday_engine,
        ["EUR_USD"],
        "K",
        "A",
        http_get=boom,
        clock=_fixed_clock(datetime(2026, 7, 20, 12, 0, tzinfo=UTC)),
    )
    assert pricer.poll_once() == {"written": 0, "pruned": 0, "instruments": 0}


# --------------------------------------------------------------------------- #
# DataProvider.get_intraday_value
# --------------------------------------------------------------------------- #


def _seed(engine, symbol, ts, mid):
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO intraday_quotes (ts, symbol, source, mid) "
                "VALUES (:ts, :s, 'oanda', :m)",
            ),
            {"ts": ts, "s": symbol, "m": mid},
        )


def test_get_intraday_value_nearest_at_or_before(intraday_engine):
    base = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    _seed(intraday_engine, "XAU_USD", base - timedelta(minutes=4), 2400.0)
    _seed(intraday_engine, "XAU_USD", base - timedelta(minutes=1), 2402.0)
    prov = DataProvider(intraday_engine)
    # Nearest at/before base is the -1min row.
    assert prov.get_intraday_value("XAU_USD", base) == pytest.approx(2402.0)
    # At/before -2min is the -4min row.
    assert prov.get_intraday_value(
        "XAU_USD",
        base - timedelta(minutes=2),
    ) == pytest.approx(2400.0)


def test_get_intraday_value_staleness_bound(intraday_engine):
    base = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    _seed(intraday_engine, "XAU_USD", base - timedelta(minutes=20), 2400.0)
    prov = DataProvider(intraday_engine)
    # 20-min-old quote is beyond a 15-min bound → None.
    assert prov.get_intraday_value("XAU_USD", base, max_staleness_minutes=15) is None
    # 30-min bound admits it.
    assert prov.get_intraday_value(
        "XAU_USD",
        base,
        max_staleness_minutes=30,
    ) == pytest.approx(2400.0)


def test_get_intraday_value_not_normalized(intraday_engine):
    # Stored under the OANDA id; a query for the daily-table name misses.
    base = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    _seed(intraday_engine, "XAU_USD", base, 2400.0)
    prov = DataProvider(intraday_engine)
    assert prov.get_intraday_value("XAU_USD", base) == pytest.approx(2400.0)
    assert prov.get_intraday_value("GOLD", base) is None  # not normalized


def test_get_intraday_value_missing_table_returns_none(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    prov = DataProvider(engine)
    # No intraday_quotes table at all → caught → None (daily fallback).
    assert (
        prov.get_intraday_value(
            "XAU_USD",
            datetime(2026, 7, 20, tzinfo=UTC),
        )
        is None
    )


# --------------------------------------------------------------------------- #
# confluence: intraday move confirms what a flat daily close would expire
# --------------------------------------------------------------------------- #


class _FakeProvider:
    """Intraday shows a +2% move; the daily close is flat."""

    def __init__(self, with_intraday: bool) -> None:
        self.with_intraday = with_intraday

    def get_intraday_value(self, symbol, as_of, max_staleness_minutes=None):
        if not self.with_intraday:
            return None
        # Older lookups (near seen_at) = 100; recent (near now) = 102.
        ref = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        return 100.0 if as_of <= ref + timedelta(minutes=30) else 102.0

    def get_latest_value(self, symbol, as_of):
        return 100.0  # flat daily close both legs

    def get_realized_vol(self, symbol, window, as_of):
        return 0.16  # annualized → daily ≈ 1%, threshold ≈ 0.25%


def _long_oanda_event(seen_at: datetime) -> dict[str, Any]:
    return {
        "id": 1,
        "seen_at": seen_at.isoformat(),
        "theme": "energy_chokepoint",
        "assessment": {
            "urgency": 9,
            "confidence": 0.9,
            "direction": "bullish",
            "affected": [
                {
                    "instrument": "XAU_USD",
                    "kind": "oanda",
                    "direction": "long",
                    "reason": "haven bid",
                },
            ],
        },
    }


def test_intraday_move_confirms_where_daily_expires():
    seen_at = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
    now = seen_at + timedelta(minutes=60)  # inside the 30-120 window
    cfg = ConfluenceConfig(intraday_max_staleness_minutes=15)

    # WITH intraday: the +2% move clears Gate B → confirmed.
    conf = EventConfluence(cfg, data_provider=_FakeProvider(with_intraday=True))
    res = conf.evaluate_and_transition(_long_oanda_event(seen_at), now=now)
    assert res.outcome == "confirmed"
    assert res.checks[0].confirmed is True

    # WITHOUT intraday: CL-hn0t makes the Gate-B reference intraday-ONLY (a
    # stale daily close as the seen_at baseline miscounted a pre-event gap as
    # confirmation). With no intraday reference there is no baseline → the leg
    # reports no_price_data and does NOT confirm, rather than confirming on a
    # daily close of unknown freshness.
    conf2 = EventConfluence(cfg, data_provider=_FakeProvider(with_intraday=False))
    res2 = conf2.evaluate_and_transition(_long_oanda_event(seen_at), now=now)
    assert res2.outcome == "pending"
    assert res2.checks[0].confirmed is False
    assert res2.checks[0].reason == "no_price_data"
