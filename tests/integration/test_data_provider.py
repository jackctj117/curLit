"""Integration test — DataProvider.get_aligned_series merges prices + macro_data (CL-5rtm).

Pre-fix: the prices-table lookup short-circuited the macro_data fallback, so
queries spanning both tables (e.g. ['EURUSD', 'US_10Y'] from yfinance plus
'DE_10Y' from FRED) silently dropped the macro_data leg. Post-fix the
provider queries both tables and union-merges the per-symbol results.

Skipped if Postgres isn't reachable (the test inserts and rolls back
fixture data into the seeded macro_data + prices tables).
"""

from __future__ import annotations

import os
import socket
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text

from src.data.provider import DataProvider
from src.runtime.run_engine import _build_db_engine


def _postgres_reachable() -> bool:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="DataProvider integration tests require Postgres running",
)


@pytest.fixture
def engine() -> None:
    return _build_db_engine()


@pytest.fixture
def fixture_data(engine):
    """Insert disposable fixture rows in a unique date range, then clean up."""
    # Use a 2010-01-XX range so we don't collide with the seeded 2015+ data.
    start = date(2010, 1, 4)
    end = date(2010, 1, 8)

    with engine.begin() as conn:
        # prices: 5 rows of EURUSD + US_10Y, daily.
        rows = []
        for i in range(5):
            d = start + timedelta(days=i)
            rows.append(
                {
                    "ts": datetime.combine(d, datetime.min.time(), tzinfo=UTC),
                    "symbol": "EURUSD",
                    "close": 1.10 + i * 0.001,
                }
            )
            rows.append(
                {
                    "ts": datetime.combine(d, datetime.min.time(), tzinfo=UTC),
                    "symbol": "US_10Y",
                    "close": 2.50 + i * 0.01,
                }
            )
        for r in rows:
            conn.execute(
                text(
                    "INSERT INTO prices (ts, symbol, source, close) "
                    "VALUES (:ts, :symbol, 'test_data_provider', :close) "
                    "ON CONFLICT DO NOTHING"
                ),
                r,
            )

        # macro_data: one DE_10Y observation in the same window.
        conn.execute(
            text("""
                INSERT INTO macro_data
                  (observation_date, release_date, series_id, value, source)
                VALUES (:obs, :rel, :sid, :val, :src)
            """),
            {
                "obs": date(2010, 1, 4),
                "rel": datetime.now(UTC),
                "sid": "IRLTLT01DEM156N",
                "val": 1.95,
                "src": "fred",
            },
        )

    yield (start, end)

    # Cleanup
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM prices WHERE source = 'test_data_provider' AND ts >= :s AND ts <= :e"
            ),
            {
                "s": datetime.combine(start, datetime.min.time(), tzinfo=UTC),
                "e": datetime.combine(end, datetime.min.time(), tzinfo=UTC),
            },
        )
        conn.execute(
            text(
                "DELETE FROM macro_data WHERE observation_date = :d AND series_id = 'IRLTLT01DEM156N'"
            ),
            {"d": date(2010, 1, 4)},
        )


def test_cross_table_query_returns_all_columns(engine, fixture_data) -> None:
    """The CL-5rtm repro: prices-only result must NOT short-circuit the
    macro_data fallback. ['EURUSD', 'US_10Y', 'DE_10Y'] must come back with
    all three columns even though DE_10Y lives only in macro_data.
    """
    start, end = fixture_data
    dp = DataProvider(engine)
    df = dp.get_aligned_series(
        ["EURUSD", "US_10Y", "DE_10Y"],
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
    )
    assert df is not None
    assert set(df.columns) == {"EURUSD", "US_10Y", "DE_10Y"}, (
        f"expected all 3 cols, got {list(df.columns)}"
    )
    assert df["EURUSD"].notna().sum() >= 1
    assert df["US_10Y"].notna().sum() >= 1
    assert df["DE_10Y"].notna().sum() >= 1


def test_prices_only_query_still_works(engine, fixture_data) -> None:
    """Regression check: the all-symbols-in-prices path is unchanged."""
    start, end = fixture_data
    dp = DataProvider(engine)
    df = dp.get_aligned_series(
        ["EURUSD", "US_10Y"],
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
    )
    assert df is not None
    assert set(df.columns) == {"EURUSD", "US_10Y"}
    assert len(df) >= 1


def test_unknown_symbols_return_none(engine, fixture_data) -> None:
    """When NO requested symbol is in either table, return None."""
    start, end = fixture_data
    dp = DataProvider(engine)
    df = dp.get_aligned_series(
        ["NONEXISTENT_FX", "ALSO_NONEXISTENT"],
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
    )
    assert df is None


def test_partial_match_returns_only_found(engine, fixture_data) -> None:
    """Mix of known + unknown — return DataFrame with just the known column."""
    start, end = fixture_data
    dp = DataProvider(engine)
    df = dp.get_aligned_series(
        ["EURUSD", "NONEXISTENT_FX"],
        datetime.combine(start, datetime.min.time()),
        datetime.combine(end, datetime.min.time()),
    )
    assert df is not None
    assert "EURUSD" in df.columns
    assert "NONEXISTENT_FX" not in df.columns
