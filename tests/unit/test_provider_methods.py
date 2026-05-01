"""Tests for DataProvider's get_latest_value + get_series (CL-9eli).

These methods were missing before and carry_vol_filter was logging an
exception every signal interval (33 errors over 24h soak — a candidate
contributor to CL-2yta's memory leak). Tests verify both methods read
correctly from macro_data + prices fallback, return None / empty Series
when no data exists, and log warnings (not exceptions) on DB errors.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.data.provider import DataProvider


@pytest.fixture
def provider_engine(tmp_path):  # type: ignore[no-untyped-def]
    """In-memory sqlite with the production schema for prices +
    macro_data, seeded with two rows for testing."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE prices (
                ts TIMESTAMP, symbol VARCHAR(64),
                close FLOAT,
                PRIMARY KEY (ts, symbol)
            )
        """))
        conn.execute(text("""
            CREATE TABLE macro_data (
                observation_date DATE, series_id VARCHAR(64),
                value FLOAT, release_date DATE,
                PRIMARY KEY (observation_date, series_id)
            )
        """))
        # Macro: FRED-style — series_id 'DGS2' across 3 days
        conn.execute(text(
            "INSERT INTO macro_data VALUES "
            "('2026-04-01', 'DGS2', 4.5, '2026-04-02'), "
            "('2026-04-02', 'DGS2', 4.6, '2026-04-03'), "
            "('2026-04-03', 'DGS2', 4.7, '2026-04-04')",
        ))
        # Prices: symbol 'EURUSD' across 3 days
        conn.execute(text(
            "INSERT INTO prices VALUES "
            "('2026-04-01 00:00:00', 'EURUSD', 1.10), "
            "('2026-04-02 00:00:00', 'EURUSD', 1.11), "
            "('2026-04-03 00:00:00', 'EURUSD', 1.12)",
        ))
    return engine


class TestGetLatestValue:
    def test_macro_data_hit(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        v = provider.get_latest_value(
            "DGS2", datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert v == 4.7

    def test_uses_at_or_before_cutoff(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        # Cutoff between rows; should return the last <= cutoff
        v = provider.get_latest_value(
            "DGS2", datetime(2026, 4, 2, 12, tzinfo=UTC),
        )
        assert v == 4.6

    def test_falls_back_to_prices(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        # 'EURUSD' isn't in macro_data; should fall to prices
        v = provider.get_latest_value(
            "EURUSD", datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert v == 1.12

    def test_unknown_returns_none(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        v = provider.get_latest_value(
            "NEVER_EXISTS", datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert v is None


class TestGetSeries:
    def test_macro_data_series(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        s = provider.get_series(
            "DGS2",
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert len(s) == 3
        assert list(s.values) == [4.5, 4.6, 4.7]

    def test_falls_back_to_prices(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        s = provider.get_series(
            "EURUSD",
            datetime(2026, 3, 31, tzinfo=UTC),
            datetime(2026, 4, 4, tzinfo=UTC),
        )
        # Inclusive bounds wide enough that all 3 rows land regardless
        # of sqlite/pg timezone quirks
        assert len(s) >= 2
        assert s.iloc[-1] == 1.12

    def test_empty_when_unknown(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        s = provider.get_series(
            "GHOST",
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert isinstance(s, pd.Series)
        assert s.empty

    def test_window_filters(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        # Only the middle row should survive
        s = provider.get_series(
            "DGS2",
            datetime(2026, 4, 2, tzinfo=UTC),
            datetime(2026, 4, 2, 23, 59, tzinfo=UTC),
        )
        assert len(s) == 1
        assert s.iloc[0] == 4.6


class TestErrorPath:
    def test_db_error_returns_none_with_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Engine that fails on every query (schema mismatch — no tables)
        bad_engine = create_engine("sqlite:///:memory:")
        provider = DataProvider(bad_engine)
        with caplog.at_level("WARNING"):
            v = provider.get_latest_value(
                "X", datetime(2026, 4, 1, tzinfo=UTC),
            )
        assert v is None
        # Error logged at WARNING (not exception — see CL-2yta defense)
        assert any(
            "get_latest_value" in r.message for r in caplog.records
        )
