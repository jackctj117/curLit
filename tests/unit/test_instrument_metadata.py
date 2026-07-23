"""Tests for src/backtest/instrument_metadata.py (CL-nt0c)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.backtest.instrument_metadata import (
    InstrumentMetadata,
    TradabilityFilter,
)


class TestIsTradable:
    def test_unknown_instrument_assumed_tradable(self) -> None:
        f = TradabilityFilter(registry={})
        assert f.is_tradable("WHATEVER", date(2024, 1, 1)) is True

    def test_pre_first_tradable_filtered(self) -> None:
        f = TradabilityFilter()
        # USDNOK first tradable 2003-01-01 in DEFAULT_REGISTRY.
        assert f.is_tradable("USDNOK", date(2002, 12, 31)) is False
        assert f.is_tradable("USDNOK", date(2003, 1, 1)) is True

    def test_synthetic_series_not_tradable_after_marker_date(self) -> None:
        f = TradabilityFilter()
        # US_2Y has last_tradable_date == 1990-01-01 in registry to
        # mark it as a feature-only series.
        assert f.is_tradable("US_2Y", date(2024, 1, 1)) is False

    def test_below_liquidity_floor_blocked(self) -> None:
        registry = {
            "TINY": InstrumentMetadata("TINY", date(2000, 1, 1), 100_000),
        }
        f = TradabilityFilter(registry=registry)
        assert f.is_tradable("TINY", date(2024, 1, 1), observed_volume=50_000) is False
        assert f.is_tradable("TINY", date(2024, 1, 1), observed_volume=200_000) is True


class TestFilterDataFrame:
    def test_drops_pre_first_tradable_rows(self) -> None:
        idx = pd.date_range("2002-01-01", "2004-01-01", freq="MS")
        df = pd.DataFrame({"close": range(len(idx))}, index=idx)
        f = TradabilityFilter()
        out = f.filter_dataframe(df, "USDNOK")
        # USDNOK first tradable 2003-01-01 → drops all 2002 rows.
        assert all(d.year >= 2003 for d in out.index)

    def test_unknown_instrument_passes_through(self) -> None:
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"close": range(10)}, index=idx)
        f = TradabilityFilter(registry={})
        out = f.filter_dataframe(df, "MYSTERY")
        assert len(out) == len(df)
