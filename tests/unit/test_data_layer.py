"""Unit tests — data layer: ingesters, vintage data, validation."""

import pandas as pd

from src.data.fred import FRED_SERIES
from src.data.stooq import SYMBOL_MAP
from src.data.validation import validate_macro_data, validate_price_data


class TestBaseIngester:
    def test_run_calls_fetch_transform_validate_upsert_order(self) -> None:
        pass  # stub — wire to mock ingester


class TestValidation:
    def test_validate_price_data_detects_nulls(self) -> None:
        df = pd.DataFrame({"ts": pd.Timestamp.utcnow(), "symbol": ["EURUSD"], "close": [None]})
        ok, failures = validate_price_data(df)
        assert not ok
        assert any("null" in f.lower() for f in failures)

    def test_validate_macro_data_passes_clean(self) -> None:
        df = pd.DataFrame(
            {
                "observation_date": [pd.Timestamp("2024-01-01")],
                "release_date": [pd.Timestamp("2024-01-01")],
                "series_id": ["DFF"],
                "value": [5.25],
                "revision": [0],
            }
        )
        ok, _ = validate_macro_data(df)
        assert ok


class TestFREDProvider:
    def test_series_list_complete(self) -> None:
        assert "DFF" in FRED_SERIES
        assert "PCEPILFE" in FRED_SERIES
        assert len(FRED_SERIES) >= 20


class TestStooqProvider:
    def test_symbol_map_complete(self) -> None:
        assert "EURUSD" in SYMBOL_MAP
        assert "GOLD" in SYMBOL_MAP
        assert "VIX" in SYMBOL_MAP
