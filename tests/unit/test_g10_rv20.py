"""Tests for the G10 RV20 computation (CL-6h6).

The math is straightforward enough that the test surface is the
boundary cases: empty input, insufficient history, and the
upsert idempotency guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scripts.compute_g10_realized_vol import (
    G10_PAIRS,
    WINDOW,
    compute_g10_rv20,
    upsert_volatility,
)
from sqlalchemy import create_engine, text


@pytest.fixture
def fx_vol_engine(tmp_path):  # type: ignore[no-untyped-def]
    db_path = tmp_path / "rv20.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE fx_volatility ("
                "  date DATE NOT NULL, index_name TEXT NOT NULL, "
                "  value REAL NOT NULL, PRIMARY KEY (date, index_name))",
            )
        )
    return engine


class TestComputeG10RV20:
    def test_empty_returns_empty(self) -> None:
        result = compute_g10_rv20(pd.DataFrame())
        assert result.empty
        assert result.name == "G10_RV20"

    def test_insufficient_history(self) -> None:
        # 5 days of one pair — not enough for a 20-day rolling std.
        idx = pd.date_range("2026-01-01", periods=5, freq="D")
        closes = pd.DataFrame({"EURUSD": [1.10, 1.11, 1.10, 1.12, 1.11]}, index=idx)
        result = compute_g10_rv20(closes)
        assert result.empty

    def test_constant_prices_zero_vol(self) -> None:
        idx = pd.date_range("2026-01-01", periods=30, freq="D")
        # Two pairs, all flat → log returns 0 → std 0 → RV20 = 0.
        closes = pd.DataFrame(
            {"EURUSD": [1.10] * 30, "GBPUSD": [1.30] * 30},
            index=idx,
        )
        result = compute_g10_rv20(closes)
        assert not result.empty
        # Last entry should be exactly 0.0 (constant series).
        assert result.iloc[-1] == pytest.approx(0.0, abs=1e-12)

    def test_known_vol_recovers_close(self) -> None:
        # Construct a single pair with iid log returns of std=0.01/day.
        # Annualized RV ≈ 0.01 * sqrt(252) ≈ 0.1587.
        rng = np.random.default_rng(seed=42)
        n = 1000  # large sample → tight std estimate
        log_ret = rng.normal(loc=0, scale=0.01, size=n)
        # Build a price series consistent with these log returns.
        prices = np.exp(np.cumsum(log_ret))
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        closes = pd.DataFrame({"EURUSD": prices}, index=idx)

        result = compute_g10_rv20(closes)
        # Expectation 0.01*sqrt(252) ≈ 0.1587. Allow slack for window noise.
        assert result.iloc[-1] == pytest.approx(0.01 * np.sqrt(252), rel=0.3)

    def test_skips_pairs_with_holes(self) -> None:
        # If one of two pairs has fewer rows, the rolling NaN pollutes
        # the cross-sectional mean (skipna=False) — that's intentional:
        # we want a coherent 20D window across the panel, not a partial.
        idx = pd.date_range("2026-01-01", periods=30, freq="D")
        a = list(np.linspace(1.10, 1.12, 30))
        b = list(np.linspace(1.30, 1.32, 30))
        closes = pd.DataFrame({"EURUSD": a, "GBPUSD": b}, index=idx)
        # Punch a hole in GBPUSD so it never has 21 contiguous obs.
        closes.iloc[-1, closes.columns.get_loc("GBPUSD")] = np.nan
        result = compute_g10_rv20(closes)
        # Trailing day's RV must be NaN-dropped because GBPUSD lacks data.
        if not result.empty:
            assert result.index.max() < idx.max()


class TestUpsertVolatility:
    def test_writes_and_idempotent(self, fx_vol_engine) -> None:  # type: ignore[no-untyped-def]
        idx = pd.date_range("2026-04-01", periods=3, freq="D")
        rv = pd.Series([0.10, 0.11, 0.12], index=idx, name="G10_RV20")

        n1 = upsert_volatility(fx_vol_engine, "G10_RV20", rv)
        n2 = upsert_volatility(fx_vol_engine, "G10_RV20", rv)
        assert n1 == n2 == 3

        with fx_vol_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM fx_volatility")).scalar()
            assert count == 3  # second run did UPDATEs, not duplicate inserts

    def test_skips_nan(self, fx_vol_engine) -> None:  # type: ignore[no-untyped-def]
        idx = pd.date_range("2026-04-01", periods=3, freq="D")
        rv = pd.Series([0.10, np.nan, 0.12], index=idx, name="G10_RV20")
        n = upsert_volatility(fx_vol_engine, "G10_RV20", rv)
        assert n == 2


def test_g10_pairs_constant_sane() -> None:
    """G10 currency set should not include USD-USD pseudo-pair."""
    assert "USDUSD" not in G10_PAIRS
    assert WINDOW > 0 and WINDOW < 252
