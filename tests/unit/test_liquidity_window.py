"""Tests for liquidity_window sizing (CL-4wi5)."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.risk.liquidity_window import (
    LiquidityProfile,
    build_profile_from_spreads,
)


def _ts(dow: int, hour: int) -> datetime:
    """Construct a Monday-based datetime; dow 0=Mon … 6=Sun."""
    base = datetime(2026, 4, 6, hour, 0)  # 2026-04-06 is a Monday
    from datetime import timedelta as _td
    return base + _td(days=dow)


class TestSizeMultiplier:
    def test_normal_spread_full_size(self) -> None:
        prof = LiquidityProfile(
            median_spread_bps={("EURUSD", 0, 12): 1.0},
            pair_median_bps={"EURUSD": 1.0},
        )
        # spread = 1.2 bps, median = 1.0 → ratio 1.2 (within thin threshold)
        m = prof.size_multiplier("EURUSD", _ts(0, 12), 1.2)
        assert m == 1.0

    def test_thin_window_half_size(self) -> None:
        prof = LiquidityProfile(
            median_spread_bps={("EURUSD", 0, 12): 1.0},
            pair_median_bps={"EURUSD": 1.0},
        )
        # spread = 1.7 bps → ratio 1.7 ≥ 1.5 thin threshold
        m = prof.size_multiplier("EURUSD", _ts(0, 12), 1.7)
        assert m == 0.5

    def test_dead_window_blocks_entry(self) -> None:
        prof = LiquidityProfile(
            median_spread_bps={("EURUSD", 0, 12): 1.0},
            pair_median_bps={"EURUSD": 1.0},
        )
        # spread = 3.0 bps → ratio 3.0 ≥ 2.0 block threshold
        m = prof.size_multiplier("EURUSD", _ts(0, 12), 3.0)
        assert m == 0.0

    def test_unknown_window_falls_back_to_pair_median(self) -> None:
        prof = LiquidityProfile(pair_median_bps={"EURUSD": 1.0})
        # No bucket key → use pair median 1.0; ratio 0.8 → full size
        assert prof.size_multiplier("EURUSD", _ts(2, 22), 0.8) == 1.0

    def test_unknown_pair_does_not_crash(self) -> None:
        prof = LiquidityProfile()
        # No data at all → conservative full-size (caller decides if
        # missing-data should block).
        assert prof.size_multiplier("MYSTERY", _ts(0, 0), 999) == 0.0


class TestBuildProfile:
    def test_bucket_with_enough_samples_is_kept(self) -> None:
        # 10 samples for one bucket → median is recorded.
        spreads = [(_ts(0, 12), "EURUSD", 1.0 + i * 0.01) for i in range(10)]
        prof = build_profile_from_spreads(spreads)
        assert ("EURUSD", 0, 12) in prof.median_spread_bps
        assert prof.median_spread_bps[("EURUSD", 0, 12)] == pytest.approx(1.045, abs=0.01)

    def test_sparse_bucket_dropped(self) -> None:
        # Only 3 samples — below n=5 floor → bucket drops, only pair
        # median survives.
        spreads = [(_ts(2, 22), "EURUSD", 5.0)] * 3
        prof = build_profile_from_spreads(spreads)
        assert ("EURUSD", 2, 22) not in prof.median_spread_bps
        assert "EURUSD" in prof.pair_median_bps
