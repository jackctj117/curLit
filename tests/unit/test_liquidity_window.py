"""Tests for liquidity_window sizing (CL-4wi5)."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.risk.liquidity_window import (
    LiquidityProfile,
    build_profile_from_spreads,
    load_profile,
    merge_profiles,
    save_profile,
    spread_bps_from_tick,
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


class TestSpreadFromTick:
    def test_two_sided_quote_yields_bps(self) -> None:
        # mid=1.0841, spread=0.0002 → 0.0002/1.0841*1e4 ≈ 1.845 bps
        bps = spread_bps_from_tick({"bid": 1.0840, "ask": 1.0842})
        assert bps == pytest.approx(1.845, abs=0.01)

    def test_missing_side_returns_none(self) -> None:
        assert spread_bps_from_tick({"bid": 1.0840}) is None
        assert spread_bps_from_tick({"ask": 1.0842}) is None

    def test_non_dict_returns_none(self) -> None:
        assert spread_bps_from_tick(None) is None
        assert spread_bps_from_tick(1.084) is None

    def test_crossed_quote_returns_none(self) -> None:
        # ask < bid is nonsense — refuse rather than emit a negative spread.
        assert spread_bps_from_tick({"bid": 1.09, "ask": 1.08}) is None

    def test_non_numeric_returns_none(self) -> None:
        assert spread_bps_from_tick({"bid": "x", "ask": "y"}) is None


class TestCanonicalMatching:
    def test_underscore_profile_answers_compact_lookup(self) -> None:
        # Profile built from OANDA-underscore ids must answer a compact
        # strategy-symbol lookup (EUR_USD bucket, EURUSD query).
        prof = build_profile_from_spreads(
            [(_ts(0, 12), "EUR_USD", 1.0 + i * 0.01) for i in range(10)],
        )
        assert ("EURUSD", 0, 12) in prof.median_spread_bps  # stored canonical
        # 3.0 bps vs ~1.045 median → ratio > 2.0 → block
        assert prof.size_multiplier("EURUSD", _ts(0, 12), 3.0) == 0.0

    def test_lookup_is_dialect_agnostic(self) -> None:
        prof = LiquidityProfile(pair_median_bps={"XAUUSD": 2.0})
        # Query in either dialect hits the same canonical bucket.
        assert prof.median_for("XAU_USD", _ts(0, 0)) == 2.0
        assert prof.median_for("XAUUSD", _ts(0, 0)) == 2.0


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path) -> None:
        prof = LiquidityProfile(
            median_spread_bps={("EURUSD", 0, 12): 1.5, ("XAUUSD", 3, 22): 4.0},
            pair_median_bps={"EURUSD": 1.2, "XAUUSD": 3.5},
            block_threshold=2.5,
            thin_threshold=1.4,
        )
        path = str(tmp_path / "liq.json")
        save_profile(prof, path)
        back = load_profile(path)
        assert back is not None
        assert back.median_spread_bps == prof.median_spread_bps
        assert back.pair_median_bps == prof.pair_median_bps
        assert back.block_threshold == 2.5
        assert back.thin_threshold == 1.4

    def test_load_missing_file_is_none(self, tmp_path) -> None:
        # Cold start: no refresh has run → None → inert gate.
        assert load_profile(str(tmp_path / "nope.json")) is None

    def test_save_is_atomic_no_tmp_left(self, tmp_path) -> None:
        prof = LiquidityProfile(pair_median_bps={"EURUSD": 1.0})
        save_profile(prof, str(tmp_path / "liq.json"))
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []


class TestMerge:
    def test_fresh_wins_stale_fills_gaps(self) -> None:
        old = LiquidityProfile(
            median_spread_bps={("EURUSD", 0, 12): 1.0, ("EURUSD", 1, 3): 9.0},
            pair_median_bps={"EURUSD": 1.0},
        )
        new = LiquidityProfile(
            median_spread_bps={("EURUSD", 0, 12): 2.0, ("XAUUSD", 4, 5): 3.0},
            pair_median_bps={"XAUUSD": 3.0},
        )
        merged = merge_profiles(old, new)
        # Re-measured bucket takes the fresh value...
        assert merged.median_spread_bps[("EURUSD", 0, 12)] == 2.0
        # ...a bucket only the old run had survives (coverage accumulates)...
        assert merged.median_spread_bps[("EURUSD", 1, 3)] == 9.0
        # ...and the new run's fresh bucket is present.
        assert merged.median_spread_bps[("XAUUSD", 4, 5)] == 3.0

    def test_merge_none_old_returns_new(self) -> None:
        new = LiquidityProfile(pair_median_bps={"EURUSD": 1.0})
        assert merge_profiles(None, new) is new
