"""Tests for the liquidity-profile refresh (CL-y412).

The DB query is isolated behind ``spreads_from_rows`` / a fake engine so no
Postgres is needed. Covers: row→spread mapping (crossed/bad quotes dropped),
full refresh writes a loadable file, and the merge accumulates coverage
across runs (the reason it's a merge, not an overwrite: intraday_quotes only
retains ~24h).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from scripts.refresh_liquidity_profile import (
    refresh,
    spreads_from_rows,
)

from src.risk.liquidity_window import load_profile


def _ts(dow: int, hour: int) -> datetime:
    from datetime import timedelta as _td
    return datetime(2026, 4, 6, hour, 0, tzinfo=UTC) + _td(days=dow)  # Mon base


class _FakeResult:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def __enter__(self) -> _FakeConn:
        return self

    def __exit__(self, *a: object) -> bool:
        return False

    def execute(self, sql: Any, params: Any) -> _FakeResult:
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows

    def connect(self) -> _FakeConn:
        return _FakeConn(self._rows)


class TestSpreadsFromRows:
    def test_maps_bid_ask_to_bps(self) -> None:
        rows = [(_ts(0, 12), "EUR_USD", 1.0840, 1.0842)]
        out = spreads_from_rows(rows)
        assert len(out) == 1
        ts, sym, bps = out[0]
        assert sym == "EUR_USD"
        assert bps > 0

    def test_drops_crossed_and_nonpositive(self) -> None:
        rows = [
            (_ts(0, 12), "EUR_USD", 1.09, 1.08),   # crossed
            (_ts(0, 12), "EUR_USD", 0.0, 1.08),    # non-positive mid side
            (_ts(0, 12), "EUR_USD", 1.0840, 1.0840),  # zero spread
        ]
        assert spreads_from_rows(rows) == []

    def test_non_numeric_skipped(self) -> None:
        rows = [(_ts(0, 12), "EUR_USD", None, 1.08)]
        assert spreads_from_rows(rows) == []


class TestRefresh:
    def test_refresh_writes_loadable_profile(self, tmp_path) -> None:
        # 6 samples in one bucket (>= n=5 floor) → bucket persists, keyed
        # canonical (EUR_USD → EURUSD).
        rows = [
            (_ts(0, 12), "EUR_USD", 1.0840, 1.0840 + 0.0002 + i * 1e-6)
            for i in range(6)
        ]
        path = str(tmp_path / "liq.json")
        stats = refresh(_FakeEngine(rows), path, lookback_hours=24)
        assert stats["new_buckets"] == 1
        prof = load_profile(path)
        assert prof is not None
        assert ("EURUSD", 0, 12) in prof.median_spread_bps

    def test_merge_accumulates_across_runs(self, tmp_path) -> None:
        path = str(tmp_path / "liq.json")
        # Run 1: Monday-noon bucket.
        rows1 = [
            (_ts(0, 12), "EUR_USD", 1.0840, 1.0842 + i * 1e-7)
            for i in range(6)
        ]
        refresh(_FakeEngine(rows1), path, lookback_hours=24)
        # Run 2: a DIFFERENT (Tuesday-3am) bucket — must not erase Monday's.
        rows2 = [
            (_ts(1, 3), "EUR_USD", 1.0840, 1.0850 + i * 1e-7)
            for i in range(6)
        ]
        refresh(_FakeEngine(rows2), path, lookback_hours=24)
        prof = load_profile(path)
        assert prof is not None
        assert ("EURUSD", 0, 12) in prof.median_spread_bps  # run 1 survived
        assert ("EURUSD", 1, 3) in prof.median_spread_bps    # run 2 added
