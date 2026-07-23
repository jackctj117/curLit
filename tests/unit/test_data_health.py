"""Tests for the data-health / series-coverage preflight (CL-q4n1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, text

from src.monitoring.data_health import (
    SeriesRequirement,
    check_series,
    format_report,
    log_startup_health,
    starved,
    strategy_readiness,
)

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)

_REQS = [
    SeriesRequirement("FULL", "macro", "rate_diff", min_rows=3, max_staleness_days=5),
    SeriesRequirement("EMPTY_S", "macro", "carry_vol", min_rows=3, max_staleness_days=5),
    SeriesRequirement("SPARSE_S", "macro", "rate_diff", min_rows=10, max_staleness_days=5),
    SeriesRequirement("STALE_PX", "price", "carry_vol", min_rows=1, max_staleness_days=5),
]


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    eng = create_engine(f"sqlite:///{tmp_path / 'h.db'}")
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE macro_data (observation_date TEXT, "
                "series_id TEXT, value FLOAT, release_date TEXT)"
            )
        )
        conn.execute(text("CREATE TABLE prices (ts TEXT, symbol TEXT, close FLOAT)"))

        def macro(sid, days_ago, n):
            for i in range(n):
                d = (NOW - timedelta(days=days_ago + i)).date().isoformat()
                conn.execute(
                    text("INSERT INTO macro_data VALUES (:d,:s,1.0,:d)"), {"d": d, "s": sid}
                )

        macro("FULL", days_ago=1, n=5)  # fresh, enough rows -> OK
        macro("SPARSE_S", days_ago=1, n=2)  # fresh but only 2 (< 10) -> SPARSE
        # EMPTY_S: nothing inserted -> EMPTY
        for i in range(3):  # price, enough rows but 30d stale
            d = (NOW - timedelta(days=30 + i)).isoformat()
            conn.execute(text("INSERT INTO prices VALUES (:t,'STALE_PX',1.0)"), {"t": d})
    return eng


def test_statuses(engine):
    by = {h.series_id: h for h in check_series(engine, _REQS, now=NOW)}
    assert by["FULL"].status == "OK"
    assert by["EMPTY_S"].status == "EMPTY"
    assert by["EMPTY_S"].rows == 0
    assert by["SPARSE_S"].status == "SPARSE"
    assert by["STALE_PX"].status == "STALE"


def test_worst_status_first(engine):
    results = check_series(engine, _REQS, now=NOW)
    # EMPTY (severity 3) must sort before OK (0).
    assert results[0].status == "EMPTY"
    assert results[-1].status == "OK"


def test_strategy_readiness(engine):
    r = strategy_readiness(check_series(engine, _REQS, now=NOW))
    # carry_vol has EMPTY_S (empty) + STALE_PX → worst is EMPTY.
    assert r["carry_vol"] == "EMPTY"
    # rate_diff has FULL (ok) + SPARSE_S (sparse) → worst is SPARSE.
    assert r["rate_diff"] == "SPARSE"


def test_starved_lists_blockers(engine):
    blocked = {h.series_id for h in starved(check_series(engine, _REQS, now=NOW))}
    assert blocked == {"EMPTY_S", "SPARSE_S"}  # STALE is not a hard blocker


def test_format_report_flags_starvation(engine):
    report = format_report(check_series(engine, _REQS, now=NOW))
    assert "STARVED" in report
    assert "EMPTY_S" in report


def test_format_report_clean_when_healthy(engine):
    reqs = [SeriesRequirement("FULL", "macro", "rate_diff", min_rows=3, max_staleness_days=5)]
    report = format_report(check_series(engine, reqs, now=NOW))
    assert "no data-starved strategies" in report


def test_log_startup_health_never_raises_on_bad_engine():
    # A broken engine must not stop the engine booting — per-series queries are
    # caught, so it returns all-EMPTY rather than raising.
    class _Bad:
        def connect(self):
            raise RuntimeError("db down")

    result = log_startup_health(_Bad(), now=NOW)
    assert isinstance(result, list)
    assert all(h.status == "EMPTY" for h in result)


def test_missing_table_is_soft(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")  # no tables
    results = check_series(eng, _REQS, now=NOW)
    # Every series unqueryable → all EMPTY, none crash.
    assert all(h.status == "EMPTY" for h in results)
