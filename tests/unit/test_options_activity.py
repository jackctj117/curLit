"""Tests for the free options-activity scanner (CL-mtum)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.events.options_activity import (
    MIN_BASELINE_SNAPSHOTS,
    activity_note,
    snapshot_tickers,
    volume_unusualness,
)


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'oa.db'}")
    sql = _strip_sql_comments(Path("migrations/015_options_activity.sql").read_text())
    sql = (sql.replace("TIMESTAMPTZ", "TEXT").replace("NUMERIC", "FLOAT")
           .replace("BIGINT", "INTEGER"))
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


def _chain(cv=1000, pv=500, iv=0.35):
    def fetch(ticker):
        return {"ticker": ticker, "call_volume": cv, "put_volume": pv,
                "call_oi": 5000, "put_oi": 4000, "atm_iv": iv,
                "expiries_sampled": 4}
    return fetch


D = date(2026, 7, 21)


def test_snapshot_writes_and_pc_ratio(engine):
    counts = snapshot_tickers(engine, ["RTX"], chain_fn=_chain(1000, 440), obs_date=D)
    assert counts["written"] == 1
    with engine.connect() as c:
        row = c.execute(text("SELECT pc_volume_ratio, call_volume FROM "
                             "options_activity WHERE ticker='RTX'")).one()
    assert row[0] == pytest.approx(0.44)


def test_snapshot_idempotent_upsert(engine):
    snapshot_tickers(engine, ["RTX"], chain_fn=_chain(100, 100), obs_date=D)
    snapshot_tickers(engine, ["RTX"], chain_fn=_chain(200, 100), obs_date=D)
    with engine.connect() as c:
        n = c.execute(text("SELECT COUNT(*) FROM options_activity")).scalar()
        cv = c.execute(text("SELECT call_volume FROM options_activity")).scalar()
    assert n == 1 and cv == 200  # updated, not duplicated


def test_snapshot_skips_oanda_ids_and_dead_chains(engine):
    counts = snapshot_tickers(
        engine, ["USD_CAD", "DEADCO"],
        chain_fn=lambda t: None, obs_date=D)
    assert counts == {"written": 0, "skipped": 2}


def test_unusualness_needs_baseline(engine):
    for i in range(1, MIN_BASELINE_SNAPSHOTS):  # too few priors
        snapshot_tickers(engine, ["RTX"], chain_fn=_chain(1000, 500),
                         obs_date=date(2026, 7, i))
    snapshot_tickers(engine, ["RTX"], chain_fn=_chain(3000, 1500), obs_date=D)
    assert volume_unusualness(engine, "RTX", D) is None  # honest: no baseline


def test_unusualness_with_baseline(engine):
    for i in range(1, MIN_BASELINE_SNAPSHOTS + 2):
        snapshot_tickers(engine, ["RTX"], chain_fn=_chain(700, 300),
                         obs_date=date(2026, 7, i))
    snapshot_tickers(engine, ["RTX"], chain_fn=_chain(2100, 900), obs_date=D)
    unusual = volume_unusualness(engine, "RTX", D)
    assert unusual == pytest.approx(3.0)  # 3000 vs 1000 baseline


def test_activity_note(engine):
    for i in range(1, MIN_BASELINE_SNAPSHOTS + 2):
        snapshot_tickers(engine, ["RTX"], chain_fn=_chain(700, 300),
                         obs_date=date(2026, 7, i))
    snapshot_tickers(engine, ["RTX"], chain_fn=_chain(2000, 600, iv=0.42),
                     obs_date=D)
    note = activity_note(engine, "RTX", D)
    assert note is not None
    assert "P/C 0.30" in note and "call-skewed" in note
    assert "x baseline" in note
    assert "ATM IV 42%" in note


def test_activity_note_absent_ticker(engine):
    assert activity_note(engine, "ZZZ", D) is None
