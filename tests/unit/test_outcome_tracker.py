"""Tests for closed-loop outcome tracking (CL-6axf).

Sqlite fixture with trade_ideas + geo_events + idea_outcomes; injected price
function (no live yfinance). Covers the niche-marker parse, direction-adjusted
win/loss/flat at horizon, open-before-horizon, no_data, and MFE/MAE tracking
across re-scorings.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.events.outcome_tracker import (
    OutcomeConfig,
    parse_niche_marker,
    score_open_ideas,
)

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'o.db'}")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE geo_events (id INTEGER PRIMARY KEY, theme TEXT)"))
        conn.execute(
            text("""
            CREATE TABLE trade_ideas (
                idea_id TEXT PRIMARY KEY, geo_event_id INTEGER, ticker TEXT,
                action TEXT, direction TEXT, confidence FLOAT, time_horizon TEXT,
                time_stop_days INTEGER, price_at_signal FLOAT, created_at TEXT,
                notes TEXT, status TEXT DEFAULT 'pending')
        """)
        )
        sql = _strip_sql_comments(Path("migrations/013_idea_outcomes.sql").read_text())
        sql = (
            sql.replace("TIMESTAMPTZ", "TEXT")
            .replace("NUMERIC", "FLOAT")
            .replace("DEFAULT FALSE", "DEFAULT 0")
        )
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


def _seed(
    engine,
    idea_id,
    ticker,
    action,
    direction,
    entry,
    days_ago,
    theme="sanctions_trade",
    time_stop_days=10,
    notes="",
    geo_id=1,
):
    created = (NOW - timedelta(days=days_ago)).isoformat()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT OR IGNORE INTO geo_events (id, theme) VALUES (:i,:t)"),
            {"i": geo_id, "t": theme},
        )
        conn.execute(
            text("""
            INSERT INTO trade_ideas (idea_id, geo_event_id, ticker, action,
                direction, confidence, time_horizon, time_stop_days,
                price_at_signal, created_at, notes)
            VALUES (:id,:g,:tk,:ac,:d,0.6,'short',:tsd,:px,:ca,:n)
        """),
            {
                "id": idea_id,
                "g": geo_id,
                "tk": ticker,
                "ac": action,
                "d": direction,
                "tsd": time_stop_days,
                "px": entry,
                "ca": created,
                "n": notes,
            },
        )


def _outcome(engine, idea_id):
    with engine.connect() as conn:
        return dict(
            conn.execute(text("SELECT * FROM idea_outcomes WHERE idea_id=:i"), {"i": idea_id})
            .mappings()
            .one()
        )


# --------------------------------------------------------------------------- #
# niche marker
# --------------------------------------------------------------------------- #


def test_parse_niche_marker():
    assert parse_niche_marker("[niche 3hop asym0.60] junior") == (True, 3)
    assert parse_niche_marker("[niche] x") == (True, None)
    assert parse_niche_marker("plain note") == (False, None)
    assert parse_niche_marker(None) == (False, None)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def test_bullish_win_at_horizon(engine):
    _seed(engine, "a1", "MP", "long", "bullish", 100.0, days_ago=20)  # past horizon
    counts = score_open_ideas(engine, price_fn=lambda t: {"MP": 112.0}, now=NOW)
    assert counts["win"] == 1
    o = _outcome(engine, "a1")
    assert o["outcome"] == "win"
    assert o["return_pct"] == pytest.approx(0.12)


def test_bearish_put_win_on_downmove(engine):
    # buy_puts + price fell 10% → direction-adjusted return is +10% → win.
    _seed(engine, "p1", "QRVO", "buy_puts", "bearish", 100.0, days_ago=20)
    counts = score_open_ideas(engine, price_fn=lambda t: {"QRVO": 90.0}, now=NOW)
    assert counts["win"] == 1
    assert _outcome(engine, "p1")["return_pct"] == pytest.approx(0.10)


def test_contradictory_action_wins_over_direction(engine):
    # CL-67m9 (P1): action=buy_puts (bearish) but direction=bullish. The OLD
    # OR-logic scored this as a LONG (inverted). Action must win: on a +10%
    # spot move a put idea is a LOSS, not a win.
    _seed(engine, "c1", "MP", "buy_puts", "bullish", 100.0, days_ago=20)
    score_open_ideas(engine, price_fn=lambda t: {"MP": 110.0}, now=NOW)
    o = _outcome(engine, "c1")
    assert o["outcome"] == "loss"
    assert o["return_pct"] == pytest.approx(-0.10)  # short sign applied


def test_idea_sign_precedence():
    from src.events.outcome_tracker import _idea_is_bullish

    # action wins over direction
    assert _idea_is_bullish("buy_puts", "bullish") is False
    assert _idea_is_bullish("buy_calls", "bearish") is True
    assert _idea_is_bullish("long", "bearish") is True
    assert _idea_is_bullish("short", "bullish") is False
    # falls back to direction when action is unknown/absent
    assert _idea_is_bullish("", "bullish") is True
    assert _idea_is_bullish("", "bearish") is False
    # indeterminate → None (scored no_data, not guessed)
    assert _idea_is_bullish("", "") is None


def test_loss_at_horizon(engine):
    _seed(engine, "l1", "MP", "long", "bullish", 100.0, days_ago=20)
    score_open_ideas(engine, price_fn=lambda t: {"MP": 90.0}, now=NOW)
    assert _outcome(engine, "l1")["outcome"] == "loss"


def test_flat_at_horizon(engine):
    _seed(engine, "f1", "MP", "long", "bullish", 100.0, days_ago=20)
    score_open_ideas(engine, price_fn=lambda t: {"MP": 101.0}, now=NOW)  # +1% < 5%
    assert _outcome(engine, "f1")["outcome"] == "flat"


def test_open_before_horizon(engine):
    _seed(engine, "o1", "MP", "long", "bullish", 100.0, days_ago=2)  # horizon 10
    counts = score_open_ideas(engine, price_fn=lambda t: {"MP": 130.0}, now=NOW)
    assert counts["open"] == 1
    o = _outcome(engine, "o1")
    assert o["outcome"] == "open"  # not finalised yet
    assert o["return_pct"] == pytest.approx(0.30)  # but return is tracked


def test_no_data_when_unpriced(engine):
    _seed(engine, "n1", "OBSCURE", "long", "bullish", 100.0, days_ago=20)
    counts = score_open_ideas(engine, price_fn=lambda t: {}, now=NOW)
    assert counts["no_data"] == 1
    assert _outcome(engine, "n1")["outcome"] == "no_data"


def test_mfe_mae_tracked_across_scorings(engine):
    _seed(engine, "m1", "MP", "long", "bullish", 100.0, days_ago=2)
    score_open_ideas(engine, price_fn=lambda t: {"MP": 120.0}, now=NOW)  # +20%
    score_open_ideas(
        engine, price_fn=lambda t: {"MP": 95.0}, now=NOW + timedelta(hours=6)
    )  # round-trips to −5%
    o = _outcome(engine, "m1")
    assert o["max_favorable_pct"] == pytest.approx(0.20)  # remembered the peak
    assert o["max_adverse_pct"] == pytest.approx(-0.05)
    assert o["scored_count"] == 2


def test_finalised_idea_not_rescored(engine):
    _seed(engine, "w1", "MP", "long", "bullish", 100.0, days_ago=20)
    score_open_ideas(engine, price_fn=lambda t: {"MP": 120.0}, now=NOW)  # win
    # A later move must NOT change a finalised outcome.
    counts = score_open_ideas(engine, price_fn=lambda t: {"MP": 60.0}, now=NOW + timedelta(days=1))
    assert counts == {"open": 0, "win": 0, "loss": 0, "flat": 0, "no_data": 0}
    assert _outcome(engine, "w1")["outcome"] == "win"  # unchanged


def test_denormalises_niche_and_red_team(engine):
    _seed(
        engine,
        "d1",
        "IDR",
        "long",
        "bullish",
        100.0,
        days_ago=2,
        notes="[niche 4hop asym0.55] junior | ⚠ survived red-team; top risk: crowded",
    )
    score_open_ideas(engine, price_fn=lambda t: {"IDR": 105.0}, now=NOW)
    o = _outcome(engine, "d1")
    assert bool(o["is_niche"]) is True
    assert o["hop_count"] == 4
    assert bool(o["red_team_survived"]) is True
    assert o["theme"] == "sanctions_trade"


def test_win_threshold_configurable(engine):
    _seed(engine, "c1", "MP", "long", "bullish", 100.0, days_ago=20)
    score_open_ideas(
        engine, price_fn=lambda t: {"MP": 103.0}, now=NOW, config=OutcomeConfig(win_threshold=0.02)
    )  # +3% ≥ 2% → win
    assert _outcome(engine, "c1")["outcome"] == "win"
