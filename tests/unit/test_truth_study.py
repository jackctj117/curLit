"""Tests for the Truth Social event-study module (CL-s9as).

Feed parse, ingest dedup, classifier storage + fast paths, and the pure
window math. Research-only invariants pinned: NULL (never zero) returns
for barless windows; posts measured exactly once.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from src.data.truth_reactions import measure_pending, measure_windows
from src.data.truth_social import ingest_posts, parse_feed
from src.events.truth_classifier import classify_pending

POSTED = datetime(2026, 7, 22, 12, 56, 52, tzinfo=UTC)

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:truth="https://truthsocial.com/ns">
  <channel>
    <item>
      <title><![CDATA[Tariffs on China will DOUBLE next week!]]></title>
      <description><![CDATA[<p>Tariffs on China will DOUBLE next week!</p>]]></description>
      <guid>https://trumpstruth.org/statuses/40212</guid>
      <pubDate>Wed, 22 Jul 2026 12:56:52 +0000</pubDate>
      <truth:originalUrl>https://truthsocial.com/@x/116963738416841583</truth:originalUrl>
      <truth:originalId>116963738416841583</truth:originalId>
    </item>
    <item>
      <title><![CDATA[[No Title] - Post from July 22, 2026]]></title>
      <description><![CDATA[<p></p>]]></description>
      <guid>https://trumpstruth.org/statuses/40213</guid>
      <pubDate>Wed, 22 Jul 2026 14:09:48 +0000</pubDate>
      <truth:originalId>116964025218558981</truth:originalId>
    </item>
  </channel>
</rss>"""


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    sql = _strip_sql_comments(Path("migrations/017_truth_study.sql").read_text())
    sql = (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("JSONB", "TEXT")
        .replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY")
        .replace("DOUBLE PRECISION", "FLOAT")
        .replace("BOOLEAN", "INTEGER")
    )
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE symbols (ticker TEXT PRIMARY KEY)"))
        conn.execute(text("INSERT INTO symbols (ticker) VALUES ('DJT')"))
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


# --------------------------------------------------------------------------- #
# feed parse + ingest
# --------------------------------------------------------------------------- #


def test_parse_feed_extracts_posts():
    posts = parse_feed(_FEED)
    assert len(posts) == 2
    assert posts[0].post_id == "116963738416841583"
    assert posts[0].posted_at == POSTED
    assert "DOUBLE next week" in posts[0].text
    assert posts[1].text == ""  # media-only ([No Title]) kept, empty text


def test_parse_feed_bad_xml_returns_empty():
    assert parse_feed("<not-xml") == []


def test_ingest_dedups(engine):
    assert ingest_posts(engine, http_get=lambda _u: _FEED) == 2
    assert ingest_posts(engine, http_get=lambda _u: _FEED) == 0  # idempotent


def test_ingest_warns_on_zero_overlap_gap(engine, caplog):
    """Feed is a sliding window: zero overlap with stored posts means the
    daemon was likely down past the window — posts in between are missing.
    Must WARN, never silently continue."""
    ingest_posts(engine, http_get=lambda _u: _FEED)
    disjoint = (
        _FEED.replace("116963738416841583", "999000000000000001")
        .replace("116964025218558981", "999000000000000002")
        .replace("statuses/40212", "statuses/50001")
        .replace("statuses/40213", "statuses/50002")
    )
    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="src.data.truth_social"):
        assert ingest_posts(engine, http_get=lambda _u: disjoint) == 2
    assert any("GAP" in r.message for r in caplog.records)
    # overlap present → no gap warning
    caplog.clear()
    with caplog.at_level(_logging.WARNING, logger="src.data.truth_social"):
        ingest_posts(engine, http_get=lambda _u: _FEED)
    assert not any("GAP" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #


class _FakeLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, messages, model, max_tokens, **kwargs):  # noqa: ANN001, ANN201, ANN003
        # CL-scup: classification is single-shot — toolset must be stripped.
        assert kwargs.get("no_tools") is True

        class _R:
            text = self._payload

        _R.text = self._payload
        return _R()


def test_classifier_stores_and_fast_paths_empty(engine):
    ingest_posts(engine, http_get=lambda _u: _FEED)
    llm = _FakeLLM(
        json.dumps(
            [
                {
                    "id": 0,
                    "relevant": True,
                    "topic": "tariffs",
                    "secondary": ["china"],
                    "tone": "threatening",
                    "entities": ["China"],
                    "explicit_market": False,
                    "confidence": 0.9,
                }
            ]
        )
    )
    n = classify_pending(engine, client=llm)
    assert n == 2  # 1 LLM-classified + 1 empty-text fast path
    with engine.connect() as c:
        row = dict(
            c.execute(
                text("SELECT * FROM truth_classifications WHERE post_id = '116963738416841583'")
            )
            .one()
            ._mapping
        )
    assert row["is_market_relevant"] in (True, 1)
    assert row["primary_topic"] == "tariffs"
    assert row["tone"] == "threatening"
    assert row["classifier_version"]
    empty = dict(c.execute if False else {})  # noqa: F841 — clarity below
    with engine.connect() as c:
        empty_rel = c.execute(
            text(
                "SELECT is_market_relevant FROM truth_classifications "
                "WHERE post_id = '116964025218558981'"
            )
        ).scalar()
    assert empty_rel in (False, 0)


def test_classifier_llm_failure_retries(engine):
    ingest_posts(engine, http_get=lambda _u: _FEED)

    class _Boom:
        def complete(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
            raise RuntimeError("down")

    n = classify_pending(engine, client=_Boom())
    assert n == 1  # only the empty-text fast path landed
    # text post still unclassified → retried next cycle
    with engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM truth_classifications")).scalar() == 1


# --------------------------------------------------------------------------- #
# window math (pure)
# --------------------------------------------------------------------------- #


def _bar(
    minutes_from_post: int,
    c: float,
    h: float | None = None,
    l: float | None = None,  # noqa: E741 — OHLC 'low', mirrors the data schema
    v: float = 1000.0,
) -> dict:
    t = POSTED + timedelta(minutes=minutes_from_post)
    return {
        "t": t.isoformat().replace("+00:00", "Z"),
        "o": c,
        "h": h if h is not None else c,
        "l": l if l is not None else c,
        "c": c,
        "v": v,
    }


def test_measure_windows_math():
    bars = (
        [_bar(m, 100.0) for m in range(-31, 1)]  # flat pre-post
        + [_bar(1, 101.0, h=101.5, l=99.5, v=3000.0)]  # pop on 3x volume
        + [_bar(m, 100.5) for m in range(2, 6)]
    )
    out = measure_windows(bars, POSTED)
    w1 = out[1]
    assert w1["start_price"] == 100.0
    assert w1["return_pct"] == pytest.approx(1.0)
    assert w1["max_favorable_pct"] == pytest.approx(1.5)
    assert w1["max_adverse_pct"] == pytest.approx(-0.5)
    assert w1["volume_ratio"] == pytest.approx(3.0)
    assert out[5]["return_pct"] == pytest.approx(0.5)
    # windows with no bars are NULL, never zero
    assert out[60]["return_pct"] == pytest.approx(0.5)  # last bar carries
    assert out[120]["return_pct"] == pytest.approx(0.5)


def test_measure_windows_no_bars_is_null():
    out = measure_windows([], POSTED)
    assert all(m["return_pct"] is None for m in out.values())


# --------------------------------------------------------------------------- #
# measurement pipeline
# --------------------------------------------------------------------------- #


def _seed_relevant(engine, post_id="116963738416841583", entities='["China", "DJT"]'):
    with engine.begin() as conn:
        # datetime objects (not .isoformat()) so sqlite's stored format
        # matches the bound-parameter format in maturity comparisons.
        conn.execute(
            text("""
            INSERT INTO truth_posts (post_id, posted_at, text, ingested_at)
            VALUES (:i, :t, 'Tariffs!', :t)
        """),
            {"i": post_id, "t": POSTED},
        )
        conn.execute(
            text("""
            INSERT INTO truth_classifications
                (post_id, is_market_relevant, primary_topic,
                 named_entities, explicit_market_language,
                 classifier_version, classified_at)
            VALUES (:i, 1, 'tariffs', :e, 0, 'test', :t)
        """),
            {"i": post_id, "e": entities, "t": POSTED},
        )


def test_measure_pending_writes_rows_and_verifies_tickers(engine):
    _seed_relevant(engine)
    calls: list[list[str]] = []

    def fake_bars(symbols, start, end):  # noqa: ANN001, ANN202
        calls.append(list(symbols))
        return {s: [_bar(-1, 100.0), _bar(1, 101.0)] for s in symbols}

    now = POSTED + timedelta(minutes=130)
    assert measure_pending(engine, fetch_bars=fake_bars, now=now) == 1
    # DJT is a literal ticker in the universe → included; "China" is not.
    assert "DJT" in calls[0] and "China" not in calls[0]
    with engine.connect() as c:
        n = c.execute(text("SELECT COUNT(*) FROM truth_market_reactions")).scalar()
    assert n == 7 * 6  # 6 ETFs + DJT, 6 windows each
    # measured exactly once
    assert measure_pending(engine, fetch_bars=fake_bars, now=now) == 0


def test_measure_pending_respects_maturity(engine):
    _seed_relevant(engine)
    too_soon = POSTED + timedelta(minutes=60)
    assert measure_pending(engine, fetch_bars=lambda *a: {}, now=too_soon) == 0
