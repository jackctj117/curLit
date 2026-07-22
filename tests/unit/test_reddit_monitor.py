"""Tests for the Reddit → geo_events bridge (CL-okww).

Canned listing JSON + sqlite geo_events + the real playbooks for theme
matching — no live Reddit, no live LLM. Covers watchlist load/validation,
listing parse (stickied skip, malformed tolerance), tier cadence, the
score/theme/cap gates, dedup on re-poll, and the reddit credibility note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from src.data.reddit_monitor import (
    RedditPost,
    WatchSubreddit,
    due_this_cycle,
    fetch_posts,
    ingest_reddit_posts,
    load_reddit_watchlist,
    parse_listing,
)
from src.events.playbooks import load_playbooks
from src.events.x_ingest import source_credibility_note

PLAYBOOKS = load_playbooks("configs/event_playbooks.yaml")


def _listing(*children: dict[str, Any]) -> dict[str, Any]:
    return {"data": {"children": [{"kind": "t3", "data": c} for c in children]}}


def _child(**over: Any) -> dict[str, Any]:
    base = {
        "id": "abc123", "title": "Strait of Hormuz tanker traffic halted",
        "selftext": "", "score": 50, "num_comments": 12,
        "created_utc": 1784800000.0, "permalink": "/r/energy/comments/abc123/x/",
    }
    base.update(over)
    return base


@pytest.fixture
def engine(tmp_path: Path) -> Any:
    eng = sa.create_engine(f"sqlite:///{tmp_path / 'r.db'}")
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE geo_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seen_at TEXT NOT NULL, source TEXT NOT NULL,
                external_id TEXT UNIQUE NOT NULL, headline TEXT NOT NULL,
                url TEXT, theme TEXT, status TEXT NOT NULL DEFAULT 'NEW',
                status_updated_at TEXT NOT NULL)
        """))
    return eng


SUB = WatchSubreddit(name="energy", tier=1, listing="new", min_score=0)


# --------------------------------------------------------------------- #
# watchlist
# --------------------------------------------------------------------- #


def test_load_real_watchlist():
    subs = load_reddit_watchlist("configs/reddit_watchlist.yaml")
    assert len(subs) >= 12
    names = {s.name for s in subs}
    assert {"geopolitics", "ShippingStocks", "energy"} <= names
    assert all(s.tier in (1, 2, 3) for s in subs)
    assert all(s.listing in ("new", "hot", "rising") for s in subs)
    # Tier 1 polls `new` (speed); worldnews is high-min_score tier 3.
    t1 = [s for s in subs if s.tier == 1]
    assert t1 and all(s.listing == "new" for s in t1)
    wn = next(s for s in subs if s.name == "worldnews")
    assert wn.tier == 3 and wn.min_score >= 50


def test_load_watchlist_rejects_garbage(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_subreddits: []\n")
    with pytest.raises(ValueError, match="subreddits"):
        load_reddit_watchlist(bad)


def test_load_watchlist_skips_malformed_entry(tmp_path: Path):
    p = tmp_path / "w.yaml"
    p.write_text(
        "subreddits:\n"
        "  - name: energy\n    tier: 9\n    listing: weird\n    min_score: -5\n"
        "  - {}\n",
    )
    subs = load_reddit_watchlist(p)
    assert len(subs) == 1
    assert subs[0].tier == 3        # clamped
    assert subs[0].listing == "new"  # invalid → default
    assert subs[0].min_score == 0    # clamped


# --------------------------------------------------------------------- #
# cadence
# --------------------------------------------------------------------- #


def test_tier_cadence():
    t1, t2, t3 = (WatchSubreddit("a", tier=t) for t in (1, 2, 3))
    assert [due_this_cycle(t1, c) for c in range(4)] == [True] * 4
    assert [due_this_cycle(t2, c) for c in range(4)] == [True, False, True, False]
    assert [due_this_cycle(t3, c) for c in range(4)] == [True, False, False, False]


# --------------------------------------------------------------------- #
# parse / fetch
# --------------------------------------------------------------------- #


def test_parse_listing_skips_stickied_and_malformed():
    payload = _listing(
        _child(id="p1"),
        _child(id="p2", stickied=True),
        _child(id="", title="no id"),
        {"weird": "shape"},
    )
    posts = parse_listing(payload, "energy")
    assert [p.id for p in posts] == ["p1"]
    assert posts[0].url == "https://www.reddit.com/r/energy/comments/abc123/x/"


def test_parse_listing_tolerates_garbage():
    assert parse_listing({}, "x") == []
    assert parse_listing({"data": {"children": "nope"}}, "x") == []


def test_fetch_posts_builds_request_and_fail_soft():
    seen: dict[str, Any] = {}

    def fake(url: str, params: dict, headers: dict) -> dict:
        seen.update(url=url, params=params, headers=headers)
        return _listing(_child())

    posts = fetch_posts(SUB, http_get=fake)
    assert len(posts) == 1
    assert seen["url"] == "https://www.reddit.com/r/energy/new.json"
    assert "User-Agent" in seen["headers"]

    def boom(url: str, params: dict, headers: dict) -> dict:
        raise RuntimeError("reddit 429")

    assert fetch_posts(SUB, http_get=boom) == []


def test_post_text_combines_title_and_selftext_head():
    p = RedditPost(id="i", subreddit="s", title="Hormuz update",
                   selftext="x" * 1000, score=1, num_comments=0,
                   created_utc=0.0, permalink="/r/s/i/")
    assert p.text.startswith("Hormuz update ")
    assert len(p.text) <= len("Hormuz update ") + 400


# --------------------------------------------------------------------- #
# ingest gates
# --------------------------------------------------------------------- #


def _post(**over: Any) -> RedditPost:
    base = {
        "id": "p1", "subreddit": "energy",
        "title": "Strait of Hormuz closed to tanker traffic",
        "selftext": "", "score": 10, "num_comments": 3,
        "created_utc": 1784800000.0, "permalink": "/r/energy/comments/p1/x/",
    }
    base.update(over)
    return RedditPost(**base)


def test_ingest_theme_matched_post(engine):
    result = ingest_reddit_posts(engine, SUB, [_post()], PLAYBOOKS)
    assert result.ingested == 1
    with engine.connect() as c:
        row = c.execute(text(
            "SELECT source, external_id, theme, status FROM geo_events")).one()
    assert row[0] == "reddit:energy"
    assert row[1] == "reddit:p1"
    assert row[2] == "energy_chokepoint"
    assert row[3] == "NEW"


def test_ingest_score_gate(engine):
    sub = WatchSubreddit(name="worldnews", tier=3, listing="hot", min_score=100)
    result = ingest_reddit_posts(engine, sub, [_post(score=40)], PLAYBOOKS)
    assert result.ingested == 0 and result.skipped_low_score == 1


def test_ingest_no_theme_gate(engine):
    result = ingest_reddit_posts(
        engine, SUB, [_post(title="My favourite soup recipes of 2026")], PLAYBOOKS)
    assert result.ingested == 0 and result.skipped_no_theme == 1


def test_ingest_cap(engine):
    posts = [_post(id=f"p{i}") for i in range(10)]
    result = ingest_reddit_posts(engine, SUB, posts, PLAYBOOKS, cap=3)
    assert result.ingested == 3


def test_ingest_dedup_on_repoll(engine):
    ingest_reddit_posts(engine, SUB, [_post()], PLAYBOOKS)
    result = ingest_reddit_posts(engine, SUB, [_post()], PLAYBOOKS)
    assert result.ingested == 0 and result.deduped == 1
    with engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM geo_events")).scalar() == 1


def test_ingest_db_failure_is_soft(engine):
    with engine.begin() as c:
        c.execute(text("DROP TABLE geo_events"))
    result = ingest_reddit_posts(engine, SUB, [_post()], PLAYBOOKS)
    assert result.ingested == 0  # logged, not raised


# --------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------- #


def test_oauth_token_cached_and_refreshed():
    from src.data.reddit_monitor import RedditOAuth

    calls: list[str] = []

    def fake_post(url, auth, data, headers):
        calls.append(url)
        assert auth == ("cid", "secret")
        assert data == {"grant_type": "client_credentials"}
        return {"access_token": f"tok{len(calls)}", "expires_in": 3600}

    clock = {"t": 1000.0}
    oauth = RedditOAuth("cid", "secret", token_post=fake_post,
                        clock=lambda: clock["t"])
    assert oauth.token() == "tok1"
    assert oauth.token() == "tok1"      # cached — no second call
    assert len(calls) == 1
    clock["t"] += 3600                   # past expiry → refresh
    assert oauth.token() == "tok2"


def test_oauth_token_failure_is_soft():
    from src.data.reddit_monitor import RedditOAuth

    def boom(url, auth, data, headers):
        raise RuntimeError("401")

    oauth = RedditOAuth("cid", "bad", token_post=boom)
    assert oauth.token() is None
    # fetch_posts with a dead oauth skips cleanly.
    assert fetch_posts(SUB, http_get=lambda *a: _listing(_child()),
                       oauth=oauth) == []


def test_fetch_posts_uses_oauth_endpoint_and_bearer():
    from src.data.reddit_monitor import RedditOAuth

    oauth = RedditOAuth(
        "cid", "secret",
        token_post=lambda *a: {"access_token": "T", "expires_in": 3600})
    seen: dict[str, Any] = {}

    def fake(url, params, headers):
        seen.update(url=url, headers=headers)
        return _listing(_child())

    posts = fetch_posts(SUB, http_get=fake, oauth=oauth)
    assert len(posts) == 1
    assert seen["url"] == "https://oauth.reddit.com/r/energy/new"
    assert seen["headers"]["Authorization"] == "bearer T"


def test_oauth_from_env(monkeypatch: pytest.MonkeyPatch):
    from src.data.reddit_monitor import oauth_from_env

    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    assert oauth_from_env() is None
    monkeypatch.setenv("REDDIT_CLIENT_ID", "cid")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "sec")
    assert oauth_from_env() is not None


def test_reddit_credibility_note():
    note = source_credibility_note("reddit:ShippingStocks")
    assert note is not None
    assert "r/ShippingStocks" in note
    assert "unverified" in note.lower()
    # X and gdelt behavior unchanged.
    assert source_credibility_note("gdelt") is None
    assert "X/@DeItaone" in source_credibility_note("x:DeItaone")
