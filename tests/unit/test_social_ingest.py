"""Tests for the social/forum fetchers (Reddit, HN, 4chan, lainchan, Twitter)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.research.ingest import FeedConfig
from src.research.social_ingest import (
    FourchanFetcher,
    HackerNewsFetcher,
    LainchanFetcher,
    RedditFetcher,
    TwitterFetcher,
    _strip_html,
    register_social_fetchers,
)


def _feed(adapter: str, query_url: str, source_label: str = "test") -> FeedConfig:
    return FeedConfig(
        name=f"{adapter}_test",
        adapter=adapter,
        query_url=query_url,
        source_label=source_label,
    )


class TestStripHtml:
    def test_strips_tags(self) -> None:
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_strips_entities(self) -> None:
        assert "amp" not in _strip_html("AT&amp;T")

    def test_handles_empty(self) -> None:
        assert _strip_html("") == ""
        assert _strip_html(None) == ""  # type: ignore[arg-type]

    def test_collapses_whitespace(self) -> None:
        assert _strip_html("<br>foo<br><br>bar") == "foo bar"


class TestRegistry:
    def test_all_five_adapters_registered(self) -> None:
        register_social_fetchers()
        from src.research.ingest import _FETCHER_REGISTRY
        for adapter in ("reddit", "hackernews", "fourchan", "lainchan", "twitter"):
            assert adapter in _FETCHER_REGISTRY, (
                f"{adapter} missing from fetcher registry"
            )


class TestHackerNews:
    def _fake_http(self, hits: list[dict]) -> object:
        def _shim(url: str) -> str:
            return json.dumps({"hits": hits})
        return _shim

    def test_pulls_recent_hits(self) -> None:
        now_ts = int(datetime.now(UTC).timestamp())
        hits = [
            {
                "objectID": "1",
                "title": "Fresh story",
                "url": "https://example.com/1",
                "created_at_i": now_ts - 3600,  # 1h old
                "author": "alice",
                "story_text": "body text",
            },
        ]
        f = HackerNewsFetcher(http_get=self._fake_http(hits))
        out = f.fetch(_feed("hackernews", "front", "HN"))
        assert len(out) == 1
        assert out[0].title == "Fresh story"

    def test_drops_stale_hits(self) -> None:
        # 48h old, default max_age is 12h.
        old_ts = int((datetime.now(UTC) - timedelta(hours=48)).timestamp())
        hits = [{
            "objectID": "2", "title": "Old", "url": "u",
            "created_at_i": old_ts, "author": "a", "story_text": "",
        }]
        f = HackerNewsFetcher(http_get=self._fake_http(hits))
        assert f.fetch(_feed("hackernews", "front", "HN")) == []


class TestFourchan:
    def _fake_http(self, pages: list) -> object:
        def _shim(url: str) -> str:
            return json.dumps(pages)
        return _shim

    def test_pulls_recent_threads(self) -> None:
        now = int(datetime.now(UTC).timestamp())
        pages = [{
            "threads": [
                {
                    "no": 12345, "time": now - 1800,
                    "sub": "EUR/USD analysis",
                    "com": "What do you think?",
                    "name": "Anonymous",
                },
            ],
        }]
        f = FourchanFetcher(http_get=self._fake_http(pages))
        out = f.fetch(_feed("fourchan", "biz", "4chan /biz/ (low-signal)"))
        assert len(out) == 1
        assert "EUR/USD" in out[0].title
        assert out[0].source_label == "4chan /biz/ (low-signal)"

    def test_drops_stale_threads(self) -> None:
        old = int((datetime.now(UTC) - timedelta(hours=24)).timestamp())
        pages = [{"threads": [{
            "no": 1, "time": old, "sub": "old", "com": "stale",
        }]}]
        f = FourchanFetcher(http_get=self._fake_http(pages))
        assert f.fetch(_feed("fourchan", "biz")) == []

    def test_handles_invalid_json(self) -> None:
        def _shim(url: str) -> str:
            return "not json"
        f = FourchanFetcher(http_get=_shim)
        assert f.fetch(_feed("fourchan", "biz")) == []


class TestLainchan:
    def _fake_http(self, pages: list) -> object:
        def _shim(url: str) -> str:
            return json.dumps(pages)
        return _shim

    def test_pulls_recent_threads(self) -> None:
        now = int(datetime.now(UTC).timestamp())
        pages = [{
            "threads": [{
                "no": 999, "time": now,
                "sub": "Privacy", "com": "discussion",
                "name": "lain",
            }],
        }]
        f = LainchanFetcher(http_get=self._fake_http(pages))
        out = f.fetch(_feed("lainchan", "tech"))
        assert len(out) == 1
        assert out[0].url.startswith("https://lainchan.org")


class TestTwitter:
    def test_missing_token_returns_empty(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TWITTER_BEARER_TOKEN", None)
            f = TwitterFetcher()
            assert f.fetch(_feed("twitter", "fed cut")) == []

    def test_api_response_parses(self) -> None:
        body = {
            "data": [{
                "id": "1234",
                "text": "Fed cut incoming",
                "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "author_id": "user-1",
            }],
            "includes": {
                "users": [{"id": "user-1", "username": "@trader"}],
            },
        }
        resp = MagicMock()
        resp.json.return_value = body
        resp.raise_for_status.return_value = None
        with (
            patch.dict(os.environ, {"TWITTER_BEARER_TOKEN": "fake"}),
            patch("httpx.get", return_value=resp),
        ):
            f = TwitterFetcher()
            out = f.fetch(_feed("twitter", "fed cut", "Twitter"))
        assert len(out) == 1
        assert "Fed cut" in out[0].title
        assert out[0].authors == ("@@trader",)  # @ already prefixed in fixture

    def test_api_error_returns_empty(self) -> None:
        with (
            patch.dict(os.environ, {"TWITTER_BEARER_TOKEN": "fake"}),
            patch("httpx.get", side_effect=RuntimeError("blame the network")),
        ):
            f = TwitterFetcher()
            assert f.fetch(_feed("twitter", "fed")) == []


class TestReddit:
    def test_missing_creds_returns_empty(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REDDIT_CLIENT_ID", None)
            os.environ.pop("REDDIT_CLIENT_SECRET", None)
            f = RedditFetcher()
            assert f.fetch(_feed("reddit", "wallstreetbets")) == []

    def test_parse_subs_handles_r_prefix(self) -> None:
        assert RedditFetcher._parse_subs("r/wallstreetbets,r/forex") == \
            ["wallstreetbets", "forex"]
        assert RedditFetcher._parse_subs("foo, bar  baz") == ["foo", "bar", "baz"]
