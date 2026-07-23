"""Tests for the RSS + Polymarket fetchers (niche-source ingestion).

The arXiv tests in test_paper_ingest.py already cover the Atom path
+ the FeedConfig + the IngestRunner integration. These tests cover
the parallel paths for substack/blog (RSS 2.0) and Polymarket
(JSON) — both register against the same _FETCHER_REGISTRY so
build_fetcher() finds them, and both produce Paper records the
existing extractor agent + idea agent can consume uniformly.
"""

from __future__ import annotations

import json

from src.research.ingest import (
    FeedConfig,
    RSSFetcher,
    build_fetcher,
)
from src.research.polymarket import PolymarketFetcher

# ---------------------------------------------------------------------- #
# RSS 2.0 (substack / blog)
# ---------------------------------------------------------------------- #


_SAMPLE_RSS_SUBSTACK = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:dc="http://purl.org/dc/elements/1.1/"
     xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Doomberg</title>
    <link>https://newsletter.doomberg.com</link>
    <description>Energy macro analysis</description>
    <item>
      <title>Why oil-USD correlation broke in Q1</title>
      <link>https://newsletter.doomberg.com/p/why-oil-usd</link>
      <pubDate>Wed, 30 Apr 2026 14:00:00 GMT</pubDate>
      <dc:creator>Doomberg</dc:creator>
      <description>The 30-year correlation between oil prices and the US dollar collapsed in Q1 2026; we trace the cause to changing OPEC+ supply policy.</description>
    </item>
    <item>
      <title>Markets digest the Fed's pause</title>
      <link>https://newsletter.doomberg.com/p/fed-pause</link>
      <pubDate>Tue, 29 Apr 2026 10:30:00 GMT</pubDate>
      <author>Doomberg</author>
      <description>Brief commentary on the Fed's hold decision and what it implies for the dollar index.</description>
    </item>
  </channel>
</rss>
"""


_RSS_NO_DESCRIPTION = """<?xml version="1.0"?>
<rss version="2.0"
     xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <item>
      <title>Body-only post</title>
      <link>https://example.com/p/1</link>
      <pubDate>2026-04-30</pubDate>
      <content:encoded><![CDATA[<p>Body via <em>content:encoded</em> with <a href="x">html</a> tags</p>]]></content:encoded>
    </item>
  </channel>
</rss>
"""


class TestRSSFetcher:
    def test_parses_substack_atom_into_papers(self) -> None:
        fetcher = RSSFetcher(http_get=lambda _u: _SAMPLE_RSS_SUBSTACK)
        feed = FeedConfig(
            name="doomberg",
            adapter="rss",
            query_url="https://newsletter.doomberg.com/feed",
            source_label="Doomberg",
        )
        papers = fetcher.fetch(feed)
        assert len(papers) == 2
        first = papers[0]
        assert first.title == "Why oil-USD correlation broke in Q1"
        assert first.url == "https://newsletter.doomberg.com/p/why-oil-usd"
        assert first.year == 2026
        assert first.authors == ("Doomberg",)
        assert "30-year correlation" in first.abstract
        assert first.source_label == "Doomberg"

    def test_falls_back_to_content_encoded_when_description_missing(
        self,
    ) -> None:
        fetcher = RSSFetcher(http_get=lambda _u: _RSS_NO_DESCRIPTION)
        feed = FeedConfig(
            name="t",
            adapter="rss",
            query_url="x",
            source_label="T",
        )
        papers = fetcher.fetch(feed)
        assert len(papers) == 1
        # HTML stripped, whitespace collapsed
        assert "<p>" not in papers[0].abstract
        assert "Body via" in papers[0].abstract
        assert "content:encoded" in papers[0].abstract

    def test_http_failure_returns_empty(self) -> None:
        def boom(_u: str) -> str:
            raise ConnectionError("network")

        fetcher = RSSFetcher(http_get=boom)
        feed = FeedConfig(
            name="bad",
            adapter="rss",
            query_url="x",
            source_label="bad",
        )
        assert fetcher.fetch(feed) == []

    def test_malformed_xml_returns_empty(self) -> None:
        fetcher = RSSFetcher(http_get=lambda _u: "<<bad")
        feed = FeedConfig(
            name="bad",
            adapter="rss",
            query_url="x",
            source_label="bad",
        )
        assert fetcher.fetch(feed) == []


class TestRSSAdapterRegistration:
    def test_rss_adapter_in_registry(self) -> None:
        f = build_fetcher("rss", http_get=lambda _u: "")
        assert isinstance(f, RSSFetcher)


# ---------------------------------------------------------------------- #
# Polymarket
# ---------------------------------------------------------------------- #


# Minimal Gamma API shape — actual responses have many more fields,
# but we only consume question / description / outcomes /
# outcomePrices / endDate / volume / slug / id.
_SAMPLE_POLY_API = json.dumps(
    [
        {
            "id": "fed-cuts-jun",
            "slug": "fed-cuts-25bps-june-2026",
            "question": "Will the Fed cut rates by 25bps in June 2026?",
            "description": "Resolves YES if the Fed FOMC cuts the federal funds rate target by 25bps at its June 2026 meeting.",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.78", "0.22"],
            "volume": 1250000,
            "endDate": "2026-06-15",
            "active": True,
            "closed": False,
        },
        {
            "id": "election",
            "slug": "us-election",
            "question": "2028 US presidential election winner?",
            "description": "Predicts the winner of the 2028 US presidential election.",
            "outcomes": ["Republican", "Democrat", "Other"],
            "outcomePrices": ["0.48", "0.49", "0.03"],
            "volume": 50000000,
            "endDate": "2028-11-08",
        },
        {
            # NOT FX/macro relevant — should be filtered out
            "id": "sports-market",
            "slug": "lebron-mvp-2027",
            "question": "Will LeBron James win MVP in the 2026-27 season?",
            "description": "Sports market.",
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0.05", "0.95"],
            "volume": 100000,
            "endDate": "2027-06-01",
        },
    ]
)


class TestPolymarketFetcher:
    def test_filters_to_fx_macro_keywords(self) -> None:
        fetcher = PolymarketFetcher(http_get=lambda _u: _SAMPLE_POLY_API)

        # Pass a duck-typed feed object (anything with .query_url +
        # .source_label attrs)
        class _Feed:
            query_url = "https://gamma-api.polymarket.com/markets"
            source_label = "Polymarket FX"

        papers = fetcher.fetch(_Feed())
        # The sports market should be filtered out (no fx/macro keywords)
        slugs = [p.url.split("/")[-1] for p in papers]
        assert "fed-cuts-25bps-june-2026" in slugs
        assert "us-election" in slugs
        assert "lebron-mvp-2027" not in slugs

    def test_format_outcomes_with_prices(self) -> None:
        fetcher = PolymarketFetcher(http_get=lambda _u: _SAMPLE_POLY_API)

        class _Feed:
            query_url = "x"
            source_label = "Polymarket"

        papers = fetcher.fetch(_Feed())
        fed_market = next(p for p in papers if "Fed" in p.title)
        assert "Yes=78.0%" in fed_market.abstract
        assert "No=22.0%" in fed_market.abstract
        assert "$1.2M" in fed_market.abstract  # 1.25M → 1.2M (banker's rounding)

    def test_handles_outcome_prices_as_json_string(self) -> None:
        # Real Gamma API sometimes returns outcomePrices as a JSON-
        # encoded string rather than a native list. Defend against it.
        body = json.dumps(
            [
                {
                    "id": "x",
                    "slug": "fed-x",
                    "question": "Will the Fed do anything?",
                    "description": "fed monetary policy",
                    "outcomes": '["Yes", "No"]',  # string, not list
                    "outcomePrices": '["0.50", "0.50"]',
                    "volume": 100,
                    "endDate": "2026-12-31",
                }
            ]
        )
        fetcher = PolymarketFetcher(http_get=lambda _u: body)

        class _Feed:
            query_url = "x"
            source_label = "P"

        papers = fetcher.fetch(_Feed())
        assert len(papers) == 1
        assert "Yes=50.0%" in papers[0].abstract

    def test_unparseable_response_returns_empty(self) -> None:
        fetcher = PolymarketFetcher(http_get=lambda _u: "not json")

        class _Feed:
            query_url = "x"
            source_label = "P"

        assert fetcher.fetch(_Feed()) == []

    def test_http_failure_returns_empty(self) -> None:
        def boom(_u: str) -> str:
            raise ConnectionError("offline")

        fetcher = PolymarketFetcher(http_get=boom)

        class _Feed:
            query_url = "x"
            source_label = "P"

        assert fetcher.fetch(_Feed()) == []


class TestPolymarketAdapterRegistration:
    def test_polymarket_adapter_in_registry(self) -> None:
        # The polymarket module self-registers on import via its
        # register() side-effect at the bottom. src/research/__init__.py
        # imports it for that purpose.
        f = build_fetcher("polymarket", http_get=lambda _u: "[]")
        assert isinstance(f, PolymarketFetcher)


class TestVolumeFormatting:
    def test_millions(self) -> None:
        # 1.25M rounds to 1.2M under banker's rounding (.1f format)
        assert PolymarketFetcher._format_volume(1_250_000) == "1.2M"
        # Unambiguous round: 1.7M
        assert PolymarketFetcher._format_volume(1_700_000) == "1.7M"

    def test_thousands(self) -> None:
        assert PolymarketFetcher._format_volume(1500) == "1.5K"

    def test_small(self) -> None:
        assert PolymarketFetcher._format_volume(50) == "50"

    def test_unparseable(self) -> None:
        assert PolymarketFetcher._format_volume("invalid") == "0"
