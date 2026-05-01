"""Unit tests for the paper-stream ingester (CL-2klj).

Mocks the HTTP fetcher with a canned arXiv Atom XML string and the LLM
agent with a canned-response driver. No network calls, no API keys.

Covers:
  * paper_hash uses DOI > URL > metadata fallback and is stable
  * ArxivFetcher parses arXiv Atom XML correctly
  * ArxivFetcher swallows HTTP failures + XML parse failures
  * load_feed_configs validates structure
  * ExtractStore writes the citation header + body
  * ExtractStore.has() detects existing extracts
  * IngestRunner skips duplicates, calls extractor only on new papers
  * IngestRunner counts feed/extract failures correctly
  * One bad feed doesn't kill an ingest run
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.research.config import (
    AgentConfig,
    ProviderConfig,
    ResearchConfig,
)
from src.research.ingest import (
    ArxivFetcher,
    ExtractStore,
    FeedConfig,
    IngestRunner,
    Paper,
    PaperExtractor,
    build_fetcher,
    load_feed_configs,
    paper_hash,
)
from src.research.llm.client import (
    Driver,
    LLMResponse,
    register_driver,
)

# Mock LLM driver -----------------------------------------------------------


class _IngestCannedDriver(Driver):
    name = "ingest-canned"

    def __init__(self, api_key: str = "x", canned_text: str = "stub") -> None:
        super().__init__(api_key)
        self.canned_text = canned_text
        self.calls: list[Any] = []

    def complete(
        self,
        messages: Any,
        model: str,
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            text=self.canned_text, model=model, provider=self.name,
            input_tokens=10, output_tokens=20, usd_cost=0.0001, elapsed_sec=0.01,
        )


register_driver("ingest-canned", _IngestCannedDriver)


# Sample Atom XML ----------------------------------------------------------

_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2026.04001v1</id>
    <updated>2026-04-25T12:00:00Z</updated>
    <published>2026-04-25T12:00:00Z</published>
    <title>Carry-trade returns under regime switches</title>
    <summary>We study carry-trade Sharpe across 1990-2024 monetary regimes.
We find regime-conditional carry persists in EM but compresses in DM after 2015.</summary>
    <author><name>Alice Doe</name></author>
    <author><name>Bob Roe</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2026.04002v1</id>
    <published>2026-04-26T08:00:00Z</published>
    <title>Microstructure of FX option spreads</title>
    <summary>Bid-ask spreads in FX options widen 4x during macro releases.</summary>
    <author><name>Carol Vee</name></author>
  </entry>
</feed>
"""

_MALFORMED_XML = "<<not xml at all>>"


# ---------------------------------------------------------------------------- #
# paper_hash
# ---------------------------------------------------------------------------- #


class TestPaperHash:
    def test_doi_takes_precedence(self) -> None:
        a = Paper(
            title="t", authors=("X",), year=2026,
            url="http://example.com/a", doi="10.1000/abc",
        )
        b = Paper(
            title="completely different",
            authors=("Different",), year=1999,
            url="http://example.com/b", doi="10.1000/abc",
        )
        # Same DOI → same hash regardless of other fields
        assert paper_hash(a) == paper_hash(b)

    def test_url_when_no_doi(self) -> None:
        a = Paper(title="t", authors=(), year=None, url="http://x.com/a")
        b = Paper(title="other", authors=(), year=None, url="http://x.com/a")
        assert paper_hash(a) == paper_hash(b)

    def test_metadata_fallback_when_no_doi_or_url(self) -> None:
        a = Paper(title="T", authors=("Alice",), year=2026, url="")
        b = Paper(title="t", authors=("alice",), year=2026, url="")
        # Same title/authors/year (case-insensitive) → same hash
        assert paper_hash(a) == paper_hash(b)

    def test_different_papers_different_hashes(self) -> None:
        a = Paper(title="A", authors=("X",), year=2026, url="http://x.com/a")
        b = Paper(title="B", authors=("X",), year=2026, url="http://x.com/b")
        assert paper_hash(a) != paper_hash(b)


# ---------------------------------------------------------------------------- #
# ArxivFetcher
# ---------------------------------------------------------------------------- #


class TestArxivFetcher:
    def test_parses_atom_into_papers(self) -> None:
        fetcher = ArxivFetcher(http_get=lambda _u: _SAMPLE_ATOM)
        feed = FeedConfig(
            name="t", adapter="arxiv",
            query_url="http://example.com/atom",
            source_label="arXiv test",
        )
        papers = fetcher.fetch(feed)
        assert len(papers) == 2
        first = papers[0]
        assert first.title == "Carry-trade returns under regime switches"
        assert first.authors == ("Alice Doe", "Bob Roe")
        assert first.year == 2026
        assert first.url == "http://arxiv.org/abs/2026.04001v1"
        assert first.source_label == "arXiv test"
        assert "regime-conditional carry persists" in first.abstract

    def test_http_failure_returns_empty(self) -> None:
        def boom(_u: str) -> str:
            raise ConnectionError("network down")

        fetcher = ArxivFetcher(http_get=boom)
        feed = FeedConfig(
            name="bad", adapter="arxiv",
            query_url="http://example.com",
            source_label="bad",
        )
        assert fetcher.fetch(feed) == []

    def test_malformed_xml_returns_empty(self) -> None:
        fetcher = ArxivFetcher(http_get=lambda _u: _MALFORMED_XML)
        feed = FeedConfig(
            name="bad", adapter="arxiv",
            query_url="http://example.com",
            source_label="bad",
        )
        assert fetcher.fetch(feed) == []

    def test_empty_atom_feed_returns_empty(self) -> None:
        empty = (
            '<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        )
        fetcher = ArxivFetcher(http_get=lambda _u: empty)
        feed = FeedConfig(
            name="empty", adapter="arxiv",
            query_url="http://example.com",
            source_label="empty",
        )
        assert fetcher.fetch(feed) == []


# ---------------------------------------------------------------------------- #
# load_feed_configs
# ---------------------------------------------------------------------------- #


class TestLoadFeedConfigs:
    def test_parses_yaml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "f.yaml"
        cfg.write_text(yaml.safe_dump({
            "feeds": {
                "test1": {
                    "adapter": "arxiv",
                    "query_url": "http://example.com/atom",
                    "source_label": "T1",
                },
            },
        }))
        feeds = load_feed_configs(cfg)
        assert len(feeds) == 1
        assert feeds[0].name == "test1"
        assert feeds[0].adapter == "arxiv"
        assert feeds[0].source_label == "T1"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_feed_configs(tmp_path / "ghost.yaml")

    def test_missing_feeds_key_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "f.yaml"
        cfg.write_text("not_feeds: {}")
        with pytest.raises(ValueError, match="missing 'feeds'"):
            load_feed_configs(cfg)

    def test_real_paper_streams_yaml_loads(self) -> None:
        # Guard the actual configs/paper_streams.yaml against drift.
        feeds = load_feed_configs("configs/paper_streams.yaml")
        assert len(feeds) >= 1
        # Every feed's adapter must be a known one. The set evolves
        # as new sources land (arxiv, rss, polymarket) — assert each
        # is registered in the runtime registry rather than pinning
        # to a single value.
        from src.research.ingest import _FETCHER_REGISTRY  # noqa: PLC0415
        # Ensure self-registering modules (polymarket) ran
        import src.research  # noqa: F401, PLC0415
        for f in feeds:
            assert f.adapter in _FETCHER_REGISTRY, (
                f"feed {f.name!r} uses unknown adapter {f.adapter!r}; "
                f"registered: {sorted(_FETCHER_REGISTRY)}"
            )


# ---------------------------------------------------------------------------- #
# build_fetcher
# ---------------------------------------------------------------------------- #


class TestBuildFetcher:
    def test_returns_arxiv(self) -> None:
        f = build_fetcher("arxiv", http_get=lambda _u: "")
        assert isinstance(f, ArxivFetcher)

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown feed adapter"):
            build_fetcher("ghost-adapter")


# ---------------------------------------------------------------------------- #
# ExtractStore
# ---------------------------------------------------------------------------- #


class TestExtractStore:
    def test_writes_extract_with_citation_header(self, tmp_path: Path) -> None:
        store = ExtractStore(root=tmp_path / "extracts")
        paper = Paper(
            title="A test paper",
            authors=("Alice", "Bob"), year=2026,
            url="http://arxiv.org/abs/test/1",
            source_label="arXiv test",
        )
        body = "## Methodology\nstub.\n## Findings\nstub.\n"
        path = store.write(paper, body)
        assert path.exists()
        text = path.read_text()
        assert "# A test paper" in text
        assert "Alice, Bob" in text
        assert "**year**: 2026" in text
        assert "http://arxiv.org/abs/test/1" in text
        assert "## Methodology" in text

    def test_has_returns_true_after_write(self, tmp_path: Path) -> None:
        store = ExtractStore(root=tmp_path / "extracts")
        paper = Paper(
            title="A", authors=(), year=2026,
            url="http://x.com/a",
        )
        assert not store.has(paper)
        store.write(paper, "## Methodology\nstub.\n")
        assert store.has(paper)


# ---------------------------------------------------------------------------- #
# IngestRunner integration
# ---------------------------------------------------------------------------- #


@pytest.fixture
def extractor_prompt(tmp_path: Path) -> Path:
    f = tmp_path / "extractor_prompt.md"
    f.write_text("You are a test extractor.")
    return f


@pytest.fixture
def cfg(
    extractor_prompt: Path, monkeypatch: pytest.MonkeyPatch,
) -> ResearchConfig:
    monkeypatch.setenv("INGEST_KEY", "fake")
    return ResearchConfig(
        providers={
            "ingest-canned": ProviderConfig(
                api_key_env="INGEST_KEY", default_model="m1",
            ),
        },
        agents={
            "paper_extractor": AgentConfig(
                provider="ingest-canned",
                role="paper_extractor",
                prompt_path=str(extractor_prompt),
                model=None,
            ),
        },
        debates={},  # type: ignore[arg-type]
    )


def _make_extractor(cfg: ResearchConfig, canned: str) -> PaperExtractor:
    a = PaperExtractor.from_config(
        name="paper_extractor", research_config=cfg,
    )
    a.client.driver.canned_text = canned  # type: ignore[attr-defined]
    return a


_CANNED_EXTRACT = (
    "## Methodology\nStub for tests.\n\n"
    "## Findings\nStub.\n\n"
    "## FX trading applicability\nDirect.\n\n"
    "## Data sources cited\n- (none specified in abstract)\n\n"
    "## Key citations\n- (none in abstract)\n"
)


class TestIngestRunner:
    def test_processes_new_papers_writes_extracts(
        self, cfg: ResearchConfig, tmp_path: Path,
    ) -> None:
        extractor = _make_extractor(cfg, _CANNED_EXTRACT)
        store = ExtractStore(root=tmp_path / "extracts")
        runner = IngestRunner(
            extractor=extractor, store=store,
            http_get=lambda _u: _SAMPLE_ATOM,
        )
        feed = FeedConfig(
            name="t", adapter="arxiv",
            query_url="http://example.com/atom",
            source_label="arXiv test",
        )
        summary = runner.run([feed])
        assert summary.feeds_total == 1
        assert summary.feeds_failed == 0
        assert summary.papers_seen == 2
        assert summary.papers_skipped_duplicate == 0
        assert summary.papers_extracted == 2
        assert summary.papers_extract_failed == 0
        assert len(summary.extract_paths) == 2
        # Each extract written contains both header + body
        for path in summary.extract_paths:
            text = path.read_text()
            assert "## Methodology" in text
            assert "**source**: arXiv test" in text

    def test_dedup_skips_already_extracted(
        self, cfg: ResearchConfig, tmp_path: Path,
    ) -> None:
        extractor = _make_extractor(cfg, _CANNED_EXTRACT)
        store = ExtractStore(root=tmp_path / "extracts")
        runner = IngestRunner(
            extractor=extractor, store=store,
            http_get=lambda _u: _SAMPLE_ATOM,
        )
        feed = FeedConfig(
            name="t", adapter="arxiv",
            query_url="http://example.com/atom",
            source_label="arXiv test",
        )
        first = runner.run([feed])
        assert first.papers_extracted == 2

        # Second run: same papers, should all dedup
        second = runner.run([feed])
        assert second.papers_seen == 2
        assert second.papers_skipped_duplicate == 2
        assert second.papers_extracted == 0
        # Extractor should NOT have been called on second run for these
        # (only the 2 calls from the first run)
        assert len(extractor.client.driver.calls) == 2  # type: ignore[attr-defined]

    def test_unknown_adapter_marks_feed_failed(
        self, cfg: ResearchConfig, tmp_path: Path,
    ) -> None:
        extractor = _make_extractor(cfg, _CANNED_EXTRACT)
        runner = IngestRunner(
            extractor=extractor, store=ExtractStore(root=tmp_path / "e"),
            http_get=lambda _u: _SAMPLE_ATOM,
        )
        feed = FeedConfig(
            name="bad", adapter="ghost",
            query_url="http://example.com", source_label="bad",
        )
        summary = runner.run([feed])
        assert summary.feeds_total == 1
        assert summary.feeds_failed == 1
        assert summary.papers_extracted == 0

    def test_extractor_failure_counts_but_doesnt_kill_run(
        self, cfg: ResearchConfig, tmp_path: Path,
    ) -> None:
        # Build an extractor whose run() always raises.
        extractor = _make_extractor(cfg, _CANNED_EXTRACT)

        def boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("LLM unavailable")

        extractor.run = boom  # type: ignore[method-assign]
        runner = IngestRunner(
            extractor=extractor, store=ExtractStore(root=tmp_path / "e"),
            http_get=lambda _u: _SAMPLE_ATOM,
        )
        feed = FeedConfig(
            name="t", adapter="arxiv",
            query_url="http://example.com/atom",
            source_label="arXiv test",
        )
        summary = runner.run([feed])
        # Both papers tried, both failed, run continued
        assert summary.papers_seen == 2
        assert summary.papers_extract_failed == 2
        assert summary.papers_extracted == 0


# ---------------------------------------------------------------------------- #
# Real prompt file
# ---------------------------------------------------------------------------- #


class TestRealPromptFile:
    def test_paper_extractor_prompt_exists(self) -> None:
        path = Path("configs/research_prompts/paper_extractor.md")
        assert path.exists()
        text = path.read_text()
        # Doctrine reference
        assert "EVIDENCE_FIRST.md" in text
        # Required sections named
        for section in (
            "## Methodology",
            "## Findings",
            "## FX trading applicability",
            "## Data sources cited",
            "## Key citations",
        ):
            assert section in text, f"missing section heading {section!r}"
