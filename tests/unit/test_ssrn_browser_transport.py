"""Unit tests for the SSRN browser-automation transport (CL-nj2h).

Playwright is fully mocked — no browser launch, no network. Covers:

  * browser_http_get drives the (mocked) playwright chain: launch →
    context (realistic UA/viewport) → goto → wait for the listing
    selector → content(), and always closes the browser
  * selector timeout degrades to returning whatever HTML rendered
  * missing playwright raises RuntimeError with install instructions
  * build_fetcher wires the ssrn adapter to the browser transport by
    default while an injected http_get still overrides it (and other
    adapters keep plain httpx)
  * SSRNFetcher parses browser-rendered listing HTML into Papers
  * SSRNFetcher.fetch log-and-skips when the transport raises (the
    missing-playwright path) — one gated feed can't kill a run
  * IngestRunner end-to-end over an ssrn feed with a canned-HTML
    transport injected via the default-transport table

The live smoke check against papers.ssrn.com is intentionally NOT in
this suite — it needs a real browser and network. See the CL-nj2h
notes for the manual invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

import src.research.ingest as ingest
from src.research import browser_transport
from src.research.browser_transport import (
    SSRN_LISTING_SELECTOR,
    browser_http_get,
)
from src.research.config import (
    AgentConfig,
    ProviderConfig,
    ResearchConfig,
)
from src.research.ingest import (
    ExtractStore,
    FeedConfig,
    IngestRunner,
    PaperExtractor,
    SSRNFetcher,
    build_fetcher,
)
from src.research.llm.client import (
    Driver,
    LLMResponse,
    register_driver,
)

# Fake playwright object graph -----------------------------------------------


class _FakePage:
    def __init__(self, html: str, selector_raises: bool = False) -> None:
        self._html = html
        self._selector_raises = selector_raises
        self.goto_calls: list[dict[str, Any]] = []
        self.wait_calls: list[dict[str, Any]] = []

    def goto(self, url: str, **kwargs: Any) -> None:
        self.goto_calls.append({"url": url, **kwargs})

    def wait_for_selector(self, selector: str, **kwargs: Any) -> None:
        self.wait_calls.append({"selector": selector, **kwargs})
        if self._selector_raises:
            msg = "Timeout 20000ms exceeded"
            raise TimeoutError(msg)

    def content(self) -> str:
        return self._html


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    def new_page(self) -> _FakePage:
        return self._page


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self._page = page
        self.context_kwargs: dict[str, Any] = {}
        self.closed = False

    def new_context(self, **kwargs: Any) -> _FakeContext:
        self.context_kwargs = kwargs
        return _FakeContext(self._page)

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser
        self.launch_kwargs: dict[str, Any] = {}

    def launch(self, **kwargs: Any) -> _FakeBrowser:
        self.launch_kwargs = kwargs
        return self._browser


class _FakePlaywrightCM:
    """Stands in for the object sync_playwright() returns."""

    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium = chromium

    def __enter__(self) -> _FakePlaywrightCM:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def _install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch,
    html: str,
    selector_raises: bool = False,
) -> tuple[_FakePage, _FakeBrowser, _FakeChromium]:
    page = _FakePage(html, selector_raises=selector_raises)
    browser = _FakeBrowser(page)
    chromium = _FakeChromium(browser)
    monkeypatch.setattr(
        browser_transport,
        "_load_sync_playwright",
        lambda: lambda: _FakePlaywrightCM(chromium),
    )
    return page, browser, chromium


# ---------------------------------------------------------------------------- #
# browser_http_get
# ---------------------------------------------------------------------------- #


class TestBrowserHttpGet:
    def test_returns_rendered_html_and_closes_browser(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        page, browser, _ = _install_fake_playwright(
            monkeypatch, "<html><div class='trow'>x</div></html>",
        )
        html = browser_http_get("https://papers.ssrn.com/sol3/x.cfm")
        assert "class='trow'" in html
        assert browser.closed
        assert page.goto_calls[0]["url"] == "https://papers.ssrn.com/sol3/x.cfm"

    def test_realistic_browser_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, browser, chromium = _install_fake_playwright(monkeypatch, "<html/>")
        browser_http_get("https://papers.ssrn.com/sol3/x.cfm")
        assert chromium.launch_kwargs["headless"] is True
        ua = browser.context_kwargs["user_agent"]
        # Akamai flags the stock "HeadlessChrome" token instantly.
        assert "Headless" not in ua
        assert "Chrome/" in ua
        viewport = browser.context_kwargs["viewport"]
        assert viewport["width"] > 0
        assert viewport["height"] > 0

    def test_waits_for_listing_selector(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        page, _, _ = _install_fake_playwright(monkeypatch, "<html/>")
        browser_http_get("https://papers.ssrn.com/sol3/x.cfm")
        assert page.wait_calls[0]["selector"] == SSRN_LISTING_SELECTOR

    def test_selector_timeout_still_returns_html(
        self, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # If the JS challenge never resolves into the listing, we hand
        # back what rendered — the parser yields [] and the runner
        # marks the feed failed, per the log-and-skip convention.
        _, browser, _ = _install_fake_playwright(
            monkeypatch, "<html>Access Denied</html>", selector_raises=True,
        )
        with caplog.at_level("WARNING"):
            html = browser_http_get("https://papers.ssrn.com/sol3/x.cfm")
        assert "Access Denied" in html
        assert browser.closed
        assert any("never rendered" in r.message for r in caplog.records)

    def test_missing_playwright_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # None in sys.modules makes `import playwright.sync_api` raise
        # ImportError even though the package is installed.
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match=r"curlit\[browser\]"):
            browser_http_get("https://papers.ssrn.com/sol3/x.cfm")


# ---------------------------------------------------------------------------- #
# Adapter wiring — build_fetcher default transports
# ---------------------------------------------------------------------------- #


class TestSSRNTransportWiring:
    def test_ssrn_defaults_to_browser_transport(self) -> None:
        f = build_fetcher("ssrn")
        assert isinstance(f, SSRNFetcher)
        assert f.http_get is ingest._browser_http_get

    def test_injected_http_get_overrides_browser_transport(self) -> None:
        def fake(_url: str) -> str:
            return ""
        f = build_fetcher("ssrn", http_get=fake)
        assert f.http_get is fake

    def test_other_adapters_keep_plain_http(self) -> None:
        f = build_fetcher("arxiv")
        assert f.http_get is ingest._default_http_get


# ---------------------------------------------------------------------------- #
# SSRNFetcher parsing + failure handling
# ---------------------------------------------------------------------------- #


# Legacy SSRN markup (pre-SPA, CL-kxcs). The parser keeps the
# div.trow selectors as a fallback so an SSRN rollback doesn't silently
# break ingestion; this fixture exercises that path.
_SSRN_LISTING_HTML = """<!DOCTYPE html>
<html><body>
<div class="tbody">
  <div class="trow">
    <div class="title-holder">
      <a class="title" href="https://ssrn.com/abstract=5012345">
        Carry Unwinds and FX Options Skew
      </a>
    </div>
    <div class="authors"><a>Jane Smith</a>, <a>John Doe</a></div>
    <div class="abstract">We document that carry-trade unwinds steepen the
    FX risk-reversal skew in G10 pairs within two trading days.</div>
    <div class="note">Last revised: 12 May 2026</div>
  </div>
  <div class="trow">
    <div class="title-holder">
      <a class="title" href="/sol3/papers.cfm?abstract_id=5023456">
        Dealer Constraints and Covered Interest Parity
      </a>
    </div>
    <div class="authors"><a>Ana Silva</a></div>
    <div class="abstract">Balance-sheet costs explain most of the CIP basis.</div>
    <div class="note">Posted: 3 Apr 2026</div>
  </div>
</div>
</body></html>
"""


# Current SSRN markup (verified live 2026-07-14): a client-side SPA
# renders papers into #network-papers > ol > li > div.paper. The stats
# spans are concatenated with no whitespace ("…2026Publication…"),
# which is why the year regex uses digit-boundary lookarounds.
_SSRN_SPA_HTML = """<!DOCTYPE html>
<html><body>
<div id="network-papers"><ol>
  <li><div class="paper">
    <div class="paper-wrap"><div class="paper-info">
      <div class="title">
        <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7035198">
          Observing Climate Risks in Financial Markets
        </a>
      </div>
      <div class="stats"><span>Number of pages: 22</span><span>Posted 14 Jul 2026</span></div>
      <div class="type status"><span>Publication Status: </span><span>Under Review</span></div>
      <div class="authors"><div><a href="/x?per_id=1">Charles Donovan</a></div></div>
    </div></div>
  </div></li>
  <li><div class="paper">
    <div class="paper-wrap"><div class="paper-info">
      <div class="title">
        <a href="/sol3/papers.cfm?abstract_id=7110978">LRISK: Systemic Liquidity Risk</a>
      </div>
      <div class="stats"><span>Number of pages: 40</span><span>Posted 2 Jun 2025</span></div>
      <div class="authors">
        <div><a href="/x?per_id=2">Tristan Jourde</a></div>
        <div><a href="/x?per_id=3">Martin Saillard</a></div>
      </div>
    </div></div>
  </div></li>
</ol></div>
</body></html>
"""


class TestSSRNFetcherSPA:
    """Parsing the current (SPA) SSRN listing markup."""

    def test_parses_spa_listing(self) -> None:
        fetcher = SSRNFetcher(http_get=lambda _u: _SSRN_SPA_HTML)
        feed = FeedConfig(
            name="ssrn_spa", adapter="ssrn",
            query_url="https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?journal_id=203",
            source_label="SSRN FEN",
        )
        papers = fetcher.fetch(feed)
        assert len(papers) == 2

        first = papers[0]
        assert first.title == "Observing Climate Risks in Financial Markets"
        assert first.url == (
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7035198"
        )
        assert first.authors == ("Charles Donovan",)
        # Year survives the whitespace-free stats concatenation.
        assert first.year == 2026

        second = papers[1]
        # Relative href expands against papers.ssrn.com.
        assert second.url == (
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7110978"
        )
        assert second.authors == ("Tristan Jourde", "Martin Saillard")
        assert second.year == 2025


class TestSSRNFetcher:
    def test_parses_browser_rendered_listing(self) -> None:
        fetcher = SSRNFetcher(http_get=lambda _u: _SSRN_LISTING_HTML)
        feed = FeedConfig(
            name="ssrn_test", adapter="ssrn",
            query_url="https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?journal_id=1",
            source_label="SSRN test",
        )
        papers = fetcher.fetch(feed)
        assert len(papers) == 2

        first = papers[0]
        assert first.title == "Carry Unwinds and FX Options Skew"
        assert first.url == "https://ssrn.com/abstract=5012345"
        assert first.authors == ("Jane Smith", "John Doe")
        assert first.year == 2026
        assert "risk-reversal skew" in first.abstract
        assert first.source_label == "SSRN test"

        # Relative hrefs expand against papers.ssrn.com.
        second = papers[1]
        assert second.url == (
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5023456"
        )
        assert second.authors == ("Ana Silva",)

    def test_transport_failure_logs_and_returns_empty(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The missing-playwright RuntimeError follows the same path as
        # any transport error: log a warning, return [], keep the run
        # alive.
        def raising_get(_url: str) -> str:
            msg = "playwright is required ... pip install 'curlit[browser]'"
            raise RuntimeError(msg)

        fetcher = SSRNFetcher(http_get=raising_get)
        feed = FeedConfig(
            name="ssrn_gated", adapter="ssrn",
            query_url="https://papers.ssrn.com/sol3/x.cfm",
            source_label="SSRN",
        )
        with caplog.at_level("WARNING"):
            papers = fetcher.fetch(feed)
        assert papers == []
        assert any("fetch failed" in r.message for r in caplog.records)

    def test_challenge_page_yields_no_papers(self) -> None:
        # Akamai interstitial HTML has none of the listing selectors —
        # the parser must return [] rather than invent records.
        fetcher = SSRNFetcher(
            http_get=lambda _u: "<html><body>Access Denied</body></html>",
        )
        feed = FeedConfig(
            name="ssrn_denied", adapter="ssrn",
            query_url="https://papers.ssrn.com/sol3/x.cfm",
            source_label="SSRN",
        )
        assert fetcher.fetch(feed) == []


# ---------------------------------------------------------------------------- #
# IngestRunner over an ssrn feed
# ---------------------------------------------------------------------------- #


class _SSRNCannedDriver(Driver):
    name = "ssrn-canned"

    def __init__(self, api_key: str = "x", canned_text: str = "stub") -> None:
        super().__init__(api_key)
        self.canned_text = canned_text

    def complete(
        self,
        messages: Any,  # noqa: ARG002
        model: str,
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            text=self.canned_text, model=model, provider=self.name,
            input_tokens=10, output_tokens=20, usd_cost=0.0001, elapsed_sec=0.01,
        )


register_driver("ssrn-canned", _SSRNCannedDriver)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ResearchConfig:
    prompt = tmp_path / "extractor_prompt.md"
    prompt.write_text("You are a test extractor.")
    monkeypatch.setenv("SSRN_INGEST_KEY", "fake")
    return ResearchConfig(
        providers={
            "ssrn-canned": ProviderConfig(
                api_key_env="SSRN_INGEST_KEY", default_model="m1",
            ),
        },
        agents={
            "paper_extractor": AgentConfig(
                provider="ssrn-canned",
                role="paper_extractor",
                prompt_path=str(prompt),
                model=None,
            ),
        },
        debates={},  # type: ignore[arg-type]
    )


class TestIngestRunnerSSRN:
    def test_default_runner_uses_browser_transport_for_ssrn(
        self, cfg: ResearchConfig, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Swap the ssrn default transport for a canned-HTML shim; the
        # runner is constructed WITHOUT http_get, exactly like the
        # production cron path.
        transport_urls: list[str] = []

        def fake_browser_get(url: str) -> str:
            transport_urls.append(url)
            return _SSRN_LISTING_HTML

        monkeypatch.setitem(
            ingest._DEFAULT_TRANSPORTS, "ssrn", fake_browser_get,
        )
        extractor = PaperExtractor.from_config(
            name="paper_extractor", research_config=cfg,
        )
        runner = IngestRunner(
            extractor=extractor,
            store=ExtractStore(root=tmp_path / "extracts"),
        )
        feed = FeedConfig(
            name="ssrn_finmarkets", adapter="ssrn",
            query_url="https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?journal_id=1",
            source_label="SSRN Financial Markets eJournal",
        )
        summary = runner.run([feed])
        assert transport_urls == [feed.query_url]
        assert summary.feeds_failed == 0
        assert summary.papers_seen == 2
        assert summary.papers_extracted == 2

    def test_injected_http_get_still_wins_for_ssrn(
        self, cfg: ResearchConfig, tmp_path: Path,
    ) -> None:
        # Tests / dry-runs inject a plain shim; the browser transport
        # must not be consulted at all.
        extractor = PaperExtractor.from_config(
            name="paper_extractor", research_config=cfg,
        )
        runner = IngestRunner(
            extractor=extractor,
            store=ExtractStore(root=tmp_path / "extracts"),
            http_get=lambda _u: _SSRN_LISTING_HTML,
        )
        feed = FeedConfig(
            name="ssrn_finmarkets", adapter="ssrn",
            query_url="https://papers.ssrn.com/sol3/JELJOUR_Results.cfm?journal_id=1",
            source_label="SSRN Financial Markets eJournal",
        )
        summary = runner.run([feed])
        assert summary.papers_seen == 2
        assert summary.papers_extracted == 2
