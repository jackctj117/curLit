"""Live-network integration tests for things that previously lacked them
(CL-gnhu, CL-kxcs, CL-qdns, CL-sy32).

These tests hit the public internet and are skipped under
``-m 'not network'``. CI can opt in via the ``network`` marker. The
goal is to catch URL drift early — every URL pattern that used to be
"educated guess" is here as an actual HTTP probe.
"""

from __future__ import annotations

import os

import httpx
import pytest

# Skip everything in this file unless explicitly run.
pytestmark = pytest.mark.skipif(
    os.environ.get("CURLIT_RUN_NETWORK_TESTS", "0") != "1",
    reason="set CURLIT_RUN_NETWORK_TESTS=1 to enable network tests",
)


def _ok(url: str, timeout: float = 15.0) -> bool:
    """True if the URL returns 2xx after redirects."""
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "curLit-research/1.0"},
        )
    except httpx.HTTPError:
        return False
    return resp.status_code < 400


class TestRSSFeedURLs:
    """The new RSS feed URLs added in CL-kxcs must resolve and parse."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.nber.org/rss/new.xml",
            "https://www.frbsf.org/feed/",
            "https://www.bis.org/doclist/wppubls.rss",
        ],
    )
    def test_resolves_to_xml(self, url: str) -> None:
        resp = httpx.get(
            url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "curLit-research/1.0"},
        )
        assert resp.status_code == 200, f"{url} returned {resp.status_code}"
        body = resp.text[:500].lower()
        # RSS 2.0 (<rss>), Atom 1.0 (<feed>), or RSS 1.0 (<rdf:rdf>).
        # BIS in particular serves RSS 1.0.
        assert any(tag in body for tag in ("<rss", "<feed", "<rdf:rdf")), (
            f"{url} returned non-feed body: {resp.text[:200]}"
        )


class TestCBHistoricalURLs:
    """The CL-qdns historical archive URLs must return 200 for a recent year."""

    @pytest.mark.parametrize(
        "url",
        [
            # ECB 2024 (post-site-rebuild)
            "https://www.ecb.europa.eu/press/pubbydate/html/index.en.html"
            "?name_of_publication=Press%20release&year=2024",
            # BoE 2024 via Taxonomies
            "https://www.bankofengland.co.uk/news/news"
            "?Taxonomies=ce90163e489841e0b66d06243d35d5cb"
            "&NewsTypes=ce90163e489841e0b66d06243d35d5cb"
            "&Direction=Latest&InfiniteScrolling=False&Page=1&Year=2024",
            # BoJ 2023
            "https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_2023/index.htm",
            # BoC 2024
            "https://www.bankofcanada.ca/news/?mtm_search_filter=monetary-policy&date_year=2024",
        ],
    )
    def test_archive_resolves(self, url: str) -> None:
        assert _ok(url), f"CB archive URL did not resolve: {url}"


class TestNTPLive:
    def test_pool_ntp_org_returns_finite_offset(self) -> None:
        from scripts.clock_drift_monitor import query_ntp_offset

        offset = query_ntp_offset("pool.ntp.org")
        # A live offset on a sane laptop is < 60 seconds. Anything
        # outside that range means either the parser is wrong or we
        # hit a misconfigured NTP server.
        assert -60 < offset < 60, f"implausible NTP offset {offset}"


class TestPDFExtractionLive:
    def test_arxiv_pdf_extracts(self) -> None:
        from src.research.pdf_extractor import extract_pdf_text

        # arXiv 2401.00001 — small open-access paper, stable URL.
        text = extract_pdf_text("https://arxiv.org/pdf/2401.00001")
        # Should pull at least a few hundred chars of recognizable
        # English. Blank or sub-100-char output means pypdf failed.
        assert len(text) > 500, f"PDF extraction yielded {len(text)} chars"
        # Sanity: should contain at least one common English word
        # in the body. arXiv abstracts always include one of these.
        assert any(word in text.lower() for word in ("the", "this", "we", "in", "of"))
