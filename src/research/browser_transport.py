"""Playwright-driven browser transport (CL-nj2h).

SSRN listing pages (papers.ssrn.com/sol3/JELJOUR_Results.cfm) sit
behind Akamai bot protection: plain httpx/curl requests get HTTP 403
even with a browser User-Agent, because the challenge demands signed
UA client hints plus real JS execution (verified 2026-05-04). Per-
paper pages (papers.ssrn.com/abstract=N) are open — only the listing
crawl needs a real browser.

``browser_http_get`` conforms to the ingester's ``HttpGet`` shim
(``Callable[[str], str]``): it launches headless Chromium via
Playwright's sync API, navigates with a realistic UA/viewport, lets
the page execute JS, waits for the listing selector to render, and
returns the fully rendered HTML. ``SSRNFetcher._parse`` in
``src/research/ingest.py`` then consumes that HTML unchanged — its
selectors were written against what a real browser sees.

playwright is an OPTIONAL dependency (the ``[browser]`` extra in
pyproject.toml, mirroring the ``[polymarket]`` extra pattern). The
import is lazy: importing this module is free, and calling the
transport without playwright installed raises a RuntimeError with the
exact install commands. Fetchers catch that, log, and skip the feed —
one missing extra doesn't kill an ingest run.

Sync-only by design: the ingest pipeline's transports are all sync
callables, and the ingester runs from cron, never inside an asyncio
loop (Playwright's sync API refuses to run inside a running loop).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# A current desktop Chrome UA. Chromium's default headless UA
# advertises "HeadlessChrome", which Akamai flags instantly; we
# override to the stock Chrome string. The major version (149) matches
# the Chromium build playwright 1.61 bundles, so the UA stays
# consistent with the sec-ch-ua client-hint brands the real engine
# sends — a UA/UA-CH version mismatch is itself a bot signal.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)

# Common laptop viewport — the Playwright default (1280x720) is a
# known automation fingerprint.
DEFAULT_VIEWPORT: dict[str, int] = {"width": 1440, "height": 900}

# Selector that marks an SSRN listing page as "rendered". SSRN is a
# client-side SPA: the #network-papers shell ships in the initial HTML
# but the paper <li> items are injected by an XHR to api.ssrn.com only
# after the Akamai/Cloudflare challenge clears. We therefore wait for a
# rendered <li> (or the legacy div.trow), NOT the shell container —
# waiting on the shell would resolve instantly and hand back an empty
# list. An empty journal legitimately renders no <li>, so this selector
# timing out is the correct signal for "nothing to ingest".
SSRN_LISTING_SELECTOR = "#network-papers ol li, div.trow"

# Navigation is the whole point here (Akamai's JS challenge can take
# several seconds), so the budget is deliberately larger than the
# plain-HTTP DEFAULT_HTTP_TIMEOUT_SEC.
DEFAULT_NAV_TIMEOUT_MS = 60_000
DEFAULT_SELECTOR_TIMEOUT_MS = 20_000


def _load_sync_playwright() -> Any:
    """Lazy import so playwright stays optional. Split out from
    ``browser_http_get`` so tests can monkeypatch the playwright
    machinery without installing a browser."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        msg = (
            "playwright is required for the SSRN browser transport but is "
            "not installed. Install the optional extra:\n"
            "  pip install 'curlit[browser]'\n"
            "  playwright install chromium"
        )
        raise RuntimeError(msg) from exc
    return sync_playwright


def browser_http_get(
    url: str,
    *,
    wait_selector: str | None = SSRN_LISTING_SELECTOR,
    nav_timeout_ms: int = DEFAULT_NAV_TIMEOUT_MS,
    selector_timeout_ms: int = DEFAULT_SELECTOR_TIMEOUT_MS,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    """Fetch ``url`` in headless Chromium and return the rendered HTML.

    Conforms to the ingester's ``HttpGet`` signature (when called with
    just the URL), so it drops into any fetcher's ``http_get`` slot.

    Behaviour:
      * navigates and waits for DOMContentLoaded, then for
        ``wait_selector`` to appear (i.e. the JS challenge resolved and
        the listing actually rendered);
      * if the selector never appears, logs a warning and returns
        whatever HTML did render — the parser then yields zero papers
        and the runner records the feed as failed, matching the
        log-and-skip convention of the other fetchers;
      * raises RuntimeError if playwright isn't installed, and lets
        Playwright's own errors (navigation timeout, browser missing)
        propagate — callers (fetchers) already catch-and-log transport
        exceptions per feed.

    A fresh browser per call is deliberate: the ingester hits one SSRN
    listing page per run, so there's nothing to amortize, and a cold
    profile avoids accumulating bot-score state across runs.
    """
    sync_playwright = _load_sync_playwright()
    with sync_playwright() as p:
        # channel="chromium" selects the full Chromium build in "new
        # headless" mode. Plain headless=True would pick the stripped
        # chromium-headless-shell, whose fingerprint (missing codecs,
        # fonts, GL stack) Akamai scores as a bot far more readily.
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            # Chromium sets navigator.webdriver and a few Blink flags
            # under automation; Akamai checks both.
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        try:
            browser = p.chromium.launch(channel="chromium", **launch_kwargs)
        except Exception:
            # Older playwright or full-chromium build not installed —
            # fall back to the default (headless shell) build.
            logger.info(
                "browser transport: chromium channel unavailable, "
                "falling back to default headless build",
            )
            browser = p.chromium.launch(**launch_kwargs)
        try:
            context = browser.new_context(
                user_agent=user_agent,
                viewport=DEFAULT_VIEWPORT,
                locale="en-US",
                timezone_id="America/New_York",
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
            if wait_selector:
                try:
                    page.wait_for_selector(
                        wait_selector, timeout=selector_timeout_ms,
                    )
                except Exception:
                    # Timeout (or challenge interstitial) — return what
                    # rendered so the caller can log title/status from
                    # the body instead of losing it to an exception.
                    logger.warning(
                        "browser transport: selector %r never rendered at %s "
                        "— returning page as-is",
                        wait_selector, url,
                    )
            html: str = page.content()
            return html
        finally:
            browser.close()
