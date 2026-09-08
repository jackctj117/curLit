"""Research tools for the tool-augmented niche pass (CL-2czc).

Turns the niche agent from pure recall into an ACTIVE researcher: between
hopping cycles, deterministic code pulls REAL data on the names found so far
and injects it into the next cycle's prompt, so hop 2/3 come from actual named
customers / suppliers / risk factors rather than the model's training memory.

Two free, keyless sources:

* SEC EDGAR — a company's latest 10-K/10-Q. We already store the CIK
  (:meth:`SymbolUniverse.get_cik`, from the SEC enrichment CL-9xha), which is
  the key to its filings. We fetch the primary document, strip it to text, and
  return the Business / Risk-Factors excerpt (where filers name their real
  customers, suppliers, competitors, and dependencies).
* yfinance — sector / industry / business summary for quick grounding.

Everything is FAIL-SOFT: any fetch/parse error yields no grounding for that
name (the model just hops from its own reasoning, as before) — never a raise.
SEC's edge blocks non-browser agents, so the User-Agent is browser-shaped and
overridable via ``SEC_EDGAR_USER_AGENT`` (same knob as CL-9xha).
"""

from __future__ import annotations

import html as _html
import json
import logging
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from src.events.research_evidence import SourceDocument

logger = logging.getLogger(__name__)

_DEFAULT_SEC_USER_AGENT = (
    "Mozilla/5.0 (curLit niche-research; set SEC_EDGAR_USER_AGENT to declare a contact)"
)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}"

#: Anchors (lower-cased) where a 10-K's supply-chain / dependency content lives;
#: searched in order to start the excerpt at the useful part.
_EXCERPT_ANCHORS = (
    "risk factors",
    "item 1a",
    "principal customers",
    "our customers",
    "our business",
    "item 1.",
    "competition",
)

#: Injectable transports so unit tests feed canned bodies (no live network).
SecHttpGet = Callable[[str, dict[str, str]], str]
ProfileFn = Callable[[str], dict[str, Any] | None]


def _default_sec_http_get(url: str, headers: dict[str, str]) -> str:
    resp = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def yfinance_profile(ticker: str) -> dict[str, Any] | None:
    """Sector / industry / business summary via yfinance ``.info``. Fail-soft
    → None (the caller degrades to SEC-only or no grounding)."""
    try:
        import yfinance as yf  # noqa: PLC0415 — deferred; cheap for non-live callers

        info = dict(yf.Ticker(ticker).info or {})
    except Exception:
        logger.debug("yfinance profile unavailable for %s", ticker, exc_info=True)
        return None
    return {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "summary": info.get("longBusinessSummary"),
    }


def html_to_text(raw_html: str) -> str:
    """Strip an HTML filing to whitespace-collapsed plain text."""
    stripped = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        raw_html,
        flags=re.I | re.S,
    )
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = _html.unescape(stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _prose_score(window: str) -> int:
    """Higher = more like body prose, lower = more like a table-of-contents
    line (page numbers + Title Case). Used to skip the TOC 'Risk Factors 10
    Item 1B ... 27' entry and land on the actual section text."""
    if not window:
        return -(10**6)
    digits = sum(c.isdigit() for c in window)
    lower = sum(c.islower() for c in window)
    return lower - 4 * digits


def anchored_excerpt(text: str, max_chars: int) -> str | None:
    """Return the useful slice of a filing: the supply-chain anchor (risk
    factors / business / customers) whose following text reads most like prose
    — skipping the table-of-contents reference — for ``max_chars``. Falls back
    to the document head. None if there's not enough text."""
    if not text or len(text) < 200:
        return None
    low = text.lower()
    best_idx: int | None = None
    best_score = -(10**7)
    for anchor in _EXCERPT_ANCHORS:
        start = 0
        while True:
            idx = low.find(anchor, start)
            if idx < 0:
                break
            score = _prose_score(text[idx : idx + 300])
            if score > best_score:
                best_score, best_idx = score, idx
            start = idx + len(anchor)
    if best_idx is not None:
        return text[best_idx : best_idx + max_chars]
    return text[:max_chars]


class ResearchTools:
    """Grounds niche ideas in real SEC filings + company profiles."""

    def __init__(
        self,
        sec_http_get: SecHttpGet | None = None,
        profile_fn: ProfileFn | None = None,
        sec_user_agent: str | None = None,
        max_excerpt_chars: int = 4000,
        max_summary_chars: int = 600,
        max_doc_process_chars: int = 600_000,
        technicals_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self._sec_http_get = sec_http_get or _default_sec_http_get
        self._profile_fn = profile_fn or yfinance_profile
        # Computed price-structure context (CL-3xoj) — injectable; the live
        # default fetches daily bars via yfinance. None-able for tests.
        if technicals_fn is not None:
            self._technicals_fn = technicals_fn
        else:
            from src.events.technical_context import compute_for_ticker  # noqa: PLC0415

            self._technicals_fn = compute_for_ticker
        self.sec_user_agent = sec_user_agent or os.environ.get(
            "SEC_EDGAR_USER_AGENT",
            _DEFAULT_SEC_USER_AGENT,
        )
        self.max_excerpt_chars = max_excerpt_chars
        self.max_summary_chars = max_summary_chars
        self.max_doc_process_chars = max_doc_process_chars

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.sec_user_agent}

    def filing_documents(
        self,
        cik: int,
        symbol: str,
        query: str = "",
        limit: int = 3,
    ) -> list[SourceDocument]:
        """Dated recent annual/quarterly/current reports, newest first.

        Current reports (8-K) include company announcements. Capture exact
        normalized-text offsets and source identity; never fetch a model URL.
        The bounded primary-document collection is not an exhaustive SEC search.
        """
        logger.info("research: retrieving recent filings for %s", symbol)
        try:
            raw = self._sec_http_get(_SUBMISSIONS_URL.format(cik=int(cik)), self._headers())
            recent = json.loads(raw).get("filings", {}).get("recent", {})
            rows = zip(
                recent.get("form", []),
                recent.get("accessionNumber", []),
                recent.get("primaryDocument", []),
                recent.get("filingDate", []),
                strict=False,
            )
            selected = sorted(
                (r for r in rows if r[0] in ("10-K", "10-Q", "8-K")),
                key=lambda r: r[3],
                reverse=True,
            )[: max(0, min(limit, 3))]  # Three documents bounds per-tool network work.
        except Exception:
            logger.warning("research: filing index unavailable for %s", symbol)
            return []
        result = []
        for form, accession, name, published in selected:
            if not re.fullmatch(r"[0-9-]+", str(accession)) or not re.fullmatch(
                r"[A-Za-z0-9_.-]+",
                str(name),
            ):
                continue
            url = _ARCHIVE_URL.format(cik=int(cik), accn=accession.replace("-", ""), doc=name)
            try:
                body = self._sec_http_get(url, self._headers())
                text = html_to_text(body[: self.max_doc_process_chars])
                terms = re.findall(r"[A-Za-z]{4,}", query.lower())[:12]
                hits = [text.lower().find(t) for t in terms if t in text.lower()]
                if hits:
                    start = max(0, min(hits) - 200)  # Retain preceding sentence context.
                    passage = text[start : start + self.max_excerpt_chars]
                else:
                    passage = anchored_excerpt(text, self.max_excerpt_chars) or ""
                    start = text.find(passage)
                if not passage:
                    continue
                # SEC filingDate has day precision. End-of-day avoids claiming
                # availability before an unknown intraday publication time.
                result.append(
                    SourceDocument(
                        symbol=symbol.upper(),
                        url=url,
                        published_at=f"{published}T23:59:59+00:00",
                        retrieved_at=datetime.now(UTC).isoformat(),
                        text=passage,
                        locator=f"{form} {accession}; normalized-text chars {start}:{start + len(passage)}",
                    )
                )
            except Exception:
                logger.warning("research: filing document unavailable for %s", symbol)
        return result

    def sec_excerpt(self, cik: int) -> str | None:
        """Latest 10-K/10-Q Business/Risk excerpt for a CIK, or None.

        submissions JSON → newest 10-K (else 10-Q) → primary doc → text →
        anchored excerpt. Every step is fail-soft.
        """
        try:
            body = self._sec_http_get(
                _SUBMISSIONS_URL.format(cik=int(cik)),
                self._headers(),
            )
            sub = json.loads(body)
        except Exception:
            logger.debug("SEC submissions failed for CIK %s", cik, exc_info=True)
            return None
        recent = (sub.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        accns = recent.get("accessionNumber") or []
        docs = recent.get("primaryDocument") or []
        pick: tuple[str, str, str] | None = None
        for want in ("10-K", "10-Q"):  # prefer the annual, fall back to quarterly
            for i, form in enumerate(forms):
                if form == want and i < len(accns) and i < len(docs) and docs[i]:
                    pick = (form, str(accns[i]), str(docs[i]))
                    break
            if pick:
                break
        if pick is None:
            return None
        form, accn, doc = pick
        url = _ARCHIVE_URL.format(
            cik=int(cik),
            accn=accn.replace("-", ""),
            doc=doc,
        )
        try:
            raw = self._sec_http_get(url, self._headers())
        except Exception:
            logger.debug("SEC doc fetch failed: %s", url, exc_info=True)
            return None
        text = html_to_text(raw[: self.max_doc_process_chars])
        excerpt = anchored_excerpt(text, self.max_excerpt_chars)
        return f"[{form}] {excerpt}" if excerpt else None

    def ground_one(
        self,
        ticker: str,
        company_name: str,
        universe: Any,
    ) -> str | None:
        """A single grounding block for one name (profile + SEC excerpt), or
        None when no real data could be gathered."""
        parts = [f"--- {company_name or ticker} ({ticker}) ---"]
        try:
            profile = self._profile_fn(ticker)
        except Exception:
            profile = None
        if profile:
            si = " / ".join(x for x in (profile.get("sector"), profile.get("industry")) if x)
            if si:
                parts.append(f"Sector: {si}")
            summary = (profile.get("summary") or "").strip()
            if summary:
                parts.append(f"Business: {summary[: self.max_summary_chars]}")
        cik = universe.get_cik(ticker) if hasattr(universe, "get_cik") else None
        if cik:
            excerpt = self.sec_excerpt(cik)
            if excerpt:
                parts.append(
                    "Latest SEC filing excerpt (ground your next hops in THIS — "
                    f"real named customers/suppliers/risks):\n{excerpt}",
                )
        # Computed technical context (CL-3xoj) — real levels/trend so entry
        # triggers and invalidations reference actual price structure.
        try:
            ctx = self._technicals_fn(ticker)
        except Exception:
            ctx = None
        if ctx is not None:
            from src.events.technical_context import format_context_block  # noqa: PLC0415

            parts.append(format_context_block(ctx))
        # Options-activity confirmation (CL-mtum) — free chain-snapshot read
        # (P/C skew, volume vs baseline). Uses the universe's DB engine when
        # present; silently absent otherwise.
        db_engine = getattr(universe, "engine", None)
        if db_engine is not None:
            try:
                from src.events.options_activity import activity_note  # noqa: PLC0415

                note = activity_note(db_engine, ticker)
            except Exception:
                note = None
            if note:
                parts.append(note)
        if len(parts) == 1:  # header only → no real data gathered
            return None
        return "\n".join(parts)

    def enrich(self, ideas: list[Any], universe: Any) -> list[str]:
        """Grounding blocks for a small set of ideas (those with a real
        ticker). Order preserved; names that yield nothing are skipped."""
        blocks: list[str] = []
        for idea in ideas:
            ticker = getattr(idea, "ticker", "") or ""
            company = getattr(idea, "company_name", "") or ""
            if not ticker:
                continue
            block = self.ground_one(ticker, company, universe)
            if block:
                blocks.append(block)
        return blocks
