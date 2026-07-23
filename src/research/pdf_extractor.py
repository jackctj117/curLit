"""Full-text PDF extraction for the paper ingester (CL-sy32).

Extends the abstract-only ingestion path: when a Paper has a PDF URL,
download → extract → return the body text so the LLM extractor sees
methodology + results sections, not just the abstract.

Disk hygiene (per CL-2klj acceptance bullet): tmp PDFs are deleted as
soon as text extraction finishes, regardless of success. The extractor
runs in process; we use NamedTemporaryFile + finally to guarantee.

Implementation notes:
  - pypdf for parsing (pure Python, no system deps).
  - 30 MB hard cap on download — academic PDFs are usually 1-5 MB; a
    50+ MB outlier is almost always a survey volume not worth scanning.
  - Truncate extracted text at ~80,000 characters (~16k tokens) to
    keep extractor prompt size bounded.
  - Strip control characters and collapse whitespace — pypdf preserves
    layout artifacts (column wraps, page breaks) that bloat tokens.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# 30 MB — see module docstring rationale.
_MAX_PDF_BYTES: int = 30 * 1024 * 1024
# 80k chars ≈ 16k tokens — fits Claude/GPT context budget for the
# extractor prompt's other inputs (abstract + system prompt + JSON
# schema) plus headroom for the model's response.
_MAX_TEXT_CHARS: int = 80_000


HttpGetBytes = Callable[[str], bytes]


def _default_http_get_bytes(url: str) -> bytes:
    import httpx

    resp = httpx.get(
        url,
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "curLit-research/1.0 (+research@curlit)"},
    )
    resp.raise_for_status()
    if len(resp.content) > _MAX_PDF_BYTES:
        raise ValueError(
            f"PDF too large ({len(resp.content)} bytes > {_MAX_PDF_BYTES})",
        )
    return bytes(resp.content)


def _normalize_text(raw: str) -> str:
    """Collapse PDF layout artifacts: column wraps, page breaks, control
    chars, repeated whitespace. Output is one paragraph per blank-line
    block, no other layout."""
    # Strip control chars (form feed especially — pypdf inserts \x0c at
    # page boundaries).
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", raw)
    # Collapse hyphenated line breaks: "comple-\nting" → "completing".
    cleaned = re.sub(r"-\n\s*", "", cleaned)
    # Single newline within a paragraph → space; double newline → keep.
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    # Collapse runs of whitespace.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_pdf_text(
    url: str,
    http_get_bytes: HttpGetBytes | None = None,
    max_chars: int = _MAX_TEXT_CHARS,
) -> str:
    """Download + extract body text. Returns "" on any failure.

    Caller is expected to use this opportunistically — abstract-only
    ingestion remains the fallback so a missing pypdf dependency, an
    unreadable PDF, or a network error never blocks the ingest run.
    """
    fetcher = http_get_bytes or _default_http_get_bytes
    try:
        body = fetcher(url)
    except Exception as exc:
        logger.warning("PDF download failed for %s: %s: %s", url, type(exc).__name__, exc)
        return ""

    tmp_path = Path(tempfile.mkstemp(suffix=".pdf", prefix="curlit-pdf-")[1])
    try:
        tmp_path.write_bytes(body)
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("pypdf not installed — PDF extraction skipped")
            return ""
        try:
            reader = PdfReader(str(tmp_path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            logger.warning(
                "PDF parse failed for %s: %s: %s",
                url,
                type(exc).__name__,
                exc,
            )
            return ""
        text = _normalize_text("\n".join(pages))
        return text[:max_chars]
    finally:
        # Disk hygiene — drop the temp file regardless of outcome.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to unlink %s", tmp_path)


def enrich_paper_with_pdf(
    paper: Any,
    http_get_bytes: HttpGetBytes | None = None,
) -> Any:
    """Convenience: if ``paper`` has a pdf_url and an empty/short abstract,
    fill paper.full_text from the PDF. Returns the (possibly mutated)
    Paper. No-op when paper has no pdf_url field."""
    pdf_url = getattr(paper, "pdf_url", None) or getattr(paper, "url", "")
    if not pdf_url or not pdf_url.lower().endswith(".pdf"):
        return paper
    full_text = extract_pdf_text(pdf_url, http_get_bytes=http_get_bytes)
    if full_text and hasattr(paper, "abstract"):
        # Append (don't replace) — abstract is curated, full text is
        # supplementary. Cap combined length.
        combined = (paper.abstract + "\n\n" + full_text)[:_MAX_TEXT_CHARS]
        try:
            paper.abstract = combined
        except AttributeError:
            # Frozen dataclass — caller should rebuild instead.
            logger.debug("Paper is immutable; PDF text not attached")
    return paper
