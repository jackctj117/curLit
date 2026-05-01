"""Paper-stream ingester (CL-2klj) — pulls academic papers from
configured feeds, runs an LLM extractor over each, and persists the
result as markdown to ``data/research/extracts/{hash}.md``. The Idea
Agent (CL-n0m8) consumes those extracts.

Architecture:

  * ``Paper``        — the canonical record (id, title, authors, year,
                       url, doi, abstract, source_label).
  * ``paper_hash``   — SHA-256 over (DOI || URL || title+authors+year).
                       Stable across reruns; used both for filename and
                       dedup.
  * ``ArxivFetcher`` — concrete fetcher for arXiv's Atom API. Other
                       sources (SSRN, NBER, Fed, BIS) are extensible by
                       registering a new fetcher class against an
                       adapter name in ``_FETCHER_REGISTRY``.
  * ``PaperExtractor`` — Agent subclass; runs the extractor system
                         prompt against a Paper and returns the markdown
                         body of the extract.
  * ``ExtractStore`` — disk-backed; writes extracts under a configurable
                       root (default ``data/research/extracts``); skips
                       already-stored hashes for idempotent reruns.
  * ``IngestRunner`` — wires fetcher → extractor → store for one run.

v1 scope:
  * arXiv-only fetcher. SSRN/NBER/Fed/BIS adapters are tracked as
    follow-ups under CL-h986; the registry pattern means each is a
    single-class addition with no plumbing changes.
  * Abstract-only extraction (no PDF download). The bead description
    spec'd PDF fetching but for v1 the LLM works from the abstract;
    full-text extraction is filed as a P2 follow-up so this loop can
    start producing extracts immediately.

The HTTP fetcher and the LLM agent are both injectable so unit tests
run with a canned XML string + a fake agent — no network, no API key.
"""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from src.research.agents.base import Agent

logger = logging.getLogger(__name__)


# Default location for stored extracts. Lives under data/, which is
# already gitignored — extracts are runtime artifacts, not source.
DEFAULT_EXTRACT_ROOT: Path = Path("data/research/extracts")

# HTTP timeout for feed fetches. Atom queries are small; if a feed
# blocks past this, we skip and try next run.
DEFAULT_HTTP_TIMEOUT_SEC: float = 30.0


# --------------------------------------------------------------------------- #
# Paper record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Paper:
    """One paper's canonical metadata. Frozen so hashes are stable."""

    title: str
    authors: tuple[str, ...]
    year: int | None
    url: str
    doi: str = ""
    abstract: str = ""
    source_label: str = ""


def paper_hash(paper: Paper) -> str:
    """SHA-256 hash for dedup + extract filename. Prefers DOI > URL >
    (title+authors+year) — DOI is the canonical citation key when
    available, URL is next-best, and the title/author/year fallback
    handles working papers without DOIs."""
    if paper.doi.strip():
        key = f"doi:{paper.doi.strip().lower()}"
    elif paper.url.strip():
        key = f"url:{paper.url.strip().lower()}"
    else:
        author_part = "|".join(a.strip().lower() for a in paper.authors)
        year_part = str(paper.year) if paper.year else ""
        key = f"meta:{paper.title.strip().lower()}::{author_part}::{year_part}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Feed config
# --------------------------------------------------------------------------- #


@dataclass
class FeedConfig:
    """One entry from configs/paper_streams.yaml."""

    name: str
    adapter: str        # 'arxiv' | (future: 'ssrn', 'nber', ...)
    query_url: str
    source_label: str


def load_feed_configs(path: Path | str) -> list[FeedConfig]:
    p = Path(path)
    if not p.exists():
        msg = f"feed config not found at {p}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict) or "feeds" not in raw:
        msg = f"feed config at {p} missing 'feeds' top-level key"
        raise ValueError(msg)
    feeds: list[FeedConfig] = []
    for name, cfg in raw["feeds"].items():
        if not isinstance(cfg, dict):
            msg = f"feed {name!r} is not a mapping"
            raise ValueError(msg)
        feeds.append(FeedConfig(
            name=name,
            adapter=str(cfg["adapter"]),
            query_url=str(cfg["query_url"]).strip(),
            source_label=str(cfg.get("source_label", name)),
        ))
    return feeds


# --------------------------------------------------------------------------- #
# Fetchers
# --------------------------------------------------------------------------- #


# Type alias for the HTTP shim. Tests inject a canned string returner.
HttpGet = Callable[[str], str]


def _default_http_get(url: str) -> str:
    """Fetch the URL and return the response body as text. Raises on
    non-2xx — the runner catches and skips the offending feed."""
    resp = httpx.get(url, timeout=DEFAULT_HTTP_TIMEOUT_SEC, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


# Atom 1.0 namespace used by arXiv's API.
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivFetcher:
    """Fetches arXiv's Atom API and yields Paper records.

    The arXiv API returns Atom 1.0 with one ``<entry>`` per paper.
    Each entry has ``<title>``, ``<summary>`` (abstract),
    ``<author><name>``, ``<published>``, and ``<id>`` (canonical URL).
    """

    http_get: HttpGet = field(default=_default_http_get)

    def fetch(self, feed: FeedConfig) -> list[Paper]:
        """Fetch the feed and parse all entries into Paper records.
        Returns an empty list if the feed is unreachable or returns
        malformed XML — failures are logged, not raised, so one bad
        feed doesn't kill an ingest run."""
        try:
            body = self.http_get(feed.query_url)
        except Exception as exc:
            logger.warning(
                "feed %r fetch failed: %s: %s",
                feed.name, type(exc).__name__, exc,
            )
            return []
        try:
            return self._parse(body, source_label=feed.source_label)
        except ET.ParseError as exc:
            logger.warning(
                "feed %r XML parse failed: %s", feed.name, exc,
            )
            return []

    @staticmethod
    def _parse(xml_body: str, source_label: str) -> list[Paper]:
        """Parse the Atom feed body into Paper records."""
        root = ET.fromstring(xml_body)  # noqa: S314 — arXiv-source XML, no DTD
        out: list[Paper] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            title_elem = entry.find("atom:title", _ATOM_NS)
            summary_elem = entry.find("atom:summary", _ATOM_NS)
            id_elem = entry.find("atom:id", _ATOM_NS)
            published_elem = entry.find("atom:published", _ATOM_NS)
            authors = tuple(
                (a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
                for a in entry.findall("atom:author", _ATOM_NS)
            )
            title = (title_elem.text if title_elem is not None else "") or ""
            abstract = (summary_elem.text if summary_elem is not None else "") or ""
            url = (id_elem.text if id_elem is not None else "") or ""
            year = ArxivFetcher._extract_year(
                published_elem.text if published_elem is not None else None,
            )
            out.append(Paper(
                title=" ".join(title.split()).strip(),
                authors=tuple(a for a in authors if a),
                year=year,
                url=url.strip(),
                doi="",  # arXiv entries usually don't carry DOI in the feed
                abstract=" ".join(abstract.split()).strip(),
                source_label=source_label,
            ))
        return out

    @staticmethod
    def _extract_year(published: str | None) -> int | None:
        if not published or len(published) < 4:
            return None
        try:
            return int(published[:4])
        except ValueError:
            return None


@dataclass
class RSSFetcher:
    """Generic RSS 2.0 fetcher for substack / quant blogs / news feeds.

    RSS 2.0 uses ``<rss><channel><item>...`` rather than Atom's
    ``<feed><entry>``. Each item has ``<title>``, ``<link>``,
    ``<description>`` (and sometimes ``<content:encoded>`` for the
    full body, which we strip to text-only summary), ``<pubDate>``,
    and either ``<author>`` or ``<dc:creator>``.

    The fetched body is passed through the same Paper record so the
    rest of the pipeline (extractor → idea agent → debate) treats blog
    posts as just another source. The paper_extractor prompt's four
    sections (methodology / findings / FX trading applicability /
    data sources / key citations) work fine on a tight blog summary
    even though the source isn't peer-reviewed — the idea agent will
    DECLINE most of them, which is correct: a blog post that doesn't
    contain a falsifiable thesis isn't a hypothesis.
    """

    http_get: HttpGet = field(default=_default_http_get)

    def fetch(self, feed: FeedConfig) -> list[Paper]:
        try:
            body = self.http_get(feed.query_url)
        except Exception as exc:
            logger.warning(
                "feed %r fetch failed: %s: %s",
                feed.name, type(exc).__name__, exc,
            )
            return []
        try:
            return self._parse(body, source_label=feed.source_label)
        except ET.ParseError as exc:
            logger.warning(
                "feed %r XML parse failed: %s", feed.name, exc,
            )
            return []

    @staticmethod
    def _parse(xml_body: str, source_label: str) -> list[Paper]:
        # RSS 2.0 has its own dc: namespace for author ("dc:creator")
        # plus content: for the full HTML body. Register both.
        ns = {
            "dc": "http://purl.org/dc/elements/1.1/",
            "content": "http://purl.org/rss/1.0/modules/content/",
        }
        root = ET.fromstring(xml_body)  # noqa: S314 — RSS feed, no DTD
        # RSS 2.0 wraps items in <channel>; some atom-flavored RSS skips
        # the channel wrapper. Try both.
        items = root.findall(".//item")
        out: list[Paper] = []
        for item in items:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = item.findtext("pubDate") or ""
            # Author: try dc:creator first (substack uses this), then
            # the standard <author> tag.
            author = (
                (item.findtext("dc:creator", default="", namespaces=ns) or "")
                or (item.findtext("author") or "")
            ).strip()
            # Description is the summary; some feeds put the full HTML
            # body in content:encoded. Prefer the description for
            # extract size; if missing, fall back to content:encoded
            # stripped of HTML tags.
            description = (item.findtext("description") or "").strip()
            if not description:
                content = item.findtext(
                    "content:encoded", default="", namespaces=ns,
                ) or ""
                description = re.sub(r"<[^>]+>", " ", content).strip()
            description = re.sub(r"\s+", " ", description)
            out.append(Paper(
                title=title,
                authors=(author,) if author else (),
                year=RSSFetcher._extract_year(pub_date),
                url=link,
                doi="",
                abstract=description[:4000],  # cap so paper_hash is stable + extract size bounded
                source_label=source_label,
            ))
        return out

    @staticmethod
    def _extract_year(pub_date: str) -> int | None:
        # RSS 2.0 pubDate format: "Tue, 30 Apr 2026 14:00:00 GMT"
        # Just look for a 4-digit year between 1990 and 2099.
        match = re.search(r"\b(19[9]\d|20\d\d)\b", pub_date)
        if match:
            return int(match.group(1))
        return None


_FETCHER_REGISTRY: dict[str, Callable[[HttpGet], Any]] = {
    "arxiv": lambda http_get: ArxivFetcher(http_get=http_get),
    "rss": lambda http_get: RSSFetcher(http_get=http_get),
}


def build_fetcher(adapter: str, http_get: HttpGet | None = None) -> Any:
    """Look up the fetcher class for an adapter name. Future SSRN/NBER
    adapters register here with no other plumbing changes."""
    if adapter not in _FETCHER_REGISTRY:
        msg = (
            f"unknown feed adapter {adapter!r}; "
            f"registered: {sorted(_FETCHER_REGISTRY)}"
        )
        raise KeyError(msg)
    factory = _FETCHER_REGISTRY[adapter]
    return factory(http_get or _default_http_get)


# --------------------------------------------------------------------------- #
# Extractor — Agent subclass
# --------------------------------------------------------------------------- #


class PaperExtractor(Agent):
    """LLM extractor. Construct via ``PaperExtractor.from_config`` with
    the ``paper_extractor`` agent name (system prompt loads from
    configs/research_prompts/paper_extractor.md)."""

    def extract(self, paper: Paper) -> str:
        """Run the LLM and return the markdown body of the extract.
        The store wraps this body with a citation header before
        persisting; the LLM only writes the four-section body."""
        resp = self.run(
            user_prompt=(
                "Produce the markdown extract for the paper below. "
                "Match the section structure specified by your system "
                "prompt exactly."
            ),
            context_files={
                "paper_metadata": _format_paper_for_prompt(paper),
            },
        )
        return resp.text.strip()


def _format_paper_for_prompt(paper: Paper) -> str:
    """Render a Paper as a labeled context block for the extractor."""
    authors_line = ", ".join(paper.authors) if paper.authors else "(unknown)"
    return (
        f"title: {paper.title}\n"
        f"authors: {authors_line}\n"
        f"year: {paper.year or '(unknown)'}\n"
        f"url: {paper.url}\n"
        f"source: {paper.source_label}\n\n"
        f"abstract:\n{paper.abstract}\n"
    )


# --------------------------------------------------------------------------- #
# Store — disk-backed extract persistence + dedup
# --------------------------------------------------------------------------- #


class ExtractStore:
    """Owns the on-disk extract directory. Hash-based filenames make
    dedup mechanical — a paper that was extracted on a previous run
    appears as an existing file and the runner skips re-extracting it."""

    def __init__(self, root: Path | str = DEFAULT_EXTRACT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def has(self, paper: Paper) -> bool:
        return self.path_for(paper).exists()

    def path_for(self, paper: Paper) -> Path:
        return self.root / f"{paper_hash(paper)}.md"

    def write(self, paper: Paper, extract_body: str) -> Path:
        """Persist extract with a citation header on top. Returns the
        path written. Idempotent — overwrite is fine because the input
        paper hash deterministically maps to the same filename."""
        out_path = self.path_for(paper)
        out_path.write_text(_render_extract(paper, extract_body))
        return out_path


def _render_extract(paper: Paper, body: str) -> str:
    """Compose the on-disk markdown: citation header + LLM body."""
    authors_line = ", ".join(paper.authors) if paper.authors else "(unknown)"
    header = (
        f"# {paper.title}\n\n"
        f"- **authors**: {authors_line}\n"
        f"- **year**: {paper.year or '(unknown)'}\n"
        f"- **url**: {paper.url}\n"
        f"- **doi**: {paper.doi or '(none)'}\n"
        f"- **source**: {paper.source_label}\n"
        f"- **paper_hash**: `{paper_hash(paper)}`\n\n"
        f"---\n\n"
    )
    return header + body.strip() + "\n"


# --------------------------------------------------------------------------- #
# Runner — orchestrates one ingest pass
# --------------------------------------------------------------------------- #


@dataclass
class IngestRunSummary:
    """Per-run summary returned by IngestRunner.run()."""

    feeds_total: int = 0
    feeds_failed: int = 0
    papers_seen: int = 0
    papers_skipped_duplicate: int = 0
    papers_extracted: int = 0
    papers_extract_failed: int = 0
    extract_paths: list[Path] = field(default_factory=list)


class IngestRunner:
    """Wires fetchers + extractor + store for one ingest pass.

    Each ``run`` call iterates the supplied feed configs, fetches each,
    dedups via the store, runs the LLM extractor on the new ones, and
    writes the resulting extract. Returns a summary suitable for cron-
    log inspection."""

    def __init__(
        self,
        extractor: PaperExtractor,
        store: ExtractStore | None = None,
        http_get: HttpGet | None = None,
    ) -> None:
        self.extractor = extractor
        self.store = store or ExtractStore()
        self.http_get = http_get or _default_http_get

    def run(self, feed_configs: list[FeedConfig]) -> IngestRunSummary:
        summary = IngestRunSummary(feeds_total=len(feed_configs))
        for feed in feed_configs:
            try:
                fetcher = build_fetcher(feed.adapter, http_get=self.http_get)
            except KeyError:
                logger.warning(
                    "feed %r adapter %r unknown — skipping",
                    feed.name, feed.adapter,
                )
                summary.feeds_failed += 1
                continue
            papers = fetcher.fetch(feed)
            if not papers:
                summary.feeds_failed += 1
                continue
            for paper in papers:
                summary.papers_seen += 1
                if self.store.has(paper):
                    summary.papers_skipped_duplicate += 1
                    continue
                try:
                    body = self.extractor.extract(paper)
                except Exception as exc:
                    logger.exception(
                        "extraction failed for paper %r from %r",
                        paper.title, feed.name,
                    )
                    _ = exc  # suppress unused-name in ruff under noqa-friendly form
                    summary.papers_extract_failed += 1
                    continue
                path = self.store.write(paper, body)
                summary.extract_paths.append(path)
                summary.papers_extracted += 1
        return summary
