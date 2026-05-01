"""Central bank document scraper — abstract base class."""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class Document:
    cb: str
    doc_type: str
    title: str
    date: datetime
    url: str
    speaker: str | None = None
    raw_html: str = ""
    raw_text: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        h = hashlib.sha256(f"{self.cb}:{self.url}".encode()).hexdigest()[:16]
        return f"{self.cb}_{self.doc_type}_{self.date.strftime('%Y%m%d')}_{h}"


class CBScraper(ABC):
    def __init__(self, raw_dir: Path, timeout: int = 30) -> None:
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (research scraper curLit)"},
            follow_redirects=True,
        )

    @property
    @abstractmethod
    def cb_name(self) -> str: ...

    @abstractmethod
    def list_documents(self, since: datetime) -> list[dict]:
        """Return list of {url, date, doc_type, title, speaker?} dicts."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def fetch_url(self, url: str) -> str:
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.text

    @abstractmethod
    def parse_document(self, html: str, meta: dict) -> Document: ...

    def save_document(self, doc: Document) -> Path:
        path = self.raw_dir / f"{doc.doc_id}.json"
        path.write_text(json.dumps({
            "cb": doc.cb,
            "doc_type": doc.doc_type,
            "title": doc.title,
            "date": doc.date.isoformat(),
            "url": doc.url,
            "speaker": doc.speaker,
            "raw_text": doc.raw_text,
            "metadata": doc.metadata,
        }, indent=2))
        return path

    def run(self, since: datetime) -> list[Document]:
        docs = []
        for meta in self.list_documents(since):
            stem = f"{self.cb_name}_{meta['doc_type']}_{meta['date'].strftime('%Y%m%d')}"
            existing = list(self.raw_dir.glob(f"{stem}_*.json"))
            if existing:
                logger.debug("Skipping already-fetched %s", stem)
                continue
            try:
                html = self.fetch_url(meta["url"])
                doc = self.parse_document(html, meta)
                self.save_document(doc)
                docs.append(doc)
                logger.info("Saved %s", doc.doc_id)
            except Exception:
                logger.exception("Failed %s", meta["url"])
        return docs
