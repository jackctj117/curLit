"""BoE, BoJ, BoC scrapers — minimal stubs; extend with site-specific parsing."""

from datetime import datetime

from .base import CBScraper, Document


class BoEStatementScraper(CBScraper):
    cb_name = "boe"
    BASE = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes"

    def list_documents(self, since: datetime) -> list[dict]:
        return []

    def parse_document(self, html: str, meta: dict) -> Document:
        return Document(cb="boe", doc_type=meta["doc_type"], title=meta["title"],
                        date=meta["date"], url=meta["url"], raw_html=html, raw_text="")


class BoJStatementScraper(CBScraper):
    cb_name = "boj"
    BASE = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"

    def list_documents(self, since: datetime) -> list[dict]:
        return []

    def parse_document(self, html: str, meta: dict) -> Document:
        return Document(cb="boj", doc_type=meta["doc_type"], title=meta["title"],
                        date=meta["date"], url=meta["url"], raw_html=html, raw_text="")


class BoCStatementScraper(CBScraper):
    cb_name = "boc"

    def list_documents(self, since: datetime) -> list[dict]:
        return []

    def parse_document(self, html: str, meta: dict) -> Document:
        return Document(cb="boc", doc_type=meta["doc_type"], title=meta["title"],
                        date=meta["date"], url=meta["url"], raw_html=html, raw_text="")
