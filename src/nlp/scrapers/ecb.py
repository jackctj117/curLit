"""ECB scrapper — monetary policy statements, press conferences, and speeches."""

from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from .base import CBScraper, Document


class ECBStatementScraper(CBScraper):
    cb_name = "ecb"
    PRESS_URL = "https://www.ecb.europa.eu/press/pr/date/html/index.en.html"

    def list_documents(self, since: datetime) -> list[dict[str, Any]]:
        html = self.fetch_url(self.PRESS_URL)
        soup = BeautifulSoup(html, "html.parser")
        docs = []
        for item in soup.select("div.date, dt, .doc-title"):
            link = item.find("a")
            if not link:
                continue
            href = str(link.get("href") or "")
            title = link.get_text(strip=True)
            if "monetary policy" not in title.lower():
                continue
            date_str = str(item.get("data-date") or "")
            try:
                d = datetime.fromisoformat(date_str[:10]) if date_str else datetime.utcnow()
            except ValueError:
                continue
            if d < since:
                continue
            url = f"https://www.ecb.europa.eu{href}" if href.startswith("/") else href
            docs.append({"url": url, "date": d, "doc_type": "statement", "title": title})
        return docs

    def parse_document(self, html: str, meta: dict[str, Any]) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else ""
        return Document(
            cb="ecb", doc_type=meta["doc_type"], title=meta["title"],
            date=meta["date"], url=meta["url"], raw_html=html, raw_text=text,
        )
