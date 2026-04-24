"""Fed scrapper — FOMC statements, minutes, and speeches from federalreserve.gov."""

import re
from datetime import datetime

from bs4 import BeautifulSoup

from .base import CBScraper, Document


class FedStatementScraper(CBScraper):
    cb_name = "fed"
    CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def list_documents(self, since: datetime) -> list[dict]:
        html = self.fetch_url(self.CALENDAR_URL)
        soup = BeautifulSoup(html, "html.parser")
        docs = []
        for link in soup.find_all("a", href=re.compile(r"monetary\d+a\.htm")):
            href = link.get("href", "")
            m = re.search(r"monetary(\d{8})a\.htm", href)
            if not m:
                continue
            date = datetime.strptime(m.group(1), "%Y%m%d")
            if date < since:
                continue
            url = f"https://www.federalreserve.gov{href}" if href.startswith("/") else href
            docs.append({"url": url, "date": date, "doc_type": "statement",
                          "title": f"FOMC Statement {date.strftime('%Y-%m-%d')}"})
        return docs

    def parse_document(self, html: str, meta: dict) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        content = soup.find("div", id="article") or soup.find("div", class_="col-md-8") or soup.find("body")
        for tag in content.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()
        paragraphs = [
            p.get_text(strip=True) for p in content.find_all("p")
            if p.get_text(strip=True) and "For release" not in p.get_text()
        ]
        return Document(
            cb="fed", doc_type=meta["doc_type"], title=meta["title"],
            date=meta["date"], url=meta["url"], raw_html=html,
            raw_text="\n\n".join(paragraphs),
        )
