"""Fed scrapper — FOMC statements, minutes, and speeches from federalreserve.gov."""

import logging
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from .base import CBScraper, Document

logger = logging.getLogger(__name__)


class FedStatementScraper(CBScraper):
    cb_name = "fed"
    # Current-year calendar plus historical archive years.
    CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    HISTORICAL_URL_FMT = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"

    def list_documents(self, since: datetime) -> list[dict[str, Any]]:
        """List FOMC statement docs since `since` across current + historical pages.

        The current calendar page covers ~current + a couple recent years; older
        years (each with ~8 FOMC statements) live at fomchistorical{year}.htm.
        We walk every year from `since.year` to current-year-1 to avoid double-
        counting (the current calendar already covers current year). Dedup on
        URL since some years overlap between sources.
        """
        urls_seen: set[str] = set()
        docs: list[dict[str, Any]] = []

        # Pages to crawl: current calendar + each historical year page.
        pages: list[str] = [self.CALENDAR_URL]
        current_year = datetime.utcnow().year
        for year in range(since.year, current_year):
            pages.append(self.HISTORICAL_URL_FMT.format(year=year))

        for page_url in pages:
            try:
                html = self.fetch_url(page_url)
            except Exception:
                logger.exception("Fed scraper: failed to fetch %s", page_url)
                continue

            soup = BeautifulSoup(html, "html.parser")
            # Match both /newsevents/pressreleases/monetaryYYYYMMDDa.htm (modern)
            # and historical /monetarypolicy/files/monetaryYYYYMMDDa.htm patterns.
            for link in soup.find_all("a", href=re.compile(r"monetary\d{8}a\.htm")):
                href = str(link.get("href") or "")
                m = re.search(r"monetary(\d{8})a\.htm", href)
                if not m:
                    continue
                date = datetime.strptime(m.group(1), "%Y%m%d")
                if date < since:
                    continue
                url = f"https://www.federalreserve.gov{href}" if href.startswith("/") else href
                if url in urls_seen:
                    continue
                urls_seen.add(url)
                docs.append(
                    {
                        "url": url,
                        "date": date,
                        "doc_type": "statement",
                        "title": f"FOMC Statement {date.strftime('%Y-%m-%d')}",
                    }
                )

        return docs

    def parse_document(self, html: str, meta: dict[str, Any]) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        content = (
            soup.find("div", id="article")
            or soup.find("div", class_="col-md-8")
            or soup.find("body")
        )
        if content is None:
            return Document(
                cb="fed",
                doc_type=meta["doc_type"],
                title=meta["title"],
                date=meta["date"],
                url=meta["url"],
                raw_html=html,
                raw_text="",
            )
        for tag in content.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()
        paragraphs = [
            p.get_text(strip=True)
            for p in content.find_all("p")
            if p.get_text(strip=True) and "For release" not in p.get_text()
        ]
        return Document(
            cb="fed",
            doc_type=meta["doc_type"],
            title=meta["title"],
            date=meta["date"],
            url=meta["url"],
            raw_html=html,
            raw_text="\n\n".join(paragraphs),
        )
