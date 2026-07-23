"""BoE, BoJ, BoC scrapers — central bank document parsing."""

import contextlib
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from .base import CBScraper, Document


class BoEStatementScraper(CBScraper):
    cb_name = "boe"
    BASE = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes"
    # CL-qdns: per-year archives. The summary-and-minutes URL doesn't
    # accept a Year query param (returns 302→404). The /news/news
    # endpoint with the "Monetary Policy" Taxonomies GUID does — same
    # listing data, different route. Verified live 2026-05-04.
    HISTORICAL_URL_FMT = (
        "https://www.bankofengland.co.uk/news/news"
        "?Taxonomies=ce90163e489841e0b66d06243d35d5cb"
        "&NewsTypes=ce90163e489841e0b66d06243d35d5cb"
        "&Direction=Latest&InfiniteScrolling=False&Page=1&Year={year}"
    )

    def list_documents(self, since: datetime) -> list[dict[str, Any]]:
        urls_seen: set[str] = set()
        docs: list[dict[str, Any]] = []
        pages = [self.BASE]
        current_year = datetime.utcnow().year
        for year in range(since.year, current_year):
            pages.append(self.HISTORICAL_URL_FMT.format(year=year))

        for page_url in pages:
            try:
                html = self.fetch_url(page_url)
            except Exception:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for item in soup.select("li.list-item, a[href*='monetary-policy'], article"):
                link = item.find("a") if item.name != "a" else item
                if link is None:
                    continue
                href = str(link.get("href") or "")
                title = link.get_text(strip=True)
                if "monetary policy" not in title.lower() and "mpc" not in title.lower():
                    continue
                date_str = ""
                date_el = item.find("time") or item.find("span", class_="date")
                if date_el:
                    date_str = str(date_el.get("datetime", "") or date_el.get_text(strip=True))
                try:
                    d = datetime.fromisoformat(date_str[:10]) if date_str else datetime.utcnow()
                except ValueError:
                    d = datetime.utcnow()
                if d < since:
                    continue
                url = f"https://www.bankofengland.co.uk{href}" if href.startswith("/") else href
                if url in urls_seen:
                    continue
                urls_seen.add(url)
                docs.append({"url": url, "date": d, "doc_type": "minutes", "title": title})
        return docs

    def parse_document(self, html: str, meta: dict[str, Any]) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup.find("div", class_="content") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else html[:2000]
        return Document(
            cb="boe",
            doc_type=meta["doc_type"],
            title=meta["title"],
            date=meta["date"],
            url=meta["url"],
            raw_html=html,
            raw_text=text,
        )


class BoJStatementScraper(CBScraper):
    cb_name = "boj"
    BASE = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"
    # CL-qdns: BoJ archives by year — separate page per calendar year.
    HISTORICAL_URL_FMT = "https://www.boj.or.jp/en/mopo/mpmsche_minu/minu_{year}/index.htm"

    def list_documents(self, since: datetime) -> list[dict[str, Any]]:
        urls_seen: set[str] = set()
        docs: list[dict[str, Any]] = []
        pages = [self.BASE]
        current_year = datetime.utcnow().year
        for year in range(since.year, current_year):
            pages.append(self.HISTORICAL_URL_FMT.format(year=year))

        for page_url in pages:
            try:
                html = self.fetch_url(page_url)
            except Exception:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.select("a[href*='mopo']"):
                href = str(link.get("href") or "")
                title = link.get_text(strip=True)
                d = datetime.utcnow()
                m = re.search(r"(\d{4})", title)
                if m:
                    d = datetime(int(m.group(1)), 1, 1)
                if d < since:
                    continue
                if href.startswith("/"):
                    href = f"https://www.boj.or.jp{href}"
                if href in urls_seen:
                    continue
                urls_seen.add(href)
                docs.append({"url": href, "date": d, "doc_type": "statement", "title": title})
        return docs

    def parse_document(self, html: str, meta: dict[str, Any]) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("div", id="contents") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else html[:2000]
        return Document(
            cb="boj",
            doc_type=meta["doc_type"],
            title=meta["title"],
            date=meta["date"],
            url=meta["url"],
            raw_html=html,
            raw_text=text,
        )


class BoCStatementScraper(CBScraper):
    cb_name = "boc"
    BASE = "https://www.bankofcanada.ca/news/"
    # CL-qdns: BoC archives via year query parameter on news listing.
    HISTORICAL_URL_FMT = (
        "https://www.bankofcanada.ca/news/?mtm_search_filter=monetary-policy&date_year={year}"
    )

    def list_documents(self, since: datetime) -> list[dict[str, Any]]:
        urls_seen: set[str] = set()
        docs: list[dict[str, Any]] = []
        pages = [self.BASE]
        current_year = datetime.utcnow().year
        for year in range(since.year, current_year):
            pages.append(self.HISTORICAL_URL_FMT.format(year=year))

        for page_url in pages:
            try:
                html = self.fetch_url(page_url)
            except Exception:
                continue
            soup = BeautifulSoup(html, "html.parser")
            for item in soup.select("article, .post, li.news-item"):
                link = item.find("a")
                if not link:
                    continue
                title = link.get_text(strip=True)
                if "rate" not in title.lower() and "monetary" not in title.lower():
                    continue
                href = str(link.get("href") or "")
                date_el = item.find("time") or item.find("span", class_="date")
                d = datetime.utcnow()
                if date_el:
                    date_str = str(date_el.get("datetime", "") or date_el.get_text(strip=True))
                    with contextlib.suppress(ValueError):
                        d = datetime.fromisoformat(date_str[:10])
                if d < since:
                    continue
                url = href if href.startswith("http") else f"https://www.bankofcanada.ca{href}"
                if url in urls_seen:
                    continue
                urls_seen.add(url)
                docs.append({"url": url, "date": d, "doc_type": "statement", "title": title})
        return docs

    def parse_document(self, html: str, meta: dict[str, Any]) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("article") or soup.find("div", class_="entry-content") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else html[:2000]
        return Document(
            cb="boc",
            doc_type=meta["doc_type"],
            title=meta["title"],
            date=meta["date"],
            url=meta["url"],
            raw_html=html,
            raw_text=text,
        )
