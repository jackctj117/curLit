"""BoE, BoJ, BoC scrapers — central bank document parsing."""

import re
from datetime import datetime

from bs4 import BeautifulSoup

from .base import CBScraper, Document


class BoEStatementScraper(CBScraper):
    cb_name = "boe"
    BASE = "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes"

    def list_documents(self, since: datetime) -> list[dict]:
        docs = []
        try:
            html = self.fetch_url(self.BASE)
            soup = BeautifulSoup(html, "html.parser")
            for item in soup.select("li.list-item, a[href*='monetary-policy'], article"):
                link = item.find("a") if item.name != "a" else item
                if link is None:
                    continue
                href = link.get("href", "")
                title = link.get_text(strip=True)
                if "monetary policy" not in title.lower() and "mpc" not in title.lower():
                    continue
                date_str = ""
                date_el = item.find("time") or item.find("span", class_="date")
                if date_el:
                    date_str = date_el.get("datetime", "") or date_el.get_text(strip=True)
                try:
                    d = datetime.fromisoformat(date_str[:10]) if date_str else datetime.utcnow()
                except ValueError:
                    d = datetime.utcnow()
                if d < since:
                    continue
                url = f"https://www.bankofengland.co.uk{href}" if href.startswith("/") else href
                docs.append({"url": url, "date": d, "doc_type": "minutes", "title": title})
        except Exception:
            pass
        return docs

    def parse_document(self, html: str, meta: dict) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup.find("div", class_="content") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else html[:2000]
        return Document(cb="boe", doc_type=meta["doc_type"], title=meta["title"],
                        date=meta["date"], url=meta["url"], raw_html=html, raw_text=text)


class BoJStatementScraper(CBScraper):
    cb_name = "boj"
    BASE = "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm"

    def list_documents(self, since: datetime) -> list[dict]:
        docs = []
        try:
            html = self.fetch_url(self.BASE)
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.select("a[href*='mopo']"):
                href = link.get("href", "")
                title = link.get_text(strip=True)
                d = datetime.utcnow()
                m = re.search(r"(\d{4})", title)
                if m:
                    d = datetime(int(m.group(1)), 1, 1)
                if d < since:
                    continue
                if href.startswith("/"):
                    href = f"https://www.boj.or.jp{href}"
                docs.append({"url": href, "date": d, "doc_type": "statement", "title": title})
        except Exception:
            pass
        return docs

    def parse_document(self, html: str, meta: dict) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("div", id="contents") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else html[:2000]
        return Document(cb="boj", doc_type=meta["doc_type"], title=meta["title"],
                        date=meta["date"], url=meta["url"], raw_html=html, raw_text=text)


class BoCStatementScraper(CBScraper):
    cb_name = "boc"
    BASE = "https://www.bankofcanada.ca/news/"

    def list_documents(self, since: datetime) -> list[dict]:
        docs = []
        try:
            html = self.fetch_url(self.BASE)
            soup = BeautifulSoup(html, "html.parser")
            for item in soup.select("article, .post, li.news-item"):
                link = item.find("a")
                if not link:
                    continue
                title = link.get_text(strip=True)
                if "rate" not in title.lower() and "monetary" not in title.lower():
                    continue
                href = link.get("href", "")
                date_el = item.find("time") or item.find("span", class_="date")
                d = datetime.utcnow()
                if date_el:
                    date_str = date_el.get("datetime", "") or date_el.get_text(strip=True)
                    try:
                        d = datetime.fromisoformat(date_str[:10])
                    except ValueError:
                        pass
                if d < since:
                    continue
                docs.append({"url": href if href.startswith("http") else f"https://www.bankofcanada.ca{href}",
                              "date": d, "doc_type": "statement", "title": title})
        except Exception:
            pass
        return docs

    def parse_document(self, html: str, meta: dict) -> Document:
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("article") or soup.find("div", class_="entry-content") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else html[:2000]
        return Document(cb="boc", doc_type=meta["doc_type"], title=meta["title"],
                        date=meta["date"], url=meta["url"], raw_html=html, raw_text=text)
