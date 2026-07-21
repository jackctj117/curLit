"""Polymarket fetcher — pulls top FX/macro-relevant prediction markets
and surfaces their current implied probabilities as research extracts.

Polymarket is a derivative data source: each market is "Will X happen
by date Y?" with a price 0–100 expressing market-implied probability.
For a research-pipeline, the value is **directional**: when a market
on "Fed cuts 25bps in June" is at 78%, that's information about
where the price action is likely to lean. The paper_extractor
agent reads each formatted market as if it were a paper abstract;
the idea agent decides whether the market's implications support a
falsifiable single-asset / cross-symbol trading hypothesis.

This is the v1 "text extract" integration. A future v2 would ingest
market price time-series into the DataProvider so strategies can
reference them as features (``symbols=['EURUSD', 'POLY:fed-cut-jun']``).
That's filed as CL-... when scoped.

API: Polymarket's public Gamma GraphQL endpoint at
``https://gamma-api.polymarket.com/markets``. No auth required for
read-only queries; rate-limited to a few queries per second. We
filter to FX/macro categories at fetch time.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.research.ingest import HttpGet, Paper

logger = logging.getLogger(__name__)


DEFAULT_POLYMARKET_API: str = "https://gamma-api.polymarket.com/markets"

# Keywords that flag a market as FX/macro-relevant. Markets that
# match any of these in title/description get included; others are
# filtered out at fetch time so we don't burn extractor budget on
# sports/election-trivia/crypto markets that don't move FX.
_FX_MACRO_KEYWORDS: tuple[str, ...] = (
    "fed", "fomc", "rate cut", "rate hike", "inflation", "cpi", "ppi",
    "ecb", "boj", "bank of japan", "bank of england", "boe",
    "treasury", "yield curve", "recession", "gdp", "unemployment",
    "nonfarm", "nfp", "jobs report", "trade deficit", "tariff",
    "election", "central bank", "monetary policy", "interest rate",
    "dollar", "euro", "yen", "pound", "currency",
)


@dataclass
class PolymarketFetcher:
    """Pulls active Polymarket markets via the public Gamma API,
    filters to FX/macro-relevant, formats each as a Paper record so
    the rest of the pipeline (extractor → idea agent) treats them
    uniformly with arXiv papers and substack posts."""

    http_get: HttpGet = field(default=lambda url: httpx.get(
        url, timeout=30.0, follow_redirects=True,
    ).text)
    keywords: tuple[str, ...] = field(default=_FX_MACRO_KEYWORDS)
    max_markets: int = 50

    def fetch(self, feed: object) -> list[Paper]:  # FeedConfig at runtime
        # The feed.query_url should point at the Gamma API with any
        # query params the operator wants (e.g. limit, active filter).
        url = getattr(feed, "query_url", DEFAULT_POLYMARKET_API)
        source_label = getattr(feed, "source_label", "Polymarket")
        try:
            body = self.http_get(url)
        except Exception as exc:
            logger.warning(
                "polymarket fetch failed: %s: %s",
                type(exc).__name__, exc,
            )
            return []
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("polymarket JSON decode failed: %s", exc)
            return []
        markets = data if isinstance(data, list) else data.get("data", [])
        if not isinstance(markets, list):
            logger.warning(
                "polymarket response unexpected shape: %s",
                type(markets).__name__,
            )
            return []
        return self._format_markets(markets, source_label)

    def _format_markets(
        self, markets: list[dict[str, Any]], source_label: str,
    ) -> list[Paper]:
        out: list[Paper] = []
        for m in markets[: self.max_markets]:
            if not isinstance(m, dict):
                continue
            title = str(m.get("question") or m.get("title") or "").strip()
            if not title:
                continue
            description = str(m.get("description") or "").strip()
            haystack = (title + " " + description).lower()
            if not any(kw in haystack for kw in self.keywords):
                continue
            # Build a tight extract-shaped abstract that the
            # paper_extractor prompt can handle. Polymarket markets
            # have current price = market-implied probability; turn
            # that into a single quantitative claim.
            outcome_prices = m.get("outcomePrices") or []
            outcomes = m.get("outcomes") or []
            price_summary = self._format_outcomes(outcomes, outcome_prices)
            volume = m.get("volume") or m.get("volumeNum") or 0
            end_date = (
                m.get("endDate") or m.get("end_date_iso") or ""
            )[:10]
            abstract = (
                f"Market: {title}\n"
                f"End date: {end_date}\n"
                f"Volume traded: ${self._format_volume(volume)}\n"
                f"Current implied probabilities: {price_summary}\n\n"
                f"Description: {description[:1500]}"
            )
            # Polymarket's user-facing URL is /event/<slug>; the old /market/
            # path 404s (CL-7j3k).
            url = (
                f"https://polymarket.com/event/"
                f"{m.get('slug', m.get('id', ''))}"
            )
            out.append(Paper(
                title=title,
                authors=(),
                year=PolymarketFetcher._year_from_iso(end_date),
                url=url,
                doi="",
                abstract=abstract[:4000],
                source_label=source_label,
            ))
        return out

    @staticmethod
    def _format_outcomes(outcomes: object, prices: object) -> str:
        # Polymarket returns outcomes + outcomePrices as either lists
        # or JSON-encoded strings depending on endpoint. Normalize.
        def _coerce_list(x: object) -> list[Any]:
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                try:
                    parsed = json.loads(x)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            return []

        outs = _coerce_list(outcomes)
        prs = _coerce_list(prices)
        if len(outs) != len(prs):
            return "(price/outcome shape mismatch)"
        try:
            pairs = [
                f"{out}={float(pr)*100:.1f}%"
                for out, pr in zip(outs, prs, strict=False)
            ]
        except (ValueError, TypeError):
            return "(unparseable)"
        return ", ".join(pairs) if pairs else "(none)"

    @staticmethod
    def _format_volume(v: object) -> str:
        try:
            n = float(v)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return "0"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return f"{n:.0f}"

    @staticmethod
    def _year_from_iso(iso: str) -> int | None:
        match = re.match(r"^(\d{4})", iso)
        return int(match.group(1)) if match else None


def register() -> None:
    """Register the Polymarket adapter under the ``polymarket`` adapter
    name so paper_streams.yaml entries can use it."""
    from src.research.ingest import _FETCHER_REGISTRY  # noqa: PLC0415
    _FETCHER_REGISTRY["polymarket"] = (
        lambda http_get: PolymarketFetcher(http_get=http_get)
    )


# Self-register on import.
register()


