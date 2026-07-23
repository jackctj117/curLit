"""Polymarket live order-book + market data (CL-poly-1).

Read-only access to Polymarket's CLOB and Gamma APIs. No auth required
for market data. Used by:
  * The paper broker (CL-poly-2) — pulls order books to model fills.
  * Strategies that want a freshly-quoted price rather than the
    nightly-seeded historical close from the prices table.
  * The live broker (CL-poly-3) for symbol resolution
    (POLY:condition_id:outcome → token_id).

API endpoints used:
  GET https://clob.polymarket.com/book?token_id=...   — full book
  GET https://clob.polymarket.com/midpoint?market=... — midpoint price
  GET https://gamma-api.polymarket.com/markets/{id}   — market metadata

Reference: Plan §1.

This module deliberately does NOT support order placement. Phase 3
(CL-poly-3) adds the EIP-712 + py-clob-client write side; this Phase 1
module is reads-only forever.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# 10s timeout for all REST calls. Polymarket's CLOB usually responds in
# <500ms; 10s is a generous ceiling that catches a stuck connection
# without hanging the strategy loop indefinitely. Strategies running
# at 5min cadence have plenty of headroom.
_DEFAULT_TIMEOUT_SEC: float = 10.0

# Production hosts. Amoy testnet uses a different host (added in
# CL-poly-3). For market data, mainnet is the only relevant one
# because there's no real liquidity to scrape on testnet.
_CLOB_HOST: str = "https://clob.polymarket.com"
_GAMMA_HOST: str = "https://gamma-api.polymarket.com"


# Tiny HTTP shim — tests inject a fake. Real production uses httpx.get.
HttpGetJson = Callable[[str, dict[str, str]], dict[str, Any]]


def _default_http_get_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    resp = httpx.get(url, params=params, timeout=_DEFAULT_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


@dataclass(frozen=True)
class BookLevel:
    """One side of one price level. Polymarket returns asks/bids as
    arrays of these."""

    price: Decimal
    size: Decimal


@dataclass(frozen=True)
class OrderBookSnapshot:
    """A single point-in-time book snapshot. Bids sorted desc, asks asc."""

    token_id: str
    bids: list[BookLevel]
    asks: list[BookLevel]

    @property
    def top_bid(self) -> BookLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def top_ask(self) -> BookLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        """Midpoint between top bid and top ask, or None if either side
        is empty (no two-sided market)."""
        if self.top_bid is None or self.top_ask is None:
            return None
        return (self.top_bid.price + self.top_ask.price) / Decimal("2")

    def depth_at_or_better(self, price: Decimal, side: str) -> Decimal:
        """Aggregate resting size at or better than the given price.

        Used by the cost model + paper broker to estimate slippage and
        fill probability. "At or better" means: for a BUY, the asks at
        or below price; for a SELL, the bids at or above.
        """
        side_u = side.upper()
        if side_u == "BUY":
            return sum(
                (lvl.size for lvl in self.asks if lvl.price <= price),
                start=Decimal("0"),
            )
        if side_u == "SELL":
            return sum(
                (lvl.size for lvl in self.bids if lvl.price >= price),
                start=Decimal("0"),
            )
        msg = f"side must be 'BUY' or 'SELL', got {side!r}"
        raise ValueError(msg)


class PolymarketDataSource:
    """Read-only CLOB + Gamma client. Stateless; constructor takes a
    custom http_get for tests."""

    def __init__(
        self,
        http_get_json: HttpGetJson | None = None,
        clob_host: str = _CLOB_HOST,
        gamma_host: str = _GAMMA_HOST,
    ) -> None:
        self.http_get_json = http_get_json or _default_http_get_json
        self.clob_host = clob_host.rstrip("/")
        self.gamma_host = gamma_host.rstrip("/")

    def get_book(self, token_id: str) -> OrderBookSnapshot:
        """Pull the full order book for one token_id.

        Polymarket returns ``{"asks": [{"price": "0.42", "size": "100"}, ...],
        "bids": [...]}``. Strings parse cleanly into Decimal — preserving
        precision matters for tick-aligned (0.01) prices.
        """
        url = f"{self.clob_host}/book"
        data = self.http_get_json(url, {"token_id": token_id})
        return _parse_book(token_id, data)

    def get_midpoint(self, token_id: str) -> Decimal | None:
        """Polymarket's own midpoint endpoint. Returns None when the
        market has no two-sided book (one side empty)."""
        url = f"{self.clob_host}/midpoint"
        try:
            data = self.http_get_json(url, {"market": token_id})
        except Exception:
            logger.warning(
                "polymarket midpoint fetch failed for %s",
                token_id,
                exc_info=True,
            )
            return None
        mid = data.get("mid")
        if mid is None:
            return None
        try:
            return Decimal(str(mid))
        except Exception:
            logger.warning("polymarket midpoint parse failed: %r", mid)
            return None

    def get_market_metadata(self, condition_id: str) -> dict[str, Any]:
        """Pull /markets/{condition_id} from Gamma. Includes the
        ``clobTokenIds`` field — the YES/NO token_ids the CLOB API
        addresses by. Used for symbol resolution."""
        url = f"{self.gamma_host}/markets/{condition_id}"
        return self.http_get_json(url, {})

    def resolve_symbol(self, symbol: str) -> str:
        """Resolve a curLit symbol to a Polymarket CLOB token_id.

        Accepts three forms:
          * ``POLY:<token_id>`` — already a token; passthrough
          * ``POLY:<condition_id>:<outcome_idx>`` — looked up via Gamma;
            outcome_idx 0 = YES, 1 = NO by Polymarket convention.
          * Anything else → ValueError.
        """
        if not symbol.startswith("POLY:"):
            msg = f"not a POLY symbol: {symbol!r}"
            raise ValueError(msg)
        body = symbol[len("POLY:") :]
        parts = body.split(":")
        if len(parts) == 1:
            return parts[0]  # already a token_id
        if len(parts) == 2:
            condition_id, outcome_idx_str = parts
            try:
                outcome_idx = int(outcome_idx_str)
            except ValueError as exc:
                msg = f"outcome index must be int, got {outcome_idx_str!r} in {symbol!r}"
                raise ValueError(msg) from exc
            meta = self.get_market_metadata(condition_id)
            tokens_raw = meta.get("clobTokenIds")
            tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
            if not isinstance(tokens, list) or outcome_idx >= len(tokens):
                msg = (
                    f"market {condition_id} has no token at outcome index "
                    f"{outcome_idx}; got tokens={tokens!r}"
                )
                raise ValueError(msg)
            return str(tokens[outcome_idx])
        msg = f"unrecognized POLY symbol shape: {symbol!r}"
        raise ValueError(msg)


def _parse_book(token_id: str, data: dict[str, Any]) -> OrderBookSnapshot:
    bids = [
        BookLevel(price=Decimal(str(b["price"])), size=Decimal(str(b["size"])))
        for b in data.get("bids", [])
    ]
    asks = [
        BookLevel(price=Decimal(str(a["price"])), size=Decimal(str(a["size"])))
        for a in data.get("asks", [])
    ]
    # Polymarket returns bids in descending price order and asks in
    # ascending. We re-sort defensively in case API output drifts.
    bids.sort(key=lambda lvl: lvl.price, reverse=True)
    asks.sort(key=lambda lvl: lvl.price)
    return OrderBookSnapshot(token_id=token_id, bids=bids, asks=asks)
