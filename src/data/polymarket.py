"""Polymarket price-history ingester (CL-3t4j v2).

Pulls per-market implied-probability time-series from Polymarket's
Gamma API and writes them to the existing ``prices`` table with
synthetic symbol ``POLY:<slug>``. Strategies declare those symbols
the same way they declare ``EURUSD`` — DataProvider.get_aligned_series
returns one column per symbol, walk-forward harness handles the rest.

Storage decision (from CL-3t4j design): reuse the ``prices`` table
rather than introducing a new ``polymarket_markets`` table. The
``close`` column carries the implied probability in [0, 1]. This
isn't semantically a price, but the existing ingest+query+harness
plumbing handles it without modification, and strategies can read
``data['POLY:fed-cut-jun-2026']`` as a probability series with no
new code paths. The downside — ``pct_change()`` on probabilities
yields odd "returns" — is irrelevant because POLY symbols are
expected to be FEATURE inputs, not the strategy's
``execution_symbol``. The walk-forward harness only computes returns
on execution_symbol's column (CL-40n2 v1), so POLY columns stay
read-only feature inputs.

Operator workflow:
  1. Curate a list of markets in ``configs/polymarket_markets.yaml``
     (condition_id + human slug + optional FX-relevance note).
  2. Run ``python -m scripts.seed_polymarket_history`` daily (or
     wire into the existing seed_historical_data.py orchestrator).
  3. Strategies declare ``symbols=['EURUSD', 'POLY:fed-cut-jun-2026']``
     and use the cross-symbol signal pattern.

Gamma API endpoint used:
  GET https://clob.polymarket.com/prices-history
      ?market=<token_id>&interval=1d&fidelity=10

Returns ``{"history": [{"t": <unix_ts>, "p": <price>}, ...]}``.
Free + rate-limited; no auth.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

from src.data.base import BaseIngester

logger = logging.getLogger(__name__)


DEFAULT_API: str = "https://clob.polymarket.com/prices-history"
DEFAULT_TIMEOUT_SEC: float = 30.0
DEFAULT_CONFIG_PATH: Path = Path("configs/polymarket_markets.yaml")


# Type alias for the HTTP shim. Tests inject a canned-response fn.
HttpGetJson = Callable[[str, dict[str, str]], dict[str, Any]]


def _default_http_get_json(url: str, params: dict[str, str]) -> dict[str, Any]:
    """Production HTTP GET returning parsed JSON. Raises on non-2xx."""
    resp = httpx.get(
        url, params=params, timeout=DEFAULT_TIMEOUT_SEC,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def load_market_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
) -> list[dict[str, str]]:
    """Load the curated list of markets to track. Each entry must have
    ``token_id`` (Polymarket's CLOB token identifier — used by the
    prices-history API) and ``slug`` (human readable, becomes the
    suffix of the synthetic symbol). Optional fields are preserved
    on the raw dict for the operator's reference."""
    p = Path(path)
    if not p.exists():
        msg = f"polymarket markets config not found at {p}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict) or "markets" not in raw:
        msg = f"polymarket markets config at {p} missing 'markets' key"
        raise ValueError(msg)
    markets = raw["markets"]
    if not isinstance(markets, list):
        msg = f"'markets' in {p} must be a list"
        raise ValueError(msg)
    out: list[dict[str, str]] = []
    for m in markets:
        if not isinstance(m, dict):
            continue
        if "token_id" not in m or "slug" not in m:
            logger.warning(
                "skipping market entry missing token_id or slug: %s", m,
            )
            continue
        out.append(m)
    return out


def to_symbol(slug: str) -> str:
    """Synthesize the DataProvider symbol from a Polymarket slug.
    Caps at 64 chars total so the ``prices.symbol`` column doesn't
    blow out (the downstream consumer is varchar)."""
    full = f"POLY:{slug}"
    return full[:64]


class PolymarketHistoryIngester(BaseIngester):
    """Pulls per-market probability history and upserts to ``prices``.

    Each tracked market becomes one symbol; each daily snapshot
    becomes one row. The ``close`` column carries the implied
    probability in [0, 1] — strategies read it directly via
    ``data[symbol]`` and use threshold logic like ``> 0.70``.
    """

    def __init__(
        self,
        db_url: str,
        markets: list[dict[str, str]] | None = None,
        http_get_json: HttpGetJson | None = None,
        api_url: str = DEFAULT_API,
    ) -> None:
        super().__init__(db_url, "polymarket")
        self.markets = markets if markets is not None else load_market_config()
        self.http_get_json = http_get_json or _default_http_get_json
        self.api_url = api_url

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch price history for every configured market in the
        window. Returns a long-format DataFrame the transform step
        normalizes to the prices-table schema."""
        rows: list[dict[str, object]] = []
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        for m in self.markets:
            token_id = m["token_id"]
            slug = m["slug"]
            try:
                data = self.http_get_json(
                    self.api_url,
                    {
                        "market": token_id,
                        "interval": "1d",
                        "fidelity": "10",
                        "startTs": str(start_ts),
                        "endTs": str(end_ts),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "polymarket history fetch failed for %s: %s: %s",
                    slug, type(exc).__name__, exc,
                )
                continue
            history = data.get("history", []) if isinstance(data, dict) else []
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                ts = entry.get("t")
                price = entry.get("p")
                if ts is None or price is None:
                    continue
                try:
                    rows.append({
                        "ts": datetime.fromtimestamp(int(ts), tz=UTC),
                        "symbol": to_symbol(slug),
                        "close": float(price),
                    })
                except (ValueError, TypeError) as exc:
                    logger.debug(
                        "polymarket: skipping malformed entry for %s: %s",
                        slug, exc,
                    )
                    continue
        return pd.DataFrame(rows)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return raw
        df = raw.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["source"] = self.source
        # OHLCV columns the prices schema expects — Polymarket has
        # neither volume nor OHLC at this granularity, so zero out.
        df["open"] = 0.0
        df["high"] = 0.0
        df["low"] = 0.0
        df["volume"] = 0.0
        keep = ["ts", "symbol", "source", "open", "high", "low", "close", "volume"]
        return df[keep]

    def _key_columns(self) -> list[str]:
        return ["ts", "symbol"]

    def upsert(self, df: pd.DataFrame) -> int:
        return self._upsert_dataframe(
            df, table_name="prices", engine=self.engine, key_cols=self._key_columns(),
        )


# Keep yaml import-only for tests; suppress unused-import lint if needed.
_ = json
