"""Intraday OANDA quote poller (CL-dz71) — feeds ``intraday_quotes`` so the
event confluence layer can confirm real intraday moves.

Confluence Gate B needs the price AT an event's seen_at and the price NOW. The
daily ``prices`` table can't answer that inside a 30-120 minute window (and its
yfinance ingest can lag days), so every event EXPIRES unconfirmed. This poller
hits OANDA's pricing endpoint — one batched call for all tradable event
instruments — every couple of minutes and writes fresh, timestamped mids into
``intraday_quotes``. We already hold OANDA practice credentials (the engine
trades there), and OANDA covers exactly these FX / metal / energy / index
instruments, INCLUDING ones with no daily series at all (XAG_USD, NATGAS_USD,
WHEAT_USD, NAS100_USD, ...).

The store is keyed by the raw OANDA instrument id, read back by
``DataProvider.get_intraday_value`` (which does NOT normalize), and pruned to a
short retention horizon each cycle — it is a rolling confirmation buffer, not a
historical archive.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_PRACTICE_URL = "https://api-fxpractice.oanda.com"
_LIVE_URL = "https://api-fxtrade.oanda.com"

#: Injectable transport: (url, headers, params) → parsed JSON dict. Lets unit
#: tests feed a canned OANDA pricing payload with no live network.
HttpGetJson = Callable[[str, dict[str, str], dict[str, str]], dict[str, Any]]


def _default_http_get_json(
    url: str, headers: dict[str, str], params: dict[str, str],
) -> dict[str, Any]:
    resp = httpx.get(url, headers=headers, params=params, timeout=10.0,
                     follow_redirects=True)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def parse_oanda_pricing(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """OANDA ``/pricing`` JSON → ``[{symbol, bid, ask, mid}]``.

    Skips non-tradeable quotes (weekend/halted — a last close stamped "now"
    would be misleading) and any row missing a bid/ask. Never raises.
    """
    out: list[dict[str, Any]] = []
    for p in payload.get("prices") or []:
        if not isinstance(p, dict):
            continue
        if p.get("tradeable") is False:
            continue
        inst = p.get("instrument")
        bids = p.get("bids") or []
        asks = p.get("asks") or []
        if not inst or not bids or not asks:
            continue
        try:
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if bid <= 0 or ask <= 0:
            continue
        out.append({
            "symbol": str(inst),
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2.0,
        })
    return out


def fetch_oanda_pricing(
    instruments: list[str],
    api_key: str,
    account_id: str,
    practice: bool = True,
    http_get: HttpGetJson | None = None,
) -> list[dict[str, Any]]:
    """One batched OANDA pricing call for ``instruments`` → parsed quotes."""
    if not instruments:
        return []
    getter = http_get or _default_http_get_json
    base = _PRACTICE_URL if practice else _LIVE_URL
    url = f"{base}/v3/accounts/{account_id}/pricing"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"instruments": ",".join(instruments)}
    payload = getter(url, headers, params)
    return parse_oanda_pricing(payload)


class IntradayPricer:
    """Polls OANDA pricing for a fixed instrument set and maintains the
    rolling ``intraday_quotes`` buffer."""

    def __init__(
        self,
        engine: Engine,
        instruments: list[str],
        api_key: str,
        account_id: str,
        practice: bool = True,
        source: str = "oanda",
        retention_hours: int = 24,
        http_get: HttpGetJson | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.engine = engine
        # De-dup while preserving order.
        self.instruments = list(dict.fromkeys(instruments))
        self.api_key = api_key
        self.account_id = account_id
        self.practice = practice
        self.source = source
        self.retention_hours = retention_hours
        self._http_get = http_get
        self._clock = clock or (lambda: datetime.now(UTC))

    def poll_once(self) -> dict[str, int]:
        """Fetch one batch, insert at a single timestamp, prune old rows.

        Fail-soft: a fetch error logs and returns zeros (the buffer keeps its
        existing rows; confluence just falls back to the daily close for any
        instrument that has no fresh quote). Returns
        ``{"written", "pruned", "instruments"}``.
        """
        ts = self._clock()
        try:
            quotes = fetch_oanda_pricing(
                self.instruments, self.api_key, self.account_id,
                practice=self.practice, http_get=self._http_get,
            )
        except Exception as exc:
            logger.warning(
                "intraday pricer: OANDA fetch failed (%d instruments) — "
                "buffer unchanged: %s", len(self.instruments), str(exc)[:200],
            )
            return {"written": 0, "pruned": 0, "instruments": 0}

        written = self._insert(quotes, ts)
        pruned = self._prune(ts)
        logger.info(
            "intraday pricer: wrote %d/%d quotes, pruned %d (ts=%s)",
            written, len(self.instruments), pruned, ts.isoformat(),
        )
        return {"written": written, "pruned": pruned, "instruments": len(quotes)}

    def _insert(self, quotes: list[dict[str, Any]], ts: datetime) -> int:
        if not quotes:
            return 0
        with self.engine.begin() as conn:
            for q in quotes:
                conn.execute(
                    text("""
                        INSERT INTO intraday_quotes
                            (ts, symbol, source, bid, ask, mid)
                        VALUES (:ts, :symbol, :source, :bid, :ask, :mid)
                        ON CONFLICT (ts, symbol, source) DO UPDATE SET
                            bid = excluded.bid,
                            ask = excluded.ask,
                            mid = excluded.mid
                    """),
                    {
                        "ts": ts, "symbol": q["symbol"], "source": self.source,
                        "bid": q["bid"], "ask": q["ask"], "mid": q["mid"],
                    },
                )
        return len(quotes)

    def _prune(self, now: datetime) -> int:
        cutoff = now - timedelta(hours=self.retention_hours)
        with self.engine.begin() as conn:
            res = conn.execute(
                text("DELETE FROM intraday_quotes WHERE ts < :cutoff"),
                {"cutoff": cutoff},
            )
        return res.rowcount if res.rowcount and res.rowcount > 0 else 0
