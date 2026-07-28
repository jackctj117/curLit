"""OANDA daily-candle backfill (CL-lb03) — gives the event instruments that
have NO daily series a realized-vol baseline, so confluence Gate B can confirm
them.

The intraday quote feed (CL-dz71) gave the 10 unmapped event instruments
(XAG_USD, XPT_USD, XPD_USD, NATGAS_USD, WHEAT_USD, CORN_USD, NAS100_USD,
USD_NOK, USD_ZAR, USD_CNH, USD_MXN, ...) a live price, but Gate B ALSO needs a
20-day realized DAILY vol to size its move threshold — and those instruments
have no row in the daily ``prices`` table at all, so confirmation still failed
with ``no_vol_data``.

This module pulls ~60 daily candles per instrument from OANDA and upserts them
into ``prices`` under the RAW OANDA instrument id and a distinct
``source='oanda_daily'``. ``DataProvider.get_realized_vol`` passes an unmapped
id through ``_normalize_symbol`` unchanged, so it reads exactly these rows.
This is genuine DAILY data (one bar per session), so — unlike the intraday
quotes — it belongs in ``prices`` and does not distort the daily-return math.

Mapped instruments (XAU_USD→GOLD, BCO_USD→OIL_WTI, ...) already resolve to a
canonical yfinance series and are intentionally NOT backfilled here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

#: OANDA stamps candle times with 9 fractional (nanosecond) digits, which
#: datetime.fromisoformat can't parse — collapse them to 6 while KEEPING the
#: timezone offset.
_NANOS_RE = re.compile(r"(\.\d{6})\d+")

_PRACTICE_URL = "https://api-fxpractice.oanda.com"
_LIVE_URL = "https://api-fxtrade.oanda.com"

#: Injectable transport: (url, headers, params) → parsed JSON dict.
HttpGetJson = Callable[[str, dict[str, str], dict[str, str]], dict[str, Any]]


def _default_http_get_json(
    url: str,
    headers: dict[str, str],
    params: dict[str, str],
) -> dict[str, Any]:
    resp = httpx.get(url, headers=headers, params=params, timeout=15.0, follow_redirects=True)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def parse_daily_candles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """OANDA ``/candles`` JSON → ``[{ts, open, high, low, close, volume}]``.

    Only COMPLETE candles are kept — the still-forming current-day bar is
    excluded so realized vol is computed on settled sessions. Never raises.
    """
    out: list[dict[str, Any]] = []
    for c in payload.get("candles") or []:
        if not isinstance(c, dict) or not c.get("complete"):
            continue
        mid = c.get("mid") or {}
        raw_time = c.get("time")
        if not raw_time or not isinstance(mid, dict):
            continue
        try:
            # RFC3339 with nanoseconds → microseconds, Z → +00:00, tz kept.
            iso = _NANOS_RE.sub(r"\1", raw_time.replace("Z", "+00:00"))
            ts = datetime.fromisoformat(iso)
            o = float(mid["o"])
            h = float(mid["h"])
            low = float(mid["l"])
            close = float(mid["c"])
        except (KeyError, TypeError, ValueError):
            continue
        vol = c.get("volume")
        try:
            volume = float(vol) if vol is not None else None
        except (TypeError, ValueError):
            volume = None
        out.append(
            {
                "ts": ts,
                "open": o,
                "high": h,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return out


def fetch_daily_candles(
    instrument: str,
    api_key: str,
    account_id: str,
    practice: bool = True,
    count: int = 60,
    http_get: HttpGetJson | None = None,
) -> list[dict[str, Any]]:
    """Fetch the last ``count`` COMPLETE daily (granularity D) mid candles."""
    getter = http_get or _default_http_get_json
    base = _PRACTICE_URL if practice else _LIVE_URL
    url = f"{base}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"granularity": "D", "count": str(count), "price": "M"}
    return parse_daily_candles(getter(url, headers, params))


def refresh_daily_candles(
    engine: Engine,
    instruments: list[str],
    api_key: str,
    account_id: str,
    practice: bool = True,
    count: int = 60,
    source: str = "oanda_daily",
    http_get: HttpGetJson | None = None,
) -> dict[str, int]:
    """Backfill daily candles for ``instruments`` into ``prices``.

    Stored under the RAW OANDA id and ``source``. Fail-soft PER instrument —
    one bad symbol (e.g. an id OANDA rejects) logs and is skipped, the rest
    proceed. Returns ``{"instruments", "bars"}``.
    """
    ok_instruments = 0
    total_bars = 0
    for instrument in instruments:
        try:
            candles = fetch_daily_candles(
                instrument,
                api_key,
                account_id,
                practice=practice,
                count=count,
                http_get=http_get,
            )
        except Exception as exc:
            logger.warning(
                "daily candles: fetch failed for %s — skipping: %s",
                instrument,
                str(exc)[:160],
            )
            continue
        if not candles:
            continue
        with engine.begin() as conn:
            for bar in candles:
                conn.execute(
                    text("""
                        INSERT INTO prices
                            (ts, symbol, source, open, high, low, close, volume)
                        VALUES (:ts, :symbol, :source, :open, :high, :low,
                                :close, :volume)
                        ON CONFLICT (ts, symbol, source) DO UPDATE SET
                            open = excluded.open, high = excluded.high,
                            low = excluded.low, close = excluded.close,
                            volume = excluded.volume
                    """),
                    {"symbol": instrument, "source": source, **bar},
                )
        ok_instruments += 1
        total_bars += len(candles)
    logger.info(
        "daily candles: backfilled %d instruments, %d bars (source=%s)",
        ok_instruments,
        total_bars,
        source,
    )
    return {"instruments": ok_instruments, "bars": total_bars}


def unmapped_tradables(instruments: list[str]) -> list[str]:
    """Subset of ``instruments`` with NO daily prices-table alias — i.e. the
    ones whose realized vol can only come from an OANDA daily backfill.
    ``_normalize_symbol`` returns a mapped id unchanged only when it has no
    alias, so ``normalize(x) == x`` identifies the unmapped ones."""
    from src.data.provider import _normalize_symbol  # noqa: PLC0415

    return [i for i in instruments if _normalize_symbol(i) == i]


# ---------------------------------------------------------------------------
# Intraday (M5) historical backfill into intraday_quotes (CL-b425)
# ---------------------------------------------------------------------------
#
# The live intraday_quotes feed only exists from the moment the pricer daemon
# first ran, so the CL-z95p event study could not price any event seen before
# that — 4,631 of 5,971 candidate legs were unmeasurable on its first run.
# OANDA's /candles endpoint serves the same venue's HISTORY at M5 with
# bid/ask/mid closes, which is enough to price event horizons at the study's
# 15-minute match tolerance. This backfill writes those candles into
# intraday_quotes under a DISTINCT source, clipped to end where the live feed
# begins, so the live region stays purely live and Gate B (which only reads
# the freshest minutes) never sees a backfilled row as "now".
#
# NO LOOKAHEAD: a candle's close is only known when the candle ENDS, so rows
# are stamped ts = candle_start + granularity. The study's entry rule
# (first quote AT/AFTER the anchor) then uses a price from strictly after the
# anchor, exactly as with live quotes.

#: Granularity label → minutes. Only the ones this backfill supports.
_GRANULARITY_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}

#: OANDA hard cap on candles per request.
_MAX_CANDLES_PER_REQUEST = 5000


def parse_mba_candles(
    payload: dict[str, Any],
    granularity_minutes: int,
) -> list[dict[str, Any]]:
    """OANDA ``price=MBA`` candles → ``[{ts, bid, ask, mid}]`` quote rows.

    ``ts`` is the candle END (start + granularity) and prices are the CLOSES —
    the no-lookahead stamping described in the module note. Incomplete candles
    and malformed rows are skipped. Never raises."""
    out: list[dict[str, Any]] = []
    for c in payload.get("candles") or []:
        if not isinstance(c, dict) or not c.get("complete"):
            continue
        raw_time = c.get("time")
        mid = c.get("mid") or {}
        bid = c.get("bid") or {}
        ask = c.get("ask") or {}
        if not raw_time or not isinstance(mid, dict):
            continue
        try:
            iso = _NANOS_RE.sub(r"\1", str(raw_time).replace("Z", "+00:00"))
            start = datetime.fromisoformat(iso)
            mid_close = float(mid["c"])
            bid_close = float(bid["c"]) if isinstance(bid, dict) and "c" in bid else None
            ask_close = float(ask["c"]) if isinstance(ask, dict) and "c" in ask else None
        except (KeyError, TypeError, ValueError):
            continue
        if mid_close <= 0:
            continue
        out.append(
            {
                "ts": start + timedelta(minutes=granularity_minutes),
                "bid": bid_close,
                "ask": ask_close,
                "mid": mid_close,
            }
        )
    return out


def fetch_intraday_history(
    instrument: str,
    api_key: str,
    account_id: str,
    *,
    start: datetime,
    end: datetime,
    granularity: str = "M5",
    practice: bool = True,
    http_get: HttpGetJson | None = None,
) -> list[dict[str, Any]]:
    """All complete MBA candles for ``instrument`` in ``[start, end]``,
    paginated past OANDA's per-request cap. Rows are quote-shaped (see
    :func:`parse_mba_candles`) and strictly ts-ascending."""
    if granularity not in _GRANULARITY_MINUTES:
        msg = f"unsupported candle granularity {granularity!r}"
        raise ValueError(msg)
    gran_min = _GRANULARITY_MINUTES[granularity]
    getter = http_get or _default_http_get_json
    base = _PRACTICE_URL if practice else _LIVE_URL
    url = f"{base}/v3/accounts/{account_id}/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {api_key}"}

    rows: list[dict[str, Any]] = []
    cursor = start
    # Bounded loop: each page advances the cursor or breaks; the ceiling is
    # generous (a year of M1 is ~75 pages) and protects against a server that
    # keeps replying without progress.
    for _ in range(200):
        if cursor >= end:
            break
        params = {
            "granularity": granularity,
            "price": "MBA",
            "from": cursor.isoformat().replace("+00:00", "Z"),
            "to": end.isoformat().replace("+00:00", "Z"),
        }
        page = parse_mba_candles(getter(url, headers, params), gran_min)
        page = [r for r in page if r["ts"] > cursor and r["ts"] <= end]
        if not page:
            break
        rows.extend(page)
        new_cursor = page[-1]["ts"]
        if new_cursor <= cursor:  # no forward progress — stop, never spin
            break
        cursor = new_cursor
    return rows


def backfill_intraday_quotes(
    engine: Engine,
    instruments: list[str],
    api_key: str,
    account_id: str,
    *,
    start: datetime,
    end: datetime,
    granularity: str = "M5",
    practice: bool = True,
    source: str = "oanda_m5_backfill",
    http_get: HttpGetJson | None = None,
) -> dict[str, int]:
    """Backfill ``intraday_quotes`` from OANDA candle history (CL-b425).

    Callers MUST pass ``end`` clipped to where the live quote feed begins so
    the live region stays purely live (the script computes that clip). Upserts
    on the (ts, symbol, source) key, so re-runs are idempotent. Fail-soft PER
    instrument — one rejected symbol logs and is skipped, the rest proceed.
    Returns ``{"instruments", "rows"}``.
    """
    ok = 0
    total = 0
    for instrument in instruments:
        try:
            rows = fetch_intraday_history(
                instrument,
                api_key,
                account_id,
                start=start,
                end=end,
                granularity=granularity,
                practice=practice,
                http_get=http_get,
            )
        except Exception:
            logger.warning("candle backfill: %s failed — skipped", instrument, exc_info=True)
            continue
        if not rows:
            logger.info("candle backfill: %s — no candles in window", instrument)
            continue
        with engine.begin() as conn:
            for r in rows:
                conn.execute(
                    text("""
                        INSERT INTO intraday_quotes (ts, symbol, source, bid, ask, mid)
                        VALUES (:ts, :symbol, :source, :bid, :ask, :mid)
                        ON CONFLICT (ts, symbol, source) DO UPDATE SET
                            bid = excluded.bid, ask = excluded.ask, mid = excluded.mid
                    """),
                    {
                        "ts": r["ts"],
                        "symbol": instrument,
                        "source": source,
                        **{k: r[k] for k in ("bid", "ask", "mid")},
                    },
                )
        ok += 1
        total += len(rows)
        logger.info("candle backfill: %s — %d rows", instrument, len(rows))
    return {"instruments": ok, "rows": total}
