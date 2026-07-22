"""Truth Social event-study reaction measurement (CL-s9as) — RESEARCH ONLY.

For each market-relevant post, measures liquid-instrument reaction over
fixed windows (1/5/15/30/60/120 min): return, volume vs pre-post
baseline, MFE/MAE. Bars come from Alpaca's free IEX minute-bar feed
(delayed data is FINE — this is an after-the-fact study, latency is
explicitly out of scope).

Instruments: broad + sector ETFs, plus a literally-written ticker symbol
(e.g. "DJT" in the text) when it verifies against the SymbolUniverse.
Company-NAME → ticker inference is deliberately not done in v1.

Honesty rules: a window with no bars (market closed, halt) stores NULL
returns — never zero; a post is measured exactly once (rows exist =
measured), and only after all windows have matured (posted_at + 125 min).
No trading fields, no signals, no alerts.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

WINDOWS_MIN = (1, 5, 15, 30, 60, 120)
ETFS = ("SPY", "QQQ", "XLE", "XLF", "XLI", "XLK")
_BASELINE_MIN = 30  # pre-post minutes used for the volume baseline
_MATURITY_MIN = 125  # all windows complete before measuring

#: Injectable bar fetch: (symbols, start, end) -> {symbol: [bar dicts]}.
#: Bar dict shape mirrors Alpaca v2: {"t": iso, "o","h","l","c","v"}.
BarsFn = Callable[
    [list[str], datetime, datetime], dict[str, list[dict[str, Any]]],
]

_DATA_BASE = "https://data.alpaca.markets"


def _default_fetch_bars(
    symbols: list[str], start: datetime, end: datetime,
) -> dict[str, list[dict[str, Any]]]:
    import httpx  # noqa: PLC0415 — lazy

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY/SECRET not set — cannot fetch bars")
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
    page_token: str | None = None
    while True:
        params: dict[str, str] = {
            "symbols": ",".join(symbols),
            "timeframe": "1Min",
            "start": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "end": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "feed": "iex",
            "limit": "10000",
        }
        if page_token:
            params["page_token"] = page_token
        resp = httpx.get(
            f"{_DATA_BASE}/v2/stocks/bars", params=params,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=30.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        for sym, bars in (payload.get("bars") or {}).items():
            out.setdefault(sym, []).extend(bars or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break
    return out


def _bar_time(bar: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(
        str(bar["t"]).replace("Z", "+00:00"),
    ).astimezone(UTC)


def measure_windows(
    bars: list[dict[str, Any]], posted_at: datetime,
) -> dict[int, dict[str, float | None]]:
    """Pure window math for ONE instrument.

    start price = close of the last bar at/before the post (the price the
    market showed when the post landed). Per window: end price = close of
    the last bar inside the window; MFE/MAE from highs/lows inside it;
    volume ratio vs the per-minute baseline of the 30 min before the
    post. All *_pct values are PERCENT. Windows with no usable bars (or
    no baseline volume) → None fields.
    """
    parsed = sorted(
        ((_bar_time(b), b) for b in bars or []), key=lambda x: x[0],
    )
    pre = [(t, b) for t, b in parsed if t <= posted_at]
    out: dict[int, dict[str, float | None]] = {}
    start_price = float(pre[-1][1]["c"]) if pre else None
    base_cut = posted_at - timedelta(minutes=_BASELINE_MIN)
    base_vols = [float(b["v"]) for t, b in pre if t > base_cut]
    base_per_min = (sum(base_vols) / _BASELINE_MIN) if base_vols else None

    for w in WINDOWS_MIN:
        end_cut = posted_at + timedelta(minutes=w)
        in_win = [(t, b) for t, b in parsed if posted_at < t <= end_cut]
        if start_price is None or not in_win:
            out[w] = {"return_pct": None, "volume_ratio": None,
                      "max_favorable_pct": None, "max_adverse_pct": None,
                      "start_price": start_price, "end_price": None}
            continue
        end_price = float(in_win[-1][1]["c"])
        highs = max(float(b["h"]) for _t, b in in_win)
        lows = min(float(b["l"]) for _t, b in in_win)
        vol = sum(float(b["v"]) for _t, b in in_win)
        out[w] = {
            "return_pct": (end_price - start_price) / start_price * 100.0,
            "volume_ratio": (
                vol / (base_per_min * w)
                if base_per_min and base_per_min > 0 else None
            ),
            "max_favorable_pct": (highs - start_price) / start_price * 100.0,
            "max_adverse_pct": (lows - start_price) / start_price * 100.0,
            "start_price": start_price,
            "end_price": end_price,
        }
    return out


def _ticker_entities(engine: Any, entities: list[str]) -> list[str]:
    """Entities that are LITERALLY ticker symbols (e.g. 'DJT'), verified
    against the symbols universe. Name→ticker inference is v2 territory."""
    cands = [
        e.strip().upper() for e in entities
        if isinstance(e, str) and 1 <= len(e.strip()) <= 5
        and e.strip().isalpha() and e.strip().isupper()
    ]
    if not cands:
        return []
    verified: list[str] = []
    try:
        with engine.connect() as conn:
            for c in dict.fromkeys(cands):
                hit = conn.execute(
                    text("SELECT 1 FROM symbols WHERE ticker = :t LIMIT 1"),
                    {"t": c},
                ).scalar()
                if hit:
                    verified.append(c)
    except Exception:
        logger.debug("truth reactions: symbols lookup failed", exc_info=True)
    return verified[:3]


def measure_pending(
    engine: Any,
    fetch_bars: BarsFn | None = None,
    now: datetime | None = None,
    limit: int = 5,
) -> int:
    """Measure up to ``limit`` matured, relevant, unmeasured posts.
    Returns posts measured. A post with NO bars anywhere still writes its
    NULL rows so it is never retried forever (honest: market was closed)."""
    now = now or datetime.now(UTC)
    fetch = fetch_bars or _default_fetch_bars
    import json  # noqa: PLC0415

    with engine.connect() as conn:
        posts = [dict(r._mapping) for r in conn.execute(text("""
            SELECT p.post_id, p.posted_at, c.named_entities
            FROM truth_posts p
            JOIN truth_classifications c ON c.post_id = p.post_id
            WHERE c.is_market_relevant
              AND p.posted_at <= :matured
              AND NOT EXISTS (SELECT 1 FROM truth_market_reactions r
                              WHERE r.post_id = p.post_id)
            ORDER BY p.posted_at
            LIMIT :lim
        """), {"matured": now - timedelta(minutes=_MATURITY_MIN),
               "lim": limit})]
    measured = 0
    for post in posts:
        posted_at = post["posted_at"]
        if isinstance(posted_at, str):  # sqlite tests; Postgres → datetime
            posted_at = datetime.fromisoformat(posted_at)
        if posted_at.tzinfo is None:
            posted_at = posted_at.replace(tzinfo=UTC)
        raw_ents = post.get("named_entities")
        ents = (json.loads(raw_ents) if isinstance(raw_ents, str)
                else (raw_ents or []))
        instruments = list(ETFS) + _ticker_entities(engine, ents)
        try:
            bars = fetch(
                instruments,
                posted_at - timedelta(minutes=_BASELINE_MIN + 5),
                posted_at + timedelta(minutes=WINDOWS_MIN[-1] + 5),
            )
        except Exception:
            logger.warning("truth reactions: bar fetch failed for %s — "
                           "retry next cycle", post["post_id"],
                           exc_info=True)
            continue
        with engine.begin() as conn:
            for sym in instruments:
                for w, m in measure_windows(
                    bars.get(sym) or [], posted_at,
                ).items():
                    conn.execute(text("""
                        INSERT INTO truth_market_reactions
                            (post_id, instrument, window_minutes,
                             return_pct, volume_ratio, max_favorable_pct,
                             max_adverse_pct, start_price, end_price,
                             measured_at)
                        VALUES (:p, :s, :w, :r, :vr, :mf, :ma, :sp, :ep, :at)
                        ON CONFLICT (post_id, instrument, window_minutes)
                        DO NOTHING
                    """), {"p": post["post_id"], "s": sym, "w": w,
                           "r": m["return_pct"], "vr": m["volume_ratio"],
                           "mf": m["max_favorable_pct"],
                           "ma": m["max_adverse_pct"],
                           "sp": m["start_price"], "ep": m["end_price"],
                           "at": now})
        measured += 1
        logger.info("truth reactions: measured post %s across %d "
                    "instruments", post["post_id"], len(instruments))
    return measured
