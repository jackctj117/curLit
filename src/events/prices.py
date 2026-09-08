"""Batch price snapshot helper for notification enrichment (CL-mgcp).

``get_prices(tickers)`` returns ``{ticker: {price, change_pct, asof}}``
for the tickers it could price; everything else is simply ABSENT from
the result — callers render "no price" by omission, and this module
never raises for a data failure.

Two resolution paths:

  * **Equities / ETFs** (plain tickers) — one batched yfinance daily
    download (last close + change vs prior close). The downloader is
    an injectable shim (same convention as
    :class:`src.scanners.relative_volume.RelativeVolumeScanner`) so
    unit tests feed canned frames, never the network. Prices are
    last-session closes — during a session they lag, on weekends they
    are Friday's close; that is honest enough for an advisory feed.
  * **OANDA-style ids** (contain ``_``, e.g. ``BCO_USD``) — the
    ``prices`` table via :class:`src.data.provider.DataProvider`. As
    of CL-mgcp the ingesters do not populate broker symbols there, so
    this path usually yields absent keys today; it exists so prices
    appear automatically once such a feed lands, with zero caller
    changes.

Results are computed once per call for the deduped ticker batch — the
pipeline fetches ONE batch per assess cycle and shares the mapping
across ledger persistence, the digest, and the alerts.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Calendar-day lookback for the daily-bars download — enough to span
#: long weekends/holidays and still find two closes.
WINDOW_DAYS = 10

#: Exchange-listed ticker shape (mirrors the RVOL scanner's filter):
#: uppercase, optional class suffix. Anything else — Polymarket slugs,
#: lowercase junk — is never sent to Yahoo.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.-][A-Z0-9]{1,3})?$")

#: Downloader shim type — injectable for tests (no live yfinance in
#: unit tests, per the project's transport-shim convention).
Downloader = Callable[[Sequence[str], datetime, datetime], pd.DataFrame]


def _yf_download(
    tickers: Sequence[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Default downloader: one batched yfinance daily-bars request."""
    import yfinance as yf  # deferred — keep import cheap for non-price callers

    logger.info("event prices: downloading %d tickers on the calling thread", len(tickers))
    return yf.download(
        list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        # CL-i3js: reuse one thread-local cache connection, as in the RVOL
        # scan; per-symbol worker resources can exhaust a long-lived daemon.
        threads=False,
    )


def _extract_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """One ticker's OHLCV sub-frame from a batch download (handles the
    three column layouts yfinance produces — see the RVOL scanner)."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker]
        if ticker in data.columns.get_level_values(1):
            return data.xs(ticker, axis=1, level=1)
        return None
    return data


def _entry_from_closes(closes: pd.Series) -> dict[str, Any] | None:
    """{price, change_pct, asof} from a time-indexed close series, or
    ``None`` when there is no usable last close."""
    closes = closes.dropna()
    if closes.empty:
        return None
    price = float(closes.iloc[-1])
    if not price > 0:
        return None
    change_pct: float | None = None
    if len(closes) >= 2:
        prev = float(closes.iloc[-2])
        if prev > 0:
            change_pct = (price / prev - 1.0) * 100.0
    asof = closes.index[-1]
    return {
        "price": price,
        "change_pct": round(change_pct, 4) if change_pct is not None else None,
        "asof": asof.to_pydatetime() if hasattr(asof, "to_pydatetime") else asof,
    }


def _equity_prices(
    tickers: list[str],
    downloader: Downloader,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    start = now - timedelta(days=WINDOW_DAYS)
    try:
        data = downloader(tickers, start, now)
    except Exception:
        logger.warning(
            "price fetch: batch download failed for %d equities; rendering without prices",
            len(tickers),
            exc_info=True,
        )
        return {}
    out: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        try:
            frame = _extract_ticker_frame(data, ticker)
            if frame is None or "Close" not in {str(c) for c in frame.columns}:
                continue
            entry = _entry_from_closes(frame["Close"])
        except Exception:
            logger.debug("price fetch: %s unusable; skipping", ticker, exc_info=True)
            continue
        if entry is not None:
            out[ticker] = entry
    return out


def _oanda_prices(
    ids: list[str],
    engine: Any,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    """OANDA-style ids via the DataProvider ``prices``-table closes.
    Any failure (no engine, table empty, DB down) → absent keys."""
    if engine is None:
        return {}
    try:
        from src.data.provider import DataProvider  # noqa: PLC0415

        provider = DataProvider(engine)
    except Exception:
        logger.debug("price fetch: DataProvider unavailable", exc_info=True)
        return {}
    out: dict[str, dict[str, Any]] = {}
    start = now - timedelta(days=WINDOW_DAYS)
    for instrument in ids:
        try:
            series = provider.get_series(instrument, start, now)
            entry = _entry_from_closes(series)
        except Exception:
            logger.debug(
                "price fetch: %s failed via DataProvider",
                instrument,
                exc_info=True,
            )
            continue
        if entry is not None:
            out[instrument] = entry
    return out


def get_prices(
    tickers: Sequence[str],
    *,
    engine: Any = None,
    downloader: Downloader | None = None,
    now: datetime | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve last close + daily change for a batch of tickers.

    Returns ``{ticker: {"price": float, "change_pct": float | None,
    "asof": datetime}}``. Tickers that could not be priced (network
    failure, unknown symbol, no DB feed) are ABSENT — never a raise,
    never a fake number. Duplicates are fetched once.
    """
    now = now or datetime.now(UTC)
    seen: set[str] = set()
    equities: list[str] = []
    oanda_ids: list[str] = []
    for raw in tickers:
        ticker = str(raw).strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        if "_" in ticker:
            oanda_ids.append(ticker)
        elif _TICKER_RE.match(ticker):
            equities.append(ticker)
        else:
            logger.debug("price fetch: skipping non-ticker %r", ticker)

    out: dict[str, dict[str, Any]] = {}
    if equities:
        out.update(_equity_prices(equities, downloader or _yf_download, now))
    if oanda_ids:
        out.update(_oanda_prices(oanda_ids, engine, now))
    return out


def parse_ts(value: Any) -> datetime | None:
    """Defensive timestamp read for DB rows: aware datetimes pass
    through, naive ones are assumed UTC, ISO strings are parsed
    (sqlite test engines store TEXT), garbage → ``None``."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def format_age(then: Any, now: datetime | None = None) -> str:
    """Compact age string for phone-first messages: ``45m`` / ``2h`` /
    ``3d``. Empty string when ``then`` is missing/unparseable — callers
    omit the age rather than show a lie."""
    then_dt = parse_ts(then)
    if then_dt is None:
        return ""
    now = now or datetime.now(UTC)
    seconds = max(0.0, (now - then_dt).total_seconds())
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60.0
    if hours < 48:
        return f"{int(round(hours))}h"
    return f"{int(round(hours / 24.0))}d"


def format_price(ticker: str, info: Mapping[str, Any] | None) -> str:
    """Render one price entry for a message: ``$24.10 (+3.2%)`` for
    equities, ``78.4 (+2.1%)`` for OANDA-style ids (no ``$`` — many are
    not dollar-quoted). Empty string when ``info`` is None/unusable —
    callers just render the bare ticker. Output is HTML-safe by
    construction (digits, ``$%().+-`` only)."""
    if not info:
        return ""
    try:
        price = float(info["price"])
    except (KeyError, TypeError, ValueError):
        return ""
    text = f"{price:.5g}" if "_" in ticker else f"${price:,.2f}"
    change = info.get("change_pct")
    if change is not None:
        with contextlib.suppress(TypeError, ValueError):
            text += f" ({float(change):+.1f}%)"
    return text
