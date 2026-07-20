"""Relative-volume scanner for the equity watch universe (CL-i4sr).

Scans every ``kind: equity_watch`` ticker across all themes in
``configs/event_playbooks.yaml`` (via the validated
:mod:`src.events.playbooks` loader — the config is re-read every scan,
so playbook edits are picked up without a restart) and computes, from
free yfinance daily bars:

  * ``rvol``             — today's volume / mean of the prior 20 sessions
  * ``price_change_pct`` — close vs prior close, in percent
  * ``is_unusual``       — ``rvol >= rvol_threshold`` (default 2.5,
                           env ``RVOL_THRESHOLD``)

Every scanned row is persisted to ``volume_spikes`` (migration 006) —
not just the unusual ones. The baseline history is what makes later
questions ("was FRO already running hot before the headline?")
answerable.

HONESTY NOTES — read before trusting a number:

  * yfinance intraday volume is DELAYED AND PARTIAL during the trading
    session: the "today" bar fills in as the session progresses, so a
    mid-session scan understates RVOL, and Yahoo's consolidated-tape
    coverage is itself best-effort. Treat RVOL as advisory
    confirmation of an event thesis, never as a trading gate on its
    own. After the close (or on weekends, when the last completed
    session is returned) the numbers are much more trustworthy.
  * OTC ADRs and foreign ordinaries (GLNCY / FQVLF / IVPAF style) have
    thin, unreliable volume on Yahoo — a handful of trades can print a
    huge RVOL that means nothing. They are still scanned (the history
    row is still useful), but any ticker whose 20d average volume is
    below ``min_avg_volume`` (default 50k shares, env
    ``RVOL_MIN_AVG_VOLUME``) is never flagged ``is_unusual``.

Failure posture: one broken ticker (delisted, renamed, no Yahoo data)
never kills the scan — it is logged and skipped. A failed batch
download degrades to per-ticker downloads before giving up.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.events.playbooks import (
    DEFAULT_PLAYBOOKS_PATH,
    Playbook,
    load_playbooks,
)

logger = logging.getLogger(__name__)

#: Default RVOL at/above which a spike is flagged unusual.
DEFAULT_RVOL_THRESHOLD = 2.5

#: 20d-average-volume floor (shares) below which ``is_unusual`` is
#: never set — thin OTC ADR tape makes RVOL meaningless down there.
DEFAULT_MIN_AVG_VOLUME = 50_000.0

#: Baseline window: mean of this many prior sessions.
BASELINE_SESSIONS = 20

#: Calendar-day download window (>= ~20 trading sessions + slack).
WINDOW_DAYS = 30

#: Minimum prior sessions with usable volume before RVOL is computed —
#: a 3-day-old listing dividing by a 2-sample mean is noise, not signal.
MIN_BASELINE_SESSIONS = 5

#: Exchange-listed ticker shape. Playbook validation already forces
#: equity_watch entries through the loader, but the universe filter is
#: defensively strict anyway: Polymarket-style slugs
#: (``strait-of-hormuz-closed-in-2026``) and other lowercase junk that
#: might drift into the config are silently skipped, never sent to
#: Yahoo. Allows BRK-B / RDS.A style class suffixes.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.-][A-Z0-9]{1,3})?$")

#: Downloader shim type — injectable for tests (no live yfinance calls
#: in unit tests, per the project's transport-shim convention).
Downloader = Callable[[Sequence[str], datetime, datetime], pd.DataFrame]


@dataclass(frozen=True)
class VolumeScanRow:
    """One scanned ticker — mirrors a ``volume_spikes`` row."""

    ticker: str
    scanned_at: datetime
    rvol: float
    volume: int | None
    avg_volume_20d: float | None
    price_change_pct: float | None
    is_unusual: bool
    source: str = "yfinance"


def equity_watch_universe(playbooks: dict[str, Playbook]) -> tuple[str, ...]:
    """Union of ``kind: equity_watch`` tickers across all themes,
    deduped and sorted. Non-ticker-shaped entries (Polymarket slugs,
    lowercase junk) are skipped with a debug log, not raised — the
    playbook config is edited by humans and agents alike."""
    universe: set[str] = set()
    for pb in playbooks.values():
        for inst in pb.instruments:
            if inst.kind != "equity_watch":
                continue
            ticker = inst.instrument.strip()
            if not _TICKER_RE.match(ticker):
                logger.debug(
                    "universe: skipping non-ticker equity_watch entry %r (theme %s)",
                    ticker, pb.key,
                )
                continue
            universe.add(ticker)
    return tuple(sorted(universe))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _yf_download(
    tickers: Sequence[str], start: datetime, end: datetime,
) -> pd.DataFrame:
    """Default downloader: one batched yfinance daily-bars request."""
    import yfinance as yf  # deferred — keep import cheap for non-scan callers

    return yf.download(
        list(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )


def _extract_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull one ticker's OHLCV sub-frame out of a batch download.

    yfinance returns ``(ticker, field)`` MultiIndex columns with
    ``group_by="ticker"``, ``(field, ticker)`` without, and flat columns
    for a single-ticker request — handle all three."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker]
        if ticker in data.columns.get_level_values(1):
            return data.xs(ticker, axis=1, level=1)
        return None
    return data


class RelativeVolumeScanner:
    """Key-free RVOL scan over the playbook equity watch universe.

    Advisory only — see the module docstring for the yfinance latency
    and thin-OTC-volume caveats before treating any flag as a signal.
    """

    def __init__(
        self,
        db_url: str,
        playbooks_path: Path | str = DEFAULT_PLAYBOOKS_PATH,
        rvol_threshold: float | None = None,
        min_avg_volume: float | None = None,
        downloader: Downloader | None = None,
    ) -> None:
        self.engine = create_engine(db_url)
        self.playbooks_path = playbooks_path
        self.rvol_threshold = (
            rvol_threshold
            if rvol_threshold is not None
            else _env_float("RVOL_THRESHOLD", DEFAULT_RVOL_THRESHOLD)
        )
        self.min_avg_volume = (
            min_avg_volume
            if min_avg_volume is not None
            else _env_float("RVOL_MIN_AVG_VOLUME", DEFAULT_MIN_AVG_VOLUME)
        )
        self.downloader = downloader or _yf_download

    # ------------------------------------------------------------- #
    # universe
    # ------------------------------------------------------------- #

    def universe(self) -> tuple[str, ...]:
        """Re-read the playbook config (it is edited live) and return
        the current equity watch universe."""
        return equity_watch_universe(load_playbooks(self.playbooks_path))

    # ------------------------------------------------------------- #
    # math
    # ------------------------------------------------------------- #

    def _compute_row(
        self, ticker: str, frame: pd.DataFrame, scanned_at: datetime,
    ) -> VolumeScanRow | None:
        """RVOL for one ticker from its daily-bars frame, or ``None``
        when the data is unusable (too new, all-NaN, zero baseline)."""
        if frame is None or frame.empty:
            return None
        cols = {str(c) for c in frame.columns}
        if "Volume" not in cols or "Close" not in cols:
            return None
        bars = frame[["Close", "Volume"]].dropna(subset=["Volume"])
        bars = bars[bars["Volume"] > 0]
        if len(bars) < MIN_BASELINE_SESSIONS + 1:
            return None

        today = bars.iloc[-1]
        prior = bars.iloc[:-1].tail(BASELINE_SESSIONS)
        avg_volume = float(prior["Volume"].mean())
        if not avg_volume > 0:
            return None

        today_volume = float(today["Volume"])
        rvol = today_volume / avg_volume

        price_change_pct: float | None = None
        prior_close = prior["Close"].dropna()
        try:
            last_prior_close = float(prior_close.iloc[-1])
            today_close = float(today["Close"])
            if last_prior_close > 0 and pd.notna(today_close):
                price_change_pct = (today_close / last_prior_close - 1.0) * 100.0
        except (IndexError, TypeError, ValueError):
            price_change_pct = None

        is_unusual = (
            rvol >= self.rvol_threshold and avg_volume >= self.min_avg_volume
        )
        return VolumeScanRow(
            ticker=ticker,
            scanned_at=scanned_at,
            rvol=round(rvol, 4),
            volume=int(today_volume),
            avg_volume_20d=round(avg_volume, 2),
            price_change_pct=(
                round(price_change_pct, 4) if price_change_pct is not None else None
            ),
            is_unusual=is_unusual,
        )

    # ------------------------------------------------------------- #
    # scan
    # ------------------------------------------------------------- #

    def scan(self, persist: bool = True) -> list[VolumeScanRow]:
        """Scan the whole watch universe; persist every computed row.

        Per-ticker failures are logged and skipped. Returns the rows
        actually computed (persisted when ``persist=True``)."""
        tickers = self.universe()
        if not tickers:
            logger.warning("rvol scan: empty equity watch universe; nothing to do")
            return []

        scanned_at = datetime.now(UTC)
        start = scanned_at - timedelta(days=WINDOW_DAYS)

        try:
            data = self.downloader(tickers, start, scanned_at)
        except Exception:
            logger.exception("rvol scan: batch download failed for %d tickers", len(tickers))
            data = None

        rows: list[VolumeScanRow] = []
        for ticker in tickers:
            try:
                frame = None
                if data is not None and not data.empty:
                    frame = _extract_ticker_frame(data, ticker)
                if frame is None or frame.dropna(how="all").empty:
                    # Batch miss (or whole-batch failure) — one retry
                    # alone before declaring the ticker unusable.
                    frame = self.downloader([ticker], start, scanned_at)
                    extracted = _extract_ticker_frame(frame, ticker)
                    frame = extracted if extracted is not None else frame
                row = self._compute_row(ticker, frame, scanned_at)
            except Exception:
                logger.warning("rvol scan: %s failed; skipping", ticker, exc_info=True)
                continue
            if row is None:
                logger.info("rvol scan: %s has no usable volume history; skipping", ticker)
                continue
            rows.append(row)

        if persist and rows:
            self._persist(rows)
        unusual = sum(1 for r in rows if r.is_unusual)
        logger.info(
            "rvol scan: %d/%d tickers scanned, %d unusual (threshold %.2f, floor %.0f)",
            len(rows), len(tickers), unusual, self.rvol_threshold, self.min_avg_volume,
        )
        return rows

    def _persist(self, rows: list[VolumeScanRow]) -> None:
        stmt = text(
            "INSERT INTO volume_spikes "
            "(ticker, scanned_at, rvol, volume, avg_volume_20d, "
            " price_change_pct, is_unusual, source) "
            "VALUES (:ticker, :scanned_at, :rvol, :volume, :avg_volume_20d, "
            " :price_change_pct, :is_unusual, :source)"
        )
        params = [
            {
                "ticker": r.ticker,
                "scanned_at": r.scanned_at,
                "rvol": r.rvol,
                "volume": r.volume,
                "avg_volume_20d": r.avg_volume_20d,
                "price_change_pct": r.price_change_pct,
                "is_unusual": r.is_unusual,
                "source": r.source,
            }
            for r in rows
        ]
        with self.engine.begin() as conn:
            conn.execute(stmt, params)
