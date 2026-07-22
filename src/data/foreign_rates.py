"""Daily foreign / derived rate-series ingester (CL-gr8o).

Fills the ``macro_data`` gap that keeps ``rate_diff_mean_reversion`` and
``carry_vol_filter`` dormant: US2Y_MINUS_DE2Y / DE2Y / CVIX / USD_3M_OIS /
EUR_3M_ESTR_OIS all had 0 rows, so the strategies (correctly) refused to
trade. This module ingests DAILY history for each and upserts under the
EXACT series ids the strategies read (``configs/live_portfolio.yaml``,
``src/strategies/carry_vol_filter.py``, ``rate_diff_mean_reversion.py``).

SOURCES (all verified LIVE on 2026-07-21; keyless except FRED):

* ``DE2Y`` — Bundesbank SDMX REST, flow ``BBSIS``, key
  ``D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A``: daily yield derived
  from the term structure (Svensson) of listed Federal securities, 2.0-year
  residual maturity. CSV format (the portal's SDMX-JSON Accept-header
  variant returned a non-JSON body on probe; CSV is stable). Values in
  percent, missing days flagged ``.``. Probe: last obs 2026-07-21 = 2.75.

* ``US2Y_MINUS_DE2Y`` — DERIVED: FRED ``DGS2`` minus ``DE2Y`` on
  inner-joined observation dates. DGS2 is re-fetched from FRED here (not
  read from macro_data) so the spread is computed from a single consistent
  snapshot; the DGS2 rows themselves are NOT re-written (FREDIngester owns
  that series id).

* ``USD_3M_OIS`` — PROXY: FRED ``SOFR90DAYAVG`` (NY Fed 90-day compounded
  average of SOFR). This is a BACKWARD-looking 3M compounded overnight
  rate, not a forward-looking dealer 3M OIS quote — documented proxy, free
  and daily. Probe: 2026-07-21 = 3.62675. NOTE: the strategies read
  ``USD_3M_OIS`` (see _RATE_SERIES_MAP in carry_vol_filter.py); the old
  data_health requirement said ``US_3M_OIS`` — that was a latent id
  mismatch, fixed alongside this module (data_health now checks
  ``USD_3M_OIS``).

* ``EUR_3M_ESTR_OIS`` — PROXY: ECB Data Portal, flow ``EST``, key
  ``B.EU000A2QQF32.CR``: the ECB's official 3-month Compounded €STR
  Average Rate. Same backward-looking-compounded methodology as the US
  leg, so the carry differential USD_3M_OIS − EUR_3M_ESTR_OIS compares
  like with like. Probe: 2026-07-20 = 2.02997.

* ``CVIX`` — PROXY: the real CVIX is a proprietary Deutsche Bank FX
  implied-vol index and has no free feed; Cboe's EVZ is dead (yfinance
  ``^EVZ`` probe on 2026-07-21: "possibly delisted; no price data").
  Proxy composition: equal-weight 20-day realized volatility across the
  G10 USD pairs already in our ``prices`` table (same math as G10_RV20 in
  ``scripts/compute_g10_realized_vol.py``: per-pair rolling std of daily
  log returns × sqrt(252), averaged across pairs), scaled ×100 into
  annualized-percent "vol points" so levels are CVIX-like (~5-15). This
  is REALIZED vol, not implied — it lags turns by up to the 20d window.
  All in-repo consumers (vol_regime.py z-score, kill_switches CVIX z>3)
  only use z-scores against the series' own rolling baseline, for which a
  realized-vol basket is a sound regime proxy. No network needed for this
  leg — it derives from our own prices table.

PROVENANCE: every row is written with a per-series ``source`` value (see
``SERIES_SOURCES``) so proxy rows are auditable/deletable independently of
the FRED/ECB ingester rows.

CONVENTIONS: fail-soft per series (one dead endpoint must not sink the
others), injectable ``HttpGet`` transport shim so unit tests feed canned
payloads (no live network), insert-only dedup on (observation_date,
series_id) matching FREDIngester's natural-key behavior (CL-mht0) — these
proxy series do not track vintage revisions.

Runner: ``scripts/refresh_rates.py`` (daily cron; --once / --loop).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from .base import BaseIngester

logger = logging.getLogger(__name__)

#: Injectable HTTP shim (full URL → response body) so unit tests feed
#: canned payloads; mirrors src/data/symbols.py.
HttpGet = Callable[[str], str]

BUNDESBANK_DE2Y_URL = (
    "https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
    "D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A"
)
ECB_ESTR_3M_URL = (
    "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2QQF32.CR"
)
FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

#: series_id → macro_data.source value (audit trail; see module docstring).
SERIES_SOURCES: dict[str, str] = {
    "DE2Y": "bundesbank_bbsis",
    "US2Y_MINUS_DE2Y": "derived_dgs2_minus_de2y",
    "USD_3M_OIS": "fred_sofr90dayavg_proxy",
    "EUR_3M_ESTR_OIS": "ecb_estr_3m_compounded",
    "CVIX": "g10_rv20_proxy",
}

#: G10 USD pairs feeding the CVIX proxy — same tuple as
#: scripts/compute_g10_realized_vol.py; pairs absent from prices are
#: silently skipped (they never become pivot columns).
CVIX_PAIRS: tuple[str, ...] = (
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF",
    "USDCAD", "AUDUSD", "NZDUSD", "USDNOK", "USDSEK",
)
CVIX_WINDOW: int = 20
_ANNUALIZATION: float = 252.0


def _default_http_get(url: str) -> str:
    """Real fetch: httpx GET, 30s timeout, follows redirects."""
    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


# --------------------------------------------------------------------------- #
# pure parsers (unit-testable with canned bodies)
# --------------------------------------------------------------------------- #

def parse_bundesbank_csv(body: str) -> pd.Series:
    """Parse a Bundesbank SDMX ``format=csv`` body → date-indexed Series.

    The body is a BOM-prefixed CSV: ~8 metadata lines (series title,
    decimals, unit, ...) then ``YYYY-MM-DD,value,flag`` rows. Missing days
    carry ``.`` as the value ("No value available") and are dropped.
    Malformed bodies yield an empty Series (caller treats as soft failure).
    """
    rows: list[tuple[pd.Timestamp, float]] = []
    for line in body.lstrip("\ufeff").splitlines():
        parts = line.split(",")
        if len(parts) < 2 or len(parts[0]) != 10:
            continue
        try:
            ts = pd.Timestamp(parts[0])
        except ValueError:
            continue
        raw = parts[1].strip().strip('"')
        if raw in ("", "."):
            continue
        try:
            rows.append((ts, float(raw)))
        except ValueError:
            continue
    if not rows:
        return pd.Series(dtype=float)
    ser = pd.Series(dict(rows), dtype=float)
    return ser.sort_index()


def parse_fred_observations(body: str) -> pd.Series:
    """Parse a FRED ``series/observations`` JSON body → date-indexed Series.

    Missing observations carry ``"."`` and are dropped. Malformed bodies
    yield an empty Series.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("FRED body is not valid JSON")
        return pd.Series(dtype=float)
    rows: dict[pd.Timestamp, float] = {}
    for obs in data.get("observations", []) or []:
        raw = str(obs.get("value", "")).strip()
        if raw in ("", "."):
            continue
        try:
            rows[pd.Timestamp(obs["date"])] = float(raw)
        except (KeyError, ValueError):
            continue
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(rows, dtype=float).sort_index()


def parse_ecb_sdmx_series(body: str) -> pd.Series:
    """Parse an ECB Data Portal ``format=jsondata`` single-series body.

    Shape: ``dataSets[0].series[<key>].observations`` maps an observation
    index (string) to ``[value, ...]``; the index resolves to a date via
    ``structure.dimensions.observation[0].values[i].id``. Only the first
    series in the body is read (our URLs pin a full key → one series).
    Malformed bodies yield an empty Series.
    """
    try:
        data = json.loads(body)
        series_map = data["dataSets"][0]["series"]
        obs_dates = data["structure"]["dimensions"]["observation"][0]["values"]
        first = series_map[next(iter(series_map))]
        observations = first["observations"]
    except (json.JSONDecodeError, TypeError, ValueError, KeyError,
            IndexError, StopIteration):
        logger.warning("ECB SDMX body has unexpected shape")
        return pd.Series(dtype=float)
    rows: dict[pd.Timestamp, float] = {}
    for idx_str, values in observations.items():
        try:
            period = obs_dates[int(idx_str)]["id"]
            value = values[0]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if value is None:
            continue
        try:
            rows[pd.Timestamp(period)] = float(value)
        except ValueError:
            continue
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(rows, dtype=float).sort_index()


def compute_cvix_proxy(closes: pd.DataFrame) -> pd.Series:
    """CVIX proxy from a wide close-price frame (index=date, cols=pair).

    Equal-weight mean across pairs of the 20-day rolling std of daily log
    returns × sqrt(252) × 100 (annualized percent, CVIX-like vol points).
    ``skipna=False`` — a date where any present pair lacks a full window
    is dropped, matching G10_RV20 (scripts/compute_g10_realized_vol.py).
    """
    if closes.empty:
        return pd.Series(dtype=float)
    # Postgres NUMERIC arrives as decimal.Decimal (object dtype) — np.log
    # chokes on it; coerce to float first (no-op for float frames).
    closes = closes.astype(float)
    log_ret = np.log(closes / closes.shift(1))
    per_pair = log_ret.rolling(CVIX_WINDOW).std(ddof=1) * np.sqrt(_ANNUALIZATION)
    blended = per_pair.mean(axis=1, skipna=False).dropna() * 100.0
    return blended


# --------------------------------------------------------------------------- #
# ingester
# --------------------------------------------------------------------------- #

class ForeignRatesIngester(BaseIngester):
    """Fetch + derive the five series and upsert into ``macro_data``.

    ``http_get`` is the injectable transport; ``fred_api_key`` falls back
    to the FRED_API_KEY env var (empty string → the FRED-backed legs fail
    soft with a warning while Bundesbank/ECB/CVIX legs still ingest).
    """

    def __init__(
        self,
        db_url: str,
        http_get: HttpGet | None = None,
        fred_api_key: str | None = None,
    ) -> None:
        super().__init__(db_url, "foreign_rates")
        self.http_get = http_get or _default_http_get
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")

    # -- per-source fetch helpers (each fail-soft) ----------------------

    def _fetch_url(
        self, label: str, url: str, parser: Callable[[str], pd.Series],
    ) -> pd.Series:
        """Fetch + parse one source; any failure → empty Series + warning.

        The url is NOT logged (the FRED variant embeds the API key)."""
        try:
            body = self.http_get(url)
            ser = parser(body)
        except Exception as exc:
            logger.warning(
                "foreign_rates: %s fetch failed: %s: %s",
                label, type(exc).__name__, exc,
            )
            return pd.Series(dtype=float)
        if ser.empty:
            logger.warning("foreign_rates: %s returned no observations", label)
        return ser

    def _fetch_bundesbank_de2y(self, start: datetime) -> pd.Series:
        url = (
            f"{BUNDESBANK_DE2Y_URL}?format=csv&lang=en"
            f"&startPeriod={start:%Y-%m-%d}"
        )
        return self._fetch_url("DE2Y(bundesbank)", url, parse_bundesbank_csv)

    def _fetch_fred(self, series_id: str, start: datetime) -> pd.Series:
        if not self.fred_api_key:
            logger.warning(
                "foreign_rates: no FRED_API_KEY — skipping %s", series_id,
            )
            return pd.Series(dtype=float)
        url = (
            f"{FRED_OBS_URL}?series_id={series_id}"
            f"&api_key={self.fred_api_key}&file_type=json"
            f"&observation_start={start:%Y-%m-%d}"
        )
        return self._fetch_url(f"{series_id}(fred)", url, parse_fred_observations)

    def _fetch_ecb_estr_3m(self, start: datetime) -> pd.Series:
        url = (
            f"{ECB_ESTR_3M_URL}?format=jsondata"
            f"&startPeriod={start:%Y-%m-%d}"
        )
        return self._fetch_url(
            "EUR_3M_ESTR_OIS(ecb)", url, parse_ecb_sdmx_series,
        )

    def _compute_cvix(self, start: datetime, end: datetime) -> pd.Series:
        """CVIX proxy from our own prices table (no network). Fetches an
        extra ~60 calendar days so the 20-trading-day window is warm at
        ``start``; fail-soft on any DB/shape problem."""
        try:
            with self.engine.connect() as conn:
                stmt = text(
                    "SELECT ts, symbol, close FROM prices "
                    "WHERE symbol IN :syms AND ts >= :start AND ts <= :end "
                    "ORDER BY ts",
                ).bindparams(bindparam("syms", expanding=True))
                df = pd.DataFrame(
                    conn.execute(stmt, {
                        "syms": list(CVIX_PAIRS),
                        "start": start - timedelta(days=60),
                        "end": end,
                    }).fetchall(),
                    columns=["ts", "symbol", "close"],
                )
        except Exception as exc:
            logger.warning(
                "foreign_rates: CVIX prices query failed: %s: %s",
                type(exc).__name__, exc,
            )
            return pd.Series(dtype=float)
        if df.empty:
            logger.warning("foreign_rates: no G10 closes for CVIX proxy")
            return pd.Series(dtype=float)
        try:
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
            df["ts"] = df["ts"].dt.normalize()
            closes = df.pivot_table(
                index="ts", columns="symbol", values="close", aggfunc="last",
            )
            cvix = compute_cvix_proxy(closes)
        except Exception as exc:
            # Fail-soft: a bad price row must only cost us the CVIX leg.
            logger.warning(
                "foreign_rates: CVIX proxy computation failed: %s: %s",
                type(exc).__name__, exc,
            )
            return pd.Series(dtype=float)
        return cvix[cvix.index >= pd.Timestamp(start.date())]

    # -- BaseIngester contract ------------------------------------------

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Long frame [observation_date, series_id, value] for all series
        that could be built; each source is fail-soft so a dead endpoint
        only drops its own series (and the spread, if a leg is missing)."""
        de2y = self._fetch_bundesbank_de2y(start)
        dgs2 = self._fetch_fred("DGS2", start)  # spread leg only, not written
        parts: dict[str, pd.Series] = {
            "DE2Y": de2y,
            "USD_3M_OIS": self._fetch_fred("SOFR90DAYAVG", start),
            "EUR_3M_ESTR_OIS": self._fetch_ecb_estr_3m(start),
            "CVIX": self._compute_cvix(start, end),
        }
        if not de2y.empty and not dgs2.empty:
            parts["US2Y_MINUS_DE2Y"] = (dgs2 - de2y).dropna()
        else:
            logger.warning(
                "foreign_rates: cannot derive US2Y_MINUS_DE2Y "
                "(DGS2 rows=%d, DE2Y rows=%d)", len(dgs2), len(de2y),
            )
        frames: list[pd.DataFrame] = []
        start_ts, end_ts = pd.Timestamp(start.date()), pd.Timestamp(end.date())
        for sid, ser in parts.items():
            if ser.empty:
                continue
            ser = ser[(ser.index >= start_ts) & (ser.index <= end_ts)]
            frame = ser.rename("value").rename_axis("observation_date").reset_index()
            frame["series_id"] = sid
            frames.append(frame)
            logger.info(
                "foreign_rates: %s — %d obs, last %s",
                sid, len(frame),
                frame["observation_date"].max().date() if len(frame) else None,
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
        # release_date = ingest time; NOT part of the dedup key (CL-mht0
        # convention from FREDIngester — reruns must not duplicate rows).
        df["release_date"] = pd.Timestamp.utcnow()
        df["revision"] = 0
        df["source"] = df["series_id"].map(SERIES_SOURCES)
        df = df[["observation_date", "release_date", "series_id", "value",
                 "revision", "source"]]
        return df.dropna(subset=["value", "source"])

    def _key_columns(self) -> list[str]:
        return ["observation_date", "series_id"]

    def upsert(self, df: pd.DataFrame) -> int:
        """Insert-only upsert on (observation_date, series_id), anti-joined
        against existing rows for OUR series ids only (cheaper than the
        generic whole-table read in BaseIngester._upsert_dataframe, and
        portable to the sqlite engines the unit tests use)."""
        if df.empty:
            return 0
        ids = sorted(df["series_id"].unique())
        with self.engine.begin() as conn:
            stmt = text(
                "SELECT observation_date, series_id FROM macro_data "
                "WHERE series_id IN :ids",
            ).bindparams(bindparam("ids", expanding=True))
            existing = pd.DataFrame(
                conn.execute(stmt, {"ids": ids}).fetchall(),
                columns=["observation_date", "series_id"],
            )
            if not existing.empty:
                existing["observation_date"] = pd.to_datetime(
                    existing["observation_date"],
                ).dt.date
                merged = df.merge(
                    existing, on=["observation_date", "series_id"],
                    how="left", indicator=True,
                )
                df = merged[merged["_merge"] == "left_only"].drop(
                    columns=["_merge"],
                )
            if df.empty:
                return 0
            df.to_sql("macro_data", conn, if_exists="append", index=False)
        return len(df)
