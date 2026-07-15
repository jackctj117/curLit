"""Alternative data ingestion — modular multi-source architecture (CL-6sp).

Provides a lightweight, DB-free counterpart to ``BaseIngester`` for
"alternative" (non-price, non-macro-release) data sources. Each source
implements the same three-step contract::

    fetch(start, end)   -> raw pd.DataFrame (source-shaped)
    transform(raw)      -> canonical long format: ts / symbol / value / source
    validate(df)        -> dropna + dedup on (ts, symbol), value sanity

and ``run(start, end)`` chains the three. The canonical output uses
synthetic symbols (``GT:US:recession``, ``ENTSOE:DE:load``) so the frames
can later be bridged into the existing ``prices``/``macro_data`` plumbing
the same way Polymarket probabilities were (see ``src/data/polymarket.py``
storage decision) without new query paths.

Modularity: sources self-register via the ``@register_source`` decorator
into a module-level registry. Adding a new source is: subclass
``AlternativeDataSource``, implement ``fetch``/``transform``, decorate.
``available_sources()``/``get_source()`` are the discovery API.

Implemented sources:
  * ``google_trends`` — Google Trends search interest via pytrends
    (free, no API key). Queries a curated set of economic anxiety /
    activity terms per currency's dominant geo (e.g. "recession",
    "unemployment benefits" in US for USD). Search-interest spikes lead
    soft data by days-to-weeks. Requires ``pip install 'curlit[altdata]'``
    (pytrends is an optional dep — import is lazy).
  * ``entsoe`` — ENTSO-E Transparency Platform actual total electricity
    load per bidding zone (free, but needs a registered API token in
    ``ENTSOE_API_TOKEN``). Electricity consumption is a real-time proxy
    for industrial activity in EUR/GBP/SEK/NOK/CHF economies. Uses the
    raw REST/XML API via httpx — deliberately NOT entsoe-py, to keep the
    dependency footprint at zero (httpx is already a core dep).

Stub sources (placeholders, raise NotImplementedError from fetch):
  * ``satellite_shipping`` — port-call / AIS congestion counts
    (Spire Maritime or MarineTraffic; both are paid APIs).
  * ``consumer_spending`` — foot-traffic derived spend proxies
    (SafeGraph Patterns or similar; paid).

Token handling: ENTSO-E's token is read from the ``ENTSOE_API_TOKEN``
env var. When absent, ``EntsoeSource.is_configured()`` is False and
``fetch`` raises ``ConfigurationError`` with setup instructions — callers
that iterate all registered sources should check ``is_configured()``
first and skip unconfigured ones rather than crash the run.
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, ClassVar

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Errors
# ---------------------------------------------------------------------- #


class AltDataError(RuntimeError):
    """Base error for the alternative-data layer."""


class ConfigurationError(AltDataError):
    """A source is missing required configuration (API token, etc.)."""


# ---------------------------------------------------------------------- #
# Base class + registry
# ---------------------------------------------------------------------- #

#: Canonical output schema every source's ``transform`` must produce.
CANONICAL_COLUMNS: list[str] = ["ts", "symbol", "value", "source"]


class AlternativeDataSource(ABC):
    """Abstract alternative-data source with a fetch/transform/validate
    contract.

    Unlike ``BaseIngester`` this is DB-free: ``run`` returns the
    validated canonical DataFrame instead of upserting, so sources are
    usable from research notebooks and backtests without a database.
    """

    #: Registry key + value of the ``source`` column. Subclasses override.
    source_name: ClassVar[str] = ""

    #: "live" for working implementations, "stub" for placeholders.
    status: ClassVar[str] = "live"

    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch raw data from the source for [start, end]."""

    @abstractmethod
    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise raw data to the canonical ts/symbol/value/source schema."""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with null keys/values, dedup on (ts, symbol), and
        drop non-finite values. Subclasses may extend with source-specific
        range checks (call super() first)."""
        if df.empty:
            return df
        missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
        if missing:
            msg = f"{self.source_name}: transform output missing columns {missing}"
            raise AltDataError(msg)
        df = df.dropna(subset=["ts", "symbol", "value"])
        df = df[pd.to_numeric(df["value"], errors="coerce").notna()]
        df = df.drop_duplicates(subset=["ts", "symbol"], keep="last")
        return df.reset_index(drop=True)

    def is_configured(self) -> bool:
        """Whether the source has everything it needs for a live fetch.
        Default True; sources needing tokens override."""
        return True

    def run(self, start: datetime, end: datetime) -> pd.DataFrame:
        """fetch -> transform -> validate. Returns the canonical frame."""
        logger.info(
            "alt-data %s: fetching [%s … %s]", self.source_name, start, end,
        )
        raw = self.fetch(start, end)
        if raw.empty:
            logger.warning("alt-data %s: no data", self.source_name)
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        df = self.validate(self.transform(raw))
        logger.info("alt-data %s: %d rows", self.source_name, len(df))
        return df


_SOURCES: dict[str, type[AlternativeDataSource]] = {}


def register_source(
    cls: type[AlternativeDataSource],
) -> type[AlternativeDataSource]:
    """Class decorator — add ``cls`` to the source registry under its
    ``source_name``. Re-registering a name overwrites (deliberate:
    lets tests/notebooks swap in doubles)."""
    if not cls.source_name:
        msg = f"{cls.__name__} must define a non-empty source_name"
        raise ValueError(msg)
    _SOURCES[cls.source_name] = cls
    return cls


def get_source(name: str) -> type[AlternativeDataSource]:
    """Look up a registered source class by name."""
    try:
        return _SOURCES[name]
    except KeyError:
        msg = f"unknown alt-data source {name!r}; known: {sorted(_SOURCES)}"
        raise KeyError(msg) from None


def available_sources(include_stubs: bool = True) -> list[str]:
    """Names of registered sources, sorted. ``include_stubs=False``
    filters to sources with a working fetch implementation."""
    return sorted(
        name for name, cls in _SOURCES.items()
        if include_stubs or cls.status == "live"
    )


# ---------------------------------------------------------------------- #
# Google Trends
# ---------------------------------------------------------------------- #

#: Economic search terms per currency, queried in that currency's
#: dominant-economy geo. Terms picked for macro signal: labour-market
#: anxiety ("unemployment benefits" style), recession fear, and
#: inflation salience. <=5 terms per geo (Google Trends payload limit).
TRENDS_TERMS_BY_CURRENCY: dict[str, dict[str, Any]] = {
    "USD": {"geo": "US", "terms": ["recession", "unemployment benefits", "inflation"]},
    "EUR": {"geo": "DE", "terms": ["rezession", "arbeitslosengeld", "inflation"]},
    "GBP": {"geo": "GB", "terms": ["recession", "universal credit", "inflation"]},
    "JPY": {"geo": "JP", "terms": ["景気後退", "失業保険", "インフレ"]},
    "AUD": {"geo": "AU", "terms": ["recession", "centrelink", "inflation"]},
    "CAD": {"geo": "CA", "terms": ["recession", "employment insurance", "inflation"]},
    "CHF": {"geo": "CH", "terms": ["rezession", "arbeitslosengeld", "inflation"]},
}


def trends_symbol(geo: str, term: str) -> str:
    """Synthetic symbol for a trends series. Spaces become underscores;
    capped at 64 chars to fit the ``prices.symbol`` varchar (same rule
    as ``polymarket.to_symbol``)."""
    return f"GT:{geo}:{term.replace(' ', '_')}"[:64]


@register_source
class GoogleTrendsSource(AlternativeDataSource):
    """Google Trends search-interest via pytrends (free, no key).

    One pytrends payload per geo (Google limits 5 terms per payload).
    A failed geo (429 rate-limit is common) is logged and skipped — one
    bad geo doesn't kill the run. ``request_delay_sec`` sleeps between
    payloads to stay under the anonymous rate limit.
    """

    source_name: ClassVar[str] = "google_trends"

    def __init__(
        self,
        currencies: list[str] | None = None,
        terms_by_currency: dict[str, dict[str, Any]] | None = None,
        client_factory: Callable[[], Any] | None = None,
        request_delay_sec: float = 1.0,
    ) -> None:
        self.terms_by_currency = (
            terms_by_currency if terms_by_currency is not None
            else TRENDS_TERMS_BY_CURRENCY
        )
        self.currencies = (
            currencies if currencies is not None
            else list(self.terms_by_currency)
        )
        unknown = [c for c in self.currencies if c not in self.terms_by_currency]
        if unknown:
            msg = f"no trends terms configured for currencies {unknown}"
            raise ConfigurationError(msg)
        self.client_factory = client_factory or self._default_client_factory
        self.request_delay_sec = request_delay_sec

    @staticmethod
    def _default_client_factory() -> Any:
        """Build the production pytrends client. Import is lazy so the
        module (and every registry consumer) works without the optional
        dep installed."""
        try:
            from pytrends.request import TrendReq
        except ImportError as exc:
            msg = (
                "pytrends is required for GoogleTrendsSource — "
                "install with: pip install 'curlit[altdata]'"
            )
            raise ImportError(msg) from exc
        # NOTE: deliberately no retries=/backoff_factor= — pytrends 4.9.2
        # forwards them to urllib3.Retry(method_whitelist=...), a kwarg
        # urllib3 2.x removed, so any retries>0 raises TypeError. Google
        # 429s anonymous clients aggressively; per-geo failures are
        # logged + skipped in fetch instead.
        return TrendReq(hl="en-US", tz=0, timeout=(10, 30))

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """One interest_over_time frame per currency geo, concatenated
        wide-format with a ``geo`` column (transform melts to long)."""
        client = self.client_factory()
        timeframe = f"{start:%Y-%m-%d} {end:%Y-%m-%d}"
        frames: list[pd.DataFrame] = []
        for i, ccy in enumerate(self.currencies):
            cfg = self.terms_by_currency[ccy]
            geo, terms = cfg["geo"], list(cfg["terms"])[:5]
            if i > 0 and self.request_delay_sec > 0:
                time.sleep(self.request_delay_sec)
            try:
                client.build_payload(terms, timeframe=timeframe, geo=geo)
                frame = client.interest_over_time()
            except Exception as exc:
                logger.warning(
                    "google_trends fetch failed for %s (%s): %s: %s",
                    ccy, geo, type(exc).__name__, exc,
                )
                continue
            if frame is None or frame.empty:
                logger.warning("google_trends: empty frame for %s (%s)", ccy, geo)
                continue
            frame = frame.copy()
            frame["geo"] = geo
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Melt the wide interest_over_time frames to canonical long
        format. Drops Google's ``isPartial`` flag column (last bucket
        is partial and revised — keep the value, drop the flag)."""
        if raw.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        df = raw.drop(columns=["isPartial"], errors="ignore")
        # pytrends names its DatetimeIndex "date"; fall back to "index"
        # for unnamed indices (test doubles).
        df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        long = df.melt(
            id_vars=["ts", "geo"], var_name="term", value_name="value",
        )
        long["symbol"] = [
            trends_symbol(g, t) for g, t in zip(long["geo"], long["term"], strict=True)
        ]
        long["ts"] = pd.to_datetime(long["ts"])
        long["value"] = long["value"].astype(float)
        long["source"] = self.source_name
        return long[CANONICAL_COLUMNS]

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Base checks + trends values are indices in [0, 100]."""
        df = super().validate(df)
        if df.empty:
            return df
        bad = (df["value"] < 0) | (df["value"] > 100)
        if bad.any():
            logger.warning(
                "google_trends: dropping %d out-of-range values", int(bad.sum()),
            )
            df = df[~bad].reset_index(drop=True)
        return df


# ---------------------------------------------------------------------- #
# ENTSO-E electricity load
# ---------------------------------------------------------------------- #

ENTSOE_API_URL: str = "https://web-api.tp.entsoe.eu/api"
ENTSOE_TOKEN_ENV: str = "ENTSOE_API_TOKEN"
DEFAULT_TIMEOUT_SEC: float = 30.0

#: EIC codes for bidding zones / control areas, keyed by short area name.
#: Chosen for FX relevance: DE/FR dominate EUR industrial output; the
#: others map to GBP, CHF, SEK, NOK.
ENTSOE_AREAS: dict[str, str] = {
    "DE": "10Y1001A1001A83F",  # Germany
    "FR": "10YFR-RTE------C",  # France
    "GB": "10YGB----------A",  # Great Britain
    "CH": "10YCH-SWISSGRIDZ",  # Switzerland
    "SE": "10YSE-1--------K",  # Sweden
    "NO": "10YNO-0--------C",  # Norway
}

#: HTTP shim — tests inject a canned-XML fn (same pattern as
#: ``polymarket.HttpGetJson``).
HttpGetText = Callable[[str, dict[str, str]], str]


def _default_http_get_text(url: str, params: dict[str, str]) -> str:
    """Production HTTP GET returning response text. Raises on non-2xx."""
    resp = httpx.get(
        url, params=params, timeout=DEFAULT_TIMEOUT_SEC, follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text


def _parse_resolution(res: str) -> timedelta:
    """Map ENTSO-E period resolutions to timedeltas."""
    table = {
        "PT15M": timedelta(minutes=15),
        "PT30M": timedelta(minutes=30),
        "PT60M": timedelta(hours=1),
        "P1D": timedelta(days=1),
    }
    try:
        return table[res]
    except KeyError:
        msg = f"unsupported ENTSO-E resolution {res!r}"
        raise AltDataError(msg) from None


@register_source
class EntsoeSource(AlternativeDataSource):
    """ENTSO-E actual total load per bidding zone — industrial-activity
    proxy for European currencies.

    Uses the raw Transparency Platform REST API (documentType=A65
    "system total load", processType=A16 "realised") and parses the
    GL_MarketDocument XML with stdlib ElementTree. No extra dependency.

    Token: free but registration-gated. Read from ``ENTSOE_API_TOKEN``
    (or passed explicitly). Without it, ``is_configured()`` is False and
    ``fetch`` raises ``ConfigurationError`` — callers looping over the
    registry should skip unconfigured sources.
    """

    source_name: ClassVar[str] = "entsoe"

    def __init__(
        self,
        areas: list[str] | None = None,
        api_token: str | None = None,
        http_get_text: HttpGetText | None = None,
        api_url: str = ENTSOE_API_URL,
    ) -> None:
        self.areas = areas if areas is not None else list(ENTSOE_AREAS)
        unknown = [a for a in self.areas if a not in ENTSOE_AREAS]
        if unknown:
            msg = f"unknown ENTSO-E areas {unknown}; known: {sorted(ENTSOE_AREAS)}"
            raise ConfigurationError(msg)
        self.api_token = api_token or os.environ.get(ENTSOE_TOKEN_ENV)
        self.http_get_text = http_get_text or _default_http_get_text
        self.api_url = api_url

    def is_configured(self) -> bool:
        return bool(self.api_token)

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Actual total load for each configured area. Long-format raw
        frame: ts / area / quantity_mw. A failed area is logged and
        skipped."""
        if not self.is_configured():
            msg = (
                "ENTSO-E API token missing — register (free) at "
                "https://transparency.entsoe.eu/ and export "
                f"{ENTSOE_TOKEN_ENV}=<token>, or pass api_token="
            )
            raise ConfigurationError(msg)
        rows: list[dict[str, object]] = []
        for area in self.areas:
            eic = ENTSOE_AREAS[area]
            params = {
                "securityToken": str(self.api_token),
                "documentType": "A65",   # system total load
                "processType": "A16",    # realised
                "outBiddingZone_Domain": eic,
                "periodStart": f"{start:%Y%m%d%H%M}",
                "periodEnd": f"{end:%Y%m%d%H%M}",
            }
            try:
                xml_text = self.http_get_text(self.api_url, params)
                rows.extend(self._parse_load_xml(xml_text, area))
            except ConfigurationError:
                raise
            except Exception as exc:
                logger.warning(
                    "entsoe fetch failed for %s: %s: %s",
                    area, type(exc).__name__, exc,
                )
                continue
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_load_xml(xml_text: str, area: str) -> list[dict[str, object]]:
        """Parse a GL_MarketDocument into ts/area/quantity_mw dicts.
        An Acknowledgement_MarketDocument (ENTSO-E's "no data / bad
        request" response) yields an AltDataError with the reason."""
        root = ET.fromstring(xml_text)
        if root.tag.endswith("Acknowledgement_MarketDocument"):
            reason = root.findtext(".//{*}Reason/{*}text") or "unknown reason"
            msg = f"ENTSO-E rejected request for {area}: {reason}"
            raise AltDataError(msg)
        # {*} namespace wildcards: the GL_MarketDocument namespace embeds
        # a schema version (…generationloaddocument:3:0) — hard-coding it
        # would silently return zero rows after an ENTSO-E version bump.
        out: list[dict[str, object]] = []
        for ts_el in root.findall("{*}TimeSeries"):
            for period in ts_el.findall("{*}Period"):
                start_text = period.findtext("{*}timeInterval/{*}start")
                res_text = period.findtext("{*}resolution")
                if not start_text or not res_text:
                    continue
                period_start = datetime.fromisoformat(
                    start_text.replace("Z", "+00:00"),
                )
                step = _parse_resolution(res_text)
                for point in period.findall("{*}Point"):
                    pos = point.findtext("{*}position")
                    qty = point.findtext("{*}quantity")
                    if pos is None or qty is None:
                        continue
                    out.append({
                        "ts": period_start + step * (int(pos) - 1),
                        "area": area,
                        "quantity_mw": float(qty),
                    })
        return out

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sub-hourly load to daily mean MW per area. Daily
        cadence matches the rest of the feature stack (yfinance/FRED);
        the intraday shape is noise for an activity proxy."""
        if raw.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        df = raw.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        daily = (
            df.set_index("ts")
            .groupby("area")["quantity_mw"]
            .resample("1D")
            .mean()
            .dropna()
            .reset_index()
        )
        daily["symbol"] = "ENTSOE:" + daily["area"] + ":load"
        daily = daily.rename(columns={"quantity_mw": "value"})
        daily["source"] = self.source_name
        return daily[CANONICAL_COLUMNS]

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Base checks + load must be positive (a zero/negative national
        load is a parse or unit bug, not a brownout)."""
        df = super().validate(df)
        if df.empty:
            return df
        bad = df["value"] <= 0
        if bad.any():
            logger.warning(
                "entsoe: dropping %d non-positive load values", int(bad.sum()),
            )
            df = df[~bad].reset_index(drop=True)
        return df


# ---------------------------------------------------------------------- #
# Placeholder stubs
# ---------------------------------------------------------------------- #


@register_source
class SatelliteShippingSource(AlternativeDataSource):
    """Placeholder — AIS/satellite shipping activity (port congestion,
    anchorage counts) as a trade-flow proxy. Candidate providers: Spire
    Maritime, MarineTraffic. Both are paid, credential-gated APIs; wire
    up once an account exists. Symbols will follow ``SHIP:<port>:<metric>``."""

    source_name: ClassVar[str] = "satellite_shipping"
    status: ClassVar[str] = "stub"

    def is_configured(self) -> bool:
        return False

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        msg = (
            "satellite_shipping is a stub — needs a Spire Maritime or "
            "MarineTraffic subscription (paid)"
        )
        raise NotImplementedError(msg)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@register_source
class ConsumerSpendingSource(AlternativeDataSource):
    """Placeholder — consumer-spending proxy from foot-traffic /
    transaction panels (SafeGraph Patterns or similar). Paid,
    credential-gated. Symbols will follow ``SPEND:<geo>:<category>``."""

    source_name: ClassVar[str] = "consumer_spending"
    status: ClassVar[str] = "stub"

    def is_configured(self) -> bool:
        return False

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        msg = (
            "consumer_spending is a stub — needs a SafeGraph (or similar "
            "spend-panel) subscription (paid)"
        )
        raise NotImplementedError(msg)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Live smoke demo (kept out of the test suite — run manually):
#   .venv/bin/python -m src.data.alternative
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    from datetime import timedelta as _td

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    end_dt = datetime.now()
    start_dt = end_dt - _td(days=90)

    print(f"registered sources: {available_sources()}")
    print(f"live sources:       {available_sources(include_stubs=False)}")

    # Google Trends — free, no key. One geo to minimise 429 risk.
    gt = GoogleTrendsSource(currencies=["USD"])
    try:
        gt_df = gt.run(start_dt, end_dt)
        print(f"\ngoogle_trends: {len(gt_df)} rows")
        print(gt_df.tail(6).to_string(index=False))
    except Exception as exc:  # 429 rate-limits are common for anonymous clients
        print(f"\ngoogle_trends live fetch FAILED: {type(exc).__name__}: {exc}")

    # ENTSO-E — only if the token is configured.
    en = EntsoeSource(areas=["DE"])
    if en.is_configured():
        try:
            en_df = en.run(end_dt - _td(days=7), end_dt)
            print(f"\nentsoe: {len(en_df)} rows")
            print(en_df.tail(6).to_string(index=False))
        except Exception as exc:
            print(f"\nentsoe live fetch FAILED: {type(exc).__name__}: {exc}")
    else:
        print(f"\nentsoe: skipped (no {ENTSOE_TOKEN_ENV} in env)")
    sys.exit(0)
