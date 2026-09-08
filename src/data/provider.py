"""Data provider — query aligned time-series data from the database."""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

logger = logging.getLogger(__name__)

# --- Event/OANDA-id → prices-table-symbol alias map (CL-5lpp) -----------
#
# CRITICAL bug fix: the current-events pipeline (geo_events assessments,
# configs/event_playbooks.yaml, configs/cross_asset_checks.yaml,
# src/strategies/event_driven.py) speaks OANDA-style instrument ids
# (XAU_USD, BCO_USD, USD_JPY, ...). The `prices` table this provider reads
# holds yfinance-style symbols (GOLD, OIL_WTI, USDJPY, ...). Because the
# two vocabularies never matched, DataProvider price/vol lookups for every
# tradable event leg returned nothing, so confluence Gate B could never
# confirm a move: all 164 confluence-eligible events EXPIRED, 0 CONFIRMED,
# 0 TRADED.
#
# This map translates an incoming OANDA-style id to its prices-table
# equivalent at the START of every read method. It is a PURE ADDITIVE
# overlay: only OANDA ids appear as keys, and none of them collide with a
# DB-native symbol, so the existing FX/macro strategies (which already
# query DB-native names like EURUSD / US_10Y / DGS2) pass straight through
# unchanged.
#
# SAFETY: this normalization is applied to PRICE READS ONLY. Order
# placement is untouched — src/strategies/event_driven.py's instrument_map
# still hands the OANDA name (e.g. BCO_USD) to the broker for the actual
# order. Only the DataProvider price lookup is normalized here.
_SYMBOL_ALIASES: dict[str, str] = {
    # Energy. No Brent series exists in the prices table, so Brent
    # (BCO_USD) is mapped to WTI (OIL_WTI) as a directional PROXY for
    # confirmation only — Brent and WTI are ~0.9+ correlated intraday, so
    # a move-in-direction confirmation on WTI is a fair stand-in. This is
    # a documented substitution, NOT an exact price; the broker still
    # trades real Brent (BCO_USD) via the strategy's instrument_map.
    "BCO_USD": "OIL_WTI",   # Brent → WTI proxy (no Brent series ingested)
    "WTICO_USD": "OIL_WTI",
    # Metals.
    "XAU_USD": "GOLD",
    "XCU_USD": "COPPER",
    # FX (OANDA underscores → yfinance no-underscore).
    "USD_JPY": "USDJPY",
    "USD_CAD": "USDCAD",
    "USD_CHF": "USDCHF",
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "AUD_USD": "AUDUSD",
    "NZD_USD": "NZDUSD",
    # Equity index.
    "SPX500_USD": "SPX",
}

# Known event/OANDA-style instrument ids with NO prices-table equivalent
# yet. They intentionally stay UNMAPPED: normalization passes them through
# unchanged, the DB lookup finds nothing, and confirmation honestly stays
# impossible until the underlying series is ingested. Listed here so the
# gap is discoverable and greppable. (XAG=silver, XPT=platinum,
# XPD=palladium, NOK/ZAR/CNH FX, NATGAS/WHEAT/CORN commodities,
# NAS100=Nasdaq index — none are in the `prices` table today.)
UNMAPPED_EVENT_INSTRUMENTS: frozenset[str] = frozenset({
    "XAG_USD", "XPT_USD", "XPD_USD",
    "USD_NOK", "USD_ZAR", "USD_CNH",
    "NATGAS_USD", "WHEAT_USD", "CORN_USD",
    "NAS100_USD",
})

# DEBUG-log a known-unmapped miss at most ONCE per distinct symbol per
# process (module-level so it survives across DataProvider instances) —
# avoids per-call spam in the tight confluence/strategy loops.
_logged_unmapped: set[str] = set()


def _normalize_symbol(sym: str) -> str:
    """Translate an OANDA/event-style instrument id to the prices-table
    symbol (CL-5lpp). Pass-through unchanged when the id isn't a known
    alias — so DB-native names (EURUSD, US_10Y, DGS2, ...) and
    known-unmapped event ids (WHEAT_USD, ...) are returned as-is.

    When the id is a KNOWN-UNMAPPED event instrument, log once at DEBUG so
    the confirmation-can't-resolve gap is discoverable without per-call
    spam. Purely additive: only OANDA ids are keys and none collide with a
    DB-native symbol, so existing strategy lookups are unaffected.
    """
    if not isinstance(sym, str):
        return sym
    mapped = _SYMBOL_ALIASES.get(sym)
    if mapped is not None:
        return mapped
    if sym in UNMAPPED_EVENT_INSTRUMENTS and sym not in _logged_unmapped:
        _logged_unmapped.add(sym)
        logger.debug(
            "symbol %s is a known event instrument with no prices-table "
            "equivalent (UNMAPPED_EVENT_INSTRUMENTS) — price/vol lookups "
            "will resolve to nothing until the series is ingested (CL-5lpp)",
            sym,
        )
    return sym


class DataProvider:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def get_aligned_series(
        self, symbols: list[str], start: datetime, end: datetime,
    ) -> pd.DataFrame | None:
        """Return a DataFrame indexed by date with one column per requested symbol.

        Queries both `prices` (yfinance daily) and `macro_data` (FRED) and
        merges the results. Symbols missing from one table fall through to
        the other (CL-5rtm — pre-fix, prices-only short-circuit dropped any
        FRED-only symbol like DE_10Y). Returns None only if NO data is found
        for any requested symbol.
        """
        # Normalize OANDA/event-style ids → prices-table symbols before
        # querying (CL-5lpp). Additive: DB-native names pass through.
        symbols = [_normalize_symbol(s) for s in symbols]

        # FRED symbol routing for symbols not found in `prices`.
        fred_map = {
            "US_2Y": "DGS2", "US_10Y": "DGS10",
            "DE_2Y": "IRLTLT01DEM156N", "DE_10Y": "IRLTLT01DEM156N",
            "EURUSD": "DEXUSEU", "US_FEDFUNDS": "DFF",
        }

        per_symbol: list[pd.Series] = []

        # Pass 1: prices table.
        try:
            df = pd.read_sql(
                text("""
                    SELECT ts, symbol, close FROM prices
                    WHERE symbol = ANY(:symbols)
                      AND ts >= :start AND ts <= :end
                    ORDER BY ts
                """),
                self.engine,
                params={"symbols": symbols, "start": start, "end": end},
            )
            if not df.empty:
                df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
                pivot = df.pivot_table(
                    index="ts", columns="symbol", values="close", aggfunc="last",
                )
                for col in pivot.columns:
                    per_symbol.append(pivot[col].astype(float))
        except Exception:
            logger.exception("get_aligned_series: prices query failed")

        # Pass 2: macro_data for any symbol still missing. Legacy aliases go
        # through fred_map (US_10Y → DGS10); anything else is looked up by
        # its OWN series_id (CL-gr8o follow-up — the old fred_map-only gate
        # silently dropped direct macro ids like US2Y_MINUS_DE2Y / CVIX /
        # USD_3M_OIS, so the rate-diff refit never saw its spread series
        # even after the data was ingested).
        found = {s.name for s in per_symbol}
        missing = [s for s in symbols if s not in found]
        if missing:
            fred_to_caller = {fred_map.get(s, s): s for s in missing}
            try:
                df = pd.read_sql(
                    text("""
                        SELECT DISTINCT ON (observation_date, series_id)
                            observation_date AS ts, series_id, value AS close
                        FROM macro_data
                        WHERE series_id = ANY(:symbols)
                          AND observation_date >= :start AND observation_date <= :end
                        ORDER BY observation_date, series_id, release_date DESC
                    """),
                    self.engine,
                    params={
                        "symbols": list(fred_to_caller.keys()),
                        "start": start.date(), "end": end.date(),
                    },
                )
                if not df.empty:
                    df["ts"] = pd.to_datetime(df["ts"])
                    pivot = df.pivot_table(
                        index="ts", columns="series_id", values="close", aggfunc="last",
                    )
                    # Rename FRED series_id → caller-facing symbol name.
                    pivot.columns = [fred_to_caller[c] for c in pivot.columns]
                    for col in pivot.columns:
                        per_symbol.append(pivot[col].astype(float))
            except Exception:
                logger.exception("get_aligned_series: macro_data query failed")

        if not per_symbol:
            return None
        # Concat with outer-join on the union of indices. Different sources
        # (daily yfinance vs monthly FRED) will be sparse outside their own
        # timestamps; callers can ffill if they want continuous coverage.
        return pd.concat(per_symbol, axis=1).sort_index()

    def get_latest_rate_spread(self) -> float | None:
        try:
            query = text("""
                SELECT us.close - de.close AS spread
                FROM prices us
                JOIN prices de ON us.ts = de.ts
                WHERE us.symbol = 'US_2Y' AND de.symbol = 'DE_2Y'
                ORDER BY us.ts DESC LIMIT 1
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()
            return float(result[0]) if result else None
        except Exception:
            logger.warning(
                "get_latest_rate_spread (US_2Y - DE_2Y) failed",
                exc_info=True,
            )
            return None

    def get_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = text("""
            SELECT ts, symbol, close FROM prices
            WHERE ts >= :start AND ts <= :end
            ORDER BY ts
        """)
        df = pd.read_sql(query, self.engine, params={"start": start, "end": end})
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        return df.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last")

    def get_latest_value(
        self, series_id: str, as_of: datetime,
    ) -> float | None:
        """Return the most recent value of ``series_id`` at or before
        ``as_of``. Looks in ``macro_data`` first (FRED-style series) then
        falls back to ``prices`` (price-like series). Returns None if
        no value exists in either table at or before the cutoff.

        Added for CL-9eli — carry_vol_filter was logging errors every
        signal interval because this method didn't exist; the missing
        methods produced 33 errors over 24h and were a candidate for
        the memory leak in CL-2yta (logger.exception retains traceback
        objects in tight loops)."""
        # Normalize OANDA/event-style ids → prices-table symbols (CL-5lpp)
        # so confluence Gate B can resolve event legs like XAU_USD → GOLD.
        series_id = _normalize_symbol(series_id)
        try:
            with self.engine.connect() as conn:
                # macro_data.observation_date + value
                row = conn.execute(
                    text("""
                        SELECT value FROM macro_data
                        WHERE series_id = :sid
                          AND observation_date <= :as_of
                        ORDER BY observation_date DESC, release_date DESC
                        LIMIT 1
                    """),
                    {"sid": series_id, "as_of": as_of.date()},
                ).fetchone()
                if row is not None and row[0] is not None:
                    return float(row[0])
                # Fall back to prices table
                row = conn.execute(
                    text("""
                        SELECT close FROM prices
                        WHERE symbol = :sid AND ts <= :as_of
                        ORDER BY ts DESC LIMIT 1
                    """),
                    {"sid": series_id, "as_of": as_of},
                ).fetchone()
                if row is not None and row[0] is not None:
                    return float(row[0])
        except Exception as exc:
            logger.warning(
                "get_latest_value(%s, %s) failed: %s: %s",
                series_id, as_of, type(exc).__name__, exc,
            )
        return None

    def get_intraday_value(
        self,
        symbol: str,
        as_of: datetime,
        max_staleness_minutes: int | None = None,
    ) -> float | None:
        """Most recent intraday mid at or before ``as_of`` from
        ``intraday_quotes`` (CL-dz71), or None.

        Keyed by the RAW OANDA instrument id (XAU_USD, BCO_USD, ...) — the
        feed's own vocabulary — so this deliberately does NOT call
        ``_normalize_symbol``; that also lets it serve instruments with no
        daily prices-table series (XAG_USD, NATGAS_USD, ...).

        ``max_staleness_minutes`` bounds how old the nearest quote may be
        (so a dead poller doesn't hand back an ancient price as "current",
        and a seen_at with no nearby quote returns None → the caller falls
        back to the daily close). A missing ``intraday_quotes`` table (pre
        migration 012) is caught and returns None.
        """
        floor = (
            as_of - timedelta(minutes=max_staleness_minutes)
            if max_staleness_minutes is not None else None
        )
        try:
            with self.engine.connect() as conn:
                if floor is not None:
                    row = conn.execute(
                        text("""
                            SELECT mid FROM intraday_quotes
                            WHERE symbol = :sid AND ts <= :as_of AND ts >= :floor
                            ORDER BY ts DESC LIMIT 1
                        """),
                        {"sid": symbol, "as_of": as_of, "floor": floor},
                    ).fetchone()
                else:
                    row = conn.execute(
                        text("""
                            SELECT mid FROM intraday_quotes
                            WHERE symbol = :sid AND ts <= :as_of
                            ORDER BY ts DESC LIMIT 1
                        """),
                        {"sid": symbol, "as_of": as_of},
                    ).fetchone()
                if row is not None and row[0] is not None:
                    return float(row[0])
        except Exception as exc:
            logger.debug(
                "get_intraday_value(%s, %s) failed (table missing?): %s: %s",
                symbol, as_of, type(exc).__name__, exc,
            )
        return None

    def get_series(
        self, series_id: str, start: datetime, end: datetime,
    ) -> pd.Series:
        """Return a time-indexed Series of ``series_id`` between
        ``start`` and ``end``. Same fallback rule as
        ``get_latest_value`` — macro_data first, prices second.
        Empty Series if no data found."""
        # Normalize OANDA/event-style ids → prices-table symbols (CL-5lpp).
        series_id = _normalize_symbol(series_id)
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text("""
                        SELECT observation_date AS ts, value AS v
                        FROM macro_data
                        WHERE series_id = :sid
                          AND observation_date >= :start
                          AND observation_date <= :end
                        ORDER BY observation_date
                    """),
                    conn,
                    params={
                        "sid": series_id,
                        "start": start.date(), "end": end.date(),
                    },
                )
                if df.empty:
                    df = pd.read_sql(
                        text("""
                            SELECT ts, close AS v FROM prices
                            WHERE symbol = :sid
                              AND ts >= :start AND ts <= :end
                            ORDER BY ts
                        """),
                        conn,
                        params={
                            "sid": series_id, "start": start, "end": end,
                        },
                    )
        except Exception as exc:
            logger.warning(
                "get_series(%s) failed: %s: %s",
                series_id, type(exc).__name__, exc,
            )
            return pd.Series(dtype=float)
        if df.empty:
            return pd.Series(dtype=float)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")["v"].astype(float)

    def get_realized_vol(
        self, pair: str, window: int = 20, as_of: datetime | None = None,
    ) -> float | None:
        """Annualized realized volatility from the most recent ``window``
        daily closes of ``pair`` in the prices table. Returns None when
        there aren't at least ``window+1`` rows available (need N+1 closes
        for N log-returns).

        Uses 252 trading days per year for annualization. Window is the
        number of daily returns; pass 20 for ~1-month vol, 63 for 3-month,
        252 for full-year. Added for CL-jxn (rate_diff_mr was the only
        consumer pre-CL-2yta; portfolio-level metrics added in CL-5lq
        also call this)."""
        # Normalize OANDA/event-style ids → prices-table symbols (CL-5lpp)
        # so the confluence Gate-B vol threshold resolves for event legs.
        pair = _normalize_symbol(pair)
        try:
            with self.engine.connect() as conn:
                params: dict[str, Any] = {"pair": pair, "n": window + 1}
                cutoff_clause = ""
                if as_of is not None:
                    cutoff_clause = "AND ts <= :cutoff"
                    params["cutoff"] = as_of
                df = pd.read_sql(
                    # CL-u59z: cutoff_clause is fixed SQL; all values bound.
                    text(f"""
                        SELECT close FROM prices
                        WHERE symbol = :pair {cutoff_clause}
                        ORDER BY ts DESC LIMIT :n
                    """),  # nosec B608
                    conn,
                    params=params,
                )
        except Exception as exc:
            logger.warning(
                "get_realized_vol(%s) failed: %s: %s",
                pair, type(exc).__name__, exc,
            )
            return None
        if len(df) < window + 1:
            return None
        # Order ASC so log-returns are chronological; sign flip on diff
        # doesn't affect std but keeps semantics readable.
        closes = df["close"].astype(float).iloc[::-1].reset_index(drop=True)
        log_ret = (closes / closes.shift(1)).map(
            lambda x: 0.0 if x is None or x <= 0 else float(np.log(x)),
        ).dropna()
        if log_ret.empty:
            return None
        return float(log_ret.std(ddof=1) * (252.0 ** 0.5))

    # ------------------------------------------------------------------
    # Batched reads (CL-9ts9 / CL-8s2a) — confluence Gate B looked prices
    # and vol up PER event × PER instrument, opening a NEW connection each
    # call → hundreds of PG round-trips per poll as the ASSESSED backlog
    # grows. These helpers collect the FULL instrument set for a tick and
    # read each metric ONCE (one connection, one query per table) with an
    # IN/ANY over the set, so the cost is O(1) connections + O(1) queries
    # per metric instead of O(events × instruments). Results are keyed by
    # the CALLER's original id (the OANDA/event id it passed) so the
    # confluence layer can look them up directly; the byte-identical
    # per-symbol path (get_latest_value / get_intraday_value /
    # get_realized_vol) is still there for cache misses and non-batched
    # callers.
    # ------------------------------------------------------------------

    def get_latest_values_batch(
        self, symbols: list[str], as_of: datetime,
    ) -> dict[str, float]:
        """Most-recent value at/before ``as_of`` for each of ``symbols``,
        keyed by the caller's ORIGINAL id. macro_data first then prices,
        exactly mirroring :meth:`get_latest_value` — but for the whole set
        in ONE connection (two IN queries). Symbols with no value are
        simply absent from the returned dict (same as None per-symbol).

        SQL is portable (expanding IN + row_number() window) so it runs on
        the pg live path AND the sqlite test fixtures — the Gate-B parity
        test exercises it against real in-memory data."""
        out: dict[str, float] = {}
        if not symbols:
            return out
        # original id → normalized DB symbol (many originals can map to one
        # normalized symbol, e.g. BCO_USD & WTICO_USD → OIL_WTI).
        norm = {s: _normalize_symbol(s) for s in symbols}
        db_symbols = sorted(set(norm.values()))
        macro: dict[str, float] = {}
        prices: dict[str, float] = {}
        try:
            with self.engine.connect() as conn:
                macro_q = text("""
                    SELECT series_id, value FROM (
                        SELECT series_id, value,
                               row_number() OVER (
                                   PARTITION BY series_id
                                   ORDER BY observation_date DESC, release_date DESC
                               ) AS rn
                        FROM macro_data
                        WHERE series_id IN :sids AND observation_date <= :as_of
                    ) t WHERE rn = 1
                """).bindparams(bindparam("sids", expanding=True))
                for row in conn.execute(
                    macro_q, {"sids": db_symbols, "as_of": as_of.date()},
                ):
                    if row[1] is not None:
                        macro[str(row[0])] = float(row[1])
                still = [s for s in db_symbols if s not in macro]
                if still:
                    prices_q = text("""
                        SELECT symbol, close FROM (
                            SELECT symbol, close,
                                   row_number() OVER (
                                       PARTITION BY symbol ORDER BY ts DESC
                                   ) AS rn
                            FROM prices
                            WHERE symbol IN :sids AND ts <= :as_of
                        ) t WHERE rn = 1
                    """).bindparams(bindparam("sids", expanding=True))
                    for row in conn.execute(
                        prices_q, {"sids": still, "as_of": as_of},
                    ):
                        if row[1] is not None:
                            prices[str(row[0])] = float(row[1])
        except Exception as exc:
            logger.warning(
                "get_latest_values_batch(%d symbols, %s) failed: %s: %s",
                len(db_symbols), as_of, type(exc).__name__, exc,
            )
            return out
        for orig, db_sym in norm.items():
            if db_sym in macro:
                out[orig] = macro[db_sym]
            elif db_sym in prices:
                out[orig] = prices[db_sym]
        return out

    def get_intraday_values_batch(
        self,
        symbols: list[str],
        as_of: datetime,
        max_staleness_minutes: int | None = None,
    ) -> dict[str, float]:
        """Nearest intraday mid at/before ``as_of`` (within the staleness
        bound) for each of ``symbols`` in ONE query — batched
        :meth:`get_intraday_value`. Keyed by the RAW OANDA id (NO
        normalization), matching the per-symbol method. Missing/absent
        symbols are omitted; a missing table returns {} (logged debug)."""
        out: dict[str, float] = {}
        if not symbols:
            return out
        uniq = sorted(set(symbols))
        floor = (
            as_of - timedelta(minutes=max_staleness_minutes)
            if max_staleness_minutes is not None else None
        )
        params: dict[str, Any] = {"sids": uniq, "as_of": as_of}
        floor_clause = ""
        if floor is not None:
            floor_clause = "AND ts >= :floor"
            params["floor"] = floor
        try:
            with self.engine.connect() as conn:
                # CL-u59z: fixed floor_clause; sids/as_of/floor bound.
                q = text(f"""
                    SELECT symbol, mid FROM (
                        SELECT symbol, mid,
                               row_number() OVER (
                                   PARTITION BY symbol ORDER BY ts DESC
                               ) AS rn
                        FROM intraday_quotes
                        WHERE symbol IN :sids AND ts <= :as_of {floor_clause}
                    ) t WHERE rn = 1
                """).bindparams(bindparam("sids", expanding=True))  # nosec B608
                for row in conn.execute(q, params):
                    if row[1] is not None:
                        out[str(row[0])] = float(row[1])
        except Exception as exc:
            logger.debug(
                "get_intraday_values_batch(%d symbols, %s) failed "
                "(table missing?): %s: %s",
                len(uniq), as_of, type(exc).__name__, exc,
            )
        return out

    def get_realized_vols_batch(
        self, symbols: list[str], window: int, as_of: datetime,
    ) -> dict[str, float]:
        """Annualized realized vol for each of ``symbols`` in ONE query,
        keyed by the caller's ORIGINAL id — batched
        :meth:`get_realized_vol`. Pulls window+1 closes per (normalized)
        symbol via a windowed row_number(), then reuses the identical
        per-symbol log-return math so a batched verdict is byte-identical
        to the per-instrument path. Symbols with < window+1 closes are
        omitted (same as None per-symbol)."""
        out: dict[str, float] = {}
        if not symbols:
            return out
        norm = {s: _normalize_symbol(s) for s in symbols}
        db_symbols = sorted(set(norm.values()))
        try:
            q = text("""
                SELECT symbol, close FROM (
                    SELECT symbol, ts, close,
                           row_number() OVER (
                               PARTITION BY symbol ORDER BY ts DESC
                           ) AS rn
                    FROM prices
                    WHERE symbol IN :sids AND ts <= :cutoff
                ) t
                WHERE rn <= :n
                ORDER BY symbol, ts DESC
            """).bindparams(bindparam("sids", expanding=True))
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    q, conn,
                    params={"sids": db_symbols, "cutoff": as_of, "n": window + 1},
                )
        except Exception as exc:
            logger.warning(
                "get_realized_vols_batch(%d symbols) failed: %s: %s",
                len(db_symbols), type(exc).__name__, exc,
            )
            return out
        if df.empty:
            return out
        per_db: dict[str, float] = {}
        for db_sym, group in df.groupby("symbol"):
            if len(group) < window + 1:
                continue
            # group is ts DESC; reverse to chronological — identical to the
            # per-symbol get_realized_vol math (iloc[::-1] + log returns).
            closes = group["close"].astype(float).iloc[::-1].reset_index(drop=True)
            log_ret = (closes / closes.shift(1)).map(
                lambda x: 0.0 if x is None or x <= 0 else float(np.log(x)),
            ).dropna()
            if log_ret.empty:
                continue
            per_db[str(db_sym)] = float(log_ret.std(ddof=1) * (252.0 ** 0.5))
        for orig, db_sym in norm.items():
            if db_sym in per_db:
                out[orig] = per_db[db_sym]
        return out
