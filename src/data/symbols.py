"""US-listed symbol universe (CL-tzug) — ticker verification + company-name
resolution foundation for the niche opportunity agent (CL-u2ph).

Ingests the two free, keyless NASDAQ Trader public files:

* ``nasdaqlisted.txt`` — NASDAQ-listed securities. Pipe-delimited::

      Symbol|Security Name|Market Category|Test Issue|Financial Status|
      Round Lot Size|ETF|NextShares

* ``otherlisted.txt`` — everything else on the US tape (NYSE, AMEX, ARCA,
  BATS, …). Pipe-delimited::

      ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|
      Test Issue|NASDAQ Symbol

Both files carry a header row and a ``File Creation Time: …`` trailer line
(both skipped), and a ``Test Issue = Y`` flag that marks a test security
(excluded from ingest). ``ETF = Y`` flags an exchange-traded fund.

The union is the full US-listed universe (~13k rows: NASDAQ + NYSE + AMEX +
ARCA + ETFs), which is exactly the Robinhood-tradeable scope.

HONEST COVERAGE GAP: this is US-listed only. Comprehensive OTC / pink-sheets
and non-US / global exchanges are NOT present in these free files (they need
OTC Markets data or a paid vendor). Some ADRs are present (they list on the
US exchanges), but a symbol *missing* from this universe means "not a
US-listed name we can see", NOT definitively "untradeable". ``robinhood_tradeable``
returning ``False`` is likewise "not in the US-listed universe", not a hard no.

The files update every trading day; refresh via ``scripts/refresh_symbols.py``
(wire into the Airflow event_ingestion DAG or a daily cron later).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

#: The trailer line both files end with; not a data row.
_TRAILER_PREFIX = "File Creation Time"

#: otherlisted.txt Exchange-code → canonical exchange label.
_EXCHANGE_CODE_MAP = {
    "N": "NYSE",
    "A": "AMEX",
    "P": "ARCA",
    "Z": "OTHER",  # BATS/CBOE
    "V": "OTHER",  # IEX
}

#: Injectable HTTP shim so unit tests feed canned file bodies (no live network).
HttpGet = Callable[[str], str]


def _default_http_get(url: str) -> str:
    """Real fetch: httpx GET with a tolerant ~15s timeout, returns the body."""
    resp = httpx.get(url, timeout=15.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.split("|")]


def parse_nasdaq_listed(body: str) -> list[dict[str, Any]]:
    """Parse ``nasdaqlisted.txt`` → normalized symbol dicts.

    Skips the header row, the ``File Creation Time`` trailer, blank lines, and
    ``Test Issue = Y`` rows. Every returned row is exchange ``NASDAQ``.
    """
    return _parse_file(
        body,
        symbol_idx=0,
        name_idx=1,
        etf_idx=6,
        test_issue_idx=3,
        exchange_fn=lambda _cols: "NASDAQ",
        source="nasdaqlisted",
        expected_header_start="Symbol",
    )


def parse_other_listed(body: str) -> list[dict[str, Any]]:
    """Parse ``otherlisted.txt`` → normalized symbol dicts.

    Uses the ``ACT Symbol`` column (col 0). Maps the Exchange code (col 2)
    N→NYSE, A→AMEX, P→ARCA, else OTHER. Skips header / trailer / blank /
    ``Test Issue = Y`` rows.
    """
    return _parse_file(
        body,
        symbol_idx=0,
        name_idx=1,
        etf_idx=4,
        test_issue_idx=6,
        exchange_fn=lambda cols: _EXCHANGE_CODE_MAP.get(
            cols[2] if len(cols) > 2 else "", "OTHER",
        ),
        source="otherlisted",
        expected_header_start="ACT Symbol",
    )


def _parse_file(
    body: str,
    *,
    symbol_idx: int,
    name_idx: int,
    etf_idx: int,
    test_issue_idx: int,
    exchange_fn: Callable[[list[str]], str],
    source: str,
    expected_header_start: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(_TRAILER_PREFIX):
            continue  # trailer
        if line.startswith(expected_header_start):
            continue  # header
        cols = _split_row(line)
        if len(cols) <= max(symbol_idx, name_idx, etf_idx, test_issue_idx):
            continue  # malformed / short line
        symbol = cols[symbol_idx]
        if not symbol:
            continue
        # Exclude test issues at ingest.
        if cols[test_issue_idx].upper() == "Y":
            continue
        rows.append({
            "symbol": symbol,
            "security_name": cols[name_idx] or None,
            "exchange": exchange_fn(cols),
            "is_etf": cols[etf_idx].upper() == "Y",
            "is_test_issue": False,
            "source": source,
        })
    return rows


class SymbolUniverse:
    """The US-listed symbol universe backed by the ``symbols`` table.

    Loads (and caches in-memory) a dict keyed by UPPER-cased ticker for fast
    ``exists()`` / ``get()`` / ``is_etf()`` lookups. ``refresh()`` re-fetches
    the NASDAQ Trader files, upserts them, and rebuilds the cache.
    """

    def __init__(
        self, engine: Engine, http_get: HttpGet | None = None,
    ) -> None:
        self.engine = engine
        self._http_get: HttpGet = http_get or _default_http_get
        #: UPPER-symbol → row dict, or None until first load.
        self._cache: dict[str, dict[str, Any]] | None = None

    # -- ingest ---------------------------------------------------------- #

    def refresh(self) -> dict[str, int]:
        """Fetch + parse both files and upsert into ``symbols``.

        Tolerant of a single-file fetch failure: if one URL is down we log it
        and ingest whatever the other file yields. Returns
        ``{"inserted", "updated", "skipped"}`` counts.
        """
        parsed: list[dict[str, Any]] = []
        for url, parser in (
            (NASDAQ_LISTED_URL, parse_nasdaq_listed),
            (OTHER_LISTED_URL, parse_other_listed),
        ):
            try:
                body = self._http_get(url)
            except Exception:
                logger.warning(
                    "symbol refresh: fetch failed for %s — continuing with "
                    "the other file", url, exc_info=True,
                )
                continue
            file_rows = parser(body)
            logger.info("symbol refresh: parsed %d rows from %s",
                        len(file_rows), url)
            parsed.extend(file_rows)

        if not parsed:
            logger.error("symbol refresh: no rows parsed from either file")
            return {"inserted": 0, "updated": 0, "skipped": 0}

        # De-dup within the batch (a ticker could appear on both tapes; last
        # write wins, which is fine — same instrument). Keep insertion order.
        deduped: dict[str, dict[str, Any]] = {}
        skipped_dupes = 0
        for row in parsed:
            key = row["symbol"].upper()
            if key in deduped:
                skipped_dupes += 1
            deduped[key] = row

        now = datetime.now(UTC)
        inserted, updated = self._upsert(list(deduped.values()), now)
        self._load_cache()  # rebuild in-memory view
        counts = {
            "inserted": inserted,
            "updated": updated,
            "skipped": skipped_dupes,
        }
        logger.info(
            "symbol refresh: inserted=%d updated=%d skipped=%d last_refreshed=%s",
            inserted, updated, skipped_dupes, now.isoformat(),
        )
        return counts

    def _upsert(
        self, rows: list[dict[str, Any]], now: datetime,
    ) -> tuple[int, int]:
        """ON CONFLICT(symbol) DO UPDATE upsert; returns (inserted, updated)."""
        if not rows:
            return 0, 0
        # Which symbols already exist → so we can attribute inserted vs updated
        # (xmax-style RETURNING is Postgres-specific; a pre-count keeps this
        # portable to the sqlite test engine).
        with self.engine.begin() as conn:
            existing = {
                r[0].upper()
                for r in conn.execute(text("SELECT symbol FROM symbols"))
            }
            for row in rows:
                conn.execute(
                    text("""
                        INSERT INTO symbols (
                            symbol, security_name, exchange, is_etf,
                            is_test_issue, source, last_refreshed
                        ) VALUES (
                            :symbol, :security_name, :exchange, :is_etf,
                            :is_test_issue, :source, :last_refreshed
                        )
                        ON CONFLICT (symbol) DO UPDATE SET
                            security_name = excluded.security_name,
                            exchange = excluded.exchange,
                            is_etf = excluded.is_etf,
                            is_test_issue = excluded.is_test_issue,
                            source = excluded.source,
                            last_refreshed = excluded.last_refreshed
                    """),
                    {**row, "last_refreshed": now},
                )
        inserted = sum(1 for r in rows if r["symbol"].upper() not in existing)
        updated = len(rows) - inserted
        return inserted, updated

    # -- cache ----------------------------------------------------------- #

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        cache: dict[str, dict[str, Any]] = {}
        with self.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT symbol, security_name, exchange, is_etf, is_test_issue "
                "FROM symbols",
            ))
            for symbol, name, exchange, is_etf, is_test in result:
                cache[symbol.upper()] = {
                    "symbol": symbol,
                    "security_name": name,
                    "exchange": exchange,
                    "is_etf": bool(is_etf),
                    "is_test_issue": bool(is_test),
                }
        self._cache = cache
        return cache

    def _ensure_cache(self) -> dict[str, dict[str, Any]]:
        if self._cache is None:
            return self._load_cache()
        return self._cache

    # -- lookups --------------------------------------------------------- #

    def exists(self, ticker: str) -> bool:
        """True if the ticker is a known (non-test) US-listed symbol.

        Case-insensitive. Test issues are never in the table so they read as
        absent.
        """
        if not ticker:
            return False
        row = self._ensure_cache().get(ticker.strip().upper())
        return row is not None and not row["is_test_issue"]

    def get(self, ticker: str) -> dict[str, Any] | None:
        """Return ``{symbol, security_name, exchange, is_etf}`` or ``None``."""
        if not ticker:
            return None
        row = self._ensure_cache().get(ticker.strip().upper())
        if row is None or row["is_test_issue"]:
            return None
        return {
            "symbol": row["symbol"],
            "security_name": row["security_name"],
            "exchange": row["exchange"],
            "is_etf": row["is_etf"],
        }

    def is_etf(self, ticker: str) -> bool:
        """True if the ticker resolves to an ETF in the universe."""
        if not ticker:
            return False
        row = self._ensure_cache().get(ticker.strip().upper())
        return bool(row and not row["is_test_issue"] and row["is_etf"])

    def robinhood_tradeable(self, ticker: str) -> bool:
        """True if the ticker is a US-listed common stock or ETF in the table.

        Every exchange in this universe (NASDAQ / NYSE / AMEX / ARCA) is
        Robinhood-tradeable, so presence in the table is the test.

        Caveat: OTC / pink-sheet and non-US / global exchanges are NOT in this
        free universe (they need OTC Markets / a paid vendor), so a ``False``
        means "not in the US-listed universe we can see", not definitively
        "untradeable".
        """
        return self.exists(ticker)

    def resolve_name(
        self, query: str, limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Company-name → candidate tickers (case-insensitive substring match).

        Handles the "LLM knows the company but guessed the wrong ticker" case
        (e.g. ``"Teva"`` → TEVA, ``"Frontline"`` → FRO). Ranked best-first:
        exact whole-word match > name starts with the query > plain substring;
        ties broken by shorter name then symbol. Returns up to ``limit`` dicts
        of ``{symbol, security_name, exchange, is_etf}``.
        """
        q = (query or "").strip().lower()
        if not q:
            return []
        scored: list[tuple[int, int, str, dict[str, Any]]] = []
        for row in self._ensure_cache().values():
            if row["is_test_issue"]:
                continue
            name = row["security_name"]
            if not name:
                continue
            lname = name.lower()
            if q not in lname:
                continue
            # Rank: 0 = exact whole-word hit, 1 = name starts with query,
            # 2 = plain substring (lower rank sorts first).
            words = lname.replace(",", " ").replace(".", " ").split()
            if q in words:
                rank = 0
            elif lname.startswith(q):
                rank = 1
            else:
                rank = 2
            scored.append((
                rank, len(name), row["symbol"],
                {
                    "symbol": row["symbol"],
                    "security_name": name,
                    "exchange": row["exchange"],
                    "is_etf": row["is_etf"],
                },
            ))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        return [item[3] for item in scored[:limit]]
