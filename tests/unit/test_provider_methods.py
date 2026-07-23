"""Tests for DataProvider's get_latest_value + get_series (CL-9eli).

These methods were missing before and carry_vol_filter was logging an
exception every signal interval (33 errors over 24h soak — a candidate
contributor to CL-2yta's memory leak). Tests verify both methods read
correctly from macro_data + prices fallback, return None / empty Series
when no data exists, and log warnings (not exceptions) on DB errors.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.data.provider import (
    _SYMBOL_ALIASES,
    UNMAPPED_EVENT_INSTRUMENTS,
    DataProvider,
    _normalize_symbol,
)


@pytest.fixture
def provider_engine(tmp_path):  # type: ignore[no-untyped-def]
    """In-memory sqlite with the production schema for prices +
    macro_data, seeded with two rows for testing."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE prices (
                ts TIMESTAMP, symbol VARCHAR(64),
                close FLOAT,
                PRIMARY KEY (ts, symbol)
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE macro_data (
                observation_date DATE, series_id VARCHAR(64),
                value FLOAT, release_date DATE,
                PRIMARY KEY (observation_date, series_id)
            )
        """)
        )
        # Macro: FRED-style — series_id 'DGS2' across 3 days
        conn.execute(
            text(
                "INSERT INTO macro_data VALUES "
                "('2026-04-01', 'DGS2', 4.5, '2026-04-02'), "
                "('2026-04-02', 'DGS2', 4.6, '2026-04-03'), "
                "('2026-04-03', 'DGS2', 4.7, '2026-04-04')",
            )
        )
        # Prices: symbol 'EURUSD' across 3 days
        conn.execute(
            text(
                "INSERT INTO prices VALUES "
                "('2026-04-01 00:00:00', 'EURUSD', 1.10), "
                "('2026-04-02 00:00:00', 'EURUSD', 1.11), "
                "('2026-04-03 00:00:00', 'EURUSD', 1.12)",
            )
        )
    return engine


class TestGetLatestValue:
    def test_macro_data_hit(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        v = provider.get_latest_value(
            "DGS2",
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert v == 4.7

    def test_uses_at_or_before_cutoff(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        # Cutoff between rows; should return the last <= cutoff
        v = provider.get_latest_value(
            "DGS2",
            datetime(2026, 4, 2, 12, tzinfo=UTC),
        )
        assert v == 4.6

    def test_falls_back_to_prices(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        # 'EURUSD' isn't in macro_data; should fall to prices
        v = provider.get_latest_value(
            "EURUSD",
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert v == 1.12

    def test_unknown_returns_none(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        v = provider.get_latest_value(
            "NEVER_EXISTS",
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert v is None


class TestGetSeries:
    def test_macro_data_series(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        s = provider.get_series(
            "DGS2",
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert len(s) == 3
        assert list(s.values) == [4.5, 4.6, 4.7]

    def test_falls_back_to_prices(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        s = provider.get_series(
            "EURUSD",
            datetime(2026, 3, 31, tzinfo=UTC),
            datetime(2026, 4, 4, tzinfo=UTC),
        )
        # Inclusive bounds wide enough that all 3 rows land regardless
        # of sqlite/pg timezone quirks
        assert len(s) >= 2
        assert s.iloc[-1] == 1.12

    def test_empty_when_unknown(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        s = provider.get_series(
            "GHOST",
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert isinstance(s, pd.Series)
        assert s.empty

    def test_window_filters(self, provider_engine) -> None:
        provider = DataProvider(provider_engine)
        # Only the middle row should survive
        s = provider.get_series(
            "DGS2",
            datetime(2026, 4, 2, tzinfo=UTC),
            datetime(2026, 4, 2, 23, 59, tzinfo=UTC),
        )
        assert len(s) == 1
        assert s.iloc[0] == 4.6


class TestGetRealizedVol:
    def test_basic(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        # Seed 22 sequential closes to ensure 21 returns for window=20.
        with provider_engine.begin() as conn:
            from sqlalchemy import text as _t

            for i in range(22):
                conn.execute(
                    _t(
                        "INSERT INTO prices VALUES "
                        f"('2026-01-{i + 1:02d} 00:00:00', 'AUDUSD', "
                        f"{1.10 + 0.001 * i})",
                    )
                )
        provider = DataProvider(provider_engine)
        v = provider.get_realized_vol("AUDUSD", window=20)
        assert v is not None
        # Annualized vol of monotonic small steps is finite, > 0.
        assert 0 < v < 5.0

    def test_too_few_rows(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        provider = DataProvider(provider_engine)
        # Default fixture has 3 EURUSD rows; window=20 needs 21.
        assert provider.get_realized_vol("EURUSD", window=20) is None

    def test_unknown_pair(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        provider = DataProvider(provider_engine)
        assert provider.get_realized_vol("GHOST", window=5) is None


class TestErrorPath:
    def test_db_error_returns_none_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Engine that fails on every query (schema mismatch — no tables)
        bad_engine = create_engine("sqlite:///:memory:")
        provider = DataProvider(bad_engine)
        with caplog.at_level("WARNING"):
            v = provider.get_latest_value(
                "X",
                datetime(2026, 4, 1, tzinfo=UTC),
            )
        assert v is None
        # Error logged at WARNING (not exception — see CL-2yta defense)
        assert any("get_latest_value" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# CL-5lpp — OANDA/event-id → prices-table symbol normalization.
#
# The event pipeline speaks OANDA ids (XAU_USD, BCO_USD, USD_JPY); the
# prices table holds yfinance symbols (GOLD, OIL_WTI, USDJPY). Without
# normalization every event-leg price/vol lookup returned nothing and
# confluence Gate B could never confirm (164 EXPIRED / 0 CONFIRMED).
# ---------------------------------------------------------------------------


class TestNormalizeSymbol:
    @pytest.mark.parametrize(
        ("oanda", "db"),
        [
            ("BCO_USD", "OIL_WTI"),  # Brent → WTI proxy (documented)
            ("WTICO_USD", "OIL_WTI"),
            ("XAU_USD", "GOLD"),
            ("XCU_USD", "COPPER"),
            ("USD_JPY", "USDJPY"),
            ("USD_CAD", "USDCAD"),
            ("USD_CHF", "USDCHF"),
            ("EUR_USD", "EURUSD"),
            ("GBP_USD", "GBPUSD"),
            ("AUD_USD", "AUDUSD"),
            ("NZD_USD", "NZDUSD"),
            ("SPX500_USD", "SPX"),
        ],
    )
    def test_maps_each_alias(self, oanda: str, db: str) -> None:
        assert _normalize_symbol(oanda) == db

    @pytest.mark.parametrize(
        "native",
        [
            "EURUSD",
            "US_10Y",
            "US_2Y",
            "DE_2Y",
            "DGS2",
            "DGS10",
            "GOLD",
            "OIL_WTI",
            "USDJPY",
            "COPPER",
            "SPX",
            "VIX",
            "DXY",
        ],
    )
    def test_db_native_passes_through(self, native: str) -> None:
        # DB-native names (and FRED series) must be returned unchanged so
        # the existing FX/macro strategies keep working.
        assert _normalize_symbol(native) == native

    @pytest.mark.parametrize(
        "unmapped",
        [
            "XAG_USD",
            "XPT_USD",
            "XPD_USD",
            "USD_NOK",
            "USD_ZAR",
            "USD_CNH",
            "NATGAS_USD",
            "WHEAT_USD",
            "CORN_USD",
            "NAS100_USD",
        ],
    )
    def test_unmapped_event_ids_pass_through(self, unmapped: str) -> None:
        # No DB equivalent yet — pass through unchanged so the lookup
        # honestly resolves to nothing (not a crash, not a wrong series).
        assert _normalize_symbol(unmapped) == unmapped
        assert unmapped in UNMAPPED_EVENT_INSTRUMENTS

    def test_no_alias_collides_with_a_db_native_key(self) -> None:
        # Purely additive: an alias KEY must never also be a mapped VALUE
        # (would risk double-translation) — keys are OANDA ids, values are
        # DB symbols, disjoint sets.
        keys = set(_SYMBOL_ALIASES)
        values = set(_SYMBOL_ALIASES.values())
        assert keys.isdisjoint(values)
        # And no OANDA id is simultaneously classed as unmapped.
        assert keys.isdisjoint(UNMAPPED_EVENT_INSTRUMENTS)

    def test_brent_to_wti_proxy_is_documented(self) -> None:
        # The Brent→WTI substitution is a proxy (no Brent series exists);
        # guard that the mapping is present and documented in the source.
        assert _SYMBOL_ALIASES["BCO_USD"] == "OIL_WTI"
        import inspect

        import src.data.provider as prov

        src = inspect.getsource(prov)
        assert "Brent" in src and "proxy" in src.lower()

    def test_unmapped_miss_logs_once_at_debug(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A known-unmapped event id logs a discoverable DEBUG line, but
        # only once per distinct symbol (no per-call spam in tight loops).
        import src.data.provider as prov

        prov._logged_unmapped.discard("CORN_USD")
        with caplog.at_level("DEBUG", logger="src.data.provider"):
            _normalize_symbol("CORN_USD")
            _normalize_symbol("CORN_USD")
        hits = [r for r in caplog.records if "CORN_USD" in r.message]
        assert len(hits) == 1


class TestReadMethodsNormalize:
    """Each read method must translate the OANDA id to the DB symbol
    BEFORE it hits the database — proven by mocking the engine and
    asserting the bound query parameter is the DB-native symbol."""

    def _mock_engine(self):  # type: ignore[no-untyped-def]
        """Engine whose connect() yields a conn recording execute() args
        and returning an empty result (we only care about the bound sym)."""
        engine = MagicMock()
        conn = MagicMock()
        result = MagicMock()
        result.fetchone.return_value = None
        conn.execute.return_value = result
        engine.connect.return_value.__enter__.return_value = conn
        return engine, conn

    def test_get_latest_value_normalizes(self) -> None:
        engine, conn = self._mock_engine()
        DataProvider(engine).get_latest_value("XAU_USD", datetime(2026, 4, 1, tzinfo=UTC))
        # First execute is macro_data (series_id), then prices (sid) — both
        # must carry the normalized DB symbol, never the OANDA id.
        bound = [c.args[1] for c in conn.execute.call_args_list if len(c.args) > 1]
        for params in bound:
            sid = params.get("sid")
            if sid is not None:
                assert sid == "GOLD"
        assert any(p.get("sid") == "GOLD" for p in bound)
        assert not any(p.get("sid") == "XAU_USD" for p in bound)

    def test_get_series_normalizes(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # get_series uses pd.read_sql; intercept it and capture params.
        captured: dict = {}

        def fake_read_sql(_sql, _conn, params=None):  # type: ignore[no-untyped-def]
            captured.setdefault("sids", []).append((params or {}).get("sid"))
            return pd.DataFrame()  # empty → falls through both branches

        engine, _ = self._mock_engine()
        monkeypatch.setattr("src.data.provider.pd.read_sql", fake_read_sql)
        DataProvider(engine).get_series(
            "USD_JPY",
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 2, tzinfo=UTC),
        )
        assert "USDJPY" in captured["sids"]
        assert "USD_JPY" not in captured["sids"]

    def test_get_realized_vol_normalizes(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        captured: dict = {}

        def fake_read_sql(_sql, _conn, params=None):  # type: ignore[no-untyped-def]
            captured["pair"] = (params or {}).get("pair")
            return pd.DataFrame()

        engine, _ = self._mock_engine()
        monkeypatch.setattr("src.data.provider.pd.read_sql", fake_read_sql)
        DataProvider(engine).get_realized_vol("BCO_USD", window=20)
        assert captured["pair"] == "OIL_WTI"  # Brent → WTI proxy

    def test_get_aligned_series_normalizes(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        captured: dict = {}

        def fake_read_sql(_sql, _engine, params=None):  # type: ignore[no-untyped-def]
            captured.setdefault("symbols", []).append((params or {}).get("symbols"))
            return pd.DataFrame()

        engine, _ = self._mock_engine()
        monkeypatch.setattr("src.data.provider.pd.read_sql", fake_read_sql)
        DataProvider(engine).get_aligned_series(
            ["XAU_USD", "EURUSD", "WHEAT_USD"],
            datetime(2026, 4, 1, tzinfo=UTC),
            datetime(2026, 4, 2, tzinfo=UTC),
        )
        # The prices-table query gets normalized ids: XAU_USD→GOLD,
        # EURUSD (native) unchanged, WHEAT_USD (unmapped) unchanged.
        prices_call = captured["symbols"][0]
        assert prices_call == ["GOLD", "EURUSD", "WHEAT_USD"]


class TestBatchReads:
    """Batched price/vol helpers (CL-9ts9 / CL-8s2a) — one connection, one
    query per metric over the whole instrument set. Each must return
    BYTE-IDENTICAL values to the per-symbol path (the batch is a pure
    round-trip reduction, never a semantic change), key by the caller's
    ORIGINAL id, honor the OANDA→DB normalization, and no-op on []."""

    def _seed_vol(self, engine) -> None:  # type: ignore[no-untyped-def]
        with engine.begin() as conn:
            for i in range(25):
                d = f"2026-02-{i + 1:02d} 00:00:00"
                conn.execute(
                    text(
                        f"INSERT INTO prices VALUES ('{d}', 'GOLD', {2400.0 + i * 1.5})",
                    )
                )
                conn.execute(
                    text(
                        f"INSERT INTO prices VALUES ('{d}', 'USDJPY', {150.0 + i * 0.05})",
                    )
                )

    def test_latest_values_batch_matches_per_symbol(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        with provider_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO prices VALUES ('2026-04-03 00:00:00', 'GOLD', 2400.0)",
                )
            )
        provider = DataProvider(provider_engine)
        as_of = datetime(2026, 4, 3, tzinfo=UTC)
        # XAU_USD→GOLD (prices), DGS2 (macro), EURUSD (prices).
        batch = provider.get_latest_values_batch(["XAU_USD", "DGS2", "EURUSD"], as_of)
        assert batch["XAU_USD"] == provider.get_latest_value("XAU_USD", as_of)
        assert batch["DGS2"] == provider.get_latest_value("DGS2", as_of)
        assert batch["EURUSD"] == provider.get_latest_value("EURUSD", as_of)
        # Keyed by the ORIGINAL id (the OANDA id, not the DB symbol GOLD).
        assert "GOLD" not in batch and batch["XAU_USD"] == 2400.0

    def test_latest_values_batch_omits_missing(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        provider = DataProvider(provider_engine)
        batch = provider.get_latest_values_batch(
            ["EURUSD", "NEVER_EXISTS"],
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert "EURUSD" in batch
        assert "NEVER_EXISTS" not in batch  # None per-symbol → absent in batch

    def test_realized_vols_batch_matches_per_symbol(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        self._seed_vol(provider_engine)
        provider = DataProvider(provider_engine)
        as_of = datetime(2026, 3, 1, tzinfo=UTC)
        batch = provider.get_realized_vols_batch(["XAU_USD", "USD_JPY"], 20, as_of)
        assert batch["XAU_USD"] == provider.get_realized_vol("XAU_USD", 20, as_of)
        assert batch["USD_JPY"] == provider.get_realized_vol("USD_JPY", 20, as_of)

    def test_realized_vols_batch_omits_thin_series(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        provider = DataProvider(provider_engine)
        # Default fixture EURUSD has 3 rows; window=20 needs 21 → omitted,
        # exactly as the per-symbol call returns None.
        batch = provider.get_realized_vols_batch(
            ["EURUSD"],
            20,
            datetime(2026, 4, 3, tzinfo=UTC),
        )
        assert "EURUSD" not in batch
        assert provider.get_realized_vol("EURUSD", 20) is None

    def test_intraday_values_batch_matches_per_symbol(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        with provider_engine.begin() as conn:
            conn.execute(
                text("""
                CREATE TABLE intraday_quotes (
                    ts TIMESTAMP, symbol TEXT, mid FLOAT
                )
            """)
            )
            conn.execute(
                text(
                    "INSERT INTO intraday_quotes VALUES "
                    "('2026-04-03 11:00:00', 'XAU_USD', 2401.0), "
                    "('2026-04-03 11:30:00', 'XAU_USD', 2402.0), "
                    "('2026-04-03 11:00:00', 'USD_CAD', 1.36)",
                )
            )
        provider = DataProvider(provider_engine)
        as_of = datetime(2026, 4, 3, 12, tzinfo=UTC)
        batch = provider.get_intraday_values_batch(["XAU_USD", "USD_CAD"], as_of, 120)
        # Nearest at/before as_of, RAW-keyed (no normalization), same as
        # the per-symbol get_intraday_value.
        assert batch["XAU_USD"] == provider.get_intraday_value("XAU_USD", as_of, 120)
        assert batch["XAU_USD"] == 2402.0
        assert batch["USD_CAD"] == provider.get_intraday_value("USD_CAD", as_of, 120)

    def test_intraday_batch_honors_staleness(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        with provider_engine.begin() as conn:
            conn.execute(
                text("""
                CREATE TABLE intraday_quotes (
                    ts TIMESTAMP, symbol TEXT, mid FLOAT
                )
            """)
            )
            conn.execute(
                text(
                    "INSERT INTO intraday_quotes VALUES ('2026-04-03 10:00:00', 'XAU_USD', 2400.0)",
                )
            )
        provider = DataProvider(provider_engine)
        as_of = datetime(2026, 4, 3, 12, tzinfo=UTC)  # quote is 120min old
        # 60min bound → too stale, omitted (matches per-symbol None).
        assert provider.get_intraday_values_batch(["XAU_USD"], as_of, 60) == {}
        assert provider.get_intraday_value("XAU_USD", as_of, 60) is None
        # 180min bound → within, returned.
        assert provider.get_intraday_values_batch(["XAU_USD"], as_of, 180) == {
            "XAU_USD": 2400.0,
        }

    def test_all_batches_noop_on_empty(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        provider = DataProvider(provider_engine)
        now = datetime(2026, 4, 3, tzinfo=UTC)
        assert provider.get_latest_values_batch([], now) == {}
        assert provider.get_realized_vols_batch([], 20, now) == {}
        assert provider.get_intraday_values_batch([], now, 120) == {}

    def test_batch_missing_table_returns_empty(self) -> None:
        # No tables at all — batch reads must fail soft (empty dict), never
        # raise (a missing intraday_quotes / prices must idle the caller).
        bad = create_engine("sqlite:///:memory:")
        provider = DataProvider(bad)
        now = datetime(2026, 4, 1, tzinfo=UTC)
        assert provider.get_latest_values_batch(["X"], now) == {}
        assert provider.get_realized_vols_batch(["X"], 20, now) == {}
        assert provider.get_intraday_values_batch(["X"], now, 120) == {}


class TestFxStrategyPathUnaffected:
    """The existing FX/macro strategy lookups query DB-native names and
    must resolve exactly as before — normalization is a no-op for them."""

    def test_native_fx_and_macro_still_resolve(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        provider = DataProvider(provider_engine)
        # EURUSD is in prices, DGS2 in macro_data — both DB-native.
        assert (
            provider.get_latest_value(
                "EURUSD",
                datetime(2026, 4, 3, tzinfo=UTC),
            )
            == 1.12
        )
        assert (
            provider.get_latest_value(
                "DGS2",
                datetime(2026, 4, 3, tzinfo=UTC),
            )
            == 4.7
        )

    def test_oanda_id_now_resolves_via_alias(self, provider_engine) -> None:  # type: ignore[no-untyped-def]
        # Seed GOLD directly, then look it up by the OANDA id XAU_USD.
        with provider_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO prices VALUES ('2026-04-03 00:00:00', 'GOLD', 2400.0)",
                )
            )
        provider = DataProvider(provider_engine)
        assert (
            provider.get_latest_value(
                "XAU_USD",
                datetime(2026, 4, 3, tzinfo=UTC),
            )
            == 2400.0
        )
