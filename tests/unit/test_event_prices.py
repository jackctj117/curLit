"""Unit tests — batch price snapshot helper (CL-mgcp).

No live network: the yfinance downloader is the injectable shim, the
OANDA-id path runs against an in-memory sqlite mirror of the
`prices`/`macro_data` tables. The failure posture under test is
"absent key, never raise, never a fake number".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy import text

from src.events.prices import (
    format_age,
    format_price,
    get_prices,
    parse_ts,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


def _yf_frame(closes_by_ticker: dict[str, list[float]]) -> pd.DataFrame:
    """(ticker, field) MultiIndex frame like yf.download(group_by='ticker')."""
    cols: dict[tuple[str, str], pd.Series] = {}
    for ticker, closes in closes_by_ticker.items():
        idx = pd.date_range(end="2026-07-18", periods=len(closes), freq="D")
        cols[(ticker, "Close")] = pd.Series(closes, index=idx)
        cols[(ticker, "Volume")] = pd.Series([1000] * len(closes), index=idx)
    return pd.DataFrame(cols)


class Recorder:
    """Downloader shim that records requested tickers."""

    def __init__(self, frame: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        self.frame = frame if frame is not None else pd.DataFrame()
        self.error = error
        self.calls: list[list[str]] = []

    def __call__(self, tickers: Any, start: Any, end: Any) -> pd.DataFrame:
        self.calls.append(list(tickers))
        if self.error is not None:
            raise self.error
        return self.frame


@pytest.fixture
def db() -> Any:
    """sqlite mirror of the provider tables the OANDA path reads.

    The prices table is keyed by DB-native (yfinance) symbols, NOT OANDA
    ids — Brent is stored as ``OIL_WTI`` (the WTI proxy), and the caller's
    OANDA id ``BCO_USD`` is translated to it by DataProvider's
    normalization layer (CL-5lpp). Before that fix, event-leg lookups
    queried the OANDA id directly, matched nothing, and no event ever
    confirmed.
    """
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE prices (ts TEXT, symbol TEXT, close REAL)"))
        conn.execute(
            text(
                "CREATE TABLE macro_data (series_id TEXT, observation_date TEXT, "
                "value REAL, release_date TEXT)",
            )
        )
        # Seed under OIL_WTI — the symbol BCO_USD normalizes to.
        for days_ago, close in ((2, 76.8), (1, 78.4)):
            conn.execute(
                text("INSERT INTO prices VALUES (:ts, 'OIL_WTI', :close)"),
                {"ts": (NOW - timedelta(days=days_ago)).isoformat(sep=" "), "close": close},
            )
    return engine


# --------------------------------------------------------------------- #
# Equity path (yfinance shim)
# --------------------------------------------------------------------- #


class TestEquities:
    def test_batch_last_close_and_change(self) -> None:
        dl = Recorder(_yf_frame({"FRO": [23.35, 24.10], "TSM": [175.6, 172.4]}))
        out = get_prices(["FRO", "TSM"], downloader=dl, now=NOW)
        assert out["FRO"]["price"] == pytest.approx(24.10)
        assert out["FRO"]["change_pct"] == pytest.approx(3.2119, abs=1e-3)
        assert out["TSM"]["change_pct"] == pytest.approx(-1.8223, abs=1e-3)
        assert out["TSM"]["asof"].year == 2026
        assert dl.calls == [["FRO", "TSM"]]

    def test_single_ticker_flat_frame(self) -> None:
        idx = pd.date_range(end="2026-07-18", periods=2, freq="D")
        flat = pd.DataFrame({"Close": [10.0, 11.0], "Volume": [1, 1]}, index=idx)
        out = get_prices(["FRO"], downloader=Recorder(flat), now=NOW)
        assert out["FRO"]["price"] == pytest.approx(11.0)

    def test_single_close_has_no_change(self) -> None:
        dl = Recorder(_yf_frame({"FRO": [24.10]}))
        out = get_prices(["FRO"], downloader=dl, now=NOW)
        assert out["FRO"]["price"] == pytest.approx(24.10)
        assert out["FRO"]["change_pct"] is None

    def test_download_failure_is_absent_not_raise(self) -> None:
        dl = Recorder(error=RuntimeError("yahoo down"))
        assert get_prices(["FRO"], downloader=dl, now=NOW) == {}

    def test_ticker_missing_from_batch_absent(self) -> None:
        dl = Recorder(_yf_frame({"FRO": [23.0, 24.0]}))
        out = get_prices(["FRO", "STNG"], downloader=dl, now=NOW)
        assert "FRO" in out
        assert "STNG" not in out

    def test_nan_closes_absent(self) -> None:
        frame = _yf_frame({"FRO": [float("nan"), float("nan")]})
        assert get_prices(["FRO"], downloader=Recorder(frame), now=NOW) == {}

    def test_non_ticker_junk_never_hits_downloader(self) -> None:
        dl = Recorder(_yf_frame({"FRO": [23.0, 24.0]}))
        out = get_prices(
            ["FRO", "strait-of-hormuz-closed", "lowercase", ""],
            downloader=dl,
            now=NOW,
        )
        assert dl.calls == [["FRO"]]
        assert list(out) == ["FRO"]

    def test_duplicates_fetched_once(self) -> None:
        dl = Recorder(_yf_frame({"FRO": [23.0, 24.0]}))
        get_prices(["FRO", "FRO", "FRO"], downloader=dl, now=NOW)
        assert dl.calls == [["FRO"]]

    def test_empty_input_no_calls(self) -> None:
        dl = Recorder()
        assert get_prices([], downloader=dl, now=NOW) == {}
        assert dl.calls == []


# --------------------------------------------------------------------- #
# OANDA-id path (DataProvider closes fallback)
# --------------------------------------------------------------------- #


class TestOandaIds:
    def test_closes_from_prices_table(self, db: Any) -> None:
        # BCO_USD (OANDA id) → OIL_WTI (prices-table symbol) via
        # DataProvider normalization (CL-5lpp). Output stays keyed by the
        # caller's original OANDA id.
        out = get_prices(["BCO_USD"], engine=db, now=NOW)
        assert out["BCO_USD"]["price"] == pytest.approx(78.4)
        assert out["BCO_USD"]["change_pct"] == pytest.approx(2.0833, abs=1e-3)

    def test_unknown_symbol_absent(self, db: Any) -> None:
        assert "XAU_USD" not in get_prices(["XAU_USD"], engine=db, now=NOW)

    def test_no_engine_absent_not_raise(self) -> None:
        assert get_prices(["BCO_USD"], engine=None, now=NOW) == {}

    def test_db_error_absent_not_raise(self) -> None:
        bare = sa.create_engine("sqlite://")  # no tables at all
        assert get_prices(["BCO_USD"], engine=bare, now=NOW) == {}

    def test_mixed_batch_routes_both_paths(self, db: Any) -> None:
        dl = Recorder(_yf_frame({"FRO": [23.0, 24.0]}))
        out = get_prices(["FRO", "BCO_USD"], engine=db, downloader=dl, now=NOW)
        assert set(out) == {"FRO", "BCO_USD"}
        assert dl.calls == [["FRO"]]  # OANDA id never sent to Yahoo


# --------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------- #


class TestFormatPrice:
    def test_equity_dollar_two_dp(self) -> None:
        assert format_price("FRO", {"price": 24.1, "change_pct": 3.2}) == "$24.10 (+3.2%)"

    def test_oanda_no_dollar_sig_figs(self) -> None:
        assert format_price("BCO_USD", {"price": 78.4, "change_pct": 2.08}) == "78.4 (+2.1%)"
        assert format_price("EUR_USD", {"price": 1.0834, "change_pct": -0.31}) == "1.0834 (-0.3%)"

    def test_missing_info_empty(self) -> None:
        assert format_price("FRO", None) == ""
        assert format_price("FRO", {}) == ""
        assert format_price("FRO", {"price": "junk"}) == ""

    def test_change_optional(self) -> None:
        assert format_price("FRO", {"price": 24.1, "change_pct": None}) == "$24.10"

    def test_thousands_separator(self) -> None:
        assert format_price("BRK-A", {"price": 731400.0, "change_pct": None}) == "$731,400.00"


class TestFormatAge:
    def test_minutes(self) -> None:
        assert format_age(NOW - timedelta(minutes=45), now=NOW) == "45m"

    def test_hours(self) -> None:
        assert format_age(NOW - timedelta(hours=2, minutes=10), now=NOW) == "2h"

    def test_days(self) -> None:
        assert format_age(NOW - timedelta(days=3), now=NOW) == "3d"

    def test_iso_string_input(self) -> None:
        then = (NOW - timedelta(hours=5)).isoformat()
        assert format_age(then, now=NOW) == "5h"

    def test_future_clamps_to_zero(self) -> None:
        assert format_age(NOW + timedelta(hours=1), now=NOW) == "0m"

    def test_unparseable_is_empty(self) -> None:
        assert format_age(None, now=NOW) == ""
        assert format_age("not a time", now=NOW) == ""


class TestParseTs:
    def test_aware_passthrough(self) -> None:
        assert parse_ts(NOW) is NOW

    def test_naive_assumed_utc(self) -> None:
        parsed = parse_ts(datetime(2026, 7, 20, 12, 0))
        assert parsed is not None and parsed.tzinfo is not None
        assert parsed == NOW

    def test_iso_string(self) -> None:
        assert parse_ts("2026-07-20T10:00:00+02:00") == datetime(
            2026,
            7,
            20,
            10,
            0,
            tzinfo=timezone(timedelta(hours=2)),
        )

    def test_garbage_none(self) -> None:
        assert parse_ts("nope") is None
        assert parse_ts(12345) is None
        assert parse_ts(None) is None
