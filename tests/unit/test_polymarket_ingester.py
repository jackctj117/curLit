"""Tests for the Polymarket history ingester (CL-3t4j v2).

Covers:
  * load_market_config validates structure + token_id/slug presence
  * to_symbol caps at 64 chars
  * Ingester.fetch parses canned Gamma API JSON into rows
  * Bad markets / failed HTTP calls are skipped (one bad market
    doesn't kill the run)
  * transform attaches the prices-table schema columns
  * end-to-end run against a sqlite engine writes to a prices table
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine, text

from src.data.polymarket import (
    PolymarketHistoryIngester,
    load_market_config,
    to_symbol,
)

# ---------------------------------------------------------------------- #
# load_market_config
# ---------------------------------------------------------------------- #


class TestLoadMarketConfig:
    def test_parses_yaml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text(yaml.safe_dump({
            "markets": [
                {"slug": "fed-cut", "token_id": "tok1", "fx_relevance": "high"},
                {"slug": "cpi", "token_id": "tok2"},
            ],
        }))
        out = load_market_config(cfg)
        assert len(out) == 2
        assert out[0]["slug"] == "fed-cut"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_market_config(tmp_path / "nope.yaml")

    def test_missing_markets_key_raises(self, tmp_path: Path) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text("not_markets: {}")
        with pytest.raises(ValueError, match="missing 'markets'"):
            load_market_config(cfg)

    def test_skips_entries_without_token_id_or_slug(
        self, tmp_path: Path,
    ) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text(yaml.safe_dump({
            "markets": [
                {"slug": "good", "token_id": "tok"},
                {"slug": "missing-token"},  # skipped
                {"token_id": "missing-slug"},  # skipped
                {},  # skipped
            ],
        }))
        out = load_market_config(cfg)
        assert len(out) == 1
        assert out[0]["slug"] == "good"

    def test_real_polymarket_markets_yaml_loads(self) -> None:
        # Guard against drift in the operator-curated config.
        cfg = load_market_config("configs/polymarket_markets.yaml")
        assert len(cfg) >= 1
        for m in cfg:
            assert "slug" in m and "token_id" in m


# ---------------------------------------------------------------------- #
# to_symbol
# ---------------------------------------------------------------------- #


class TestToSymbol:
    def test_short_slug(self) -> None:
        assert to_symbol("fed-cut-jun-2026") == "POLY:fed-cut-jun-2026"

    def test_long_slug_truncated(self) -> None:
        long_slug = "a" * 100
        sym = to_symbol(long_slug)
        assert sym.startswith("POLY:")
        assert len(sym) == 64


# ---------------------------------------------------------------------- #
# PolymarketHistoryIngester.fetch
# ---------------------------------------------------------------------- #


def _canned_history(prices: list[tuple[int, float]]) -> dict[str, Any]:
    """Build a Gamma API-shaped response dict."""
    return {"history": [{"t": t, "p": p} for t, p in prices]}


class TestFetch:
    def test_parses_history_into_rows(self, tmp_path: Path) -> None:
        # Use sqlite for the test engine — BaseIngester only needs an
        # engine to upsert; fetch doesn't touch it.
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        ts1 = int(datetime(2026, 4, 1, tzinfo=UTC).timestamp())
        ts2 = int(datetime(2026, 4, 2, tzinfo=UTC).timestamp())
        recorded_calls: list[dict[str, str]] = []

        def fake_http(url: str, params: dict[str, str]) -> dict[str, Any]:
            recorded_calls.append(dict(params))
            if params["market"] == "tok1":
                return _canned_history([(ts1, 0.65), (ts2, 0.72)])
            if params["market"] == "tok2":
                return _canned_history([(ts1, 0.30)])
            return {"history": []}

        ingester = PolymarketHistoryIngester(
            db_url=db_url,
            markets=[
                {"slug": "fed-cut", "token_id": "tok1"},
                {"slug": "cpi", "token_id": "tok2"},
            ],
            http_get_json=fake_http,
        )
        df = ingester.fetch(
            start=datetime(2026, 4, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, tzinfo=UTC),
        )
        assert len(df) == 3
        # Both markets queried
        assert {c["market"] for c in recorded_calls} == {"tok1", "tok2"}
        # Rows have synthetic POLY: symbols
        symbols = sorted(df["symbol"].unique())
        assert symbols == ["POLY:cpi", "POLY:fed-cut"]
        # Probabilities preserved as raw [0,1] floats
        fed_rows = df[df["symbol"] == "POLY:fed-cut"].sort_values("ts")
        assert list(fed_rows["close"]) == [0.65, 0.72]

    def test_one_bad_market_doesnt_kill_run(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{tmp_path / 'test.db'}"

        def flaky_http(url: str, params: dict[str, str]) -> dict[str, Any]:
            if params["market"] == "broken":
                raise ConnectionError("api down")
            return _canned_history([(
                int(datetime(2026, 4, 1, tzinfo=UTC).timestamp()), 0.5,
            )])

        ingester = PolymarketHistoryIngester(
            db_url=db_url,
            markets=[
                {"slug": "ok", "token_id": "good"},
                {"slug": "bad", "token_id": "broken"},
            ],
            http_get_json=flaky_http,
        )
        df = ingester.fetch(
            start=datetime(2026, 4, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, tzinfo=UTC),
        )
        # Only the good market produced rows
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "POLY:ok"

    def test_empty_history_returns_empty_df(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        ingester = PolymarketHistoryIngester(
            db_url=db_url,
            markets=[{"slug": "x", "token_id": "tok"}],
            http_get_json=lambda u, p: {"history": []},
        )
        df = ingester.fetch(
            start=datetime(2026, 4, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, tzinfo=UTC),
        )
        assert df.empty

    def test_skips_malformed_history_entries(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{tmp_path / 'test.db'}"

        def http(url: str, params: dict[str, str]) -> dict[str, Any]:
            return {
                "history": [
                    {"t": 1700000000, "p": 0.5},      # ok
                    {"t": "not-an-int", "p": 0.6},    # skip — bad ts
                    {"t": 1700000001, "p": "junk"},   # skip — bad price
                    {"t": None, "p": 0.7},            # skip
                    None,                              # skip
                ],
            }

        ingester = PolymarketHistoryIngester(
            db_url=db_url,
            markets=[{"slug": "x", "token_id": "tok"}],
            http_get_json=http,
        )
        df = ingester.fetch(
            start=datetime(2026, 4, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, tzinfo=UTC),
        )
        assert len(df) == 1
        assert df.iloc[0]["close"] == 0.5


# ---------------------------------------------------------------------- #
# transform + end-to-end run
# ---------------------------------------------------------------------- #


class TestTransform:
    def test_attaches_schema_columns(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        ingester = PolymarketHistoryIngester(
            db_url=db_url,
            markets=[{"slug": "x", "token_id": "t"}],
            http_get_json=lambda u, p: {},
        )
        raw = pd.DataFrame({
            "ts": [datetime(2026, 4, 1, tzinfo=UTC)],
            "symbol": ["POLY:fed-cut"],
            "close": [0.65],
        })
        out = ingester.transform(raw)
        assert list(out.columns) == [
            "ts", "symbol", "source", "open", "high", "low", "close", "volume",
        ]
        assert out.iloc[0]["source"] == "polymarket"
        assert out.iloc[0]["close"] == 0.65
        assert out.iloc[0]["volume"] == 0.0

    def test_empty_df_passes_through(self, tmp_path: Path) -> None:
        db_url = f"sqlite:///{tmp_path / 'test.db'}"
        ingester = PolymarketHistoryIngester(
            db_url=db_url, markets=[],
            http_get_json=lambda u, p: {},
        )
        out = ingester.transform(pd.DataFrame())
        assert out.empty


class TestEndToEnd:
    def test_run_writes_to_sqlite(self, tmp_path: Path) -> None:
        # Set up an in-memory sqlite DB with a prices table that
        # matches the production Postgres schema.
        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE prices (
                    ts TIMESTAMP,
                    symbol VARCHAR(64),
                    source VARCHAR(32),
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume FLOAT,
                    PRIMARY KEY (ts, symbol)
                )
            """))

        ts = int(datetime(2026, 4, 1, tzinfo=UTC).timestamp())
        ingester = PolymarketHistoryIngester(
            db_url=db_url,
            markets=[{"slug": "fed-cut", "token_id": "tok"}],
            http_get_json=lambda u, p: _canned_history([(ts, 0.78)]),
        )
        rows = ingester.run(
            start=datetime(2026, 4, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, tzinfo=UTC),
        )
        assert rows == 1

        # Verify it's queryable via a normal SQL select
        with engine.begin() as conn:
            result = conn.execute(
                text("SELECT symbol, close FROM prices"),
            ).fetchone()
            assert result is not None
            assert result[0] == "POLY:fed-cut"
            assert result[1] == 0.78
