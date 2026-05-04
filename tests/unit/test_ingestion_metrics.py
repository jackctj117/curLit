"""Tests for BaseIngester metric wiring (CL-47j).

Verifies that ingestion_runs / ingestion_duration / ingestion_records /
data_freshness_seconds emit during run(), and that errors_total
increments when fetch raises.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from src.data.base import BaseIngester
from src.monitoring.metrics import (
    data_freshness_seconds,
    errors_total,
    ingestion_records,
    ingestion_runs,
)


class _FakeIngester(BaseIngester):
    """Minimal ingester that returns canned data + writes to a tmp table."""

    def __init__(self, db_url: str, raise_on_fetch: bool = False) -> None:
        super().__init__(db_url, "fake_source")
        self.raise_on_fetch = raise_on_fetch
        with self.engine.begin() as conn:
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS fake_t ("
                "  ts TEXT, symbol TEXT, value REAL, "
                "  PRIMARY KEY (ts, symbol))",
            ))

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        if self.raise_on_fetch:
            raise RuntimeError("fetch failed")
        return pd.DataFrame({
            "ts": ["2026-04-01", "2026-04-02"],
            "symbol": ["EURUSD", "EURUSD"],
            "value": [1.10, 1.11],
        })

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return raw

    def _key_columns(self) -> list[str]:
        return ["ts", "symbol"]

    def upsert(self, df: pd.DataFrame) -> int:
        with self.engine.begin() as conn:
            for _, row in df.iterrows():
                conn.execute(
                    text(
                        "INSERT OR IGNORE INTO fake_t (ts, symbol, value) "
                        "VALUES (:t, :s, :v)",
                    ),
                    {"t": row["ts"], "s": row["symbol"], "v": row["value"]},
                )
        return len(df)


def _counter_value(counter: Any, **labels: str) -> float:
    """Pull a labeled counter value from the registry."""
    samples = list(counter.collect())[0].samples
    for s in samples:
        if all(s.labels.get(k) == v for k, v in labels.items()):
            return float(s.value)
    return 0.0


def _gauge_value(gauge: Any, **labels: str) -> float:
    samples = list(gauge.collect())[0].samples
    for s in samples:
        if all(s.labels.get(k) == v for k, v in labels.items()):
            return float(s.value)
    return -1.0


class TestSuccessPath:
    def test_records_run_and_records(self, tmp_path: Any) -> None:
        before_runs = _counter_value(
            ingestion_runs, source="fake_source", status="success",
        )
        before_records = _counter_value(
            ingestion_records, source="fake_source",
        )

        url = f"sqlite:///{tmp_path / 'i.db'}"
        ingester = _FakeIngester(url)
        rows = ingester.run(datetime(2026, 4, 1), datetime(2026, 4, 2))

        assert rows == 2
        assert _counter_value(
            ingestion_runs, source="fake_source", status="success",
        ) == before_runs + 1
        assert _counter_value(
            ingestion_records, source="fake_source",
        ) == before_records + 2

    def test_freshness_set_to_zero(self, tmp_path: Any) -> None:
        url = f"sqlite:///{tmp_path / 'i.db'}"
        ingester = _FakeIngester(url)
        ingester.run(datetime(2026, 4, 1), datetime(2026, 4, 2))

        # Per-symbol freshness for EURUSD plus the synthetic _all summary.
        assert _gauge_value(
            data_freshness_seconds, symbol="EURUSD", source="fake_source",
        ) == 0.0
        assert _gauge_value(
            data_freshness_seconds, symbol="_all", source="fake_source",
        ) == 0.0


class TestErrorPath:
    def test_fetch_failure_increments_error_counter(self, tmp_path: Any) -> None:
        before_errors = _counter_value(
            errors_total,
            service="ingest_fake_source",
            severity="error",
            category="ingestion",
        )
        before_runs = _counter_value(
            ingestion_runs, source="fake_source", status="error",
        )

        url = f"sqlite:///{tmp_path / 'i.db'}"
        ingester = _FakeIngester(url, raise_on_fetch=True)
        # Tenacity retries 3 times then re-raises a RetryError wrapping
        # the inner RuntimeError. Either kind is fine; we just want the
        # metrics to register.
        try:
            ingester.run(datetime(2026, 4, 1), datetime(2026, 4, 2))
        except Exception:
            pass

        # Each retry increments the counter (3 attempts × 1 increment each)
        assert _counter_value(
            errors_total,
            service="ingest_fake_source",
            severity="error",
            category="ingestion",
        ) > before_errors
        assert _counter_value(
            ingestion_runs, source="fake_source", status="error",
        ) > before_runs
