"""Tests for research meta-learning (CL-fsj)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from src.research.meta_learning import (
    MetaLearner,
    _compute_multiplier,
    _detect_topic,
)


@pytest.fixture
def meta_engine(tmp_path):  # type: ignore[no-untyped-def]
    engine = create_engine(f"sqlite:///{tmp_path / 'meta.db'}")
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE research_papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT, abstract TEXT, source TEXT,
                ingested_at TEXT,
                implementation_priority INTEGER DEFAULT 0,
                evaluation_data TEXT
            )
        """)
        )
        conn.execute(
            text("""
            CREATE TABLE strategy_fills (
                strategy_id TEXT, fill_id TEXT, symbol TEXT,
                quantity REAL, fill_price REAL, ts TEXT,
                PRIMARY KEY (strategy_id, fill_id)
            )
        """)
        )
    return engine


class TestTopicDetection:
    def test_carry_topic(self) -> None:
        t = _detect_topic(
            title="Carry Trade Returns",
            abstract="evidence on currency carry profits",
        )
        assert t == "carry"

    def test_volatility_topic(self) -> None:
        t = _detect_topic(
            title="VRP and Currency Returns",
            abstract="volatility risk premium drives FX",
        )
        assert t == "volatility"

    def test_other_topic_when_no_match(self) -> None:
        assert _detect_topic("Random title", "no relevant keywords") == "other"


class TestMultiplier:
    def test_no_implementations_returns_one(self) -> None:
        assert _compute_multiplier(implemented=0, total_pnl=0) == 1.0

    def test_negative_pnl_downweights(self) -> None:
        assert _compute_multiplier(implemented=2, total_pnl=-100) < 1.0

    def test_positive_pnl_upweights(self) -> None:
        m = _compute_multiplier(implemented=1, total_pnl=10_000)
        assert m > 1.0

    def test_capped_at_max(self) -> None:
        m = _compute_multiplier(implemented=1, total_pnl=10_000_000)
        assert m == pytest.approx(1.5)


class TestGather:
    def test_no_papers_returns_empty(self, meta_engine) -> None:  # type: ignore[no-untyped-def]
        learner = MetaLearner(meta_engine)
        report = learner.gather()
        assert report.sources == {}

    def test_aggregates_by_source(self, meta_engine) -> None:  # type: ignore[no-untyped-def]
        with meta_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO research_papers "
                    "(paper_id, title, abstract, source, ingested_at, "
                    "implementation_priority) VALUES "
                    "('p1', 'Carry Returns', 'carry trade alpha', 'arXiv q-fin.PM', "
                    " '2026-04-01', 1),"
                    "('p2', 'Vol Premium', 'fx volatility', 'arXiv q-fin.PM', "
                    " '2026-04-02', 1),"
                    "('p3', 'NBER Test', 'test', 'NBER', '2026-04-03', 0)",
                )
            )
            conn.execute(
                text(
                    "INSERT INTO strategy_fills "
                    "(strategy_id, fill_id, symbol, quantity, fill_price, ts) "
                    "VALUES "
                    "('p1', 'f1', 'EURUSD', 1000, 1.10, '2026-04-15')",
                )
            )

        report = MetaLearner(meta_engine).gather()
        assert "arXiv q-fin.PM" in report.sources
        s = report.sources["arXiv q-fin.PM"]
        assert s.n_papers == 2
        assert s.n_implemented == 2
        # NBER source has one un-implemented paper
        assert report.sources["NBER"].n_implemented == 0


class TestPersistAndOverlay:
    def test_overlay_round_trip(self, meta_engine) -> None:  # type: ignore[no-untyped-def]
        with meta_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO research_papers "
                    "(paper_id, title, abstract, source, ingested_at, "
                    "implementation_priority) VALUES "
                    "('p1', 'Carry', 'carry trade', 'arXiv q-fin.PM', "
                    " '2026-04-01', 1)",
                )
            )

        learner = MetaLearner(meta_engine)
        report = learner.gather()
        overlay = learner.persist_and_emit_overlay(report)

        assert "arXiv q-fin.PM" in overlay
        # No fills → no PnL → multiplier should be 1.0 (implemented but
        # unrealized has zero pnl → falls into the negative-pnl 0.85 branch).
        # Actually: implemented=1, total_pnl=0 → returns 0.85 per
        # _compute_multiplier rule "implemented but 0/negative → 0.85".
        assert overlay["arXiv q-fin.PM"]["*"] == pytest.approx(0.85)

        # Check that the rollup row was persisted.
        with meta_engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT evaluation_data FROM research_papers "
                    "WHERE paper_id LIKE 'meta:learning:%'",
                )
            ).fetchone()
            assert row is not None
