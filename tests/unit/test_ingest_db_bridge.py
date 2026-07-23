"""Tests for the IngestRunner → research_papers bridge (CL-28j follow-up).

Covers:
  - DB write happens BEFORE extract (so failed extracts don't hide
    papers from the operator)
  - ON CONFLICT DO NOTHING — second run with same paper is a no-op
  - relevance_scorer.score_and_update is called on insert when supplied
  - db_engine=None preserves the legacy disk-only behavior
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text

from src.research.ingest import (
    FeedConfig,
    IngestRunner,
    Paper,
    _insert_paper_row,
    paper_hash,
)


@pytest.fixture
def papers_engine(tmp_path: Any) -> Any:
    """Sqlite shim with research_papers schema matching production."""
    engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    with engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE research_papers (
                paper_id TEXT PRIMARY KEY,
                source TEXT, title TEXT, authors TEXT,
                abstract TEXT, url TEXT, pdf_url TEXT,
                published_date TEXT, ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                read_status TEXT DEFAULT 'unread',
                implementation_priority INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 0
            )
        """)
        )
    return engine


def _paper(title: str = "Carry Trades") -> Paper:
    return Paper(
        title=title,
        authors=("Sarno", "Lustig"),
        year=2012,
        url="https://example.com/p1",
        doi="",
        abstract="We document carry trade alpha and global vol risk.",
        source_label="arXiv q-fin.PM",
    )


class TestInsertRowDirectly:
    def test_first_insert_returns_true(self, papers_engine: Any) -> None:
        assert _insert_paper_row(papers_engine, _paper()) is True
        with papers_engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM research_papers")).scalar()
            assert n == 1

    def test_second_insert_returns_false_on_conflict(
        self,
        papers_engine: Any,
    ) -> None:
        p = _paper()
        assert _insert_paper_row(papers_engine, p) is True
        # Same paper → same hash → conflict → no-op.
        assert _insert_paper_row(papers_engine, p) is False
        with papers_engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM research_papers")).scalar()
            assert n == 1

    def test_paper_id_is_paper_hash(self, papers_engine: Any) -> None:
        p = _paper()
        _insert_paper_row(papers_engine, p)
        with papers_engine.connect() as conn:
            pid = conn.execute(
                text(
                    "SELECT paper_id FROM research_papers",
                )
            ).scalar()
            assert pid == paper_hash(p)


class TestRelevanceScorerWiring:
    def test_scorer_called_on_new_insert(self, papers_engine: Any) -> None:
        scorer = MagicMock()
        # score_and_update returns the float total, but we only check
        # it was called with the right paper_id + paper.
        scorer.score_and_update.return_value = 5.0

        p = _paper()
        result = _insert_paper_row(papers_engine, p, relevance_scorer=scorer)
        assert result is True
        scorer.score_and_update.assert_called_once()
        called_engine, called_pid, called_paper = scorer.score_and_update.call_args[0]
        assert called_pid == paper_hash(p)
        assert called_paper is p

    def test_scorer_not_called_on_conflict(self, papers_engine: Any) -> None:
        scorer = MagicMock()
        scorer.score_and_update.return_value = 5.0

        _insert_paper_row(papers_engine, _paper(), relevance_scorer=scorer)
        scorer.reset_mock()
        # Second insert → conflict → don't re-score.
        result = _insert_paper_row(papers_engine, _paper(), relevance_scorer=scorer)
        assert result is False
        scorer.score_and_update.assert_not_called()

    def test_scorer_failure_does_not_block_ingest(
        self,
        papers_engine: Any,
    ) -> None:
        scorer = MagicMock()
        scorer.score_and_update.side_effect = RuntimeError("scorer broke")

        # The row should still get in even if scoring raises.
        result = _insert_paper_row(
            papers_engine,
            _paper(),
            relevance_scorer=scorer,
        )
        assert result is True
        with papers_engine.connect() as conn:
            n = conn.execute(text("SELECT COUNT(*) FROM research_papers")).scalar()
            assert n == 1


class TestRunnerIntegration:
    """End-to-end: feed → fetch → DB write → extract."""

    def test_runner_writes_to_db_before_extract(
        self,
        papers_engine: Any,
        tmp_path: Any,
    ) -> None:
        # Build a fake fetcher that returns a Paper, plus a fake
        # extractor that records call order vs the DB row.
        target_paper = _paper()
        events: list[str] = []

        class _FakeExtractor:
            def extract(self, paper: Paper) -> str:
                # At extract time, the DB row should already exist.
                with papers_engine.connect() as conn:
                    n = conn.execute(
                        text(
                            "SELECT COUNT(*) FROM research_papers WHERE paper_id=:p",
                        ),
                        {"p": paper_hash(paper)},
                    ).scalar()
                events.append(f"extract:db_rows={n}")
                return "extracted body text"

        from src.research.ingest import _FETCHER_REGISTRY, ExtractStore

        # Register a one-shot fake fetcher in the registry.
        def _fake_fetcher_factory(http_get: Any) -> Any:
            mock = MagicMock()
            mock.fetch.return_value = [target_paper]
            return mock

        _FETCHER_REGISTRY["fake_for_test"] = _fake_fetcher_factory
        try:
            runner = IngestRunner(
                extractor=_FakeExtractor(),
                store=ExtractStore(root=tmp_path / "extracts"),
                db_engine=papers_engine,
            )
            feed = FeedConfig(
                name="t",
                adapter="fake_for_test",
                query_url="x",
                source_label="test",
            )
            summary = runner.run([feed])
        finally:
            _FETCHER_REGISTRY.pop("fake_for_test", None)

        assert summary.papers_db_inserted == 1
        assert summary.papers_extracted == 1
        # Ordering invariant: extract saw the DB row already in.
        assert events == ["extract:db_rows=1"]

    def test_runner_skips_db_when_engine_none(
        self,
        tmp_path: Any,
    ) -> None:
        """Default behavior: no engine → no DB writes; disk extracts only."""
        from src.research.ingest import _FETCHER_REGISTRY, ExtractStore

        target_paper = _paper()

        class _FakeExtractor:
            def extract(self, paper: Paper) -> str:
                return "body"

        def _fake_fetcher_factory(http_get: Any) -> Any:
            mock = MagicMock()
            mock.fetch.return_value = [target_paper]
            return mock

        _FETCHER_REGISTRY["fake_no_db"] = _fake_fetcher_factory
        try:
            runner = IngestRunner(
                extractor=_FakeExtractor(),
                store=ExtractStore(root=tmp_path / "extracts"),
                db_engine=None,  # explicit
            )
            feed = FeedConfig(
                name="t",
                adapter="fake_no_db",
                query_url="x",
                source_label="test",
            )
            summary = runner.run([feed])
        finally:
            _FETCHER_REGISTRY.pop("fake_no_db", None)

        assert summary.papers_db_inserted == 0
        assert summary.papers_extracted == 1
