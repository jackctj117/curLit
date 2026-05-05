"""Tests for scripts.promote_papers_to_idea_agent (CL-28j wire).

Covers the read → call → write → transition cycle that the dashboard's
"For impl" button depends on. IdeaGenerator is mocked so we don't hit
a live LLM in CI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text

from scripts.promote_papers_to_idea_agent import (
    _find_extract_path,
    _select_promotable_papers,
    _set_status,
)
from src.research.agents.idea import IdeaResult, IdeaStatus


@pytest.fixture
def papers_engine(tmp_path):  # type: ignore[no-untyped-def]
    """Sqlite shim seeded with one for_implementation row + one read row."""
    engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE research_papers (
                paper_id TEXT PRIMARY KEY,
                source TEXT, title TEXT, authors TEXT,
                abstract TEXT, url TEXT, pdf_url TEXT,
                published_date TEXT, ingested_at TEXT,
                read_status TEXT DEFAULT 'unread',
                implementation_priority INTEGER DEFAULT 0,
                relevance_score REAL DEFAULT 0,
                my_notes TEXT
            )
        """))
        conn.execute(text("""
            INSERT INTO research_papers
                (paper_id, title, source, read_status,
                 implementation_priority, relevance_score)
            VALUES
                ('hash_a', 'Carry trade alpha', 'arXiv',
                 'for_implementation', 1, 18.5),
                ('hash_b', 'Volatility risk premium', 'NBER',
                 'for_implementation', 2, 16.5),
                ('hash_c', 'Some other paper', 'NBER',
                 'read', 0, 5.0)
        """))
    return engine


@pytest.fixture
def extract_root(tmp_path):  # type: ignore[no-untyped-def]
    root = tmp_path / "extracts"
    root.mkdir()
    # Only paper A has an extract on disk.
    (root / "hash_a.md").write_text(
        "# Carry trade alpha\n\n## Source extract\n...",
    )
    return root


class TestSelectPromotable:
    def test_only_for_implementation_returned(self, papers_engine) -> None:  # type: ignore[no-untyped-def]
        rows = _select_promotable_papers(papers_engine, limit=10)
        assert len(rows) == 2
        assert {r["paper_id"] for r in rows} == {"hash_a", "hash_b"}

    def test_priority_orders_first(self, papers_engine) -> None:  # type: ignore[no-untyped-def]
        # hash_a priority 1, hash_b priority 2 → hash_a first.
        rows = _select_promotable_papers(papers_engine, limit=10)
        assert rows[0]["paper_id"] == "hash_a"
        assert rows[1]["paper_id"] == "hash_b"

    def test_limit_caps(self, papers_engine) -> None:  # type: ignore[no-untyped-def]
        rows = _select_promotable_papers(papers_engine, limit=1)
        assert len(rows) == 1
        # Highest priority wins the only slot.
        assert rows[0]["paper_id"] == "hash_a"


class TestExtractLookup:
    def test_present_extract_returns_path(self, extract_root: Path) -> None:
        path = _find_extract_path("hash_a", extract_root)
        assert path is not None
        assert path.name == "hash_a.md"

    def test_missing_extract_returns_none(self, extract_root: Path) -> None:
        assert _find_extract_path("hash_b", extract_root) is None


class TestSetStatus:
    def test_writes_new_status(self, papers_engine) -> None:  # type: ignore[no-untyped-def]
        _set_status(papers_engine, "hash_a", "in_pipeline")
        with papers_engine.connect() as conn:
            v = conn.execute(text(
                "SELECT read_status FROM research_papers WHERE paper_id='hash_a'",
            )).scalar()
        assert v == "in_pipeline"


class TestEndToEndCycle:
    """Drive scripts.promote_papers_to_idea_agent.main with a mocked
    IdeaGenerator to verify the full read→call→transition flow.
    """

    def _mock_idea(self, status: IdeaStatus, hypothesis_path: Path | None = None,
                   reason: str = "") -> MagicMock:
        result = IdeaResult(
            status=status,
            strategy_slug="carry-trade-alpha",
            extract_path=Path("data/research/extracts/hash_a.md"),
            response=MagicMock(),
            raw_text="...",
            hypothesis_path=hypothesis_path,
            reason=reason,
        )
        agent = MagicMock()
        agent.ideate.return_value = result
        return agent

    def test_proposed_writes_hypothesis_and_transitions(
        self, papers_engine, extract_root: Path, tmp_path: Path,
    ) -> None:  # type: ignore[no-untyped-def]
        from scripts import promote_papers_to_idea_agent as mod

        hypothesis_dir = tmp_path / "hyp"
        hyp_path = hypothesis_dir / "carry-trade-alpha.md"
        # The mock pretends IdeaGenerator already wrote the file. We
        # also create it on disk so the assertion that the path
        # exists isn't a tautology.
        hypothesis_dir.mkdir()
        hyp_path.write_text("hypothesis content")

        idea_agent = self._mock_idea(IdeaStatus.PROPOSED, hypothesis_path=hyp_path)
        with (
            patch("src.runtime.run_engine._build_db_engine",
                  return_value=papers_engine),
            patch("src.research.agents.idea.IdeaGenerator.from_config",
                  return_value=idea_agent),
        ):
            rc = mod.main([
                "--extract-root", str(extract_root),
                "--hypothesis-dir", str(hypothesis_dir),
                "--max-papers", "10",
            ])

        assert rc == 0
        # hash_a had an extract → ideate ran → in_pipeline.
        # hash_b had NO extract → script synthesized one from the DB
        # row, ideate ran on the stub → in_pipeline (mock returns
        # PROPOSED for both calls, so both transition together).
        with papers_engine.connect() as conn:
            states = dict(conn.execute(text(
                "SELECT paper_id, read_status FROM research_papers"
            )).fetchall())
        assert states["hash_a"] == "in_pipeline"
        assert states["hash_b"] == "in_pipeline"  # synthesized + processed
        assert states["hash_c"] == "read"          # untouched
        # Synthesized extract should now exist on disk under hash_b.md
        assert (extract_root / "hash_b.md").exists()

    def test_declined_transitions_to_idea_declined(
        self, papers_engine, extract_root: Path, tmp_path: Path,
    ) -> None:  # type: ignore[no-untyped-def]
        from scripts import promote_papers_to_idea_agent as mod

        idea_agent = self._mock_idea(
            IdeaStatus.DECLINED, reason="paper too theoretical",
        )
        with (
            patch("src.runtime.run_engine._build_db_engine",
                  return_value=papers_engine),
            patch("src.research.agents.idea.IdeaGenerator.from_config",
                  return_value=idea_agent),
        ):
            rc = mod.main([
                "--extract-root", str(extract_root),
                "--hypothesis-dir", str(tmp_path / "hyp"),
                "--max-papers", "10",
            ])

        assert rc == 0
        with papers_engine.connect() as conn:
            v = conn.execute(text(
                "SELECT read_status FROM research_papers WHERE paper_id='hash_a'",
            )).scalar()
        assert v == "idea_declined"

    def test_idea_agent_raises_leaves_status_unchanged(
        self, papers_engine, extract_root: Path, tmp_path: Path,
    ) -> None:  # type: ignore[no-untyped-def]
        """An LLM blip shouldn't lose the operator's promotion. The
        row stays in for_implementation so the next run retries."""
        from scripts import promote_papers_to_idea_agent as mod

        idea_agent = MagicMock()
        idea_agent.ideate.side_effect = RuntimeError("LLM hiccup")
        with (
            patch("src.runtime.run_engine._build_db_engine",
                  return_value=papers_engine),
            patch("src.research.agents.idea.IdeaGenerator.from_config",
                  return_value=idea_agent),
        ):
            rc = mod.main([
                "--extract-root", str(extract_root),
                "--hypothesis-dir", str(tmp_path / "hyp"),
                "--max-papers", "10",
            ])
        assert rc == 0
        with papers_engine.connect() as conn:
            v = conn.execute(text(
                "SELECT read_status FROM research_papers WHERE paper_id='hash_a'",
            )).scalar()
        assert v == "for_implementation"

    def test_synthesize_extract_uses_abstract(
        self, papers_engine, tmp_path: Path,
    ) -> None:  # type: ignore[no-untyped-def]
        """Direct test of the synthesis helper: an extract built from a
        DB row should contain the title, abstract, and a stable
        synthesized-from-DB note so audit can tell it apart from a real
        LLM extract."""
        from scripts.promote_papers_to_idea_agent import (
            _fetch_paper_row, _synthesize_extract,
        )

        # Add abstract to hash_b so we have something to synthesize from.
        with papers_engine.begin() as conn:
            conn.execute(text("""
                UPDATE research_papers
                SET abstract = 'A real abstract about volatility risk premium.',
                    url = 'https://example.com/vrp'
                WHERE paper_id = 'hash_b'
            """))

        row = _fetch_paper_row(papers_engine, "hash_b")
        assert row is not None
        path = _synthesize_extract("hash_b", row, tmp_path)
        body = path.read_text()
        assert "Volatility risk premium" in body          # title
        assert "real abstract about volatility" in body   # abstract content
        assert "synthesized from DB row" in body          # provenance marker

    def test_dry_run_doesnt_call_llm_or_transition(
        self, papers_engine, extract_root: Path,
    ) -> None:  # type: ignore[no-untyped-def]
        from scripts import promote_papers_to_idea_agent as mod

        with patch(
            "src.runtime.run_engine._build_db_engine",
            return_value=papers_engine,
        ):
            rc = mod.main([
                "--extract-root", str(extract_root),
                "--dry-run",
            ])
        assert rc == 0
        # Statuses unchanged.
        with papers_engine.connect() as conn:
            states = dict(conn.execute(text(
                "SELECT paper_id, read_status FROM research_papers"
            )).fetchall())
        assert states["hash_a"] == "for_implementation"
        assert states["hash_b"] == "for_implementation"
