"""Tests for the research-triage Streamlit dashboard helpers (CL-28j).

We don't try to run the Streamlit UI under pytest (the framework
expects a real browser session). What we DO test are the pure SQL
helpers — load_papers, update_paper_status, update_paper_priority,
update_paper_notes, list_sources — which the UI calls. The UI logic
itself (sliders, buttons, table) gets exercised by visual review and
by Streamlit's own snapshot tooling if/when we add it.

The DB-backed tests use sqlite with the same schema as Postgres
(JSONB → TEXT, TIMESTAMPTZ → TEXT) so we don't need a live Postgres.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy import create_engine, text


@pytest.fixture
def papers_engine(tmp_path):  # type: ignore[no-untyped-def]
    """Sqlite with the research_papers schema + some seed rows."""
    engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    with engine.begin() as conn:
        # JSONB → TEXT, TIMESTAMPTZ → TEXT, NUMERIC → REAL
        conn.execute(
            text("""
            CREATE TABLE research_papers (
                paper_id TEXT PRIMARY KEY,
                source TEXT, title TEXT, authors TEXT,
                abstract TEXT, url TEXT, pdf_url TEXT,
                published_date TEXT, ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                keywords TEXT, categories TEXT,
                relevance_score REAL DEFAULT 0,
                read_status TEXT DEFAULT 'unread',
                my_notes TEXT,
                implementation_priority INTEGER DEFAULT 0,
                evaluation_data TEXT
            )
        """)
        )
        # 3 papers spanning sources + scores
        conn.execute(
            text("""
            INSERT INTO research_papers
                (paper_id, source, title, authors, abstract, relevance_score,
                 read_status, implementation_priority)
            VALUES
                ('arxiv:1', 'arXiv q-fin.PM', 'Carry trade returns',
                 '["Sarno", "Lustig"]', 'We document carry trade alpha.',
                 18.5, 'unread', 0),
                ('nber:1', 'NBER',
                 'Theoretical model of asset prices',
                 '["Smith"]', 'Pure equilibrium derivation, no empirics.',
                 -2.0, 'unread', 0),
                ('frbsf:1', 'FRBSF',
                 'FOMC communication and market reaction',
                 '["Jones", "Kim"]', 'Fed statements and EUR/USD jumps.',
                 8.5, 'discarded', 0)
        """)
        )
    return engine


@pytest.fixture
def patched_engine(papers_engine):  # type: ignore[no-untyped-def]
    """Patch the @cache_resource _engine() to return our test engine."""
    with patch("research.dashboard._engine", return_value=papers_engine):
        yield papers_engine


class TestLoadPapers:
    def test_default_filters_returns_unread(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import _STATUS_UNREAD, load_papers

        df = load_papers(statuses=[_STATUS_UNREAD])
        assert len(df) == 2
        assert set(df["paper_id"]) == {"arxiv:1", "nber:1"}

    def test_min_score_filter(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import load_papers

        df = load_papers(min_score=10.0)
        # Only arxiv:1 has score >= 10. nber:1 is -2, frbsf:1 is 8.5.
        assert list(df["paper_id"]) == ["arxiv:1"]

    def test_source_filter(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import load_papers

        df = load_papers(
            statuses=["unread", "discarded"],
            sources=["NBER"],
        )
        assert list(df["paper_id"]) == ["nber:1"]

    def test_orders_by_relevance_descending(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import load_papers

        df = load_papers(statuses=["unread", "discarded"])
        # 18.5 > 8.5 > -2.0 → arxiv:1, frbsf:1, nber:1
        assert list(df["paper_id"]) == ["arxiv:1", "frbsf:1", "nber:1"]

    def test_limit_caps_results(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import load_papers

        df = load_papers(statuses=["unread", "discarded"], limit=2)
        assert len(df) == 2


class TestStatusMutations:
    def test_update_paper_status_persists(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import update_paper_status

        update_paper_status("arxiv:1", "for_implementation")
        with patched_engine.connect() as conn:
            v = conn.execute(
                text(
                    "SELECT read_status FROM research_papers WHERE paper_id='arxiv:1'",
                )
            ).scalar()
        assert v == "for_implementation"

    def test_update_paper_priority_persists(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import update_paper_priority

        update_paper_priority("arxiv:1", 1)
        with patched_engine.connect() as conn:
            v = conn.execute(
                text(
                    "SELECT implementation_priority FROM research_papers WHERE paper_id='arxiv:1'",
                )
            ).scalar()
        assert v == 1

    def test_update_paper_notes_persists(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import update_paper_notes

        update_paper_notes("arxiv:1", "Replication target — Menkhoff 2012.")
        with patched_engine.connect() as conn:
            v = conn.execute(
                text(
                    "SELECT my_notes FROM research_papers WHERE paper_id='arxiv:1'",
                )
            ).scalar()
        assert v == "Replication target — Menkhoff 2012."


class TestSidebarHelpers:
    def test_list_sources_returns_distinct_sorted(self, patched_engine) -> None:  # type: ignore[no-untyped-def]
        from research.dashboard import list_sources

        sources = list_sources()
        assert sources == sorted(sources)
        assert "arXiv q-fin.PM" in sources
        assert "NBER" in sources
        assert "FRBSF" in sources


class TestAuthorsCoercion:
    def test_authors_list_renders_comma_separated(self) -> None:
        from research.dashboard import _coerce_authors

        assert _coerce_authors(["A", "B", "C"]) == "A, B, C"

    def test_authors_json_string_parses(self) -> None:
        from research.dashboard import _coerce_authors

        assert _coerce_authors('["X", "Y"]') == "X, Y"

    def test_authors_long_list_truncates(self) -> None:
        from research.dashboard import _coerce_authors

        out = _coerce_authors([f"a{i}" for i in range(10)])
        assert out.endswith("…")
        # Up to 6 names before ellipsis
        assert out.count(",") == 5

    def test_authors_none_yields_empty(self) -> None:
        from research.dashboard import _coerce_authors

        assert _coerce_authors(None) == ""


class TestStatusVocabulary:
    def test_all_statuses_form_consistent_set(self) -> None:
        from research.dashboard import _ALL_STATUSES

        # Must include the operator's full triage vocabulary so
        # filters + buttons can both reference the same constants.
        assert {"unread", "skim_later", "read", "discarded", "for_implementation"} <= set(
            _ALL_STATUSES
        )


class TestRenderDetailContract:
    """Regression test for the KeyError that hit the live UI:
    main() does df.set_index('paper_id') before slicing, which makes
    'paper_id' a Series.name rather than an indexable column. The
    detail renderer must not assume paper['paper_id'] resolves."""

    def test_paper_series_after_set_index_does_not_have_paper_id_key(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "paper_id": "p1",
                    "title": "Foo",
                    "abstract": "x",
                    "source": "S",
                    "authors": "[]",
                    "url": "u",
                    "pdf_url": None,
                    "relevance_score": 1.0,
                    "read_status": "unread",
                    "implementation_priority": 0,
                    "my_notes": "",
                },
            ]
        )
        paper = df.set_index("paper_id").loc["p1"]
        with pytest.raises(KeyError):
            _ = paper["paper_id"]
        # The fix: paper.name carries the index value, and the public
        # signature passes paper_id explicitly.
        assert paper.name == "p1"

    def test_render_detail_signature_takes_paper_id_separately(self) -> None:
        # Locking the public contract: paper_id is a required first
        # arg. Reverting to the buggy "find paper_id inside paper"
        # pattern would change the signature, which this test catches.
        import inspect

        from research.dashboard import _render_detail

        sig = inspect.signature(_render_detail)
        params = list(sig.parameters)
        assert params == ["paper_id", "paper"], f"_render_detail signature regressed: {params}"
