"""Integration tests for KnowledgeRetriever (CL-bcr4).

Inserts disposable rows in a unique test source_id, exercises the SQL
paths, then cleans up. Skipped if Postgres isn't reachable. Embedder
is stubbed with deterministic vectors so we don't need a live OpenAI
key — the SQL <=> cosine-distance operator works on whatever vectors
go in.
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import text

from src.research.knowledge import (
    ANTHROPIC_TOOL_DEFINITIONS,
    KnowledgeChunk,
    KnowledgeRetriever,
    dispatch_tool_call,
    openai_tool_spec,
)
from src.runtime.run_engine import _build_db_engine


def _postgres_reachable() -> bool:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="KnowledgeRetriever integration tests require Postgres",
)


# Small deterministic embeddings — vector(1536) zero-padded with a
# non-zero leading section. Different per topic so cosine-distance
# differs and ordering is testable.
def _stub_vec(seed: int) -> list[float]:
    """Return a deterministic 1536-dim vector. Different seeds produce
    vectors at different angles so cosine distance separates them."""
    vec = [0.0] * 1536
    # Pack the seed-dependent signal into 8 dims; rest stays zero.
    base_idx = (seed * 13) % 1500
    for i in range(8):
        vec[base_idx + i] = 1.0 / (i + 1)
    return vec


class _StubEmbedder:
    """Deterministic-vector embedder for testing. Maps each query to a
    seed via Python's built-in hash() so identical queries embed the
    same way."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_stub_vec(abs(hash(t)) % 100) for t in texts]


# =============================================================================
# Fixture: insert 3 sources × ~3 chunks each into the DB
# =============================================================================


@pytest.fixture
def fixture_data() -> Iterator[tuple[Any, list[str]]]:
    engine = _build_db_engine()
    # Use a UUID prefix so concurrent test runs don't collide.
    prefix = "test_ret_" + uuid.uuid4().hex[:8] + "_"
    src_a = prefix + "a"
    src_b = prefix + "b"
    src_c = prefix + "c"

    with engine.begin() as conn:
        # Source A — behavioral-finance topic
        conn.execute(
            text("""
            INSERT INTO knowledge_sources (source_id, title, author, year, source_type, citation, topic_tags)
            VALUES (:sid, 'Test Behavioral', 'A. Author', 2020, 'book',
                    'A. Author (2020). Test Behavioral.',
                    ARRAY['behavioral-finance', 'cognitive-biases'])
        """),
            {"sid": src_a},
        )
        for i, (txt, page) in enumerate(
            [
                ("Overconfidence makes traders overestimate their edge.", "ch.1"),
                ("Anchoring biases pull estimates toward salient values.", "ch.2"),
            ]
        ):
            vec = _stub_vec(seed=10 + i)
            emb = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
            conn.execute(
                text("""
                INSERT INTO knowledge_chunks (source_id, chunk_idx, chunk_text, embedding, page_ref)
                VALUES (:sid, :idx, :t, CAST(:emb AS vector), :p)
            """),
                {"sid": src_a, "idx": i, "t": txt, "emb": emb, "p": page},
            )

        # Source B — crisis-history topic
        conn.execute(
            text("""
            INSERT INTO knowledge_sources (source_id, title, author, year, source_type, citation, topic_tags)
            VALUES (:sid, 'Test Crises', 'B. Author', 1955, 'book',
                    'B. Author (1955). Test Crises.',
                    ARRAY['crisis-history', '1929-crash'])
        """),
            {"sid": src_b},
        )
        for i, (txt, page) in enumerate(
            [
                ("Bank panics propagate through the deposit system.", "ch.4"),
                ("Margin debt amplifies every crash.", "ch.5"),
            ]
        ):
            vec = _stub_vec(seed=50 + i)
            emb = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
            conn.execute(
                text("""
                INSERT INTO knowledge_chunks (source_id, chunk_idx, chunk_text, embedding, page_ref)
                VALUES (:sid, :idx, :t, CAST(:emb AS vector), :p)
            """),
                {"sid": src_b, "idx": i, "t": txt, "emb": emb, "p": page},
            )

        # Source C — overlapping behavioral-finance + adds reflexivity tag
        conn.execute(
            text("""
            INSERT INTO knowledge_sources (source_id, title, author, year, source_type, citation, topic_tags)
            VALUES (:sid, 'Test Reflexivity', 'C. Author', 1987, 'book',
                    'C. Author (1987). Test Reflexivity.',
                    ARRAY['reflexivity', 'behavioral-finance'])
        """),
            {"sid": src_c},
        )
        # No embedding here — tests the IS NOT NULL filter
        conn.execute(
            text("""
            INSERT INTO knowledge_chunks (source_id, chunk_idx, chunk_text, page_ref)
            VALUES (:sid, 0, 'Reflexive feedback distorts fundamentals.', 'ch.1')
        """),
            {"sid": src_c},
        )

    yield engine, [src_a, src_b, src_c]

    # Cleanup
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM knowledge_sources WHERE source_id = ANY(:sids)"),
            {"sids": [src_a, src_b, src_c]},
        )


# =============================================================================
# Topic-only retrieval (no embedder needed)
# =============================================================================


class TestTopicSearch:
    def test_topic_filter_returns_matching_chunks(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        engine, sids = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=None)
        chunks = retriever.search_by_topics(["crisis-history"], top_k=10)
        # Only Source B should match
        sources_returned = {c.source_id for c in chunks}
        assert sids[1] in sources_returned  # src_b
        assert sids[0] not in sources_returned  # src_a behavioral-finance only
        for c in chunks:
            assert "crisis-history" in c.topic_tags

    def test_topic_overlap_array_matches_any(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        """Sources A and C both have 'behavioral-finance' tag — both
        should match a behavioral-finance topic filter."""
        engine, sids = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=None)
        chunks = retriever.search_by_topics(["behavioral-finance"], top_k=10)
        sources_returned = {c.source_id for c in chunks}
        assert sids[0] in sources_returned
        assert sids[2] in sources_returned

    def test_empty_topic_list_returns_empty(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        engine, _ = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=None)
        assert retriever.search_by_topics([], top_k=10) == []


# =============================================================================
# search() — with embedder + with topics
# =============================================================================


class TestSearchWithEmbedder:
    def test_search_with_topics_filter_excludes_non_matching(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        engine, sids = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=_StubEmbedder())  # type: ignore[arg-type]
        # Filter to behavioral-finance only — src_b crisis chunks excluded
        chunks = retriever.search(
            query="cognitive biases",
            top_k=5,
            topics=["behavioral-finance"],
        )
        for c in chunks:
            assert "behavioral-finance" in c.topic_tags

    def test_search_skips_chunks_with_null_embedding(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        """Source C's chunk has NULL embedding — must NOT appear in
        embedding-based search results."""
        engine, sids = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=_StubEmbedder())  # type: ignore[arg-type]
        chunks = retriever.search(query="anything", top_k=50)
        sources_returned = {c.source_id for c in chunks}
        assert sids[2] not in sources_returned  # src_c has NULL embedding

    def test_search_returns_distance_when_embedder_present(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        engine, _ = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=_StubEmbedder())  # type: ignore[arg-type]
        chunks = retriever.search(query="behavioral", top_k=3)
        for c in chunks:
            assert c.distance is not None
            assert c.distance >= 0.0


class TestSearchWithoutEmbedder:
    def test_no_embedder_falls_back_to_topic_filter(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        engine, sids = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=None)
        chunks = retriever.search(
            query="anything",
            top_k=5,
            topics=["crisis-history"],
        )
        for c in chunks:
            assert "crisis-history" in c.topic_tags
            assert c.distance is None  # no similarity computed

    def test_no_embedder_no_topics_returns_recent(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        """Without embedder or topics, fallback returns most-recently-
        added chunks (degraded mode)."""
        engine, _ = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=None)
        chunks = retriever.search(query="anything", top_k=10)
        assert len(chunks) > 0


# =============================================================================
# Citation formatting
# =============================================================================


class TestCitation:
    def test_citation_includes_year_and_page(self) -> None:
        c = KnowledgeChunk(
            chunk_id=1,
            source_id="x",
            chunk_text="…",
            page_ref="ch.3",
            title="My Book",
            author="A. Author",
            year=2020,
            topic_tags=[],
        )
        assert c.citation == "A. Author (2020). My Book, ch.3"

    def test_citation_handles_missing_year(self) -> None:
        c = KnowledgeChunk(
            chunk_id=1,
            source_id="x",
            chunk_text="…",
            page_ref="ch.3",
            title="My Book",
            author="A. Author",
            year=None,
            topic_tags=[],
        )
        assert c.citation == "A. Author. My Book, ch.3"


# =============================================================================
# Tool spec + dispatcher (no DB needed for some)
# =============================================================================


class TestToolSpec:
    def test_anthropic_tool_definitions_well_formed(self) -> None:
        for t in ANTHROPIC_TOOL_DEFINITIONS:
            assert t["name"]
            assert t["description"]
            assert "input_schema" in t
            assert t["input_schema"]["type"] == "object"

    def test_openai_tool_spec_round_trip(self) -> None:
        oai = openai_tool_spec()
        assert len(oai) == len(ANTHROPIC_TOOL_DEFINITIONS)
        for entry in oai:
            assert entry["type"] == "function"
            assert "function" in entry
            assert entry["function"]["name"]
            assert "parameters" in entry["function"]


class TestToolDispatch:
    def test_dispatch_search_returns_chunk_dicts(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        engine, _ = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=None)
        out = dispatch_tool_call(
            retriever,
            "knowledge_search_by_topics",
            {"topics": ["behavioral-finance"], "top_k": 10},
        )
        assert isinstance(out, list)
        for o in out:
            assert "citation" in o
            assert "chunk_text" in o
            assert "topic_tags" in o

    def test_dispatch_unknown_tool_raises(
        self,
        fixture_data: tuple[Any, list[str]],
    ) -> None:
        engine, _ = fixture_data
        retriever = KnowledgeRetriever(engine=engine, embedder=None)
        with pytest.raises(ValueError, match="unknown knowledge tool"):
            dispatch_tool_call(retriever, "knowledge_evil", {})
