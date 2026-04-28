"""KnowledgeRetriever — vector + topic-tag retrieval over the knowledge
archive (CL-bcr4).

Two retrieval modes:
  * `search(query, top_k, topics=None)`   — embeds the query and runs
       cosine-similarity against ``knowledge_chunks.embedding``.
       Falls back to topic-only filter (recency-ordered) if no
       embedder is configured — graceful degradation lets the agents
       work with whatever's in the DB.

  * `search_by_topics(topics, top_k)`     — explicit tag-only filter.
       Used when the agent already knows the topic axis and just
       wants representative chunks.

Tool spec exported as ``TOOL_DEFINITIONS`` so agents can call into
the retriever via Anthropic tool_use OR OpenAI function_call without
the orchestrator caring which provider's payload format is used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.research.embedding import Embedder

logger = logging.getLogger(__name__)


_DEFAULT_TOP_K: int = 5
_MAX_TOP_K: int = 50


@dataclass
class KnowledgeChunk:
    """One retrieved chunk with citation metadata.

    Always carry the citation string + page_ref so agents can quote
    inline ("per Kahneman 2011, ch. 12, …") without a second DB call.
    """

    chunk_id: int
    source_id: str
    chunk_text: str
    page_ref: str
    title: str
    author: str
    year: int | None
    topic_tags: list[str]
    distance: float | None = None  # cosine distance, lower = closer; None for topic-only

    @property
    def citation(self) -> str:
        if self.year is not None:
            base = f"{self.author} ({self.year}). {self.title}"
        else:
            base = f"{self.author}. {self.title}"
        return f"{base}, {self.page_ref}" if self.page_ref else base


class KnowledgeRetriever:
    """Query the knowledge_chunks corpus."""

    def __init__(
        self,
        engine: Engine,
        embedder: Embedder | None = None,
    ) -> None:
        self.engine = engine
        self.embedder = embedder

    @classmethod
    def from_env(cls, engine: Engine) -> KnowledgeRetriever:
        """Construct with the embedder auto-configured from env (or None)."""
        return cls(engine=engine, embedder=Embedder.from_env())

    # ------------------------------------------------------------------
    # Search APIs
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = _DEFAULT_TOP_K,
        topics: list[str] | None = None,
    ) -> list[KnowledgeChunk]:
        """Vector + optional topic-tag retrieval.

        If the retriever has no embedder configured, falls back to
        topic-only filter (random order — embeddings haven't populated
        yet, so similarity is meaningless). Returns up to top_k chunks.
        """
        top_k = min(max(top_k, 1), _MAX_TOP_K)
        if self.embedder is None:
            logger.debug("no embedder — falling back to topic-only retrieval")
            return self._search_no_embed(top_k=top_k, topics=topics)

        embedding = self.embedder.embed([query])[0]
        emb_str = "[" + ",".join(f"{x:.7f}" for x in embedding) + "]"

        sql = """
            SELECT c.chunk_id, c.source_id, c.chunk_text, c.page_ref,
                   s.title, s.author, s.year, s.topic_tags,
                   (c.embedding <=> CAST(:emb AS vector)) AS distance
            FROM knowledge_chunks c
            JOIN knowledge_sources s ON s.source_id = c.source_id
            WHERE c.embedding IS NOT NULL
        """
        params: dict[str, Any] = {"emb": emb_str, "k": top_k}
        if topics:
            sql += " AND s.topic_tags && CAST(:topics AS TEXT[])"
            params["topics"] = topics
        sql += " ORDER BY distance ASC LIMIT :k"

        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def search_by_topics(
        self,
        topics: list[str],
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[KnowledgeChunk]:
        """Topic-tag-only retrieval — no embedding required."""
        if not topics:
            return []
        top_k = min(max(top_k, 1), _MAX_TOP_K)
        sql = """
            SELECT c.chunk_id, c.source_id, c.chunk_text, c.page_ref,
                   s.title, s.author, s.year, s.topic_tags,
                   NULL::float AS distance
            FROM knowledge_chunks c
            JOIN knowledge_sources s ON s.source_id = c.source_id
            WHERE s.topic_tags && CAST(:topics AS TEXT[])
            ORDER BY c.chunk_id ASC LIMIT :k
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"topics": topics, "k": top_k}).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def _search_no_embed(
        self,
        top_k: int,
        topics: list[str] | None,
    ) -> list[KnowledgeChunk]:
        """Fallback when no embedder is configured: topic-only or all-recent."""
        if topics:
            return self.search_by_topics(topics, top_k=top_k)
        sql = """
            SELECT c.chunk_id, c.source_id, c.chunk_text, c.page_ref,
                   s.title, s.author, s.year, s.topic_tags,
                   NULL::float AS distance
            FROM knowledge_chunks c
            JOIN knowledge_sources s ON s.source_id = c.source_id
            ORDER BY s.added_ts DESC, c.chunk_id ASC LIMIT :k
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"k": top_k}).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    @staticmethod
    def _row_to_chunk(row: Any) -> KnowledgeChunk:
        cid, sid, ctext, page, title, author, year, tags, dist = row
        return KnowledgeChunk(
            chunk_id=int(cid),
            source_id=sid,
            chunk_text=ctext,
            page_ref=page or "",
            title=title,
            author=author,
            year=int(year) if year is not None else None,
            topic_tags=list(tags) if tags else [],
            distance=float(dist) if dist is not None else None,
        )

    def cite(self, chunk: KnowledgeChunk) -> str:
        """Format a chunk as a one-line citation string."""
        return chunk.citation


# =============================================================================
# Tool spec — exposed to agents
# =============================================================================


# Anthropic tool-use schema. The OpenAI function-calling schema (used by
# DeepSeek and Grok) is structurally identical with a wrapper layer; the
# adapter at openai_tool_spec() converts.
ANTHROPIC_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "knowledge_search",
        "description": (
            "Retrieve summary chunks from the curated knowledge archive of "
            "behavioral finance, market history, and trading-philosophy "
            "works. Returns up to top_k chunks ranked by semantic "
            "similarity to the query, optionally filtered by topic tags. "
            "Each chunk carries citation metadata (author, year, title, "
            "page_ref) for inline quoting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language query, e.g. 'overconfidence in "
                        "rate cycles' or 'reflexivity feedback loops'."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max chunks to return (1–50, default 5).",
                    "default": _DEFAULT_TOP_K,
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional topic tag filter (e.g. ['behavioral-"
                        "finance', 'crisis-history']). Sources are tagged "
                        "via the corpus YAML."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "knowledge_search_by_topics",
        "description": (
            "Retrieve representative chunks from sources tagged with the "
            "given topics — no semantic similarity, just tag overlap. "
            "Use when you know the topic axis and want a sample of what "
            "the archive contains."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topic tags (any-of match).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max chunks (1–50, default 5).",
                    "default": _DEFAULT_TOP_K,
                },
            },
            "required": ["topics"],
        },
    },
]


def openai_tool_spec() -> list[dict[str, Any]]:
    """Convert the Anthropic-format tool list to OpenAI function-calling
    format, used by DeepSeek + Grok via the OpenAI-compat SDK."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in ANTHROPIC_TOOL_DEFINITIONS
    ]


def dispatch_tool_call(
    retriever: KnowledgeRetriever,
    name: str,
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run a tool call against the retriever and return JSON-serializable
    chunks. Used by the orchestrator to handle tool_use blocks from any
    provider — the chunk dicts are provider-agnostic."""
    if name == "knowledge_search":
        chunks = retriever.search(
            query=arguments["query"],
            top_k=int(arguments.get("top_k", _DEFAULT_TOP_K)),
            topics=arguments.get("topics"),
        )
    elif name == "knowledge_search_by_topics":
        chunks = retriever.search_by_topics(
            topics=arguments["topics"],
            top_k=int(arguments.get("top_k", _DEFAULT_TOP_K)),
        )
    else:
        msg = f"unknown knowledge tool: {name!r}"
        raise ValueError(msg)
    return [
        {
            "chunk_id": c.chunk_id,
            "citation": c.citation,
            "title": c.title,
            "author": c.author,
            "year": c.year,
            "page_ref": c.page_ref,
            "topic_tags": c.topic_tags,
            "chunk_text": c.chunk_text,
            "distance": c.distance,
        }
        for c in chunks
    ]
