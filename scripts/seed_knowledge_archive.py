#!/usr/bin/env python3
"""Seed the knowledge archive (CL-jg43).

For each entry in docs/research/knowledge_corpus.yaml: produce summary
chunks via the configured extract_source mode, optionally embed them,
upsert to the knowledge_sources + knowledge_chunks tables. Idempotent
on rerun (skip already-ingested source_ids).

Original PDFs/EPUBs are NEVER stored — only the structured summaries.
Same disk-saving discipline as the paper-stream ingester.

Usage:
    .venv/bin/python scripts/seed_knowledge_archive.py \\
        [--corpus docs/research/knowledge_corpus.yaml] \\
        [--importance P1|P2|all] \\
        [--llm-provider claude|deepseek] \\
        [--no-embed] [--dry-run] [--rebuild]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

# Make src.* imports work when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.embedding import Embedder, embedding_dim
from src.research.llm.client import Message, get_client
from src.runtime.run_engine import _build_db_engine

logger = logging.getLogger(__name__)


# =============================================================================
# Source loading
# =============================================================================


@dataclass
class CorpusEntry:
    title: str
    author: str
    year: int
    source_type: str
    importance: str
    topic_tags: list[str]
    rationale: str
    extract_strategy: str
    extract_source: str
    chunks_per_source: int

    @property
    def source_id(self) -> str:
        """sha256 of (title || author || year) — stable across reruns,
        matches the schema's PRIMARY KEY definition."""
        h = hashlib.sha256()
        h.update(f"{self.title}|{self.author}|{self.year}".encode())
        return h.hexdigest()

    @property
    def citation(self) -> str:
        return f"{self.author} ({self.year}). {self.title}."


def _load_corpus(path: Path) -> list[CorpusEntry]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "sources" not in raw:
        msg = f"corpus file at {path} missing 'sources' top-level key"
        raise ValueError(msg)
    return [CorpusEntry(**s) for s in raw["sources"]]


# =============================================================================
# Extraction strategies
# =============================================================================


@dataclass
class Chunk:
    chunk_idx: int
    chunk_text: str
    page_ref: str  # 'chapter N' / 'p. 145' / 'section …'


# Prompt template for llm-prior-knowledge extraction. Asks for a JSON
# array so we don't have to do delicate text parsing on the response.
_PRIOR_KNOWLEDGE_PROMPT = """\
You are producing structured summary notes for a curated knowledge archive
that AI agents will query when reasoning about market behavior.

Source: "{title}" by {author} ({year})
Number of distinct summary chunks to produce: {n_chunks}
Topic tags relevant to this work: {topics}

For each chunk, produce a 200–400 word summary that captures distinct
key claims, methodologies, frameworks, or examples from the work. Cover
DIFFERENT chapters or themes per chunk — do not repeat content. Be
honest about what you cannot verify: if you are uncertain whether a
specific quote is verbatim, paraphrase rather than fabricate.

Output ONLY a single JSON array of objects, with no preamble or trailing
prose. Each object has these fields:

  {{
    "chunk_idx":  integer 0-indexed
    "page_ref":   string like "Chapter 3" or "Part II ch. 5" — use the
                   real chapter structure of the work; never invent
                   page numbers
    "chunk_text": the 200–400 word summary
  }}

Output exactly {n_chunks} chunks. Output ONLY the JSON array.\
"""


def _strip_code_fence(s: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite instructions.
    Strip if present."""
    s = s.strip()
    if s.startswith("```"):
        # Remove leading ``` (maybe followed by language) and trailing ```.
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _extract_llm_prior_knowledge(
    entry: CorpusEntry,
    llm_provider: str,
    llm_model: str | None,
) -> list[Chunk]:
    client = get_client(provider=llm_provider)
    prompt = _PRIOR_KNOWLEDGE_PROMPT.format(
        title=entry.title,
        author=entry.author,
        year=entry.year,
        n_chunks=entry.chunks_per_source,
        topics=", ".join(entry.topic_tags),
    )
    model = llm_model or _default_model_for_provider(llm_provider)
    resp = client.complete(
        messages=[Message(role="user", content=prompt)],
        model=model,
        max_tokens=8192,
        temperature=0.2,  # low but not zero — different chunks should differ
    )
    raw = _strip_code_fence(resp.text)
    try:
        chunks_json = json.loads(raw)
    except json.JSONDecodeError:
        logger.exception("LLM did not return valid JSON for %r", entry.title)
        return []

    if not isinstance(chunks_json, list):
        logger.error("Expected JSON array, got %s for %r", type(chunks_json), entry.title)
        return []

    chunks: list[Chunk] = []
    for c in chunks_json:
        try:
            chunks.append(
                Chunk(
                    chunk_idx=int(c["chunk_idx"]),
                    chunk_text=str(c["chunk_text"]),
                    page_ref=str(c.get("page_ref", "")),
                )
            )
        except (KeyError, ValueError):
            logger.warning("Skipping malformed chunk in %r: %r", entry.title, c)
    logger.info(
        "Extracted %d chunks for %r via %s (cost $%.4f)",
        len(chunks),
        entry.title,
        resp.provider,
        resp.usd_cost,
    )
    return chunks


def _extract_notes_file(entry: CorpusEntry, notes_path: Path) -> list[Chunk]:
    """Operator-written markdown — one ## section becomes one chunk."""
    if not notes_path.exists():
        msg = f"notes file not found: {notes_path}"
        raise FileNotFoundError(msg)
    text_content = notes_path.read_text()
    sections: list[tuple[str, str]] = []  # (heading, body)
    current_heading = ""
    current_body: list[str] = []
    for line in text_content.splitlines():
        if line.startswith("## "):
            if current_heading or current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_heading or current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))
    sections = [(h, b) for h, b in sections if b]
    return [
        Chunk(chunk_idx=i, chunk_text=body, page_ref=heading or f"section {i}")
        for i, (heading, body) in enumerate(sections)
    ]


def _default_model_for_provider(provider: str) -> str:
    """Pick a sensible default model for the LLM-prior-knowledge call."""
    return {
        "claude": "claude-opus-4-7",
        "deepseek": "deepseek-chat",
        "grok": "grok-4",
    }.get(provider, "")


def _produce_chunks(
    entry: CorpusEntry,
    llm_provider: str,
    llm_model: str | None,
) -> list[Chunk]:
    """Dispatch to the right extractor based on extract_source mode."""
    src = entry.extract_source
    if src == "llm-prior-knowledge":
        return _extract_llm_prior_knowledge(entry, llm_provider, llm_model)
    if src.startswith("notes:"):
        return _extract_notes_file(entry, Path(src.removeprefix("notes:").strip()))
    if src.startswith("pdf:"):
        # Out of scope for v1 — file as a follow-up. Real PDF chunking +
        # per-chunk LLM summarization is doable but we want the v1
        # pipeline working with llm-prior-knowledge first.
        logger.warning(
            "pdf: extract_source not implemented in v1 — skipping %r. "
            "Switch this entry to notes: or llm-prior-knowledge for now.",
            entry.title,
        )
        return []
    logger.error("unknown extract_source %r for %r", src, entry.title)
    return []


# =============================================================================
# DB persistence
# =============================================================================


def _source_already_ingested(engine: Any, source_id: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM knowledge_sources WHERE source_id = :sid"),
            {"sid": source_id},
        ).fetchone()
    return row is not None


def _delete_source(engine: Any, source_id: str) -> None:
    """Used by --rebuild to wipe + re-ingest a source. Cascades to chunks."""
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM knowledge_sources WHERE source_id = :sid"),
            {"sid": source_id},
        )


def _upsert_source(engine: Any, entry: CorpusEntry) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO knowledge_sources
                  (source_id, title, author, year, source_type, citation, topic_tags)
                VALUES (:sid, :title, :author, :year, :stype, :cit, :tags)
                ON CONFLICT (source_id) DO UPDATE SET
                  title = EXCLUDED.title,
                  topic_tags = EXCLUDED.topic_tags
            """),
            {
                "sid": entry.source_id,
                "title": entry.title,
                "author": entry.author,
                "year": entry.year,
                "stype": entry.source_type,
                "cit": entry.citation,
                "tags": entry.topic_tags,
            },
        )


def _upsert_chunks(
    engine: Any,
    source_id: str,
    chunks: list[Chunk],
    embeddings: list[list[float]] | None,
) -> int:
    if embeddings is not None and len(embeddings) != len(chunks):
        msg = (
            f"embeddings count {len(embeddings)} != chunks count "
            f"{len(chunks)} for source {source_id}"
        )
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []
    for i, c in enumerate(chunks):
        emb = embeddings[i] if embeddings is not None else None
        # Format pgvector literal: "[0.1, 0.2, …]"
        emb_str = "[" + ",".join(f"{x:.7f}" for x in emb) + "]" if emb is not None else None
        rows.append(
            {
                "sid": source_id,
                "idx": c.chunk_idx,
                "text": c.chunk_text,
                "emb": emb_str,
                "page": c.page_ref,
            }
        )

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO knowledge_chunks
                  (source_id, chunk_idx, chunk_text, embedding, page_ref)
                VALUES (:sid, :idx, :text, CAST(:emb AS vector), :page)
                ON CONFLICT (source_id, chunk_idx) DO UPDATE SET
                  chunk_text = EXCLUDED.chunk_text,
                  embedding = EXCLUDED.embedding,
                  page_ref = EXCLUDED.page_ref
            """),
            rows,
        )
    return len(rows)


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed the knowledge archive from a curated corpus YAML.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("docs/research/knowledge_corpus.yaml"),
    )
    parser.add_argument(
        "--importance",
        choices=["P1", "P2", "all"],
        default="P1",
        help="Filter which entries to ingest (default: P1 only).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["claude", "deepseek", "grok"],
        default="claude",
        help="Which LLM produces the chunks for llm-prior-knowledge entries.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Specific model — falls back to provider default.",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding step. Chunks stored with embedding=NULL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without calling LLM or DB.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete + re-ingest each entry (skips idempotency check).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    corpus = _load_corpus(args.corpus)
    if args.importance != "all":
        corpus = [e for e in corpus if e.importance == args.importance]
    logger.info("Corpus: %d entries (after %s filter)", len(corpus), args.importance)

    if args.dry_run:
        for e in corpus:
            print(
                f"  {e.importance} {e.title!r:<55} ({e.author}, {e.year})  "
                f"chunks={e.chunks_per_source}  src={e.extract_source}"
            )
        return 0

    engine = _build_db_engine()
    embedder = None if args.no_embed else Embedder.from_env()
    if embedder is None and not args.no_embed:
        logger.info(
            "OPENAI_API_KEY not set; chunks will be stored with NULL embeddings. "
            "Set the key and rerun with --rebuild to backfill."
        )

    summary: list[dict[str, Any]] = []
    for entry in corpus:
        t0 = time.time()
        sid = entry.source_id
        if _source_already_ingested(engine, sid) and not args.rebuild:
            logger.info("Skipping already-ingested: %r", entry.title)
            continue
        if args.rebuild:
            _delete_source(engine, sid)

        chunks = _produce_chunks(entry, args.llm_provider, args.llm_model)
        if not chunks:
            logger.warning("No chunks produced for %r — skipping persistence", entry.title)
            continue

        embeddings: list[list[float]] | None = None
        if embedder is not None:
            try:
                embeddings = embedder.embed([c.chunk_text for c in chunks])
                if len(embeddings[0]) != embedding_dim():
                    logger.error(
                        "Embedding dim %d != schema dim %d — schema needs migration",
                        len(embeddings[0]),
                        embedding_dim(),
                    )
                    embeddings = None
            except Exception:
                logger.exception("Embedding failed for %r — storing without", entry.title)

        _upsert_source(engine, entry)
        n = _upsert_chunks(engine, sid, chunks, embeddings)
        summary.append(
            {
                "title": entry.title,
                "chunks": n,
                "embedded": embeddings is not None,
                "elapsed_sec": round(time.time() - t0, 1),
            }
        )

    print()
    if not summary:
        print("(no entries ingested — all already present, or no chunks produced)")
        return 0
    print(f"{'title':<55}  {'chunks':>7}  {'embedded':>9}  elapsed")
    print("-" * 90)
    for s in summary:
        print(
            f"{s['title']:<55}  {s['chunks']:>7}  "
            f"{'yes' if s['embedded'] else 'no':>9}  {s['elapsed_sec']:>6}s"
        )
    print(
        f"\nTotal ingested: {sum(s['chunks'] for s in summary)} chunks across {len(summary)} sources"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
