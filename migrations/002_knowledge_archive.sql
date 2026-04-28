-- Knowledge archive (CL-mjgr) — RAG corpus for the multi-agent research
-- pipeline (parent CL-h986). Two tables:
--   * knowledge_sources — citation metadata (one row per book/paper)
--   * knowledge_chunks  — embedded text chunks for vector retrieval
--
-- Original PDFs/EPUBs are NEVER stored — only LLM-extracted chunks plus
-- their embeddings plus the citation needed to cite them. Same disk-
-- saving discipline as the paper-stream ingester.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_sources (
    -- sha256 of (title || author || year), 64 hex chars. Lets us dedup
    -- across reruns and reference back from chunks without orphaning.
    source_id    TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    author       TEXT NOT NULL,
    year         INT,
    -- 'book' | 'paper' | 'newsletter' | 'speech' | 'transcript'
    source_type  TEXT NOT NULL,
    -- pre-formatted citation string the agents quote when retrieving
    citation     TEXT NOT NULL,
    -- topic taxonomy: behavioral-finance, market-microstructure,
    -- mass-psychology, factor-investing, crisis-history, …
    topic_tags   TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    added_ts     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id     BIGSERIAL PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE CASCADE,
    chunk_idx    INT NOT NULL,
    chunk_text   TEXT NOT NULL,
    -- 1536 = OpenAI text-embedding-3-small dimension. Anthropic doesn't
    -- expose embeddings — if the embedder is swapped, alter the dim and
    -- re-seed the chunks table.
    embedding    vector(1536),
    page_ref     TEXT,
    UNIQUE (source_id, chunk_idx)
);

-- IVFFlat is the cheapest pgvector index and is fine for our corpus
-- size (~50 books × ~50 chunks = a few thousand rows). Lists=100 is
-- the default rule-of-thumb for tables under ~1M rows.
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding
    ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source
    ON knowledge_chunks (source_id);
