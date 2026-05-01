-- CL-1fm: research_papers table — academic paper triage
-- Tracks papers ingested from arXiv/NBER/SSRN/BIS for the research
-- pipeline. relevance_score is set by RelevanceScorer (CL-366);
-- read_status / my_notes / implementation_priority are operator-edited.
CREATE TABLE IF NOT EXISTS research_papers (
    paper_id                TEXT PRIMARY KEY,
    source                  TEXT,
    title                   TEXT,
    authors                 JSONB,
    abstract                TEXT,
    url                     TEXT,
    pdf_url                 TEXT,
    published_date          TIMESTAMPTZ,
    ingested_at             TIMESTAMPTZ DEFAULT NOW(),
    keywords                JSONB,
    categories              JSONB,
    relevance_score         NUMERIC DEFAULT 0,
    read_status             TEXT DEFAULT 'unread',
    my_notes                TEXT,
    implementation_priority INT DEFAULT 0,
    evaluation_data         JSONB
);

-- Triage view: unread, sorted by relevance — the dashboard query.
CREATE INDEX IF NOT EXISTS idx_research_papers_triage
    ON research_papers (read_status, relevance_score DESC);

-- Recency view for "what came in today" reports.
CREATE INDEX IF NOT EXISTS idx_research_papers_published
    ON research_papers (published_date DESC);
