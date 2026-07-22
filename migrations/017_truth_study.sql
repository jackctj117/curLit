-- CL-s9as: Truth Social event-study research module — RESEARCH ONLY.
-- Measures short-horizon market reaction to public posts. Deliberately
-- contains NO trading fields, NO sizing, NO portfolio/holdings columns —
-- the operator-approved scope is post -> classification -> measured
-- reaction, full stop. A "no durable edge" result is a valid outcome.
CREATE TABLE IF NOT EXISTS truth_posts (
    post_id     TEXT PRIMARY KEY,          -- Truth Social status id
    posted_at   TIMESTAMPTZ NOT NULL,
    text        TEXT NOT NULL,             -- '' for media-only posts
    url         TEXT,
    raw_json    JSONB,
    ingested_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_truth_posts_posted ON truth_posts (posted_at);

-- One classification per post, stamped with the classifier version so a
-- reclassification pass can be compared against the old labels.
CREATE TABLE IF NOT EXISTS truth_classifications (
    post_id            TEXT PRIMARY KEY REFERENCES truth_posts(post_id),
    is_market_relevant BOOLEAN NOT NULL,
    primary_topic      TEXT,      -- tariffs|china|energy|defense|appointments|
                                  -- broad_market|self_referential|trade|
                                  -- monetary|other
    secondary_topics   JSONB,
    tone               TEXT,      -- positive|negative|threatening|
                                  -- de-escalatory|neutral
    named_entities     JSONB,
    explicit_market_language BOOLEAN NOT NULL DEFAULT FALSE,
    confidence         DOUBLE PRECISION,
    classifier_version TEXT NOT NULL,
    classified_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_truth_class_relevant
    ON truth_classifications (is_market_relevant);

-- Event-study measurements. All *_pct columns are PERCENT (−0.42 means
-- −0.42%), matching how the numbers read in reports. NULL return = no
-- bars in the window (market closed / halted) — honest, never zero.
CREATE TABLE IF NOT EXISTS truth_market_reactions (
    id                BIGSERIAL PRIMARY KEY,
    post_id           TEXT NOT NULL REFERENCES truth_posts(post_id),
    instrument        TEXT NOT NULL,       -- SPY, QQQ, XLE, ...
    window_minutes    INTEGER NOT NULL,    -- 1, 5, 15, 30, 60, 120
    return_pct        DOUBLE PRECISION,
    volume_ratio      DOUBLE PRECISION,    -- window vol vs pre-post baseline
    max_favorable_pct DOUBLE PRECISION,
    max_adverse_pct   DOUBLE PRECISION,
    start_price       DOUBLE PRECISION,
    end_price         DOUBLE PRECISION,
    measured_at       TIMESTAMPTZ NOT NULL,
    UNIQUE (post_id, instrument, window_minutes)
);

CREATE INDEX IF NOT EXISTS idx_truth_reactions_post
    ON truth_market_reactions (post_id);
