-- CL-ynuh: persistent hop-graph edge memory (SHADOW research only).
--
-- Stores relationship edges the hop-graph verifier resolved against an exact
-- captured filing passage, so a repeated theme (e.g. Hormuz) reuses hop-1
-- facts and spends its bounded budget on hops 2-3. Contradicted edges are
-- kept so traversal never re-proposes them while unexpired.
--
-- Additive: no existing table is touched and nothing in the trading path
-- reads this table. Freshness: expires_at = as_of + 730 days, the same two
-- annual filing cycles as research_evidence.MAX_SOURCE_AGE_DAYS.
-- source_record is the JSON SourceDocument (TEXT so the sqlite test shim
-- needs only TIMESTAMPTZ -> TEXT); its SHA-256 identity must equal
-- source_hash, which the loader re-verifies before reuse.
CREATE TABLE IF NOT EXISTS niche_edges (
    src_id          TEXT NOT NULL,
    src_kind        TEXT NOT NULL,
    src_ticker      TEXT,
    src_label       TEXT NOT NULL,
    dst_id          TEXT NOT NULL,
    dst_kind        TEXT NOT NULL,
    dst_ticker      TEXT,
    dst_label       TEXT NOT NULL,
    relation        TEXT NOT NULL CHECK (relation IN
                        ('supplies','buys_from','competes_with','substitutes_for',
                         'depends_on_route','hedged_by','priced_off')),
    status          TEXT NOT NULL CHECK (status IN ('sourced','contradicted')),
    claim           TEXT NOT NULL,
    source_hash     TEXT NOT NULL,
    passage         TEXT NOT NULL,
    passage_locator TEXT NOT NULL,
    source_record   TEXT NOT NULL,
    event_theme     TEXT NOT NULL,
    as_of           TIMESTAMPTZ NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    recorded_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (src_id, dst_id, relation, event_theme),
    CHECK (expires_at > as_of)
);
CREATE INDEX IF NOT EXISTS idx_niche_edges_theme_status
    ON niche_edges (event_theme, status, expires_at);
CREATE INDEX IF NOT EXISTS idx_niche_edges_status_expiry
    ON niche_edges (status, expires_at);
