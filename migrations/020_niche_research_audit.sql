-- CL-27s0: research evidence is independent of mutable event execution state.
-- No cascading event FK: deleting an event must not delete its audit evidence.
CREATE TABLE IF NOT EXISTS niche_research_audit (
    invocation_id TEXT PRIMARY KEY,
    event_id BIGINT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_snapshot JSONB NOT NULL,
    report JSONB NOT NULL,
    payload_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_niche_research_audit_event
    ON niche_research_audit (event_id, recorded_at);
