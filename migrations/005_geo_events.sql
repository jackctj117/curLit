-- CL-6iu7: geo_events — current-events trading pipeline (shared schema).
-- Ingestion half (GDELT poll + Event Impact Agent) writes NEW → ASSESSED;
-- the consumer half (confirmation + strategy) advances ASSESSED →
-- CONFIRMED | EXPIRED | TRADED | DISMISSED. `external_id` is the dedup
-- key (sha256 of the normalized article URL, prefixed by source).
-- `assessment` holds the Event Impact Agent's strict-JSON verdict:
--   {core_event, direction, urgency, horizon, confidence,
--    affected: [{instrument, kind, direction, reason}], rationale}
CREATE TABLE IF NOT EXISTS geo_events (
    id                BIGSERIAL PRIMARY KEY,
    seen_at           TIMESTAMPTZ NOT NULL,
    source            TEXT NOT NULL,
    external_id       TEXT UNIQUE NOT NULL,
    headline          TEXT NOT NULL,
    url               TEXT,
    theme             TEXT,
    assessment        JSONB,
    status            TEXT NOT NULL DEFAULT 'NEW'
        CHECK (status IN ('NEW', 'ASSESSED', 'CONFIRMED',
                          'EXPIRED', 'TRADED', 'DISMISSED')),
    status_updated_at TIMESTAMPTZ NOT NULL
);

-- The pipeline's working query: "give me NEW (or ASSESSED) rows, newest
-- first" — both the impact agent and the consumer poll on this pair.
CREATE INDEX IF NOT EXISTS idx_geo_events_status_seen
    ON geo_events (status, seen_at);
