-- CL-ut1d: bounded research retries independent of NEW event assessment.
-- Epoch seconds avoid local/session timezone ambiguity for leases/deadlines.
-- The snapshot is immutable input, never a model-authored execution approval.
CREATE TABLE IF NOT EXISTS niche_jobs (
    job_id TEXT PRIMARY KEY,
    event_id BIGINT NOT NULL,
    policy_version TEXT NOT NULL,
    input_snapshot TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN
        ('pending', 'running', 'retry', 'completed', 'failed', 'expired', 'superseded')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    lease_until DOUBLE PRECISION,
    invocation_id TEXT,
    reason TEXT NOT NULL DEFAULT '',
    updated_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_niche_jobs_due ON niche_jobs (state, next_attempt);
CREATE INDEX IF NOT EXISTS idx_niche_jobs_event ON niche_jobs (event_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_niche_jobs_invocation ON niche_jobs (invocation_id);
