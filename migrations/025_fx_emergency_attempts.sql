-- CL-pksi: durable record of every FX kill-switch emergency order attempt.
--
-- Before this, the fences that stop a second close while an emergency order's
-- outcome is unknown (CL-o9sq) lived only in process memory: a restart forgot
-- them and the next kill-switch tick could submit a duplicate close on top of
-- an order that had in fact filled. Each attempt is written BEFORE the broker
-- call (status SUBMITTING) and updated on every status change, so a crash at
-- any point leaves a row that startup recovery re-fences and resolves from
-- broker order/transaction evidence.
--
-- Cumulative fills are recorded independently of status: a partial fill
-- followed by a cancel is PARTIAL_TERMINAL (fenced until an operator
-- reconciles it), never "done" and never "untouched". last_evidence holds a
-- JSON document (TEXT for sqlite/Postgres portability) including the per-
-- transaction fill map that makes duplicate fill delivery idempotent.
--
-- Additive only: a new table, no change to existing ones.
CREATE TABLE IF NOT EXISTS fx_emergency_attempts (
    intent_id            TEXT PRIMARY KEY,
    client_order_id      TEXT NOT NULL UNIQUE,
    action               TEXT NOT NULL,
    symbol               TEXT NOT NULL,
    route_symbol         TEXT NOT NULL,
    original_qty         DOUBLE PRECISION NOT NULL,
    target               DOUBLE PRECISION NOT NULL,
    requested_qty        DOUBLE PRECISION NOT NULL CHECK (requested_qty >= 0),
    status               TEXT NOT NULL CHECK (status IN
                             ('SUBMITTING','WORKING','UNKNOWN','PARTIAL_TERMINAL',
                              'FILLED','REJECTED','CANCELLED',
                              'NOT_SUBMITTED','OPERATOR_RELEASED')),
    broker_order_id      TEXT,
    cumulative_fill_qty  DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (cumulative_fill_qty >= 0),
    last_evidence        TEXT NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL,
    updated_at           TIMESTAMPTZ NOT NULL,
    episode_id           TEXT
);

-- Startup recovery and every health tick read the unresolved rows.
CREATE INDEX IF NOT EXISTS fx_emergency_attempts_status_idx
    ON fx_emergency_attempts (status, symbol);
CREATE INDEX IF NOT EXISTS fx_emergency_attempts_episode_idx
    ON fx_emergency_attempts (episode_id);

-- The FIXED per-leg targets of each active kill-switch action (flatten_all /
-- reduce_50pct), written before any of its orders. A restart restores them so
-- a leg that already completed its reduction is never reduced again from the
-- smaller post-fill position. targets is a JSON object (TEXT for
-- portability): canonical symbol -> [route symbol, target position]. An
-- episode is CLOSED when the daily re-arm / operator resume drops the action
-- and no attempt of it is unresolved.
CREATE TABLE IF NOT EXISTS fx_emergency_episodes (
    episode_id   TEXT PRIMARY KEY,
    action       TEXT NOT NULL,
    targets      TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('OPEN','CLOSED')),
    created_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS fx_emergency_episodes_status_idx
    ON fx_emergency_episodes (status, action);
