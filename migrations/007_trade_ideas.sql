-- CL-mgcp: trade_ideas — persisted ledger of the impact agent's
-- ADVISORY trade ideas (assessment.trade_ideas[], CL-01zt). Rows are
-- written by src/events/idea_ledger.py from each assess cycle:
-- `idea_id` = sha1(geo_event_id:ticker:action)[:16] so re-assessing the
-- same event upserts idempotently (ON CONFLICT DO NOTHING).
-- `preferred_instrument` / `instrument_reason` / `stop_loss_pct` are
-- gap-filled by the horizon-based normalizer
-- (src/events/instrument_selector.py) when the LLM omitted them — the
-- LLM's own `action` is never overridden, only annotated in `notes`.
-- `price_at_signal` is the last close at persist time when the price
-- helper could resolve one (NULL is honest: no price source reached).
-- Lifecycle: pending → expired happens automatically in the pipeline
-- cycle once `time_stop_days` elapses; taken/cancelled/closed are
-- reserved for future operator-driven transitions (v1 is read-only
-- beyond auto-expiry).
CREATE TABLE IF NOT EXISTS trade_ideas (
    id                   BIGSERIAL PRIMARY KEY,
    idea_id              TEXT UNIQUE NOT NULL,
    geo_event_id         BIGINT NOT NULL,
    ticker               TEXT NOT NULL,
    action               TEXT NOT NULL,
    direction            TEXT,
    confidence           DOUBLE PRECISION,
    time_horizon         TEXT,
    holding_period_days  TEXT,
    time_stop_days       INT,
    stop_loss_pct        DOUBLE PRECISION NULL,
    preferred_instrument TEXT,
    instrument_reason    TEXT,
    rationale            TEXT,
    suggested_entry      TEXT,
    notes                TEXT,
    price_at_signal      DOUBLE PRECISION NULL,
    created_at           TIMESTAMPTZ NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'taken', 'expired',
                          'cancelled', 'closed')),
    status_updated_at    TIMESTAMPTZ NOT NULL
);

-- The working queries: "open ideas, newest first" (bot listing +
-- auto-expiry sweep) and per-ticker history.
CREATE INDEX IF NOT EXISTS idx_trade_ideas_status_created
    ON trade_ideas (status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_trade_ideas_ticker
    ON trade_ideas (ticker);
