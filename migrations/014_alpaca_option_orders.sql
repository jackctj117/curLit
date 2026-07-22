-- CL-ldd2: alpaca_option_orders — records each advisory options idea the
-- Alpaca paper executor acted on (or deliberately skipped), keyed by idea_id
-- so an idea is never double-executed and the daily cap can be counted.
--
-- status: submitted | skipped_premium | no_contract | no_quote | no_price |
--         error. Only 'submitted' rows are real paper positions.
CREATE TABLE IF NOT EXISTS alpaca_option_orders (
    idea_id         TEXT PRIMARY KEY,
    ticker          TEXT,
    occ_symbol      TEXT,
    opt_type        TEXT,          -- call | put ("right" is a reserved word)
    qty             INTEGER,
    premium_est     NUMERIC,       -- ask * 100 * qty at submit time
    alpaca_order_id TEXT,
    status          TEXT NOT NULL,
    detail          TEXT,
    submitted_at    TIMESTAMPTZ NOT NULL
);

-- Count today's submissions fast (daily cap) + audit by ticker.
CREATE INDEX IF NOT EXISTS idx_alpaca_orders_submitted
    ON alpaca_option_orders (submitted_at);
