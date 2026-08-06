-- CL-ncbq: alpaca_equity_orders — the SHARES expression of the SAME advisory
-- event ideas the options executor (mig 014/016) trades as contracts.
--
-- WHY a second book: the CL-4c7o review measured the desk's equity ideas
-- getting DIRECTION right on the underlying 80% of the time (41/51) while the
-- short-dated OTM options expressing them won only 7% — spread + theta ate the
-- 1-2% moves the theses actually produced. This table records the plain-shares
-- A/B of those identical ideas, so the two books can be compared on the same
-- signal set. An idea may appear in BOTH tables: that is the point, not a bug,
-- and neither executor marks the trade_ideas row on entry.
--
-- side:   buy        = long shares  (from a buy_calls idea)
--         sell_short = short shares (from a buy_puts idea)
-- status: submitted | skipped_expiring | skipped_short_disabled |
--         skipped_price_too_high. Only 'submitted' rows are real positions,
--         and only they count against the daily/hourly caps.
--
-- Deliberately NO entry_mid / entry_spread_pct twins of mig 018: equities
-- quote in pennies, so the fill IS the honest basis — there is no spread
-- artifact to cancel out, which is the whole reason this book exists.
CREATE TABLE IF NOT EXISTS alpaca_equity_orders (
    idea_id         TEXT PRIMARY KEY,   -- UNIQUE dedup key (one act per idea)
    ticker          TEXT,
    side            TEXT,               -- buy | sell_short
    qty             INTEGER,
    notional_est    NUMERIC,            -- qty * price at submit time
    entry_price     NUMERIC,            -- fill price (falls back to the
                                        -- submit-time last price when Alpaca
                                        -- has not reported a fill yet)
    alpaca_order_id TEXT,
    status          TEXT NOT NULL,
    detail          TEXT,
    submitted_at    TIMESTAMPTZ NOT NULL,
    -- Exit side on the SAME row (mirrors mig 016): one row tells the whole
    -- life of a position — what we bought, why we closed, what it made.
    -- exit_status: NULL = open | submitted = close order placed | closed
    -- exit_reason: thesis_invalidated | time_stop | stop_loss |
    --              profit_target | stale | closed_external
    exit_status     TEXT,
    exit_reason     TEXT,
    exit_order_id   TEXT,
    exit_price      NUMERIC,
    pnl_pct         NUMERIC,            -- SIGNED for the side (a short that
                                        -- fell is positive)
    exited_at       TIMESTAMPTZ
);

-- Count today's/this hour's submissions fast (daily + hourly caps).
CREATE INDEX IF NOT EXISTS idx_alpaca_equity_submitted
    ON alpaca_equity_orders (submitted_at);

-- The exit manager's working set: open positions only.
CREATE INDEX IF NOT EXISTS idx_alpaca_equity_open
    ON alpaca_equity_orders (status, exit_status);
