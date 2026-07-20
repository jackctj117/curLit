-- CL-jiqq: concrete, actionable trade-idea levels. The advisory
-- trade_ideas ledger (migration 007) recorded the LLM's directional
-- call + horizon; the operator asked for "more detail in the puts and
-- shorts like tickers price points dates and everything I would need to
-- actually put money into it." These columns persist the GROUNDED
-- trade-card numbers computed by src/events/trade_card.py from the
-- LLM's percentages and the LIVE price fetched per assess cycle:
--   stop_price       real dollar stop level, direction-aware
--                    (below spot for longs/calls, above for shorts/puts)
--   target_prices    JSONB list of 1-2 dollar profit targets
--   risk_reward      |target1 - entry| / |entry - stop| (first target)
--   entry_trigger    the CONDITION to enter (LLM words, not a price)
--   invalidation     the observable fact that kills the thesis
--   dte_window       option days-to-expiry WINDOW (e.g. "2-4 weeks") —
--                    never a fabricated calendar expiry date
--   suggested_strike option strike LEVEL (~5% OTM) — advisory only; the
--                    operator picks the nearest listed strike/expiry.
-- All are NULLable: an idea with no live price (price feed miss) stores
-- the percentages only; a legacy row from before this migration keeps
-- NULLs until recomputed. The share-price stop for OPTION ideas reflects
-- the underlying move, not the premium fraction (a premium % is not a
-- price); see the trade_card module docstring.
ALTER TABLE trade_ideas
    ADD COLUMN IF NOT EXISTS stop_price       DOUBLE PRECISION NULL,
    ADD COLUMN IF NOT EXISTS target_prices    JSONB NULL,
    ADD COLUMN IF NOT EXISTS risk_reward      DOUBLE PRECISION NULL,
    ADD COLUMN IF NOT EXISTS entry_trigger    TEXT NULL,
    ADD COLUMN IF NOT EXISTS invalidation     TEXT NULL,
    ADD COLUMN IF NOT EXISTS dte_window       TEXT NULL,
    ADD COLUMN IF NOT EXISTS suggested_strike DOUBLE PRECISION NULL;
