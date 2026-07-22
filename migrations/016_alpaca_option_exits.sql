-- CL-3rho: options EXIT manager — completes the Alpaca paper loop
-- (idea -> entry -> monitoring -> exit). The entry executor (CL-ldd2) keys
-- everything by idea_id in alpaca_option_orders; the exit manager appends
-- the close side to the SAME row, so one row tells the whole life of a
-- position: what we bought, why we sold, and what it made.
--
-- exit_status: NULL      = position open (or row is not a real position)
--              submitted = sell-to-close placed, awaiting fill
--              closed    = position gone (fill confirmed / expired / external)
-- exit_reason: thesis_invalidated | time_stop | stop_loss | profit_target |
--              expiry_protect | stale | expired_worthless | closed_external
-- pnl_pct is the realized premium return (e.g. -0.42 = lost 42% of premium).
--
-- Kept as separate columns (not a status overload) so every existing query
-- — daily-cap counting, idea dedup — is untouched.
ALTER TABLE alpaca_option_orders ADD COLUMN IF NOT EXISTS exit_status   TEXT;
ALTER TABLE alpaca_option_orders ADD COLUMN IF NOT EXISTS exit_reason   TEXT;
ALTER TABLE alpaca_option_orders ADD COLUMN IF NOT EXISTS exit_order_id TEXT;
ALTER TABLE alpaca_option_orders ADD COLUMN IF NOT EXISTS exit_premium  NUMERIC;
ALTER TABLE alpaca_option_orders ADD COLUMN IF NOT EXISTS pnl_pct       DOUBLE PRECISION;
ALTER TABLE alpaca_option_orders ADD COLUMN IF NOT EXISTS exited_at     TIMESTAMPTZ;
