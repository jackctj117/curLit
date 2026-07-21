-- CL-dz71: intraday_quotes — a short-horizon, high-frequency quote store that
-- lets the event confluence layer (src/events/confluence.py Gate B) see an
-- ACTUAL intraday move since an event's seen_at.
--
-- The problem it fixes: the `prices` table is DAILY (and its yfinance ingest
-- can lag days). Within confluence's 30-120 min confirmation window,
-- get_latest_value(symbol, seen_at) and get_latest_value(symbol, now) return
-- the SAME daily close, so the observed move is ~0 and EVERY event EXPIRES
-- unconfirmed. A real-time quote feed gives confluence a fresh, timestamped
-- reference price at seen_at AND a current price now.
--
-- Why a SEPARATE table (not more rows in `prices`): DataProvider.get_realized_vol
-- computes DAILY returns straight off `prices`. Interleaving 2-minute quotes
-- there would corrupt the vol baseline for the rate-diff / carry strategies.
-- This table is read only by the intraday-aware path (DataProvider.get_intraday_value)
-- and never by the daily series/vol reads.
--
-- KEYED BY THE OANDA INSTRUMENT ID (XAU_USD, BCO_USD, USD_JPY, ...), NOT the
-- normalized prices-table symbol — the OANDA pricing feed is the source and it
-- covers instruments that have NO daily series at all (XAG_USD, NATGAS_USD,
-- WHEAT_USD, NAS100_USD, ...), so confluence reads it with the raw event
-- instrument id (no _normalize_symbol()).
CREATE TABLE IF NOT EXISTS intraday_quotes (
    ts       TIMESTAMPTZ NOT NULL,
    symbol   TEXT NOT NULL,            -- OANDA instrument id (XAU_USD, BCO_USD)
    source   TEXT NOT NULL DEFAULT 'oanda',
    bid      NUMERIC,
    ask      NUMERIC,
    mid      NUMERIC NOT NULL,
    PRIMARY KEY (ts, symbol, source)
);

SELECT create_hypertable('intraday_quotes', 'ts', if_not_exists => TRUE);

-- Nearest-quote-at-or-before lookups (get_intraday_value).
CREATE INDEX IF NOT EXISTS idx_intraday_quotes_symbol
    ON intraday_quotes (symbol, ts DESC);
