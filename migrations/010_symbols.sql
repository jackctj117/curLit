-- CL-tzug: symbols — the full US-listed symbol universe ingested from the
-- free NASDAQ Trader public files (nasdaqlisted.txt + otherlisted.txt),
-- refreshed daily by scripts/refresh_symbols.py / src.data.symbols.
--
-- Purpose: answer "does this ticker exist?", resolve a company name to its
-- ticker (the LLM-knows-the-company-but-guesses-the-wrong-symbol case), and
-- scope the niche opportunity agent (CL-u2ph) to Robinhood-tradeable names.
--
-- HONEST SCOPE: this is US-listed common stock + ETFs only (NASDAQ, NYSE,
-- AMEX, ARCA — all Robinhood-tradeable), ~13k rows. Comprehensive OTC /
-- pink-sheets and non-US/global exchanges are NOT covered by the free NASDAQ
-- Trader files (those need OTC Markets / a paid vendor). A symbol absent
-- from this table therefore means "not in the US-listed universe", not
-- definitively "untradeable".
--
-- Test issues (Test Issue = Y in the source files) are excluded at ingest
-- time; the is_test_issue column exists for completeness / auditability but
-- the ingester never writes Y rows.
CREATE TABLE IF NOT EXISTS symbols (
    id             BIGSERIAL PRIMARY KEY,
    symbol         TEXT NOT NULL,
    security_name  TEXT,
    -- NASDAQ | NYSE | AMEX | ARCA | OTHER
    exchange       TEXT,
    is_etf         BOOLEAN NOT NULL DEFAULT FALSE,
    is_test_issue  BOOLEAN NOT NULL DEFAULT FALSE,
    -- Which NASDAQ Trader file the row came from: 'nasdaqlisted' | 'otherlisted'
    source         TEXT NOT NULL,
    last_refreshed TIMESTAMPTZ NOT NULL,
    UNIQUE (symbol)
);

-- Fast case-insensitive company-name search (resolve_name substring match).
CREATE INDEX IF NOT EXISTS idx_symbols_lower_name
    ON symbols (lower(security_name));

-- Ticker lookup (exists / get / is_etf / robinhood_tradeable).
CREATE INDEX IF NOT EXISTS idx_symbols_symbol
    ON symbols (symbol);
