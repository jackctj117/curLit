-- CL-9xha: enrich the symbols universe with SEC EDGAR official company names
-- + CIK, for a sharper company-name -> ticker resolution in the niche
-- opportunity agent (CL-u2ph).
--
-- Source: SEC's free, keyless company_tickers.json
-- (https://www.sec.gov/files/company_tickers.json) — ~10k SEC-registered
-- issuers, each {cik_str, ticker, title}. The `title` is the official filer
-- name (often a cleaner / differently-phrased legal name than the verbose
-- NASDAQ Trader "Security Name", which improves name-match ranking); `cik` is
-- kept for future EDGAR filing cross-reference.
--
-- ENRICHMENT ONLY: refresh_sec_names() UPDATEs sec_name/cik on rows already in
-- the US-listed universe (ingested from the NASDAQ Trader files). It never
-- INSERTs SEC-only tickers — those lack exchange/ETF metadata and aren't
-- Robinhood-tradeable US-listed names anyway, so they'd only pollute the
-- universe. A SEC ticker that doesn't match an existing row is simply skipped.
--
-- Ticker-format note: SEC uses '-' for share classes (BRK-B) where the NASDAQ
-- Trader files use '.' (BRK.B), so a minority of class shares won't match and
-- keep a NULL sec_name. That's an acceptable, logged gap for an accuracy boost.
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS sec_name TEXT;
ALTER TABLE symbols ADD COLUMN IF NOT EXISTS cik BIGINT;

-- Case-insensitive search over the SEC official name — resolve_name() now
-- matches security_name OR sec_name and takes the better-ranked hit.
CREATE INDEX IF NOT EXISTS idx_symbols_lower_sec_name
    ON symbols (lower(sec_name));

-- CIK lookup for future EDGAR filing cross-reference.
CREATE INDEX IF NOT EXISTS idx_symbols_cik
    ON symbols (cik);
