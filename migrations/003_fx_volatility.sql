-- CL-43l: fx_volatility table — volatility index tracking
-- Stores daily values for VIX-like indices (CVIX, JPYVIX, MOVE, etc.)
-- consumed by the carry+vol strategy and portfolio kill-switches.
CREATE TABLE IF NOT EXISTS fx_volatility (
    date        DATE NOT NULL,
    index_name  TEXT NOT NULL,
    value       NUMERIC NOT NULL,
    PRIMARY KEY (date, index_name)
);

-- Common-case query: latest value per index, or a window of one index.
CREATE INDEX IF NOT EXISTS idx_fx_volatility_index_date
    ON fx_volatility (index_name, date DESC);
