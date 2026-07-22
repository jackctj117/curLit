-- CL-mtum: options_activity — daily per-ticker options-chain snapshots from
-- the FREE yfinance chain data (per-contract volume / open interest / IV,
-- aggregated across the nearest expiries). After ~2-3 weeks of snapshots the
-- table is its own baseline: "options volume 3x its 20-snapshot average" is
-- the unusual-activity signal, P/C volume ratio the direction read.
--
-- HONEST SCOPE: this is delayed, aggregate positioning data. It shows THAT
-- options are active and which way volume skews — NOT sweeps or aggressor
-- side (buyer vs seller), which need paid tick-level trade data (the
-- documented upgrade path: Unusual Whales API / OPRA feed swaps in as the
-- source; the table and consumers stay).
CREATE TABLE IF NOT EXISTS options_activity (
    obs_date         DATE NOT NULL,
    ticker           TEXT NOT NULL,
    call_volume      BIGINT,
    put_volume       BIGINT,
    call_oi          BIGINT,
    put_oi           BIGINT,
    pc_volume_ratio  NUMERIC,   -- put_vol / max(call_vol, 1)
    atm_iv           NUMERIC,   -- nearest-expiry ATM implied vol (NULL if junk)
    expiries_sampled INTEGER,
    source           TEXT NOT NULL DEFAULT 'yfinance',
    created_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (obs_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_options_activity_ticker
    ON options_activity (ticker, obs_date DESC);
