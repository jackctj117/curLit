-- CL-i4sr: volume_spikes — relative-volume scan history for the event
-- system's equity watch universe. The RelativeVolumeScanner
-- (src/scanners/relative_volume.py) persists EVERY scanned ticker per
-- cycle (not just unusual ones — the baseline history is useful for
-- later analysis); `is_unusual` marks rvol >= threshold AND
-- avg_volume_20d above the thin-ADR floor. The Telegram digest
-- (src/events/digest.py) annotates Watch-line tickers from the latest
-- unusual rows within 24h.
CREATE TABLE IF NOT EXISTS volume_spikes (
    id               BIGSERIAL PRIMARY KEY,
    ticker           TEXT NOT NULL,
    scanned_at       TIMESTAMPTZ NOT NULL,
    rvol             DOUBLE PRECISION NOT NULL,
    volume           BIGINT,
    avg_volume_20d   DOUBLE PRECISION,
    price_change_pct DOUBLE PRECISION,
    is_unusual       BOOLEAN NOT NULL DEFAULT FALSE,
    source           TEXT NOT NULL DEFAULT 'yfinance'
);

-- Per-ticker history reads ("show me FRO's recent scans").
CREATE INDEX IF NOT EXISTS idx_volume_spikes_ticker_scanned
    ON volume_spikes (ticker, scanned_at DESC);

-- The digest's working query: "latest unusual spikes in the last 24h".
CREATE INDEX IF NOT EXISTS idx_volume_spikes_unusual_scanned
    ON volume_spikes (is_unusual, scanned_at DESC);
