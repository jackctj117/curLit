-- CL-r1ep: poly_market_probs — time-series of Polymarket prediction-market
-- YES probabilities for the tracked GEOPOLITICAL markets discovered by
-- scripts/discover_polymarket_markets.py (theme-tagged, written to
-- configs/polymarket_geo_markets.yaml).
--
-- The event pipeline's --poly step polls each tracked market's current YES
-- probability every cycle (gamma outcomePrices[0] or the CLOB midpoint) and
-- appends one row here. src/events/polymarket_signal.py reads the series to
-- detect rapid shifts (|Δ| >= threshold within a window → Telegram alert like
-- the trade cards) and to surface the latest per-theme prob to the digest as
-- corroboration ("Prediction mkt: Hormuz-closure 18% ↑").
--
-- `notified_shift` is the dedup marker: once a shift has been alerted, the
-- observation row that anchored it is stamped so the same shift is not
-- re-notified on every subsequent cycle (CL-r1ep dedup requirement).
CREATE TABLE IF NOT EXISTS poly_market_probs (
    id             BIGSERIAL PRIMARY KEY,
    slug           TEXT NOT NULL,
    question       TEXT,
    theme          TEXT,
    yes_prob       DOUBLE PRECISION NOT NULL,
    observed_at    TIMESTAMPTZ NOT NULL,
    source         TEXT NOT NULL DEFAULT 'gamma',
    notified_shift TEXT NULL
);

-- The working query: "latest observations for a slug, newest first" (shift
-- detection window scan + latest-prob-for-theme corroboration lookup).
CREATE INDEX IF NOT EXISTS idx_poly_market_probs_slug_observed
    ON poly_market_probs (slug, observed_at DESC);
