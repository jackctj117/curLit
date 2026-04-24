-- curLit database schema
-- Run: psql -U fx -d fx -f migrations/001_schema.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ======================================================================
-- Price data (FX spot, futures, yields, equities, commodities)
-- ======================================================================
CREATE TABLE IF NOT EXISTS prices (
    ts          TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    source      TEXT NOT NULL,
    open        NUMERIC,
    high        NUMERIC,
    low         NUMERIC,
    close       NUMERIC,
    volume      NUMERIC,
    PRIMARY KEY (ts, symbol, source)
);

SELECT create_hypertable('prices', 'ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices (symbol, ts DESC);

-- ======================================================================
-- Macro economic data with vintage releases (point-in-time)
-- ======================================================================
CREATE TABLE IF NOT EXISTS macro_data (
    observation_date DATE NOT NULL,
    release_date     TIMESTAMPTZ NOT NULL,
    series_id        TEXT NOT NULL,
    value            NUMERIC,
    revision         INT DEFAULT 0,
    source           TEXT NOT NULL,
    PRIMARY KEY (observation_date, release_date, series_id)
);

CREATE INDEX IF NOT EXISTS idx_macro_series    ON macro_data (series_id, observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_macro_release   ON macro_data (release_date);
CREATE INDEX IF NOT EXISTS idx_macro_asof      ON macro_data (series_id, release_date, observation_date);

-- ======================================================================
-- Rate curves (OIS, treasuries, etc.) — snapshot by date
-- ======================================================================
CREATE TABLE IF NOT EXISTS rate_curves (
    ts          TIMESTAMPTZ NOT NULL,
    curve_id    TEXT NOT NULL,
    tenor_days  INT NOT NULL,
    rate        NUMERIC,
    PRIMARY KEY (ts, curve_id, tenor_days)
);

SELECT create_hypertable('rate_curves', 'ts', if_not_exists => TRUE);

-- ======================================================================
-- COT positioning data (weekly)
-- ======================================================================
CREATE TABLE IF NOT EXISTS cot_positioning (
    report_date   DATE NOT NULL,
    symbol        TEXT NOT NULL,
    category      TEXT NOT NULL,  -- lev_funds, asset_mgr, dealer, other_report, non_report
    longs         INT,
    shorts        INT,
    spreads       INT,
    open_interest INT,
    PRIMARY KEY (report_date, symbol, category)
);

CREATE INDEX IF NOT EXISTS idx_cot_date ON cot_positioning (report_date DESC);

-- ======================================================================
-- FX options data (IV, risk reversals, butterflies)
-- ======================================================================
CREATE TABLE IF NOT EXISTS fx_options (
    ts        TIMESTAMPTZ NOT NULL,
    pair      TEXT NOT NULL,
    tenor     TEXT NOT NULL,  -- 1W, 1M, 3M, 6M, 1Y
    atm_iv    NUMERIC,
    rr_25d    NUMERIC,
    bf_25d    NUMERIC,
    rr_10d    NUMERIC,
    bf_10d    NUMERIC,
    PRIMARY KEY (ts, pair, tenor)
);

SELECT create_hypertable('fx_options', 'ts', if_not_exists => TRUE);

-- ======================================================================
-- Computed features (cached)
-- ======================================================================
CREATE TABLE IF NOT EXISTS features (
    ts           TIMESTAMPTZ NOT NULL,
    symbol       TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value        NUMERIC,
    PRIMARY KEY (ts, symbol, feature_name)
);

SELECT create_hypertable('features', 'ts', if_not_exists => TRUE);

-- ======================================================================
-- CB sentiment — per-sentence scores
-- ======================================================================
CREATE TABLE IF NOT EXISTS cb_sentiment (
    ts             TIMESTAMPTZ NOT NULL,
    doc_id         TEXT NOT NULL,
    cb             TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    sentence_idx   INT NOT NULL,
    sentence       TEXT NOT NULL,
    lex_hawkish    INT,
    lex_dovish     INT,
    lex_net        NUMERIC,
    tfm_dovish     NUMERIC,
    tfm_neutral    NUMERIC,
    tfm_hawkish    NUMERIC,
    tfm_score      NUMERIC,
    PRIMARY KEY (doc_id, sentence_idx)
);

CREATE INDEX IF NOT EXISTS idx_cb_sent_date  ON cb_sentiment (cb, ts DESC);
CREATE INDEX IF NOT EXISTS idx_cb_sent_doc   ON cb_sentiment (doc_id);

-- ======================================================================
-- CB diff events — statement-to-statement shifts
-- ======================================================================
CREATE TABLE IF NOT EXISTS cb_diff_events (
    ts               TIMESTAMPTZ NOT NULL,
    cb               TEXT NOT NULL,
    doc_id           TEXT NOT NULL,
    prev_doc_id      TEXT NOT NULL,
    net_shift        NUMERIC,
    added_hawkish    NUMERIC,
    removed_hawkish  NUMERIC,
    change_ratio     NUMERIC,
    PRIMARY KEY (doc_id)
);

CREATE INDEX IF NOT EXISTS idx_cb_diff_date ON cb_diff_events (cb, ts DESC);

-- ======================================================================
-- Strategy signals — signal values over time
-- ======================================================================
CREATE TABLE IF NOT EXISTS strategy_signals (
    ts          TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    signal_data JSONB,
    PRIMARY KEY (ts, strategy_id)
);

-- ======================================================================
-- Strategy trades — entry/exit log
-- ======================================================================
CREATE TABLE IF NOT EXISTS strategy_trades (
    ts          TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    action      TEXT NOT NULL,  -- entry, exit, halt
    details     JSONB,
    PRIMARY KEY (ts, strategy_id, action)
);

-- ======================================================================
-- Materialized view: daily CB sentiment aggregates
-- ======================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS cb_daily_sentiment AS
SELECT
    DATE(ts) AS date,
    cb,
    doc_type,
    AVG(tfm_score) AS avg_tfm_score,
    AVG(lex_net)   AS avg_lex_score,
    COUNT(*)       AS sentence_count
FROM cb_sentiment
GROUP BY DATE(ts), cb, doc_type;

CREATE UNIQUE INDEX IF NOT EXISTS idx_cb_daily_sent ON cb_daily_sentiment (date, cb, doc_type);
