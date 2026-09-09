-- CL-jk7i: durable per-theme pending windows and source-wide retry/cadence state.
-- Payload is versioned JSON text so offline SQLite failure fixtures exercise
-- the same bound SQL. Dates are UTC ISO strings; scheduling clocks are Unix UTC.
CREATE TABLE IF NOT EXISTS gdelt_ingest_cursors (
    theme TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
