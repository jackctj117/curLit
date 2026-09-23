-- CL-0deu.2: durable account-wide entry halt shared by every execution path.
--
-- Before this, "halt" meant three unrelated things: an in-memory OMS flag
-- (lost on restart, invisible to the Alpaca daemons), /api/system/halt
-- flipping that same flag, and a boot-time ALPACA_LEDGER_CLOSE_ONLY env var.
-- None of them covered the whole account, none survived a restart as a
-- recorded decision, and none could tell the operator whether every writer
-- had actually honored it.
--
-- One row holds the CURRENT requested mode. ``version`` increments on every
-- change so a path can acknowledge exactly which request it observed.
CREATE TABLE IF NOT EXISTS trading_halt_state (
    id          INTEGER PRIMARY KEY CHECK (id = 1),   -- singleton
    mode        TEXT NOT NULL CHECK (mode IN
                    ('ACTIVE','PAUSE_ENTRIES','CLOSE_ONLY','EMERGENCY_FLATTEN')),
    version     INTEGER NOT NULL CHECK (version >= 1),
    reason      TEXT NOT NULL,
    source      TEXT NOT NULL,
    changed_by  TEXT NOT NULL,
    changed_at  TIMESTAMPTZ NOT NULL
);

-- Fail-closed seed: a fresh or newly migrated database starts with entries
-- PAUSED. Resuming is an explicit, logged operator act (TradingHaltStore
-- .resume) — deploying this migration can never silently unpause a book.
-- ON CONFLICT keeps an existing decision untouched on re-run.
INSERT INTO trading_halt_state (id, mode, version, reason, source, changed_by, changed_at)
VALUES (1, 'PAUSE_ENTRIES', 1,
        'initial state from migration 024: explicit operator resume required',
        'migration', 'migration-024', CURRENT_TIMESTAMP)
ON CONFLICT (id) DO NOTHING;

-- Append-only audit of every requested change (who, why, when, from where).
CREATE TABLE IF NOT EXISTS trading_halt_events (
    version     INTEGER PRIMARY KEY,
    mode        TEXT NOT NULL,
    reason      TEXT NOT NULL,
    source      TEXT NOT NULL,
    changed_by  TEXT NOT NULL,
    changed_at  TIMESTAMPTZ NOT NULL
);

-- Per-path acknowledgement: the latest version each writer observed at a
-- quiescent point (none of its own entry submissions in flight). A halt is
-- only reported APPLIED once every path acked the current version or reported
-- itself visibly unavailable.
CREATE TABLE IF NOT EXISTS trading_halt_acks (
    path        TEXT PRIMARY KEY,
    version     INTEGER NOT NULL,
    mode        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('applied','unavailable')),
    detail      TEXT,
    acked_at    TIMESTAMPTZ NOT NULL
);
