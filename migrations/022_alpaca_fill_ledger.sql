-- CL-0deu.3: additive evidence ledger. Legacy rows remain audit/decision records.
-- Unknown costs are NULL, not zero. No legacy P&L becomes verified implicitly.
CREATE TABLE IF NOT EXISTS alpaca_ledger_accounts (
    account_scope TEXT PRIMARY KEY,
    entries_paused BOOLEAN NOT NULL DEFAULT TRUE,
    reconciled_at TIMESTAMPTZ,
    snapshot_hash TEXT,
    version INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS alpaca_ledger_intents (
    account_scope TEXT NOT NULL REFERENCES alpaca_ledger_accounts(account_scope),
    intent_id TEXT NOT NULL,
    book TEXT NOT NULL CHECK (book IN ('options', 'equities')),
    idea_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    purpose TEXT NOT NULL CHECK (purpose IN ('entry', 'exit')),
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity NUMERIC NOT NULL CHECK (quantity > 0),
    multiplier NUMERIC NOT NULL CHECK (multiplier > 0),
    currency TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    detail TEXT NOT NULL,
    PRIMARY KEY (account_scope, intent_id)
);
CREATE TABLE IF NOT EXISTS alpaca_ledger_attempts (
    account_scope TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    broker_order_id TEXT,
    state TEXT NOT NULL,
    filled_quantity NUMERIC NOT NULL DEFAULT 0 CHECK (filled_quantity >= 0),
    broker_updated_at TIMESTAMPTZ,
    payload TEXT,
    PRIMARY KEY (account_scope, client_order_id),
    UNIQUE (account_scope, broker_order_id),
    FOREIGN KEY (account_scope, intent_id)
        REFERENCES alpaca_ledger_intents(account_scope, intent_id)
);
CREATE TABLE IF NOT EXISTS alpaca_ledger_activities (
    account_scope TEXT NOT NULL REFERENCES alpaca_ledger_accounts(account_scope),
    activity_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    PRIMARY KEY (account_scope, activity_id)
);
CREATE TABLE IF NOT EXISTS alpaca_ledger_fills (
    account_scope TEXT NOT NULL,
    activity_id TEXT NOT NULL,
    broker_order_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity NUMERIC NOT NULL CHECK (quantity > 0),
    price NUMERIC NOT NULL CHECK (price >= 0),
    executed_at TIMESTAMPTZ NOT NULL,
    fees NUMERIC,
    PRIMARY KEY (account_scope, activity_id),
    FOREIGN KEY (account_scope, activity_id)
        REFERENCES alpaca_ledger_activities(account_scope, activity_id)
);
CREATE TABLE IF NOT EXISTS alpaca_ledger_allocations (
    account_scope TEXT NOT NULL REFERENCES alpaca_ledger_accounts(account_scope),
    book TEXT NOT NULL,
    idea_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signed_quantity NUMERIC NOT NULL,
    entry_quantity NUMERIC NOT NULL,
    exit_quantity NUMERIC NOT NULL,
    entry_average NUMERIC,
    gross_realized NUMERIC,
    net_realized NUMERIC,
    costs_status TEXT NOT NULL,
    evidence_status TEXT NOT NULL,
    management_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    original_entered_at TIMESTAMPTZ,
    projection TEXT NOT NULL,
    PRIMARY KEY (account_scope, book, idea_id)
);
CREATE TABLE IF NOT EXISTS alpaca_ledger_repairs (
    repair_id TEXT PRIMARY KEY,
    account_scope TEXT NOT NULL,
    book TEXT NOT NULL,
    idea_id TEXT NOT NULL,
    before_payload TEXT NOT NULL,
    after_payload TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL
);
