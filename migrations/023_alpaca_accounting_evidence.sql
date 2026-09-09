-- CL-z97c: known broker fees do not imply complete per-trade billing.
CREATE TABLE IF NOT EXISTS alpaca_ledger_fee_links (
    account_scope TEXT NOT NULL,
    fee_activity_id TEXT NOT NULL,
    fill_activity_id TEXT,
    currency TEXT,
    known_cost NUMERIC,
    status TEXT NOT NULL CHECK(status IN ('linked','unallocated','pending','invalid')),
    reason TEXT,
    method TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(account_scope,fee_activity_id),
    FOREIGN KEY(account_scope,fee_activity_id)
        REFERENCES alpaca_ledger_activities(account_scope,activity_id),
    FOREIGN KEY(account_scope,fill_activity_id)
        REFERENCES alpaca_ledger_activities(account_scope,activity_id)
);

-- Proven liquidation plus an explicit accounting convention, not invented fills.
CREATE TABLE IF NOT EXISTS alpaca_ledger_historical_attributions (
    account_scope TEXT NOT NULL,
    book TEXT NOT NULL,
    idea_id TEXT NOT NULL,
    exit_order_id TEXT NOT NULL,
    allocated_quantity NUMERIC NOT NULL CHECK(allocated_quantity > 0),
    allocated_cash NUMERIC NOT NULL,
    closed_at TIMESTAMPTZ NOT NULL,
    method TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(account_scope,book,idea_id,exit_order_id),
    FOREIGN KEY(account_scope,book,idea_id)
        REFERENCES alpaca_ledger_allocations(account_scope,book,idea_id)
);
ALTER TABLE alpaca_ledger_allocations ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ;
ALTER TABLE alpaca_ledger_allocations ADD COLUMN IF NOT EXISTS attribution_method TEXT;
