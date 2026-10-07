-- CL-7kuu: structured, write-once research status for executable ideas.
--
-- Before this, the Alpaca executors decided "niche, red-team-survived" by
-- matching free text in trade_ideas.notes (LIKE '%niche%' / '%red-team%').
-- Any writer of notes, including an LLM-authored impact idea, could satisfy
-- that convention, and it was uncoupled from the evidence/review statuses the
-- niche pipeline actually records.
--
-- One row per trade_ideas.idea_id, written ONCE by the idea ledger in the SAME
-- transaction that inserts the idea, and only for an idea the niche merge
-- created from a research-eligible NicheIdea (src/events/research_status.py).
-- The executors select on this row (research_eligible + statuses) when their
-- require_niche / require_red_team policy flags are on. They never write it.
--
-- Additive only: no existing table or row is changed. Ideas persisted before
-- this migration have no row and are therefore NOT executable under the policy
-- flags unless the explicit ALPACA_LEGACY_NOTE_MATCH=1 transition shim is set.
CREATE TABLE IF NOT EXISTS idea_research_status (
    idea_id                TEXT PRIMARY KEY REFERENCES trade_ideas (idea_id),
    discovery_status       TEXT NOT NULL,
    evidence_status        TEXT NOT NULL,
    review_status          TEXT NOT NULL,
    liquidity_status       TEXT NOT NULL,
    research_eligible      BOOLEAN NOT NULL,
    source_hashes          JSONB NOT NULL,
    score_version          TEXT NOT NULL,
    research_invocation_id TEXT,
    recorded_at            TIMESTAMPTZ NOT NULL
);

-- Immutability: an UPDATE of a recorded status is silently discarded, so no
-- later writer (executor, operator script, replay) can promote a row. A
-- plpgsql trigger cannot be used because migrations/run.py splits statements
-- on semicolons; a single-statement rule has the same effect. DELETE is left
-- possible on purpose: removing a row can only make an idea NON-executable.
-- Postgres-only statement: sqlite test shims skip it (see the test helpers).
CREATE OR REPLACE RULE idea_research_status_no_update AS
    ON UPDATE TO idea_research_status DO INSTEAD NOTHING;
