-- CL-6axf: idea_outcomes — the closed-loop track record. For every surfaced
-- trade idea (trade_ideas, migration 007) that captured an entry price
-- (price_at_signal), this table records the forward outcome: current price,
-- signed direction-adjusted return, best/worst excursion (MFE/MAE), and a
-- final win/loss/flat verdict once the idea's horizon elapses.
--
-- It is the foundation for the reflective self-tuning loop (CL-#2): idea
-- attributes are DENORMALISED here at score time (theme, action, direction,
-- confidence, niche/hop, red-team-survived) so performance can be aggregated
-- by dimension with a plain GROUP BY — no re-join or notes-parsing needed.
--
-- Keyed by idea_id (FK-in-spirit to trade_ideas.idea_id); one row per idea,
-- upserted each scoring cycle until it finalises.
CREATE TABLE IF NOT EXISTS idea_outcomes (
    idea_id            TEXT PRIMARY KEY,
    ticker             TEXT NOT NULL,
    action             TEXT,
    direction          TEXT,
    theme              TEXT,
    time_horizon       TEXT,
    confidence         NUMERIC,
    is_niche           BOOLEAN NOT NULL DEFAULT FALSE,
    hop_count          INTEGER,
    red_team_survived  BOOLEAN NOT NULL DEFAULT FALSE,
    -- entry snapshot (copied from trade_ideas at first scoring)
    entry_price        NUMERIC,
    entry_at           TIMESTAMPTZ,
    -- forward measurement (updated each cycle until finalised)
    last_price         NUMERIC,
    last_at            TIMESTAMPTZ,
    return_pct         NUMERIC,   -- signed, in the idea's direction
    max_favorable_pct  NUMERIC,   -- best signed return seen (MFE)
    max_adverse_pct    NUMERIC,   -- worst signed return seen (MAE)
    horizon_days       INTEGER,
    -- open | win | loss | flat | no_data
    outcome            TEXT NOT NULL DEFAULT 'open',
    scored_count       INTEGER NOT NULL DEFAULT 0,
    created_at         TIMESTAMPTZ,
    updated_at         TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_idea_outcomes_outcome
    ON idea_outcomes (outcome);
CREATE INDEX IF NOT EXISTS idx_idea_outcomes_theme
    ON idea_outcomes (theme);
