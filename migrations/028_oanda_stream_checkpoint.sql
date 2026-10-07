-- CL-pksi (stream catch-up): durable cursor for the OANDA transaction stream.
--
-- The transaction stream only delivers transactions created while it is
-- connected. Before this, fills that happened during a disconnect (network
-- blip, stall, engine restart) were seen only by the ~300 s position poll,
-- without per-order attribution. The engine now records the id of the last
-- transaction whose fill was JOURNALED here, and on every (re)connect replays
-- GET /v3/accounts/{id}/transactions/sinceid?id=<last_transaction_id>
-- through the same fill path (deduplicated by transaction id).
--
-- One row per OANDA account. The cursor only moves forward (the writer's
-- upsert refuses a smaller id). Additive only: a new table, no change to
-- existing ones. Plain types so the same file applies to sqlite in tests.
CREATE TABLE IF NOT EXISTS oanda_stream_checkpoint (
    account_id           TEXT PRIMARY KEY,
    last_transaction_id  BIGINT NOT NULL CHECK (last_transaction_id >= 0),
    updated_at           TIMESTAMPTZ NOT NULL
);
