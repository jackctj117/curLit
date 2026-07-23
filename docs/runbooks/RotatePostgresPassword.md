# RotatePostgresPassword

## Severity: HIGH (security debt — CL-1esk / E10)

## What it means
The Postgres role `fx` is still on the well-known default password
`changeme`. `src/data/db_env.py` logs a once-per-process WARNING at every
daemon start while this is true (`build_db_url()` centralizes the check).
Rotation is **operator-gated**: it must happen in one maintenance window
because ~23 call sites and the dockerized Airflow read the same credential,
and every live connection breaks the instant the role password changes until
each consumer picks up the new value.

## Preconditions
- A maintenance window (no critical entries pending). Everything is PAPER,
  so a brief halt is safe — the reconciler restores open positions on
  restart.
- Access to run `psql` as a Postgres superuser (or the `fx` role itself).
- The DB is the host Postgres `curlit-postgres-soak` on `127.0.0.1:5432`
  (see `docker-compose.yml`); the engine and Airflow share it.

## Runbook

1. **Generate a strong password** (store it in your password manager first):
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Rotate the role password** in Postgres:
   ```bash
   psql -h 127.0.0.1 -U fx -d fx -c "ALTER USER fx WITH PASSWORD '<new-pw>';"
   ```
   (Quote/escape the value; `token_urlsafe` output is shell-safe.)

3. **Update every place that reads it — in the same window:**
   - `.env` → `POSTGRES_PASSWORD=<new-pw>` (line ~10). This one value feeds
     all native daemons via `build_db_url()`.
   - `docker-compose.yml` Airflow services inherit `${POSTGRES_PASSWORD}`
     from the same `.env`; the `AIRFLOW_DB_PASSWORD` (Airflow's OWN metadata
     DB) is separate and does NOT need to change here.
   - If any deploy uses a sealed vault, re-seal it with the new value
     (`scripts/rotate_secrets.py`); the live broker path reads env, so the
     `.env` update is the one that matters on this box.

4. **Restart every consumer:**
   ```bash
   ./scripts/daemons.sh stop && ./scripts/daemons.sh start   # all 13 native daemons
   docker compose restart airflow-scheduler airflow-webserver # + any airflow-worker
   ```

5. **Verify:**
   - No `WELL-KNOWN default password` warning in fresh logs
     (`grep -c WELL-KNOWN logs/*.log` on rows after the restart).
   - Engine reconciliation clean: `grep "Reconciliation complete" logs/engine.log | tail -1`
     shows `mismatches=False`, and `/api/system` shows `oms_halted:false`.
   - Ingest works: a fresh `event_pipeline` cycle logs `ingest: N new`.
   - Airflow can reach the DB (a DAG task run succeeds, or the webserver
     shows no connection errors).

## Rollback
If a consumer can't authenticate after the change, set the role password
back to the OLD value with the same `ALTER USER` and revert `.env`, then
restart. Because the change is a single credential, rollback is symmetric —
there is no schema or data migration to undo.

## Notes
- The code-side half of CL-1esk (loud startup WARNING) is already shipped in
  `src/data/db_env.py`; this runbook is the operator-side other half.
- Do NOT commit the new password. `.env` is git-ignored; keep it that way.
