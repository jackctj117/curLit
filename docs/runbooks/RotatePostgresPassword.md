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
   # NOT `docker compose restart` — that REUSES the existing container config
   # and will NOT pick up the changed .env, so the container silently keeps
   # the OLD password. Force a recreate:
   docker compose up -d --force-recreate --no-deps \
     airflow-scheduler airflow-webserver postgres_exporter
   ```
   `postgres_exporter` is easy to miss — it embeds the password in
   `DATA_SOURCE_NAME` (docker-compose.yml), so it needs the same recreate.

5. **Verify:**
   - No `WELL-KNOWN default password` warning in fresh logs. Filter by TIME,
     not date — earlier restarts the same day will otherwise show up:
     ```bash
     grep -h WELL-KNOWN logs/*.log | awk '{print $1,$2}' | sort | tail -3
     ```
     The newest one must PREDATE the restart.
   - Engine reconciliation clean: `grep "Reconciliation complete" logs/engine.log | tail -1`
     shows `mismatches=False`, and `/api/system` shows `oms_halted:false`.
   - No auth errors after the restart:
     ```bash
     grep -rh "password authentication failed" logs/*.log | wc -l
     ```
     A couple of hits timestamped BETWEEN the `ALTER USER` and the daemon
     restart are EXPECTED — old processes still hold the old credential in
     memory. Anything after the restart is a real failure.
   - Ingest works: a fresh `intraday_pricer` line logs `wrote N/N quotes`.
   - Airflow can reach the DB: containers report `(healthy)` and
     `docker logs curlit-airflow-scheduler --since 5m` shows no
     `password authentication failed` / `OperationalError`.

   > **Verifying the old password is dead: do it from the HOST, not inside
   > the container.** `pg_hba.conf` in this image has
   > `host all all 127.0.0.1/32 trust`, so a `docker exec ... psql -h 127.0.0.1`
   > skips password auth entirely and the OLD password will appear to still
   > work — a false negative. Host connections arrive via the docker gateway
   > and match `host all all all scram-sha-256`, which is the path the daemons
   > actually use. Check it the way they connect:
   > ```bash
   > .venv/bin/python -c "
   > import psycopg2
   > try:
   >     psycopg2.connect(host='127.0.0.1', port=5432, user='fx',
   >                      dbname='fx', password='changeme', connect_timeout=5)
   >     print('OLD PASSWORD STILL WORKS — rotation did not take')
   > except Exception:
   >     print('old password refused — good')"
   > ```

## Rollback
If a consumer can't authenticate after the change, set the role password
back to the OLD value with the same `ALTER USER` and revert `.env`, then
restart. Because the change is a single credential, rollback is symmetric —
there is no schema or data migration to undo.

## Notes
- The code-side half of CL-1esk (loud startup WARNING) is already shipped in
  `src/data/db_env.py`; this runbook is the operator-side other half.
- Do NOT commit the new password. `.env` is git-ignored; keep it that way.
