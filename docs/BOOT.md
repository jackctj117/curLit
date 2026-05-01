# BOOT — single-command startup/shutdown for curLit (CL-2ho · CL-kgie)

For agents and operators that just need to bring the system up or down.
No prose; copy-paste only. For human operations & alert response, see
`docs/OPERATOR.md`. For long-form architecture, see `docs/ARCHITECTURE.md`.

## Prerequisites

- `.env` exists at repo root with at least:
  - `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`
  - `FRED_API_KEY`
  - `POSTGRES_PASSWORD`
  - `ANTHROPIC_API_KEY` (optional — research loop only)
- Docker daemon running.
- `.venv/` exists (`make install` if not).

## Startup — paper mode (default for fresh boots)

```bash
docker compose up -d --wait && \
  source .venv/bin/activate && \
  python -m migrations.run && \
  python -m src.runtime.run_engine --broker paper >/tmp/curlit-engine.log 2>&1 &
```

## Startup — OANDA practice (live broker, fake money)

```bash
docker compose up -d --wait && \
  source .venv/bin/activate && \
  python -m migrations.run && \
  python -m src.runtime.run_engine --broker oanda-practice >/tmp/curlit-engine.log 2>&1 &
```

## Startup — OANDA live (real money — requires explicit confirmation)

```bash
docker compose up -d --wait && \
  source .venv/bin/activate && \
  python -m migrations.run && \
  python -m src.runtime.run_engine --broker oanda-live --confirm-live >/tmp/curlit-engine.log 2>&1 &
```

## Verify healthy

```bash
docker compose ps                                     # all services Up
curl -s http://localhost:8200/api/system | jq .       # engine: running
curl -s http://localhost:8099/metrics | head -5       # Prometheus scrape
tail -n 30 /tmp/curlit-engine.log                     # no recent ERROR
```

Expected ports:
- `8200` — Web API (REST)
- `8099` — Prometheus metrics scrape
- `3000` — Grafana dashboards
- `8080` — Airflow scheduler UI (research loop)
- `9090` — Prometheus
- `5432` — Postgres
- `8200/api/system` returns `{"engine":"running","oms_halted":false}`

## Logs

- Engine: `/tmp/curlit-engine.log`
- Docker services: `docker compose logs --tail=100 <service>`
- Trade journal (canonical fill log): query `trade_journal_events` table

## Shutdown

```bash
pkill -f 'src.runtime.run_engine' && \
  docker compose down
```

## Hard reset (lose state — only for dev)

```bash
docker compose down -v   # drops volumes, including Postgres data
rm -rf data/             # local artifacts
```

## Troubleshooting matrix

| Symptom                                   | First place to look                             |
| ----------------------------------------- | ----------------------------------------------- |
| Engine refuses to start                   | `/tmp/curlit-engine.log` final 50 lines        |
| Postgres unreachable                      | `docker compose logs postgres`                  |
| OANDA auth errors                         | `.env` OANDA_API_KEY scope (practice vs live)   |
| Stale prices kill switch fired            | `docker compose ps` for stream-side health      |
| Reconciliation alert                      | `reports/reconciliation/YYYY-MM-DD.json`        |
| Strategy not trading                      | `bd show <strategy bead>` + paper_mode flag     |

For each-symptom-deeper-investigation, see `docs/OPERATOR.md`.
