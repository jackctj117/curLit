# BOOT — single-command startup/shutdown for curLit (CL-2ho · CL-kgie)

> **Fleet launcher now exists**: `./scripts/daemons.sh start|stop|status`
> covers all 9 native daemons. Fresh device: [`BOOTSTRAP.md`](BOOTSTRAP.md).
> This doc's per-process commands remain valid for individual restarts.

For agents and operators that just need to bring the system up or down.
No prose; copy-paste only. For human operations & alert response, see
`docs/OPERATOR.md`. For long-form architecture, see `docs/ARCHITECTURE.md`.

## Prerequisites

- `.env` exists at repo root with at least:
  - `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`
  - `FRED_API_KEY`
  - `POSTGRES_PASSWORD`
  - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (alerts + gate approvals)
- `claude` CLI logged in on the operator's subscription (research
  pipeline + event impact agent run headless via the `claude-code`
  driver; no `ANTHROPIC_API_KEY` needed for them).
- Docker daemon running (observability + Airflow live in docker compose;
  the engine and bots run natively on the host).
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

## Startup — OANDA practice, aggressive risk profile (current soak pattern)

Practice broker + `aggressive` profile from `configs/risk_profile.yaml`
(env var wins over the `active:` key). `nohup` keeps the engine alive
after the shell exits.

```bash
docker compose up -d --wait && \
  source .venv/bin/activate && \
  python -m migrations.run && \
  CURLIT_RISK_PROFILE=aggressive nohup python -m src.runtime.run_engine \
    --broker oanda-practice >/tmp/curlit-engine.log 2>&1 &
```

## Startup — companion daemons (alongside the engine)

```bash
# Telegram gate-approval bot — long-polls the chat, applies
# approve/reject/skip replies to data/research/state.json
nohup .venv/bin/python scripts/telegram_approval_bot.py \
  >/tmp/curlit-telegram-bot.log 2>&1 &

# Current-events pipeline — GDELT ingest + LLM impact assessment
# every 15 minutes
nohup .venv/bin/python scripts/event_pipeline.py --ingest --assess --loop 900 \
  >/tmp/curlit-events.log 2>&1 &
```

## Startup — dashboards (on demand)

```bash
# Research-triage dashboard (paper queue) — http://127.0.0.1:8501
.venv/bin/python -m streamlit run research/dashboard.py --server.port 8501

# Weekly scorecard (P&L, kill switches, LLM spend, data freshness)
.venv/bin/python -m streamlit run scripts/scorecard.py --server.headless true
```

## Startup — OANDA live (real money — requires explicit confirmation)

```bash
docker compose up -d --wait && \
  source .venv/bin/activate && \
  python -m migrations.run && \
  python -m src.runtime.run_engine --broker oanda-live --confirm-live >/tmp/curlit-engine.log 2>&1 &
```

## Startup — Polymarket paper (CL-poly-2)

```bash
docker compose up -d --wait && \
  source .venv/bin/activate && \
  python -m src.runtime.run_engine --broker polymarket-paper >/tmp/curlit-engine.log 2>&1 &
```

## Startup — Polymarket Amoy testnet (CL-poly-3)

Requires `pip install '.[polymarket]'` and the `POLYMARKET_AMOY_*` env
vars (or vault entries under `secret/trading/polymarket/amoy/`).

```bash
pip install '.[polymarket]'    # installs py-clob-client + web3 + eth-account
python -m src.runtime.run_engine --broker polymarket-amoy >/tmp/curlit-engine.log 2>&1 &
```

## Startup — Polymarket mainnet (HARD-GATED)

Real-money on Polymarket. Run only after CL-poly-3 acceptance gates
pass — see `docs/runbooks/PolymarketTrading.md` and `bd show CL-poly-3`.

```bash
export POLYMARKET_MAINNET_UNLOCK=1
python -m src.runtime.run_engine \
    --broker polymarket-mainnet --confirm-live >/tmp/curlit-engine.log 2>&1 &
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
- `8080` — Airflow UI (daily ingestion DAGs — writes to the HOST
  postgres on 127.0.0.1:5432, see `docker-compose.yml` comments)
- `9090` — Prometheus
- `5432` — Postgres
- `8501` — research-triage dashboard (Streamlit, if launched)
- `8200/api/system` returns `{"engine":"running","oms_halted":false}`

## Logs

- Engine: `/tmp/curlit-engine.log`
- Telegram approval bot: `/tmp/curlit-telegram-bot.log`
- Event pipeline: `/tmp/curlit-events.log`
- Docker services: `docker compose logs --tail=100 <service>`
- Trade journal (canonical fill log): query `trade_journal_events` table

## Shutdown

```bash
pkill -f 'src.runtime.run_engine'; \
  pkill -f 'scripts/telegram_approval_bot.py'; \
  pkill -f 'scripts/event_pipeline.py'; \
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
| Polymarket mainnet refused to start       | `bd show CL-poly-3` — mainnet is hard-gated     |
| Polymarket Amoy preflight fails           | check vault entries / RPC URL / chain id        |

For each-symptom-deeper-investigation, see `docs/OPERATOR.md`.
