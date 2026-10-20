# curLit

Algorithmic FX trading system: pulls live ticker values for currencies, cryptocurrencies, and rare metals, then uses AI (NLP + quantitative models) to analyze treasury bond ETFs/trusts against news and world events to recommend short/put positions.

## Status

**Paper-trading system with execution and research hardening in progress.**
The repository contains OANDA practice and Alpaca paper execution paths.
Entry restrictions, release identities and verification dates are recorded in
[`docs/CURRENT_OPERATIONS.md`](docs/CURRENT_OPERATIONS.md); those records are not
a fresh broker or process-health check. The platform-dependent fleet roster is
defined in [`scripts/daemons.sh`](scripts/daemons.sh), including macOS-only
`keep_awake`. Roster membership does not prove a daemon is running.

Signal-direction outcomes are not executed option returns or proof of an edge.
Reflective research proposes changes for operator approval, never automatic
production tuning. Rollout and recovery remain tracked under CL-koeg.
Fresh-device setup: [`docs/BOOTSTRAP.md`](docs/BOOTSTRAP.md). Fleet operations
require separate operator approval; never start a competing execution writer.

## Development workflow

Claude implements one assigned Beads issue, checks verify the patch, Codex reviews
it read-only, and the operator approves publishing and merging. Fable is only a
task-specific escalation. See [the supervised setup guide](docs/AI_DEVELOPMENT_WORKFLOW.md).
Use a credential-free development environment; this setup does not change runtime
research models or authorize access to the paper-trading installation.

## Features

- **Autonomous research pipeline**: multi-source ingest (arXiv preprints + quant substacks + Polymarket prediction markets) → idea agent → operator GATE 1 → implementer → walk-forward backtest → Bull/Bear debate → verdict engine → operator GATE 2 → paper-shadow registration (allocation=0). All LLM roles run on headless Claude Code (`claude-fable-5`) via the operator's subscription — the Grok/DeepSeek drivers are kept only for the provider-comparison harness. Daily cron, full transcripts, Telegram alerts. Transient LLM failures (timeouts/rate limits) retry with an attempt cap instead of wedging. See [`docs/research/RUNBOOK.md`](docs/research/RUNBOOK.md).
- **Telegram gate approvals**: research gates are approved by replying `approve <id>` / `reject <id>` in the alert chat (chat-id locked bot, `scripts/telegram_approval_bot.py`); notifications carry a `Trades:` line listing the instruments each candidate would touch
- **Current-events pipeline**: GDELT (16 themed playbooks incl. Iran/Israel, Hormuz, Russia-Ukraine, Africa minerals, Taiwan semis, pharma APIs) + a 44-account X watchlist feed `geo_events` → **Haiku batch triage** (cheap relevance pre-filter, fail-open) → Sonnet impact agent (playbook-grounded assessment with concrete stops/targets/triggers) → **intraday market-confirmation gate** (real OANDA quotes, 30–120 min window) + **cross-asset entry gate** (contradictory related-asset moves veto machine legs) → `EventDrivenStrategy` paper-trades the FX legs with tight risk caps, concentration/haven caps, and a persistent event-book loss freeze
- **Niche opportunity stack**: the CL-zhkg development patch selects Claude discovery/refinement with captured evidence and an evidence-aware Claude critic; activation is separately tracked in CL-cdov. Kimi remains available for explicit research/shadow comparisons, not an automatic fallback. Ticker identity, dated sources, factual versus inferred claims, review and liquidity status stay explicit; incomplete leads remain research-only. See [research evidence and shadow comparisons](docs/NICHE_RESEARCH_EVIDENCE.md). Source changes are not deployment or profitability evidence.
- **Idea grounding**: every candidate is grounded in computed facts — SEC filing excerpts, **computed technical context** (trend vs 20/50d SMA, swing support/resistance in dollars, breakout/approaching states, level-test counts — deliberately no pattern-name vocabulary), and **free options-activity snapshots** (put/call volume skew, volume vs self-built baseline, ATM IV)
- **Paper execution infrastructure**: OANDA practice plus Alpaca options/equities. Entry restrictions take precedence over historical entry policies. Source knobs and dated rollout evidence are documented in [`docs/CURRENT_OPERATIONS.md`](docs/CURRENT_OPERATIONS.md); README defaults do not establish effective settings.
- **Closed learning loop**: a daily outcome tracker scores every surfaced idea's signed forward return (MFE/MAE, win/loss/flat at horizon, denormalised by theme/action/hop/confidence/niche/red-team) and a **reflective reviewer** aggregates the finalised record and proposes conservative, data-cited config tuning for operator approval — never auto-applied
- **Multi-source data ingestion**: FRED (US macro, yields), **Bundesbank/ECB daily foreign rates** (German 2Y, €STR — feeding US2Y_MINUS_DE2Y, OIS carry legs, a realized-vol CVIX proxy), ECB/BoJ/BoE, Yahoo Finance (FX spot, commodities, indices, option chains), **OANDA intraday quotes + daily candles** (19 event instruments), CFTC COT (positioning), CME SOFR futures, GDELT (news events), Polymarket (prediction-market probabilities); a **startup data-health preflight** makes any starved series a loud banner instead of a silently dormant strategy
- **Rate differential models**: Rolling OLS regression on yield spreads for FX pair fair-value estimation, with carry/momentum/vol-regime entry filters
- **NLP pipeline**: Fine-tuned FinBERT on central bank statements — hawkish/dovish sentiment scoring and diff analysis (Fed, ECB, BoE, BoJ, BoC)
- **OIS curve construction**: Bootstrapped discount factors and forward rates from futures prices
- **Walk-forward backtesting**: Non-overlapping IS/OOS windows with bootstrap confidence intervals, per-pair spreads, and overnight funding costs
- **Risk management**: configured sizing, concentration limits and kill switches; verify the armed startup set and effective profile, not a static README count. Cross-path controls and remaining gaps are tracked in CL-0deu.2.
- **Live trading engine**: Async event loop with price streaming, signal generation, OMS, reconciliation
- **Research tooling**: Optuna walk-forward hyperopt (`scripts/hyperopt_rate_model.py`), weekly Streamlit scorecard (`scripts/scorecard.py`), paper-triage dashboard (`research/dashboard.py`)
- **Observability**: Prometheus + Grafana + Loki + Alertmanager (Telegram alerts)
- **Security**: wolfcrypt AES-256-GCM encrypted vault with paper backup recovery (BIP39 seed)
- **Web UI**: FastAPI backend + browser dashboard for positions, signals, P&L, config, manual trades

## Architecture

Full technical design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

```
fx-system/
├── src/
│   ├── data/          Data providers (FRED, yfinance, CME, CFTC, ECB)
│   ├── features/      Feature engineering, signal computation
│   ├── models/        Rate diff model, reaction functions, correlation
│   ├── rates/         OIS curve bootstrapping, day-count conventions
│   ├── nlp/           CB scrapers, lexicon scorer, diff analyzer, inference
│   ├── backtest/      Walk-forward runner, analytics, bootstrap CIs
│   ├── risk/          Sizing, kill switches, regime monitor, stress tests
│   ├── execution/     Broker interface, PaperBroker, OANDA, Polymarket, OMS
│   ├── strategies/    Rate diff MR, CB sentiment shift, carry vol filter, event-driven
│   ├── events/        Current-events layer: playbooks, LLM impact agent, confluence
│   ├── research/      Autonomous research pipeline, LLM drivers, Telegram approvals
│   ├── runtime/       Live engine event loop, entrypoint
│   ├── monitoring/    Prometheus metrics, structured JSON logging
│   ├── security/      Vault agent, vault client (wolfcrypt)
│   └── web/           FastAPI REST backend
├── airflow/dags/      Airflow DAGs for ingestion, NLP, retraining
├── tests/             Unit + integration + property-based tests
├── labeling/          Streamlit app for CB sentence classification
├── training/          FinBERT fine-tuning pipeline
├── configs/           Strategy configs, systemd units, sshd config
└── docs/              Architecture, security, runbooks
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 8GB+ RAM

### 1. Clone and configure

This operational quick start requires a separately authorized installation.
Coding agents use a credential-free development environment instead; never copy
the running installation's environment, account credentials or databases.

```bash
git clone https://github.com/jackctj117/curLit.git
cd curLit
cp .env.example .env
# Edit .env with your API keys (at minimum: FRED_API_KEY)
```

### 2. Start infrastructure

```bash
docker compose up -d --wait
```

This starts PostgreSQL (TimescaleDB), Prometheus, Grafana, Loki, Alertmanager, Airflow, node_exporter. All services bind to `localhost`.

| Port | Service |
|------|---------|
| 5432 | PostgreSQL |
| 3000 | Grafana (admin/admin) |
| 9090 | Prometheus |
| 8080 | Airflow (admin/admin) |
| 3100 | Loki (log querying) |
| 9093 | Alertmanager |

### 3. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
```

### 4. Initialize database

```bash
python -m migrations.run
```

### 5. Seed data

```bash
# Via CLI (quick start with key series):
python -c "
from src.data.fred import FREDIngester
from src.data.yfinance_provider import YFinanceIngester
from datetime import datetime
db = 'postgresql+psycopg2://fx:changeme@localhost:5432/fx'
FREDIngester(db).run(datetime(2024,1,1), datetime(2026,4,24))
YFinanceIngester(db).run(datetime(2024,1,1), datetime(2026,4,24))
"
```

Or trigger Airflow DAGs manually at `http://localhost:8080`.

### 6. Start the engine (paper trading)

```bash
python -m src.runtime.run_engine --broker paper
```

The engine runs the live event loop — strategies evaluate signals, manage positions, and record trades. No real money is used.

### 7. View dashboards

- **Web API**: `http://localhost:8200/api/account` requires the `X-API-Key`
  header matching a strong, operator-configured `WEB_API_SECRET`. Missing and
  default secrets are rejected; query-string authentication is unsupported.
- **Grafana Glance**: `http://localhost:3000` → "curLit — Glance"
- **Airflow**: `http://localhost:8080`
- **Soak dashboard**: `http://127.0.0.1:8201/` (see below — must be launched separately)

For an authorized local read, with the secret already supplied securely:

```bash
curl --fail --header "X-API-Key: ${WEB_API_SECRET:?set a strong API secret}" \
  http://127.0.0.1:8200/api/account
```

Startup configuration evidence is recorded by
[`src/runtime/release_manifest.py`](src/runtime/release_manifest.py) in ignored
`data/runtime_manifests/`. Missing fields stay unknown. The checkout's current
Git hash does not identify code already loaded by a running process. Complete
per-path effective configuration and baseline verification remain CL-0deu.12
and CL-0deu.14; reading this README does not satisfy those gates.

### Soak-test dashboard

When the engine is running in paper mode for stability validation (24h soak runs), `scripts/soak_dashboard.py` provides a self-refreshing HTML view of engine health. Runs independently of the engine — start, stop, or restart it any time without disturbing the soak.

```bash
.venv/bin/python scripts/soak_monitor.py    # writes logs/soak_test.jsonl every 10min
.venv/bin/python scripts/soak_dashboard.py  # serves http://127.0.0.1:8201/ (default port)
```

The page auto-refreshes every 10s. Health panels:

| Panel | What it tells you |
|---|---|
| **verdict** | GREEN / YELLOW / RED rollup with reason |
| **engine** | ALIVE/DEAD + python PID + uptime + OMS halt state |
| **account · paper** | Equity, margin used (from broker via engine API) |
| **memory · cpu · fds** | RSS, CPU%, threads, file descriptors + sparkline of last 100 samples |
| **monitor** | Sample count + age of latest sample |
| **db rows** | `trade_journal_events` + `feature_snapshots` counts + latest event type |
| **recent errors** | Last 20 ERROR/CRITICAL/Traceback lines from the engine log |

Trade-activity panels (full-width tables):

| Panel | What it tells you |
|---|---|
| **positions** | Per-symbol open positions: qty, avg price, unrealized P&L (color-coded long/short) |
| **strategies** | Configured strategies and the symbols each trades |
| **recent trade-journal events** | Last 15 events with seq, ts, event type, strategy, symbol, and a per-event-type detail (delta for INTENT_SUBMITTED, side+qty for ORDER_PLACED/FILLED, rejection class for ORDER_REJECTED, summary for RECONCILIATION_REPORT) |

The trade-activity panels proxy through the engine's web API at `:8200` (so they require the engine to be up — they degrade gracefully to "engine api unreachable" if it isn't). The dashboard uses the same `WEB_API_SECRET` env var the engine does, so any custom secret in `.env` works automatically.

Verdict thresholds: **RED** if engine is dead, monitor stale >25min, or memory has doubled · **YELLOW** if monitor stale 15–25min or memory grew >50% · **GREEN** otherwise.

`GET /api/soak` returns the same data as JSON for external monitoring or scripting. Override the port via `SOAK_DASHBOARD_PORT=8500`. Bound to `127.0.0.1` only — read-only and local-network.

## API Keys Required

| Key | Source | Required? |
|-----|--------|-----------|
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org) | ✅ Required — US macro data |
| `OANDA_API_KEY` | [oanda.com](https://developer.oanda.com) | For live trading (paper mode works without) |
| `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` | [@BotFather](https://t.me/BotFather) | For alerts + interactive research-gate approvals |

General research roles and the impact agent use the local authenticated Claude
CLI. CLI authentication alone does not establish zero cost. The pending niche
switch requires subscription verification and an operator attestation that paid
overflow is disabled; quota/auth failures do not fall back to a paid provider.
The standalone Kimi comparison path still requires a funded Moonshot account.

## Testing

```bash
make test          # All tests
make test-unit     # Unit tests only
make test-slow     # Property-based tests (hypothesis)
make lint          # Ruff linter
make typecheck     # Mypy type checker
```

## Issue Tracking

This project uses [`bd` (beads)](https://github.com/steveyegge/beads) for all issue
tracking. Run `bd --sandbox ready --json` to see available work. Close only
completed acceptance criteria; unfinished implementation, external approvals,
unverified rollouts and forward studies must remain visible.

## License

Proprietary. All rights reserved.
