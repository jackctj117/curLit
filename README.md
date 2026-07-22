# curLit

Algorithmic FX trading system: pulls live ticker values for currencies, cryptocurrencies, and rare metals, then uses AI (NLP + quantitative models) to analyze treasury bond ETFs/trusts against news and world events to recommend short/put positions.

## Status

**Live paper trading on two venues** — the full event→idea→execution→scorecard
loop runs unattended across 9 daemons: OANDA practice machine-trades FX legs of
confirmed geopolitical events; Alpaca paper auto-buys the top options ideas
daily; every surfaced idea's forward return is scored into a track record that
a weekly reflective review will tune the system from. 140+ tracked issues
closed. **Operational ground truth: [`docs/CURRENT_OPERATIONS.md`](docs/CURRENT_OPERATIONS.md).**

## Features

- **Autonomous research pipeline**: multi-source ingest (arXiv preprints + quant substacks + Polymarket prediction markets) → idea agent → operator GATE 1 → implementer → walk-forward backtest → Bull/Bear debate → verdict engine → operator GATE 2 → paper-shadow registration (allocation=0). All LLM roles run on headless Claude Code (`claude-fable-5`) via the operator's subscription — the Grok/DeepSeek drivers are kept only for the provider-comparison harness. Daily cron, full transcripts, Telegram alerts. Transient LLM failures (timeouts/rate limits) retry with an attempt cap instead of wedging. See [`docs/research/RUNBOOK.md`](docs/research/RUNBOOK.md).
- **Telegram gate approvals**: research gates are approved by replying `approve <id>` / `reject <id>` in the alert chat (chat-id locked bot, `scripts/telegram_approval_bot.py`); notifications carry a `Trades:` line listing the instruments each candidate would touch
- **Current-events pipeline**: GDELT (16 themed playbooks incl. Iran/Israel, Hormuz, Russia-Ukraine, Africa minerals, Taiwan semis, pharma APIs) + a 44-account X watchlist feed `geo_events` → **Haiku batch triage** (cheap relevance pre-filter, fail-open) → Sonnet impact agent (playbook-grounded assessment with concrete stops/targets/triggers) → **intraday market-confirmation gate** (real OANDA quotes, 30–120 min window) + **cross-asset entry gate** (contradictory related-asset moves veto machine legs) → `EventDrivenStrategy` paper-trades the FX legs with tight risk caps, concentration/haven caps, and a persistent event-book loss freeze
- **Niche opportunity stack**: high-urgency events get a multi-hop second/third-order pass — an **agentic Kimi K3 tool-loop** that calls ticker-verification / company-resolution / profile / **SEC 10-K** tools mid-reasoning — verified against a 13k-symbol US-listed universe (NASDAQ Trader + SEC EDGAR names), scored for asymmetry under a dollar-volume liquidity floor, then attacked by an **adversarial red-team critic** (independent model) that kills refuted ideas and attaches the surviving bear case to the rest
- **Idea grounding**: every candidate is grounded in computed facts — SEC filing excerpts, **computed technical context** (trend vs 20/50d SMA, swing support/resistance in dollars, breakout/approaching states, level-test counts — deliberately no pattern-name vocabulary), and **free options-activity snapshots** (put/call volume skew, volume vs self-built baseline, ATM IV)
- **Two-venue paper execution**: OANDA practice (FX event legs, max 3 concurrent, restart-safe reconciliation) + **Alpaca paper options executor** (top-5 ideas/day by confidence, 1 contract each, $500 premium cap, market-hours aware, technical-alignment gate, idea-level dedup)
- **Closed learning loop**: a daily outcome tracker scores every surfaced idea's signed forward return (MFE/MAE, win/loss/flat at horizon, denormalised by theme/action/hop/confidence/niche/red-team) and a **reflective reviewer** aggregates the finalised record and proposes conservative, data-cited config tuning for operator approval — never auto-applied
- **Multi-source data ingestion**: FRED (US macro, yields), **Bundesbank/ECB daily foreign rates** (German 2Y, €STR — feeding US2Y_MINUS_DE2Y, OIS carry legs, a realized-vol CVIX proxy), ECB/BoJ/BoE, Yahoo Finance (FX spot, commodities, indices, option chains), **OANDA intraday quotes + daily candles** (19 event instruments), CFTC COT (positioning), CME SOFR futures, GDELT (news events), Polymarket (prediction-market probabilities); a **startup data-health preflight** makes any starved series a loud banner instead of a silently dormant strategy
- **Rate differential models**: Rolling OLS regression on yield spreads for FX pair fair-value estimation, with carry/momentum/vol-regime entry filters
- **NLP pipeline**: Fine-tuned FinBERT on central bank statements — hawkish/dovish sentiment scoring and diff analysis (Fed, ECB, BoE, BoJ, BoC)
- **OIS curve construction**: Bootstrapped discount factors and forward rates from futures prices
- **Walk-forward backtesting**: Non-overlapping IS/OOS windows with bootstrap confidence intervals, per-pair spreads, and overnight funding costs
- **Risk management**: Volatility-targeted sizing, regime-aware position adjustment, 11 automatic kill switches (incl. equity trailing stop + open-position correlation) wired into the live engine's health tick; risk profiles switchable via `CURLIT_RISK_PROFILE`
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

- **Web UI**: `http://localhost:8200/api/account?secret=curlit-dev` (JSON REST — positions, signals, P&L, manual trades)
- **Grafana Glance**: `http://localhost:3000` → "curLit — Glance"
- **Airflow**: `http://localhost:8080`
- **Soak dashboard**: `http://127.0.0.1:8201/` (see below — must be launched separately)

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

The research pipeline and event impact agent need no API key — they run
on the local `claude` CLI logged into the operator's subscription.

## Testing

```bash
make test          # All tests
make test-unit     # Unit tests only
make test-slow     # Property-based tests (hypothesis)
make lint          # Ruff linter
make typecheck     # Mypy type checker
```

## Issue Tracking

This project uses [`bd` (beads)](https://github.com/steveyegge/beads) for issue tracking with Dolt-backed version control. Run `bd ready` to see available work.

## License

Proprietary. All rights reserved.
