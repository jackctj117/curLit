# curLit

Algorithmic FX trading system: pulls live ticker values for currencies, cryptocurrencies, and rare metals, then uses AI (NLP + quantitative models) to analyze treasury bond ETFs/trusts against news and world events to recommend short/put positions.

## Status

**Active development** — architecture complete, all core modules built, data pipeline ingesting real market data, engine running in paper-trading mode. 115 tracked issues closed.

## Features

- **Multi-source data ingestion**: FRED (US macro, yields), ECB/BoJ/BoE, Yahoo Finance (FX spot, commodities, indices), CFTC COT (positioning), CME SOFR futures
- **Rate differential models**: Rolling OLS regression on yield spreads for FX pair fair-value estimation
- **NLP pipeline**: Fine-tuned FinBERT on central bank statements — hawkish/dovish sentiment scoring and diff analysis (Fed, ECB, BoE, BoJ, BoC)
- **OIS curve construction**: Bootstrapped discount factors and forward rates from futures prices
- **Walk-forward backtesting**: Non-overlapping IS/OOS windows with bootstrap confidence intervals
- **Risk management**: Volatility-targeted sizing, regime-aware position adjustment, 6 automatic kill switches
- **Live trading engine**: Async event loop with price streaming, signal generation, OMS, reconciliation
- **Observability**: Prometheus + Grafana + Loki + Alertmanager (Pushover + Telegram alerts)
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
│   ├── execution/     Broker interface, PaperBroker, OANDA, OMS
│   ├── strategies/    Rate diff MR, CB sentiment shift
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
python -m src.runtime.run_engine --practice
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

The page auto-refreshes every 10s and shows:

| Panel | What it tells you |
|---|---|
| **verdict** | GREEN / YELLOW / RED rollup with reason |
| **engine** | ALIVE/DEAD + python PID + uptime |
| **memory · cpu · fds** | RSS, CPU%, threads, file descriptors + sparkline of last 100 samples |
| **monitor** | Sample count + age of latest sample |
| **latest sample** | Most recent `soak_test.jsonl` row |
| **db rows** | `trade_journal_events` + `feature_snapshots` counts + latest event type |
| **recent errors** | Last 20 ERROR/CRITICAL/Traceback lines from the engine log |

Verdict thresholds: **RED** if engine is dead, monitor stale >25min, or memory has doubled · **YELLOW** if monitor stale 15–25min or memory grew >50% · **GREEN** otherwise.

`GET /api/soak` returns the same data as JSON for external monitoring or scripting. Override the port via `SOAK_DASHBOARD_PORT=8500`. Bound to `127.0.0.1` only — read-only and local-network.

## API Keys Required

| Key | Source | Required? |
|-----|--------|-----------|
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org) | ✅ Required — US macro data |
| `OANDA_API_KEY` | [oanda.com](https://developer.oanda.com) | For live trading (paper mode works without) |
| `PUSHOVER_USER_KEY` | [pushover.net](https://pushover.net) | For critical alert push notifications |
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) | For secondary alert channel |

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
