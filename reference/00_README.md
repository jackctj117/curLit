# FX Trading System — Code Reference

This repository contains code excerpts organized by subsystem. Each file is a standalone reference for one concern.

## File Index

| File | Subsystem | Contents |
|------|-----------|----------|
| `01_data_layer.md` | Data | Database schema, ingestion base classes, FRED/Stooq/CME clients |
| `02_features_and_models.md` | Analytics | Feature store, rate differential model, OIS curve, reaction function |
| `03_nlp_pipeline.md` | NLP | CB scraper, preprocessing, lexicon scorer, transformer scorer, diff analysis |
| `04_backtest_framework.md` | Backtesting | Walk-forward runner, performance analytics, event-driven backtest |
| `05_strategies.md` | Strategies | All 6 strategies: rate diff MR, CB sentiment, carry+vol, momentum, value, COT |
| `06_portfolio.md` | Portfolio | Coordinator, risk parity, correlation monitor, attribution |
| `07_execution.md` | Execution | Broker abstraction, OANDA impl, IBKR impl, OMS, paper broker |
| `08_runtime.md` | Runtime | Live engine, signal generation, price streaming, graceful shutdown |
| `09_security.md` | Security | Vault, credential management, wolfCrypt integration, SSH hardening |
| `10_monitoring.md` | Observability | Prometheus metrics, logging setup, Grafana configs, alert rules |
| `11_risk_management.md` | Risk | Position sizing, kill switches, correlation regime, stress testing |
| `12_deployment.md` | Deployment | Systemd units, Docker compose, backup scripts, deployment workflow |
| `13_research_workflow.md` | Research | Paper ingestion, relevance scoring, evaluation rubric, labeling tools |
| `14_edge_testing.md` | Edge Testing | Null hypothesis framework, multiple testing correction, live tracker, decay detection, regime edge analysis |

## Directory Structure (Target)

```
fx-system/
├── src/
│   ├── data/              # Ingestion and data providers (file 01)
│   ├── features/          # Feature computation (file 02)
│   ├── models/            # Statistical models (file 02)
│   ├── rates/             # OIS curve, day count, calendar (file 02)
│   ├── nlp/               # Central bank NLP pipeline (file 03)
│   ├── backtest/          # Backtesting framework (file 04)
│   ├── strategies/        # Trading strategies (file 05)
│   ├── portfolio/         # Portfolio coordinator and risk (file 06)
│   ├── execution/         # Broker integration and OMS (file 07)
│   ├── runtime/           # Live engine (file 08)
│   ├── security/          # Vault and credentials (file 09)
│   ├── monitoring/        # Metrics and logging (file 10)
│   ├── risk/              # Risk management (file 11)
│   ├── research/          # Research workflow (file 13)
│   └── edge_testing/      # Edge verification (file 14)
├── scripts/               # One-off scripts referenced across files
├── tests/                 # Tests mentioned in various files
├── configs/               # YAML configs
├── docker/                # Observability stack (file 12)
└── systemd/               # Service units (file 12)
```

## Reading Order for AI Assistants

1. Start with `00_README.md` for structure
2. Read `01_data_layer.md` to understand data model
3. Read `07_execution.md` for the Broker abstraction
4. Read `05_strategies.md` for trading logic
5. Read remaining files as needed per task

## Notes

- Code is in Python 3.12
- Database is PostgreSQL with TimescaleDB extension
- All secrets managed via vault (file 09), never hardcoded
- All times internally UTC
- FX pair convention: 6-char uppercase (EURUSD, USDJPY)
