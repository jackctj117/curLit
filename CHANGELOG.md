# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Polymarket integration (CL-poly-1 / 2 / 3):
  - Phase 1 (read-only): `PolymarketDataSource` + `POLY:condition_id:outcome` symbol resolver. POLY symbols flow through `DataProvider.get_aligned_series` as features.
  - Phase 2 (paper): `PolymarketPaperBroker`, fractional-Kelly sizer, cost model. New broker mode `--broker polymarket-paper`.
  - Phase 3 (live, GATED): vault loader + chain client + EIP-712 broker shell + UMA oracle monitor + on-chain reconciler + preflight. New broker modes `--broker polymarket-amoy` (testnet) and `--broker polymarket-mainnet`. Mainnet hard-gated behind `POLYMARKET_MAINNET_UNLOCK=1` until CL-poly-3 acceptance gates pass.
  - New runbook at `docs/runbooks/PolymarketTrading.md`.
  - `pip install '.[polymarket]'` extras pulls `py-clob-client`, `web3`, `eth-account`.
- Initial project skeleton and architecture documentation
- Docker Compose observability stack (Prometheus, Grafana, Loki, Alertmanager)
- Airflow scheduling infrastructure
- Data ingestion pipeline (FRED, Stooq, CME, CFTC, ECB, BoJ, BoE)
- Database schema (PostgreSQL + TimescaleDB)
- Rate differential and reaction function models
- OIS curve bootstrapping and day-count conventions
- NLP pipeline (CB scrapers, lexicon scorer, statement diff analyzer)
- Backtesting infrastructure (walk-forward, bootstrap CI, event backtester)
- Risk management (position sizing, kill switches, regime-aware adjuster)
- Execution layer (broker abstraction, paper broker, OANDA integration, OMS)
- Live engine and watchdog services
- Structured logging with Prometheus metrics
- Web UI (FastAPI backend + React frontend)
- Security vault with wolfcrypt encryption and paper backup recovery
- Dependency management with lockfiles and automated update checking
