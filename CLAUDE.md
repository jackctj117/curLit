# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

Python >= 3.11, virtualenv at `.venv/` (`make install`, or `make dev-install` for dev/train/viz extras).

```bash
.venv/bin/pytest tests/unit -q       # fast unit tests — run these first
.venv/bin/pytest tests/ -q           # full suite (integration needs Postgres up)
.venv/bin/ruff check src/ tests/     # lint
.venv/bin/mypy src/                  # typecheck
```

Make targets (see `Makefile`): `test`, `test-unit`, `test-integration`, `test-slow` (hypothesis property tests), `lint`, `typecheck`, `format`, `check` (lint + typecheck + test), `clean`. Pytest markers are strict (`unit` / `integration` / `slow` — see `pytest.ini`); unit tests must not need network or a database.

## Architecture Overview

FX trading system: async live engine + strategy plugins + autonomous LLM research pipeline. Layers under `src/`:

- **data/** — ingesters (FRED, yfinance, CFTC, CME, ECB/BoJ/BoE, GDELT, Polymarket, alt-data). Airflow DAGs (`airflow/`) run daily ingest and write to the **host** Postgres (`curlit-postgres-soak`, 127.0.0.1:5432) — the same DB the engine reads; see the comment block in `docker-compose.yml`. Additions: `symbols.py` (13k US-listed universe from NASDAQ Trader + SEC EDGAR names/CIK — niche verification + `get_cik` for filings, migrations 010/011), `intraday_pricer.py` + `oanda_candles.py` (OANDA quotes every 120s into `intraday_quotes` mig 012 + daily candle vol backfill for unmapped instruments), `foreign_rates.py` (Bundesbank DE2Y / ECB €STR / SOFR → `DE2Y`, `US2Y_MINUS_DE2Y`, `USD_3M_OIS`, `EUR_3M_ESTR_OIS`, realized-vol `CVIX` proxy — CL-gr8o), `x_monitor.py` (44-account tiered watchlist → `x_ingest`), `reddit_monitor.py` (built; BLOCKED on Reddit API approval, CL-okww).
- **features/ models/ rates/ nlp/ signals/** — feature computation, rate-diff OLS models, OIS curves, FinBERT CB-sentiment pipeline.
- **backtest/** — walk-forward runner; `cost_model.py` (per-pair spreads + overnight funding), `swap_model.py` (daily swap, triple-Wednesday).
- **strategies/** — `rate_diff_mean_reversion`, `cb_sentiment_shift`, `carry_vol_filter`, `event_driven` (+ shared `vol_regime.py` helpers). Registered in `configs/live_portfolio.yaml`. Rate-diff entries are gated by carry/momentum/vol-regime filters (each toggleable in config).
- **risk/** — kill switches (`kill_switches.py`, 11 switches incl. `equity_trailing_stop` and `open_position_correlation`; built by `build_kill_switch_manager()` in `src/runtime/run_engine.py`, evaluated in the live-engine health tick every 60 s), sizing, `risk_profile.py` (profiles in `configs/risk_profile.yaml`, override via `CURLIT_RISK_PROFILE`).
- **execution/** — Broker ABC, OANDA / paper / Polymarket brokers, OMS, rejection policy. `alpaca_options.py` + `alpaca_options_executor.py` (CL-ldd2/3xoj): Alpaca PAPER options auto-execution of advisory buy_calls/buy_puts ideas — moneyness/DTE parse → nearest listed contract, premium + daily caps, market-hours guard, technical-alignment gate, idea-level dedup in `alpaca_option_orders` (mig 014). Daemon: `scripts/execute_options.py --loop 300`; master switch ALPACA_OPTIONS_ENABLED.
- **runtime/** — `run_engine` entrypoint (`--broker {paper,oanda-practice,oanda-live,polymarket-*}`) + `LiveEngine` asyncio loop.
- **events/** — current-events layer: `playbooks.py` (16 themes incl. Iran/Israel cluster) → `triage.py` (Haiku batch relevance pre-filter, fail-open, EVENT_TRIAGE_ENABLED — CL-cunh) → `impact_agent.py` (Sonnet assessment: affected instruments + advisory trade_ideas with stops/targets/triggers) → `confluence.py` (Gate A quality + Gate B INTRADAY market confirmation via `intraday_quotes`, + cross-asset read) → `EventDrivenStrategy` (cross-asset ENTRY GATE CL-6mzn, max 3 legs, phantom-position pruning CL-v9g4). Niche stack per urgency≥7 event: `niche_agent.py` (multi-hop; iterative cycles CL-2dnf) with discovery via `kimi_tool_agent.py` (agentic Kimi K3 tool-loop, Moonshot API, NICHE_TOOL_AGENT_ENABLED — CL-ddzt), grounded by `research_tools.py` (SEC 10-K excerpts via stored CIK, profiles), `technical_context.py` (computed trend/levels/breakout — NO pattern names, CL-3xoj) and `options_activity.py` (free yfinance chain snapshots → P/C skew + volume-vs-baseline, mig 015 — CL-mtum); verified against `SymbolUniverse`, asymmetry+liquidity gated, then attacked by `adversarial_critic.py` (red-team: refuted ideas die, survivors carry the bear case — CL-3v56). Learning loop: `outcome_tracker.py` (daily forward-return scoring into `idea_outcomes` mig 013 — CL-6axf) + `reflective_review.py` (aggregates the finalised record, proposes data-cited tuning, ADVISORY only — CL-g8jl). Rows live in `geo_events` (mig 005), lifecycle NEW → ASSESSED → CONFIRMED/EXPIRED/TRADED/DISMISSED. Producer: `scripts/event_pipeline.py --ingest --assess --loop 900`.
- **research/** — autonomous research pipeline (ingest → idea agent → GATE 1 → implementer → backtest → bull/bear debate → verdict → GATE 2 → paper-shadow registrar). All LLM roles use provider `claude-code` (`src/research/llm/claude_code.py` — headless `claude` CLI on the operator's subscription, model `claude-fable-5`); role map in `configs/research_agents.yaml`; grok/deepseek drivers are kept only for `scripts/compare_llm_providers.py`. Operator gates are approved by replying in Telegram (`scripts/telegram_approval_bot.py`) or via `scripts/research_approve.py`; state in `data/research/state.json`.
- **monitoring/ web/ portfolio/ security/** — Prometheus metrics + JSON logging, FastAPI REST (`:8200`; `/api/*` auth via `?secret=<WEB_API_SECRET>`), allocation/coordination, encrypted vault. `monitoring/data_health.py` (CL-q4n1): startup series-coverage preflight — a starved strategy input is a loud WARN banner at engine boot, plus `scripts/data_health.py` (exits non-zero on starvation). `portfolio/reconciler.py` consults multi-leg strategy `open_positions` books so engine restarts don't flatten event legs (CL-8s1e).

Key configs: `configs/live_portfolio.yaml` (strategy registry), `configs/risk_profile.yaml`, `configs/event_playbooks.yaml` (16 themes), `configs/research_agents.yaml`, `configs/x_watchlist.yaml` (44 accounts), `configs/reddit_watchlist.yaml`, `configs/retail_proxies.yaml`, `configs/polymarket_geo_markets.yaml`, `configs/paper_streams*.yaml`. DB is Postgres + TimescaleDB; schema via `python -m migrations.run` (15 migrations).

**Live daemons (9)** — engine (`-m src.runtime.run_engine --broker oanda-practice`, ALWAYS with `CURLIT_RISK_PROFILE=aggressive`), `event_pipeline --ingest --assess --loop 900`, `intraday_pricer --loop 120`, `refresh_rates --loop 86400`, `x_monitor --loop 300`, `execute_options --loop 300`, `options_activity --loop 86400`, `score_outcomes --loop 86400`, `telegram_approval_bot`. Fleet control: `./scripts/daemons.sh start|stop|status` (idempotent). Fresh-device bring-up: `docs/BOOTSTRAP.md`. Full operational state, policies, and restart rules: `docs/CURRENT_OPERATIONS.md` — update it when changing any of this.

## Conventions & Patterns

- **Bead IDs in docstrings** — every module/feature docstring cites its bead, e.g. `"""GDELT Doc 2.0 ingester (CL-6iu7) — ..."""`. Keep this when adding code.
- **Dataclass configs** — each component takes a frozen/plain `@dataclass` config (`RateDiffMRConfig`, `KillSwitchConfig`, `Playbook`, ...) populated from YAML; no dict-groping in business logic.
- **Module logger** — `logger = logging.getLogger(__name__)` at module top; structured/contextual logging, no prints.
- **Fail loud** — raise on missing or corrupt config/state instead of silently defaulting (e.g. `EquityTrailingStop` refuses to start on a corrupt state file; playbook validation rejects bad instrument kinds).
- **Atomic state files** — persist JSON state via tmp file + `os.replace` (kill-switch state, event book, `data/research/state.json`). Never write state in place.
- **Injectable transport shims** — network access goes through `Callable` params like `HttpGet = Callable[[str], str]` (`src/research/ingest.py`, `pdf_extractor.py`) so unit tests inject canned responses; no live HTTP in unit tests.
