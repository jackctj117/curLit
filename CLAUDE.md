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

- **data/** — ingesters (FRED, yfinance, CFTC, CME, ECB/BoJ/BoE, GDELT, Polymarket, alt-data). Airflow DAGs (`airflow/`) run daily ingest and write to the **host** Postgres (`curlit-postgres-soak`, 127.0.0.1:5432) — the same DB the engine reads; see the comment block in `docker-compose.yml`.
- **features/ models/ rates/ nlp/ signals/** — feature computation, rate-diff OLS models, OIS curves, FinBERT CB-sentiment pipeline.
- **backtest/** — walk-forward runner; `cost_model.py` (per-pair spreads + overnight funding), `swap_model.py` (daily swap, triple-Wednesday).
- **strategies/** — `rate_diff_mean_reversion`, `cb_sentiment_shift`, `carry_vol_filter`, `event_driven` (+ shared `vol_regime.py` helpers). Registered in `configs/live_portfolio.yaml`. Rate-diff entries are gated by carry/momentum/vol-regime filters (each toggleable in config).
- **risk/** — kill switches (`kill_switches.py`, 11 switches incl. `equity_trailing_stop` and `open_position_correlation`; built by `build_kill_switch_manager()` in `src/runtime/run_engine.py`, evaluated in the live-engine health tick every 60 s), sizing, `risk_profile.py` (profiles in `configs/risk_profile.yaml`, override via `CURLIT_RISK_PROFILE`).
- **execution/** — Broker ABC, OANDA / paper / Polymarket brokers, OMS, rejection policy.
- **runtime/** — `run_engine` entrypoint (`--broker {paper,oanda-practice,oanda-live,polymarket-*}`) + `LiveEngine` asyncio loop.
- **events/** — current-events layer: `playbooks.py` (loads `configs/event_playbooks.yaml`, 6 themes) → `impact_agent.py` (LLM assessment) → `confluence.py` (quality + market-confirmation gates) → `EventDrivenStrategy`. Rows live in the `geo_events` table (migration 005), lifecycle NEW → ASSESSED → CONFIRMED/EXPIRED/TRADED/DISMISSED. Producer: `scripts/event_pipeline.py --ingest --assess --loop 900`.
- **research/** — autonomous research pipeline (ingest → idea agent → GATE 1 → implementer → backtest → bull/bear debate → verdict → GATE 2 → paper-shadow registrar). All LLM roles use provider `claude-code` (`src/research/llm/claude_code.py` — headless `claude` CLI on the operator's subscription, model `claude-fable-5`); role map in `configs/research_agents.yaml`; grok/deepseek drivers are kept only for `scripts/compare_llm_providers.py`. Operator gates are approved by replying in Telegram (`scripts/telegram_approval_bot.py`) or via `scripts/research_approve.py`; state in `data/research/state.json`.
- **monitoring/ web/ portfolio/ security/** — Prometheus metrics + JSON logging, FastAPI REST (`:8200`), allocation/coordination, encrypted vault.

Key configs: `configs/live_portfolio.yaml` (strategy registry), `configs/risk_profile.yaml`, `configs/event_playbooks.yaml`, `configs/research_agents.yaml`, `configs/paper_streams*.yaml` (research feeds; `paper_streams_none.yaml` = backlog-only runs). DB is Postgres + TimescaleDB; schema via `python -m migrations.run`.

## Conventions & Patterns

- **Bead IDs in docstrings** — every module/feature docstring cites its bead, e.g. `"""GDELT Doc 2.0 ingester (CL-6iu7) — ..."""`. Keep this when adding code.
- **Dataclass configs** — each component takes a frozen/plain `@dataclass` config (`RateDiffMRConfig`, `KillSwitchConfig`, `Playbook`, ...) populated from YAML; no dict-groping in business logic.
- **Module logger** — `logger = logging.getLogger(__name__)` at module top; structured/contextual logging, no prints.
- **Fail loud** — raise on missing or corrupt config/state instead of silently defaulting (e.g. `EquityTrailingStop` refuses to start on a corrupt state file; playbook validation rejects bad instrument kinds).
- **Atomic state files** — persist JSON state via tmp file + `os.replace` (kill-switch state, event book, `data/research/state.json`). Never write state in place.
- **Injectable transport shims** — network access goes through `Callable` params like `HttpGet = Callable[[str], str]` (`src/research/ingest.py`, `pdf_extractor.py`) so unit tests inject canned responses; no live HTTP in unit tests.
