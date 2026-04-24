# AGENTS.md — curLit

## Project
curLit is an algorithmic FX trading system: pull live ticker values for currencies, cryptocurrencies, and rare metals, then use AI (NLP + quantitative models) to analyze treasury bond ETFs/trusts (e.g. FXY, TLT) against news/world events and recommend short/put positions.

Full technical architecture and detailed design: `docs/ARCHITECTURE.md`

## Status
Architecture & planning phase. Tech stack chosen, detailed design documented. No production code yet — agents should not attempt to build, run, lint, or test code until a project skeleton exists.

## Tech Stack
- **Language**: Python 3.11+
- **Database**: PostgreSQL + TimescaleDB (time-series)
- **NLP**: FinBERT fine-tuned on CB statements, spaCy, HuggingFace Transformers
- **Modeling**: statsmodels (OLS), scikit-learn
- **Scheduling**: Apache Airflow → cron for development
- **Observability**: Prometheus + Grafana + Loki + Alertmanager
- **Broker**: OANDA v20 API (start) → Interactive Brokers (scale)
- **Process**: systemd (trading services), Docker Compose (observability)
- **Testing**: pytest + hypothesis (property-based)
- **Issue tracking**: bd (beads) — see Beads section below
- **CI/CD**: GitHub Actions or Jenkins for scheduled data pipelines

## Key Files
- `docs/ARCHITECTURE.md` — Full system design (data, models, NLP, backtesting, risk, deployment)
- `curLit-idea.md` — Original project scope
- `training/` — FinBERT fine-tuning scripts (once built)
- `labeling/` — CB statement labeling tool (once built)

## Reference Code (`reference/`)

The `reference/` directory contains **14 markdown files (9,219 lines)** with reference implementations organized by subsystem. These are the canonical code patterns that should be used when implementing each subsystem. **Always consult the relevant reference file before writing code for a task.**

### File mapping to phases / tasks

| Reference File | Subsystem | Phase/Issue prefix |
|---------------|-----------|-------------------|
| `reference/01_data_layer.md` | DB schema, ingestion, providers | Phase 2, CL-nfc, CL-9kv, CL-ft3 |
| `reference/02_features_and_models.md` | Feature store, OIS curve, rate diff, reaction function | Phase 3, CL-rvr, CL-8a9, CL-7x6 |
| `reference/03_nlp_pipeline.md` | CB scrapers, preprocessing, lexicon, transformers, diffs | Phase 4, CL-4jm, CL-dc1, CL-kc9 |
| `reference/04_backtest_framework.md` | Walk-forward runner, analytics, event backtest, bootstrap | Phase 5, CL-gkk, CL-cyk, CL-5yd |
| `reference/05_strategies.md` | All 6 strategies (rate diff MR, CB sentiment, carry+vol, momentum, value, COT) | Phase 8, Phase A, CL-7d6, CL-6m6, CL-c77 |
| `reference/06_portfolio.md` | Portfolio coordinator, risk parity, correlation monitor, attribution | Phase D, CL-6td, CL-6tu |
| `reference/07_execution.md` | Broker ABC, OANDA, IBKR, OMS, paper broker | Phase 7, CL-s0v, CL-6q0, CL-po0 |
| `reference/08_runtime.md` | Live engine, signal generation, price streaming, shutdown | Phase 9, CL-9l7, CL-xns |
| `reference/09_security.md` | Vault, wolfCrypt, credential mgmt, SSH hardening | Phase 14, CL-446, CL-gm3, CL-0sg |
| `reference/10_monitoring.md` | Prometheus metrics, structured logging, Grafana, alert rules | Phase 10, CL-98s, CL-2zx, CL-c0x |
| `reference/11_risk_management.md` | Position sizing, kill switches, correlation, stress tests | Phase 6, CL-e0z, CL-a2p, CL-jn6 |
| `reference/12_deployment.md` | Systemd, Docker Compose, backups, deploy workflow | Phase E, CL-cly, CL-2e2 |
| `reference/13_research_workflow.md` | Paper ingestion, relevance scoring, evaluation rubric | Phase B, Phase C, CL-5b6, CL-366 |

### Reading order for new tasks
1. Check the table above for the relevant reference file
2. Read that file's code patterns before writing implementation
3. `reference/00_README.md` provides the full index and directory structure

## Development Conventions

### Test Integrity
- NEVER modify, delete, skip, or weaken tests to make them pass
- NEVER hardcode expected values, mock results, or contrive a passing test result
- Fix the CODE, not the tests. If the code cannot be fixed within scope, escalate
- Every test must have an independent oracle: known test vectors from an external source, cross-validation between two independent implementations, or bit-exact comparison against a reference path

### Code Quality Standards
- **Logging**: Add extensive logging — more than you think you need. Every state change, every decision boundary, every external call. Use structured JSON logging via `LogContext`. Log at INFO for state changes, DEBUG for detailed flow.
- **Assertions**: Add assertions at function boundaries for invariants. Check preconditions on inputs, postconditions on return values, and class invariants at method entry/exit. Assertions are documentation — they tell future readers what must be true.
- **Property-based tests**: Write `hypothesis` tests alongside unit tests for every numerical function. Test monotonicity, boundedness, sign consistency, and round-trip properties. Use decorators: `@given(st.floats(...), st.floats(...))`.
- **Type hints**: All functions must have explicit type hints on parameters and return values. Use `mypy --strict` mode. Use `| None` not `Optional`, `list[dict]` not `List[Dict]`. No `Any` except at system boundaries.
- **Magic numbers**: Document every magic number and constant with a comment explaining WHY that specific value was chosen. Link to the source (paper, empirical study, architecture doc section). No unexplained numeric literals.
- **Mypy must pass**: `make typecheck` must exit 0 before any commit. CI enforces this.
- **Log BEFORE the action, not after**: "Placing order..." before the API call, "Order filled" after. You need the BEFORE log when the action crashes.

### No Fabrication
- NEVER report status, results, or completion that does not reflect work actually performed
- If uncertain whether a step succeeded, say so explicitly; do not paper over uncertainty

### Exit Code Discipline
- EVERY shell command's exit code must be checked
- NEVER proceed after a silent failure — a command that failed and was ignored is not a completed step

## Workflow
1. Check for ready work: `bd ready`
2. Claim an issue: `bd update <id> --claim`
3. Do the work
4. File issues for anything discovered: `bd create "..." -t bug -p 1 --deps discovered-from:CL-<parent>`
5. Complete: `bd close <id> --reason "Done"`

<!-- BEGIN BEADS INTEGRATION v:1 profile:full hash:f65d5d33 -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Quality
- Use `--acceptance` and `--design` fields when creating issues
- Use `--validate` to check description completeness

### Lifecycle
- `bd defer <id>` / `bd supersede <id>` for issue management
- `bd stale` / `bd orphans` / `bd lint` for hygiene
- `bd human <id>` to flag for human decisions
- `bd formula list` / `bd mol pour <name>` for structured workflows

### Auto-Sync

bd automatically syncs via Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- No manual export/import needed!

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

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
