# AGENTS.md — curLit

## AI development workflow

Work only on the issue explicitly assigned by the operator. This is a
development workflow, separate from curLit's runtime research agents.
See [the setup and handoff guide](docs/AI_DEVELOPMENT_WORKFLOW.md).

### Roles

- **Implementer — Claude Code / Opus 5:** inspect current code, make the smallest
  coherent patch, add regression tests, and update relevant documentation.
  Report exact commands, exit codes, results, and remaining limitations.
- **Escalation — Claude Code / Fable 5.1:** only for an operator-selected difficult
  design or debugging task; never a mandatory step on every change. Respect the
  assigned scope, including read-only analysis when requested.
- **Reviewer — Codex / GPT-6 Astra:** independently review a stable patch without
  modifying files or issue state. Stop the implementer's editing session first.
- **Tests and CI:** required evidence regardless of either model's assessment.
  Report baseline failures separately; passing tests alone is not approval.
- **Operator:** decides scope, resolves disagreements, and approves publishing,
  merging, and deployment. No automatic promotion from a model verdict.

### Operational boundaries

- Work in a credential-free development environment. A checkout and virtualenv
  are not security isolation; use a container, VM, or separate OS account without
  access to trading credentials, production state, or operational service sockets.
- Do not access broker accounts or production databases, including paper accounts.
- Do not start, stop, or restart trading daemons. Operational runbooks below are
  architecture context, not authorization to operate services.
- Do not push, merge, deploy, or perform destructive Git cleanup (including stash
  deletion and branch pruning). Leave publishing and deployment to the operator.
- Do not change trading parameters or runtime model providers unless assigned.
- Do not hide a failed check, skip it silently, or relax permissions to bypass it.
- Reviewer commands must be read-only; do not claim/close Beads, run auto-fix
  commands, or invoke write-capable connectors in a review session.

### Review priorities

Prioritize unknown account state permitting exposure; duplicate orders after
timeouts/restarts; partial-fill and cancellation accounting; exits that create
or reverse exposure; estimates reported as realized results; and missing
regression coverage for changed behavior.

Each finding must identify triggering conditions, incorrect behavior, its
consequence, and a file location. Distinguish confirmed defects from questions
and optional improvements. Do not invent findings to fill a quota.

### Completion

Provide the patch summary, exact test evidence, unresolved concerns, and Beads
status. The implementer files follow-up issues and updates local issue state;
the reviewer reports findings without mutations. Leave a stable patch for review
and leave publishing, merging, and deployment to the operator. Do not run Git or
Dolt pushes as a session-completion step.

## Project
curLit is an algorithmic FX trading system: pull live ticker values for currencies, cryptocurrencies, and rare metals, then use AI (NLP + quantitative models) to analyze treasury bond ETFs/trusts (e.g. FXY, TLT) against news/world events and recommend short/put positions.

Canonical operational docs — READ THESE (not the historical design in `reference/`):
- `CLAUDE.md` — architecture overview + conventions for the LIVE code under `src/`
- `docs/CURRENT_OPERATIONS.md` — full operational state, daemons, policies, restart rules
- `bd prime` — issue-tracker workflow

## Status
LIVE — paper trading. There IS production code: ~217 modules under `src/`,
~3,100 unit tests, and a 13-daemon fleet run by
`./scripts/daemons.sh start|stop|status` (operator only). Development checks:
`.venv/bin/pytest tests/unit -q`, `.venv/bin/mypy src/`,
`.venv/bin/ruff check src/ tests/`. Four strategies are live (see
`configs/live_portfolio.yaml`): rate_diff_mean_reversion, cb_sentiment_shift,
carry_vol_filter, event_driven. Everything is PAPER; OANDA practice + Alpaca paper.

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
- `CLAUDE.md` — live architecture + build/test commands (authoritative)
- `docs/CURRENT_OPERATIONS.md` — operational state, daemons, restart rules
- `configs/live_portfolio.yaml` — the strategy registry that actually runs
- `scripts/daemons.sh` — fleet control (start/stop/status)

## Reference Code (`reference/`) — HISTORICAL, do NOT implement from it

The `reference/` markdown (14 files) predates the build and describes an earlier
`fx-system/` design — including subsystems and strategies (momentum, value, COT)
that were **never built and do not exist in `src/`**. It is retained for design
history only and diverges from the live code. **Do not treat it as canonical and
do not implement from it.** The authoritative source is the live code under
`src/` plus `CLAUDE.md`; consult those before writing code.

## Development Conventions

### Test Integrity
- Never weaken, delete, or skip tests to conceal a defect.
- When requirements intentionally change, explain changes to existing test
  expectations and add coverage for the new requirement.
- Never fabricate results or contrive a passing test. Mock broker responses at
  external boundaries; assert behavior against an independent requirement oracle.
- Fix the CODE, not the tests. If the code cannot be fixed within scope, escalate
- Every test must have an independent oracle: known test vectors from an external source, cross-validation between two independent implementations, or bit-exact comparison against a reference path

### Code Quality Standards
- **Logging**: Add extensive logging — more than you think you need. Every state change, every decision boundary, every external call. Use structured JSON logging via `LogContext`. Log at INFO for state changes, DEBUG for detailed flow.
- **Assertions**: Add assertions at function boundaries for invariants. Check preconditions on inputs, postconditions on return values, and class invariants at method entry/exit. Assertions are documentation — they tell future readers what must be true.
- **Property-based tests**: Write `hypothesis` tests alongside unit tests for every numerical function. Test monotonicity, boundedness, sign consistency, and round-trip properties. Use decorators: `@given(st.floats(...), st.floats(...))`.
- **Type hints**: All functions must have explicit type hints on parameters and return values. Use `mypy --strict` mode. Use `| None` not `Optional`, `list[dict]` not `List[Dict]`. No `Any` except at system boundaries.
- **Magic numbers**: Document every magic number and constant with a comment explaining WHY that specific value was chosen. Link to the source (paper, empirical study, architecture doc section). No unexplained numeric literals.
- **Mypy must pass**: `make typecheck` must exit 0 before an operator-approved commit. CL-0deu.5.1 defines the PR checks; operator activation/required checks and remaining CI work stay tracked in CL-0deu.5. See `docs/CI.md`; do not assume remote enforcement is active.
- **Log BEFORE the action, not after**: "Placing order..." before the API call, "Order filled" after. You need the BEFORE log when the action crashes.

### No Fabrication
- NEVER report status, results, or completion that does not reflect work actually performed
- If uncertain whether a step succeeded, say so explicitly; do not paper over uncertainty

### Exit Code Discipline
- EVERY shell command's exit code must be checked
- NEVER proceed after a silent failure — a command that failed and was ignored is not a completed step

## Workflow
Implementers use the Beads workflow below for the assigned issue. Reviewers only
inspect issues using `bd --sandbox --readonly ... --json`.

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
bd --sandbox ready --json
```

**Create new issues:**

```bash
bd --sandbox create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd --sandbox create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd --sandbox update <id> --claim --json
bd --sandbox update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd --sandbox close bd-42 --reason "Completed" --json
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

1. **Check ready work**: `bd --sandbox ready` shows unblocked issues
2. **Claim your task atomically**: `bd --sandbox update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd --sandbox create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd --sandbox close <id> --reason "Done"`

### Quality
- Use `--acceptance` and `--design` fields when creating issues
- Use `--validate` to check description completeness

### Lifecycle
- `bd --sandbox defer <id>` / `bd --sandbox supersede <id>` for issue management
- `bd --sandbox stale` / `bd --sandbox orphans` / `bd --sandbox lint` for hygiene
- `bd --sandbox human <id>` to flag for human decisions
- `bd --sandbox formula list` / `bd --sandbox mol pour <name>` for structured workflows

### Local issue state and operator-controlled sync

Use `bd --sandbox` for development commands to disable automatic synchronization.
Hooks use
`bd --sandbox --readonly prime`; `.beads/PRIME.md` replaces the default completion
protocol. Do not use `bd prime --export` as session instructions: it intentionally
ignores the project override. Preserve the override when updating Beads integration.

Dolt is the local issue store, not permission to publish. Backup Git pushing is
disabled in `.beads/config.yaml`. Remote synchronization is operator-only.

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd --sandbox ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

<!-- END BEADS INTEGRATION -->
