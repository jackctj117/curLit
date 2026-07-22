# OPERATOR runbook (CL-mf6o)

> **See [`CURRENT_OPERATIONS.md`](CURRENT_OPERATIONS.md) first** — it is the
> living map of the 9 running daemons, the two paper venues (OANDA FX events
> + Alpaca options), every gate/cap/env knob, and the restart rules. This
> runbook covers the older engine-centric procedures.

Human-oriented operations guide. **For copy-paste startup/shutdown, see
`docs/BOOT.md`.** This document covers what to do once the system is
running: morning checks, alert response, dashboard reading, kill-switch
manual invocation.

The operator is not assumed to be the original system author. Every
section is written so a new operator with general SRE instincts can act
without reading the source code first.

---

## Daily morning checks (before market open, 06:00 UTC)

Run in this order. Each step takes <2 minutes.

1. **Engine health**. `curl -s http://localhost:8200/api/system | jq .` →
   expect `{"engine":"running","oms_halted":false}`. If `oms_halted=true`,
   check the journal for the last `INTENT_SUBMITTED` event before the halt
   — typical cause: a kill switch fired overnight (see §Alerts).
2. **Reconciliation**. Open today's `reports/reconciliation/$(date +%F).json`.
   If `mismatches` is non-empty, **do not start trading** — investigate first
   (see §Reconciliation mismatches below).
3. **Data freshness**. Grafana → "Data freshness" panel. Each FX pair and
   FRED series should be <24h old. Stale = ingestion job failed; check
   Airflow at http://localhost:8080.
4. **Drawdown**. Grafana → "Portfolio" panel → `fx_portfolio_drawdown_pct`.
   If <-15%, you're approaching the -20% kill-switch threshold. Brief the
   PM before any new positions.
5. **Open intents pending fill**. Query `trade_journal_events` for events
   in the last hour with `event_type=intent_submitted` and no following
   `order_filled`. >5 minutes pending suggests broker connectivity drift.

---

## Alert taxonomy & response

Alerts arrive via Telegram. Severity: 🟢 info, 🟡 warn, 🔴 page.

### 🔴 Kill switch fired

**Symptom**: `KILL SWITCH: <name> triggered` in engine log.

**Switches and what each means**:

| Switch                          | What broke                                       | Default action     | Operator response |
| ------------------------------- | ------------------------------------------------ | ------------------ | ----------------- |
| daily_loss_limit                | -3% on the day                                   | halt_new           | Review last 4h trades; resume after PM sign-off |
| drawdown_limit                  | -20% from peak                                   | flatten_all        | Stop trading; full incident review |
| vix_spike                       | VIX>35 AND +50% intraday                         | reduce_50pct       | Confirm via macro headlines; consider further reduction |
| fx_vol_spike                    | CVIX z-score > 3                                 | halt_new           | Wait for vol regime to mean-revert |
| reconciliation_failure          | Internal book ≠ broker book                      | halt_new           | See §Reconciliation mismatches |
| stale_prices                    | No tick for >10 min on tradeable pair            | halt_new           | Check broker stream connection |
| portfolio_correlation_crisis    | Cross-strategy corr regime = "crisis"            | reduce_50pct       | Review which strategies are correlating |
| strategy_correlation_spike      | Max pairwise > 0.9                               | reduce_50pct       | Same — likely shared factor exposure |
| single_strategy_drawdown        | One strategy at -25%                             | halt_strategy      | Liquidate that strategy or set paper_mode |
| equity_trailing_stop            | Equity -10% below persisted all-time peak        | halt_new           | Cooldown (see below); resume only after PM review |
| open_position_correlation       | Mean direction-adjusted corr of open positions > 0.85 | reduce_50pct | Open book is one trade in disguise — review overlap |

Switches are built by `build_kill_switch_manager()` (`src/runtime/run_engine.py`)
and evaluated by the live engine's health tick every 60 seconds. Thresholds
come from the active risk profile (`configs/risk_profile.yaml`; override
with `CURLIT_RISK_PROFILE`).

**Kill-switch state on disk**:

- `data/equity_trailing_stop_state.json` — `{peak_equity, cooldown_until}`,
  written atomically, survives restarts. After a trigger the switch halts
  new trades for the cooldown window (default 7 days); the peak stays
  frozen during cooldown and only resets after the cooldown expires AND a
  fresh equity mark arrives. A corrupt file makes the engine refuse to
  start (deliberate — repair or remove it consciously, never blindly).
- `data/event_book_state.json` — cumulative realized P&L of event-driven
  trades (created on the first event trade). If the loss breaches the
  configured `event_book_max_loss_pct`, new event entries are blocked and
  a CRITICAL is logged.

**Manual kill switch** — to halt all new trades (no liquidation):
```bash
curl -X POST -H "X-API-Secret: $WEB_API_SECRET" \
  http://localhost:8200/api/system/halt
```

To resume after investigation:
```bash
curl -X POST -H "X-API-Secret: $WEB_API_SECRET" \
  http://localhost:8200/api/system/resume
```

### 🟡 Reconciliation mismatches

**Symptom**: daily reconciliation report has non-empty `mismatches`.

**Mismatch kinds**:
- `missing_internal` — OANDA shows a fill our journal didn't record. Most
  often a partial fill split into two transactions on broker side.
  Resolution: cross-check OANDA web UI; if the fill is real, the
  internal-side gap is the audit failure to investigate (likely a crash
  during fill notification).
- `missing_broker` — our journal shows a fill OANDA doesn't.
  **Critical** — implies we acted on a fill that never happened. Halt
  and investigate before any trading.
- `qty_drift` — units differ. Usually rounding (OANDA truncates to
  whole units). >1 unit drift is a real bug.
- `price_drift` — prices differ by >1e-4. Typically: broker requote we
  didn't record. Check the OANDA transaction history for that intent.

### 🟡 Data ingest failures

Airflow → DAG `daily_ingest`. Failed task → retry first; if repeating,
check the source provider (FRED rate limit, yfinance ban, etc).

### 🟡 Event signals (current-events pipeline)

Two alerts come from `EventDrivenStrategy` (see `docs/ARCHITECTURE.md`
§13 for the pipeline):

- **"Event confirmed"** — a geopolitical event passed both the quality
  gate (urgency + confidence) and the market-confirmation gate (price
  actually moved). The message lists the headline, trades taken with
  sizes, skipped instruments, stop/time-stop settings, urgency,
  confidence, and the watch list. Positions are already on with tight
  risk — verify sizing looks sane and note the time stop
  (`event_max_holding_hours`).
- **"Event expired unconfirmed"** — a high-urgency event (urgency ≥ 8 by
  default) passed its confirmation window without a market move; **no
  trade was taken**. Informational: worth a headline scan in case the
  market is late rather than indifferent. Capped at one alert per run.

---

## Research gate approvals (Telegram)

GATE 1 (implement this hypothesis?) and GATE 2 (deploy this strategy as
paper-shadow?) are approved by **replying in the Telegram chat** that
receives the gate notifications. The notification is HTML-formatted and
includes a `Trades:` line listing the instruments the candidate would
touch (FX pairs + `POLY:` market ids; `(unknown)` if the brief exposes
none) — read it before approving.

Reply grammar (case-insensitive; trailing punctuation from mobile
keyboards is stripped):

```
approve <id> [reason]    # GATE 1: run implementer · GATE 2: deploy paper-shadow
reject  <id> [reason]    # decline (alias: skip)
pending                  # list all open approvals with their Trades lines
help                     # show this grammar
```

`<id>` is the **6-character short id** printed in the GATE 1
notification (prefix of the extract hash), or the full **strategy slug**
for GATE 2. The bot only accepts messages from the chat id in
`TELEGRAM_CHAT_ID` — anything else is logged and dropped.

Mechanics: `scripts/telegram_approval_bot.py` long-polls Telegram and
writes decisions atomically to `data/research/state.json` — the same
state file the research loop reads at the start of its next run. The
loop never blocks on a gate; approvals take effect on its next
invocation.

CLI fallback (same state file, works with the bot down):

```bash
.venv/bin/python -m scripts.research_approve --list
.venv/bin/python -m scripts.research_approve --slug <slug> --action GO
.venv/bin/python -m scripts.research_approve --gate 2 --slug <slug> --action SKIP --reason "..."
```

---

## Dashboard reading

### Regime panel
- `fx_correlation_regime` ∈ {normal, stressed, crisis}. Sustained
  "stressed" means strategies are crowding into a common factor.
- `fx_portfolio_correlation_max` >0.85 means at least one strategy pair
  is essentially the same trade. Investigate before next rebalance.

### Attribution panel
- `fx_strategy_attributed_pnl` per strategy, kinds = {realized, unrealized,
  signal_alpha, spread_cost, slippage_cost, swap_cost}.
- **Read order**: total → check signal_alpha vs costs. If signal_alpha
  ≈ costs, the strategy has no edge net of execution. Candidate for
  removal or paper mode.

### Exposure panel
- `fx_portfolio_gross_leverage` — total absolute exposure / equity.
  Alert at >2.0×.
- `fx_strategy_allocation` — target weights summing to 1.0. Rebalance
  triggers visible here as a step change.

### Reconciliation panel
- `fx_reconciliation_mismatches` — should be 0 daily.
- `fx_reconciliation_clean_days` — counter; track streak.

---

## Common symptoms → first place to look

| Symptom                                | Where to check first                                         |
| -------------------------------------- | ------------------------------------------------------------ |
| Engine logs noisy with WARNING         | `/tmp/curlit-engine.log` — filter for repeating series_id   |
| Strategy emits zero signals            | `bd show <strategy bead>` — paper_mode? min_r_squared gate? |
| Carry strategy: empty `get_series`     | macro_data row count for the FRED series — re-run ingest    |
| Sharpe collapsed week-over-week        | Attribution panel: which cost component spiked              |
| OANDA spread blown out                 | TCA report — `fx_tca_*_bps` percentiles                      |
| Mystery position appears               | trade_journal_events — find the unmatched fill              |
| Prometheus 5xx on /metrics             | engine restart — duplicate-timeseries on hot reload           |

---

## Polymarket trading

Polymarket integration ships in three phases. Read-only data flows
through DataProvider as `POLY:` symbols. Paper trading runs via
`--broker polymarket-paper`. Live trading is hard-gated until the
CL-poly-3 acceptance gates pass — see
[`docs/runbooks/PolymarketTrading.md`](runbooks/PolymarketTrading.md).
**Polymarket positions are categorically different from FX:** no
stop-losses, max loss = position cost, Kelly sizing not vol-target.
Read the runbook before placing real money.

## Configuration locations

- Strategies: `configs/live_portfolio.yaml` (registry: id, class,
  enabled, per-strategy config) and per-strategy dataclasses in
  `src/strategies/*.py` (e.g. `RateDiffMRConfig`).
- Risk profile (sizing + kill-switch thresholds + strategy gates):
  `configs/risk_profile.yaml` — `active:` key selects the profile
  (conservative / aggressive / aggressive_short); `CURLIT_RISK_PROFILE`
  env var overrides it.
- Risk parity bounds + rebalance cadence:
  `src/portfolio/coordinator.py` constants (top of file, all named).
- Allocation policy: `src/portfolio/allocation_policy.py` `PolicyConfig`
  defaults; per-strategy overrides via `PolicyConfig.overrides`.
- Kill switch implementations: `src/risk/kill_switches.py`
  `_build_switches()`. Each is documented inline.
- Event playbooks (themes, watch terms, instruments):
  `configs/event_playbooks.yaml`.
- Research agents (roles → LLM provider/model):
  `configs/research_agents.yaml` — everything runs on `claude-code`.
- Cost model defaults: `src/execution/cost_model.py` (live) and
  `src/backtest/cost_model.py` (per-pair spreads + overnight funding).
- Data ingest sources: `src/data/fred.py` `FRED_SERIES`,
  `src/data/yfinance_provider.py` ticker list.

---

## Escalation

1. Self: rerun ingest / restart engine / consult this doc.
2. PM (portfolio manager): drawdown > -10%, kill switch fired,
   reconciliation `missing_broker` mismatch.
3. Quant lead: model fits no longer producing signal; correlation
   regime sustained "crisis" >24h.
4. SRE on-call: server unreachable, postgres down, secrets compromised.

---

## Strategy lifecycle (operator commands)

The CLI lives at `scripts/manage_strategies.py`. See `docs/runbooks/`
for the full lifecycle narrative; quick reference:

```bash
.venv/bin/python -m scripts.manage_strategies add --strategy <id> --paper-days 90
.venv/bin/python -m scripts.manage_strategies evaluate --strategy <id>
.venv/bin/python -m scripts.manage_strategies promote --strategy <id> --weight 0.05
.venv/bin/python -m scripts.manage_strategies remove --strategy <id> --confirm
```
