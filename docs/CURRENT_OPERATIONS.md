# CURRENT OPERATIONS — what curLit is practicing right now

_Last full revision: 2026-07-21. This is the living map of the running system:
which venues trade, under what policies, which daemons run, how ideas flow
from headline to fill to scorecard, and every operator knob. When behavior
and this doc disagree, trust the code + `bd show <bead>` and fix this doc._

---

## 1. The two paper venues

Everything is PAPER. No real money moves anywhere.

### 1a. OANDA practice — machine-traded FX event legs
- **Account**: OANDA fxTrade Practice (`OANDA_ACCOUNT_ID` in `.env`), started
  at $100,000. FX-ONLY: the account cannot trade CFDs (oil/gold/index legs
  are excluded from the instrument map for this reason — CL-v9g4).
- **What trades**: `EventDrivenStrategy` legs from CONFIRMED geo events —
  the 21 mapped FX pairs (USD_NOK, USD_CAD, USD_JPY, …). Confluence still
  *confirms* on the best signal (e.g. Brent via the intraday feed); the
  engine trades the tradeable FX proxy.
- **Risk policy** (configs/live_portfolio.yaml `event_driven`):
  - Gate A: urgency ≥ 7 AND confidence ≥ **0.60** (lowered from 0.75 —
    CL-0qur — because the impact agent is calibrated conservatively).
  - Gate B: a real intraday move ≥ 0.25σ (20d daily vol) in the assessed
    direction inside a 30–120 min window, read from `intraday_quotes`.
  - Cross-asset ENTRY GATE (CL-6mzn): a positively-contradictory
    cross-asset read vetoes the machine legs (`cross_asset_veto`);
    missing data does NOT veto (documented fail-open posture).
  - 50 bps risk per leg at a 1% stop; max **3** concurrent legs (CL: raised
    from 2 — slot starvation); 4 h hard time stop; 2% event-book loss
    freeze; per-instrument (0.55) + haven (0.60) concentration caps.
- **Restart safety**: the reconciler consults multi-leg strategy books
  (CL-8s1e), so an engine restart no longer flattens open event legs. The
  phantom-position pruner (CL-v9g4) drops stale entries the broker doesn't
  hold (120 s grace).
- **Kill switches (CL-i4tx)**: 8 armed switches, fed a REAL context each
  60 s health tick by `src/risk/risk_context.RiskContextBuilder` (daily
  PnL vs persisted UTC day-start equity, drawdown vs persisted peak, VIX
  level/1d change, CVIX z-score, price-stream age inside the trading
  window, broker-vs-books mismatch from the 300 s alignment check).
  `flatten_all` / `reduce_50pct` now really submit target-0 / halved
  intents through the OMS (bypassing a prior halt) before halting new
  trades. Daily-loss/drawdown thresholds come from the ACTIVE risk
  profile (aggressive: -10% daily, -40% DD). Fired switches re-arm at
  UTC-day rollover; 3 consecutive evaluation failures of one switch fail
  CLOSED (halt new). Engine boot logs one ARMED/UNARMED line per switch
  — trust that log. State: `data/risk_context_state.json` (day-start +
  peak equity; a corrupt file refuses boot — repair or remove it
  deliberately).
- **Other engine strategies**: rate_diff is DATA-FED (n≈252 daily rows) but
  SIGNAL-GATED — it refuses to trade while rolling R² < 0.25 (currently far
  below; refits weekly). carry_vol has its OIS/CVIX inputs and rebalances
  monthly. cb_sentiment awaits a detected CB shift. This is fail-closed by
  design, not breakage — see `scripts/data_health.py`.

### 1b. Alpaca paper — machine-bought options on advisory ideas
- **Account**: Alpaca paper (`ALPACA_API_KEY/SECRET`), $100,000, options
  level 3. Executor: `scripts/execute_options.py --loop 300` (market-hours
  aware; a clean no-op when closed).
- **Entry exposure interlock (CL-0deu.1.1, 2026-09-07)**: both options and
  equities require a successfully read, validated position list before a new
  entry. Null/malformed responses, missing position-list capability, invalid
  quantities, duplicate symbols, and unidentifiable asset classes block entry
  with `blocked_exposure` (structured reason/detail in logs). Only an explicit
  empty list means flat. Blocked ideas remain eligible for retry; no terminal
  order row is written. Fractional equity holdings count as exposure. Options
  compare **held + proposed contracts** against both contract and underlying
  caps. Adjusted option roots that need an underlying mapping also block
  entries until that exposure can be identified. Invalid direct contract
  quantities/count caps reject configuration; zero count caps
  deliberately disable entries. Exit rules have no new entry interlock, and
  malformed broker snapshots raise instead of being mistaken for disappeared
  positions. **Remaining CL-0deu.1 work:** working-order exposure, snapshot
  freshness metadata and durable reservations across concurrent workers are
  not yet included. This first interlock does not establish a complete
  portfolio pre-trade gate. The interlock was loaded by both paper executors
  on 2026-09-08; see the bounded rollout record below. No paper/live mode or
  strategy parameter was changed.
- **Policy (paper phase — deliberately widened to build a sample, CL-ldd2)**:
  - pool: pending `buy_calls`/`buy_puts` ideas, confidence ≥ **0.45**
    (`ALPACA_OPT_MIN_CONFIDENCE`); niche/red-team gates OFF
    (`ALPACA_OPT_REQUIRE_NICHE/RED_TEAM=0`) — re-tighten before any
    real-money move;
  - up to `ALPACA_OPT_MAX_PER_DAY` (10) buys/day, PACED at `ALPACA_OPT_MAX_PER_HOUR` (2) so entries spread across the session instead of a single open burst — intraday events can still be bought in the afternoon (CL-h02l). Top-by-confidence first;
    1 contract each (`ALPACA_OPT_QTY`), skipping any contract whose premium
    exceeds $500 (`ALPACA_OPT_MAX_PREMIUM`);
  - contract selection parses the idea's moneyness band + DTE window
    ("slightly OTM ~5%, 3-5 weeks") into a target strike/expiry and picks
    the nearest LISTED contract (nearest expiry, then strike);
  - **technical-alignment gate** (CL-3xoj): computed price structure that
    scores below `ALPACA_OPT_MIN_ALIGNMENT` (−0.4) against the thesis
    skips the idea (no calls into a confirmed downtrend at the lows);
    fail-open when no history is computable;
  - **open-spread entry delay**: no entries in the first 15 minutes of
    the session (`ALPACA_OPT_ENTRY_DELAY_MIN`) — opening option spreads
    registered instant −40%+ marks on day one; ideas simply re-evaluate
    on the next 5-min cycle (~9:45 entry). Override: confidence ≥ 0.80
    (`ALPACA_OPT_ENTRY_DELAY_OVERRIDE_CONF`) enters immediately;
  - dedup via `alpaca_option_orders` (migration 014) keyed by idea_id;
    terminal decisions recorded, transient misses retried.
- **Sizing philosophy**: 1 contract, breadth over size — edge measurement is
  size-independent; more distinct positions = more information, bigger
  positions = only more variance.
- **Exit manager (CL-3rho)**: every cycle, BEFORE entries, each open option
  position is matched back to its idea row and run through prioritized
  rules — first hit sells-to-close the full position:
  1. `thesis_invalidated` (originating geo_event DISMISSED or idea
     cancelled — NOT event EXPIRED, which is just the ~2h intraday FX
     gate lapsing), 2. `time_stop` (idea's `time_stop_days`, default 10d,
     `ALPACA_OPT_DEFAULT_TIME_STOP_DAYS`), 3. `stop_loss` (premium −40%,
     `ALPACA_OPT_STOP_LOSS_PCT`; **disabled on the entry day** — opening
     spreads masquerade as losses — except the −60% extreme-move valve,
     `ALPACA_OPT_ENTRY_DAY_EXTREME_STOP`) — and even that valve is suppressed for the first `ALPACA_OPT_ENTRY_SETTLE_MIN` (15) min after entry so opening spread on cheap contracts can't trip it (CL-h02l), 4. `profit_target` (premium
     +80%, `ALPACA_OPT_PROFIT_TARGET_PCT`), 5. `expiry_protect` (≤4 DTE
     and not up ≥25%, `ALPACA_OPT_EXPIRY_PROTECT_DAYS` /
     `ALPACA_OPT_EXPIRY_MIN_PROFIT`; ≤1 DTE closes regardless — date-based,
     fires even with missing quotes), 6. `stale` safety net (time_stop + 2d).
  Master switch `ALPACA_OPT_EXIT_ENABLED` (default on). Exit side recorded
  on the SAME `alpaca_option_orders` row (mig 016: exit_status/reason/
  order_id/premium/pnl_pct/exited_at); the trade idea mirrors to
  `closed`. Unmatched Alpaca positions are flagged and NEVER auto-managed;
  vanished positions are finalized honestly (`expired_worthless` −100% /
  `closed_external`); sells reuse crash-safe `curlit-exit-<idea_id>`
  dedup. Positions therefore no longer accumulate to expiry — the book is
  self-clearing.

### 1c. Explicitly NOT auto-traded
- **Equities and options remain advisory-first**: every idea still flows to
  Telegram for manual Robinhood execution (`configs/retail_proxies.yaml`
  maps CFD/FX ideas to ETF proxies). Alpaca execution is additive.
- **Polymarket**: signals/notifications only; the real-money broker is
  hard-gated (CL-983f acceptance checklist + `POLYMARKET_MAINNET_UNLOCK`).

---

## 2. The idea lifecycle (headline → fill → scorecard)

```
GDELT (16 themed queries)  ─┐
X watchlist (44 accounts)  ─┼→ geo_events(NEW)
[Reddit — built, blocked]  ─┘        │
                    Haiku TRIAGE (batch relevance 0-10; <4 dismissed;
                    fail-open) — EVENT_TRIAGE_ENABLED          [CL-cunh]
                              │
                    Sonnet IMPACT AGENT (playbook-grounded assessment:
                    affected instruments + advisory trade_ideas with
                    stops/targets/triggers)                    [CL-6iu7…]
                              │ urgency ≥ 7
                    NICHE PASS — Kimi K3 agentic tool-loop drives
                    check_ticker / resolve_company / profile / dated SEC
                    tools mid-generation (API-billed, Moonshot)  [CL-ddzt]
                    → SymbolUniverse verification (13k US-listed + SEC
                      names; unverified tickers DROPPED)       [CL-tzug/9xha]
                    → source-backed fact/inference claims + evidence
                      coverage + dated $-volume liquidity floor
                    → evidence reviewer (claude-code): supported /
                      contradicted / insufficient / unavailable
                    → only complete, supported, liquid candidates
                      enter the trading feed                    [CL-eh28]
                              │
              ┌───────────────┼──────────────────┐
        CONFLUENCE      trade_ideas ledger   Telegram digest
        Gate A+B +      (migration 007/8,    (grounded levels,
        cross-asset     price_at_signal)     RH proxies, niche
        gate            │                    tags, bear cases)
              │         ├→ Alpaca options executor (top-5/day)
        OANDA FX legs   └→ OUTCOME TRACKER (daily): signed fwd
        (max 3)            return, MFE/MAE, win/loss/flat at
                           horizon (migration 013)             [CL-6axf]
                                    │
                    REFLECTIVE REVIEW (weekly, ≥20 finalised):
                    aggregates by theme/action/hop/confidence/niche/
                    red-team and proposes conservative, data-cited
                    tuning — ADVISORY, operator approves       [CL-g8jl]
```

Niche research outcomes, including failures and abstention, are retained in
`assessment.niche_research`. Source matching establishes provenance, not semantic
truth; missing review or unknown liquidity cannot approve a candidate.
See [the evidence contract](NICHE_RESEARCH_EVIDENCE.md).

Grounding injected into the niche pass per candidate (CL-eh28/3xoj/mtum):
targeted recent SEC 10-K/10-Q/8-K excerpts · yfinance profile ·
**computed technicals** (trend vs 20/50d SMA, swing S/R in dollars,
breakout state incl. approaching_*, level-test counts, volume ratio — never
pattern names; a test bans "flag/head/shoulders/wedge") · **options
activity** (P/C volume skew, volume vs self-built baseline, ATM IV).

---

## 3. The daemons (14 core + a macOS-only keep_awake = 15 on macOS)

`daemons.sh` prepends `keep_awake` (caffeinate) **only on macOS** — on Linux it
is skipped (a server does not sleep), so the fleet is 14 there.

| Daemon | Command | Cadence | Log |
|---|---|---|---|
| Keep awake (macOS only) | `caffeinate -dims` | held while the fleet runs | — |
| Trading engine | `CURLIT_RISK_PROFILE=aggressive .venv/bin/python -m src.runtime.run_engine --broker oanda-practice` | async loop | `logs/engine_stdout.log`, `logs/live_engine.jsonl` |
| Event pipeline | `scripts/event_pipeline.py --ingest --assess --loop 900` | 15 min | `logs/event_pipeline.log` |
| Intraday pricer | `scripts/intraday_pricer.py --loop 120` | 2 min + daily candle backfill | `logs/intraday_pricer.log` |
| Foreign rates | `scripts/refresh_rates.py --loop 86400` | daily | `logs/refresh_rates.log` |
| X monitor | `scripts/x_monitor.py --loop 300` | 5 min (tiered priorities) | `logs/x_monitor.log` |
| Options executor | `scripts/execute_options.py --loop 300` | 5 min (market-hours aware) | `logs/execute_options.log` |
| Equity executor | `scripts/execute_equities.py --loop 300` | 5 min (market-hours aware) — the SHARES A/B of the SAME advisory ideas the options executor trades (CL-ncbq): exits then entries, $1k notional sleeve per idea, `buy_calls`→long / `buy_puts`→short, 5% stop / 10% target (or the idea's own advisory levels), dedup in `alpaca_equity_orders` (mig 019). Rationale (CL-4c7o): the ideas were right on DIRECTION 80% of the time (41/51) while the short-dated OTM options expressing them won 7% — spread + theta ate the moves. **Master switch `ALPACA_EQUITY_ENABLED`, default OFF** — unset, the daemon logs "disabled" and idles (so it is safe in the roster before the operator turns it on) | `logs/execute_equities.log` |
| Options activity | `scripts/options_activity.py --loop 86400` | daily chain snapshots | `logs/options_activity.log` |
| Outcome scorer | `scripts/score_outcomes.py --loop 86400` | daily | `logs/score_outcomes.log` |
| Telegram bot | `scripts/telegram_approval_bot.py` | long-poll | `logs/` |
| Truth-post event study | `scripts/truth_monitor.py --loop 300` | 5 min — RESEARCH ONLY (CL-s9as): ingest public trumpstruth.org archive → Haiku classification (topic/tone/entities/buy-language) → measure SPY/QQQ/sector-ETF reaction at 1–120min windows (mig 017). NO orders, NO alerts, NO family-holdings logic — the deliverable is `scripts/truth_report.py` (topic×window returns, reversal rate, decay). "No durable edge" is a valid result | `logs/truth_monitor.log` |
| Fleet watchdog | `scripts/health_watch.py --loop 300` | 5 min — Telegram page on: (1) daemon up→down transitions (once per onset + recovery notice; born from the 34-min silent x_monitor gap); (2) X-ingest staleness (newest x-sourced event > `X_INGEST_STALE_HOURS`, default 3h); (3) **engine halt** — OMS new-trades halt going active (via `/api/system` `oms_halted`, enriched with the triggering kill-switch), once per onset + a recovery notice, so a halt (bug OR legit VIX/drawdown/desync trip) reaches Telegram in ≤5 min instead of waiting for someone to ask. Nothing watches the watchdog by design — simplest process in the fleet (CL-fmqp) | `logs/health_watch.log` |
| Morning digest | `scripts/morning_digest.py --loop 300` | once per trading morning at 09:15 ET (`MORNING_DIGEST_TIME_ET`) — TWO Telegram messages: (1) FULL positions, long and short with economic reading + closed-last-24h realized P&L + balances; (2) LONG ideas — the pipeline's pending bullish shopping list by confidence, independent of execution (CL-ydp8) | `logs/morning_digest.log` |
| Weekly event study | `scripts/weekly_event_study.py --loop 21600` | every 7 days runs the CL-z95p event study (`--days 30 --options`), writes the full report to `data/research/event_study_YYYYMMDD.md`, Telegrams the placebo verdicts + headline-vs-confirmation topline (CL-s1gb) | `logs/weekly_event_study.log` |

**Fleet control: `./scripts/daemons.sh start|stop|status|restart [name]`**
(idempotent — start skips running daemons; logs to `logs/<name>.log`). All
four verbs take an optional daemon name; an unknown name is a hard error
(exit 2), never a silent whole-fleet operation. To deploy a code change to
one daemon use `restart <name>` — it waits for the old process to die,
starts a new one, and fails loudly (exit 1) unless the surviving pid
differs from the old one (CL-obgy: a wrong pid-file guess once left
`execute_options` running pre-fix code while everything looked green).
Fresh-device
bring-up from zero: [`BOOTSTRAP.md`](BOOTSTRAP.md). Every entrypoint runs
the interpreter-health canary (pyexpat, CL-169t) and the engine additionally
runs the DATA-HEALTH preflight (CL-q4n1) at boot — a starved series is a
loud WARN banner, never a silent dormant strategy.

**Restart rules**
- Engine: safe with open positions since CL-8s1e — extended to the non-event
  strategies (rate_diff/CB/carry) in CL-bccy, which now persist their book
  (rate_diff→`StrategyStateStore`; CB/carry→`data/{cb_sentiment,carry_vol}_state.json`)
  and reconcile it from the broker each tick (CL-0h30), so the cold-start
  reconcile matches their legs instead of flattening them. Still check
  `/api/positions` first out of caution. Always preserve
  `CURLIT_RISK_PROFILE=aggressive`.
- Fills: on OANDA the engine also consumes the transaction stream
  (`_transaction_stream_task`) for real-time ORDER_FILLED / pending-clear
  (CL-vj74); the 300s position poll is the backstop, so a restart never loses
  a fill.
- Pipeline/monitors: restart freely; state (since_ids, event book, research
  state) persists in files/DB.

### 2026-09-08 bounded Alpaca paper rollout (CL-idfh)

Runtime source release: `935f935` (includes exposure interlock `8e0fc24`
and the individual Bandit fixes/dispositions). Hosted
[CI](https://github.com/jackctj117/curLit/actions/runs/34255853451) and
[Security Scan](https://github.com/jackctj117/curLit/actions/runs/34255853035)
passed. Linux and local macOS each passed 3,661 unit tests with the same
three pre-existing skips. Local focused concurrency/vault tests passed
20 repetitions; deliberate shared-client and response-leak mutations fail
their independent assertions. No production thread-cache or vault-auth
behavior changed to resolve the Linux fixture failures.

The preflight captured a redacted manifest in ignored
`data/deployments/2026-09-08_935f935_alpaca_baseline.json`: source commit,
Python 3.14.6, all 297 installed distribution versions and inventory hash,
effective entry/exit settings, hashed account identity and broker snapshot
counts. Before stopping either book, paper mode was verified and the broker
reported no working orders; 10 option and 9 equity positions were readable.
This is NOT the full broker-history export, backup/restore proof or
accounting cutover rehearsal required by CL-0deu.14/18.

Process ownership changed during this rollout. The named options restart
had very slow `pgrep` discovery and its briefly verified replacement exited
when the command session ended (cause not yet proven; CL-wv3v). Options was
restored under user launchd. A temporary submitted job was replaced with an
explicit `KeepAlive=false` job after a completed no-order cycle. Equities
then used the same session-independent approach, with the old PID verified
dead before bootstrap. Final PIDs: options **21444** (previous 8555),
equities **21457** (previous 30892). FX **28922** was not restarted; its
two flagged provider/attribution edits are comments only. Drift warnings
remain visible.

Both final processes completed exit/entry cycles by 17:33 UTC, with zero
accepted entries or exit submissions. Options reported zero errors. Equities
reported one HTTP 422 rejection for unsupported ticker `GLO.TO`, also present
before restart (CL-wtl6); this is not a newly introduced failure or a clean cycle.
Seven unmatched equity holdings remained visible and were not automatically
assigned or closed. Exactly one writer per book was verified independently
of the launcher; broker positions remained readable with zero working orders.
The 17:34 UTC verification is saved in ignored
`data/deployments/2026-09-08_935f935_alpaca_postcheck.json`.

The two active user jobs are `com.curlit.paper.execute-options` and
`com.curlit.paper.execute-equities` in `gui/501`. Their local, ignored
manifests live in `data/deployments/`; each directly runs the existing
executor with `-u --loop 300`, explicitly sets paper mode, and has
`RunAtLoad=true`, `KeepAlive=false`. They are not installed as new login
agents, and no full-fleet boot/recovery configuration was changed.
Unbuffered output changes logging only. For an operator-authorized restart
of these currently registered jobs, inspect the job/PID first, then use
`launchctl kickstart -k gui/501/com.curlit.paper.execute-options` (or the
equities label), verifying the old PID is gone and the new cycle completes.
Do not concurrently start a second writer through `daemons.sh`. Fleet stop
will not trigger automatic respawn because KeepAlive is disabled.

The deployed environment is older than fresh CI's dependency resolution;
its separate advisory audit/lock remains CL-e1nr. Six legacy options
`exit_pending` records were observed despite zero working broker orders;
they remain visibly unresolved under CL-0deu.3, not reconciled by a restart.
No ledger repair, DB migration, account reset, runtime-library upgrade or
research-provider switch was performed. On a deployment fault, stop the
affected writer and investigate before resuming; blindly rolling back to
pre-interlock code would restore fail-open entries. This rollout does not
establish full operational readiness or authorize real-money trading.

---

### 2026-09-08 niche research rollout (CL-294s)

Operator authorized publishing and deployment after replenishing Kimi credits.
Source release **`bab4835`** passed hosted
[CI](https://github.com/jackctj117/curLit/actions/runs/34267118787) and
[Security Scan](https://github.com/jackctj117/curLit/actions/runs/34267118821).
Both local macOS and hosted Linux passed **3,712 unit tests**, with three
existing skips and one existing sklearn convergence warning. Ruff, source
typechecking, comparison-CLI typechecking, Bandit and the staged secret scan
also passed locally.

The redacted baseline is in ignored
`data/deployments/2026-09-08_niche_preflight.json` (Python/package inventory,
source hashes, effective allowlisted research settings and provider check).
Kimi remains `kimi-k3`; discovery and Claude criticism are both enabled.
A single capped synthetic Kimi request completed with 500 total tokens.
The old pipeline subsequently completed a real discovery with 21 tool calls.
These verify provider access, not research accuracy or profitability.

After the old batch completed at **19:15:41 UTC**, PID **23860** was terminated
and verified gone before bootstrapping PID **45317** at **19:15:46 UTC**.
Only `event_pipeline` restarted. FX **28922**, options **21444** and equities
**21457** remained running. The source release was unchanged through cutover.
No broker calls, account reset, migration, dependency upgrade, model switch,
trading-limit change or historical-ledger repair was performed for this rollout.

The event pipeline now runs as user launchd job
`gui/501/com.curlit.paper.event-pipeline`, with its existing
`--ingest --assess --loop 900` arguments and unbuffered output.
Its ignored manifest is `data/deployments/com.curlit.paper.event-pipeline.plist`;
`RunAtLoad=true`, `KeepAlive=false`, and PATH explicitly includes the existing
Claude CLI. It loads the existing project environment. The job is not a new
login agent and does not change full-fleet boot configuration.

At **19:17 UTC**, singleton verification passed, the old PID was absent, the
new loop had started, database reads succeeded, and there were no new ERROR,
CRITICAL or WARNING log lines. **The new process had not completed its first
full cycle or persisted a new evidence report at this checkpoint.** Do not
confuse startup verification with a completed end-to-end research canary.
Cutover and checkpoint records are saved alongside the baseline as
`2026-09-08_bab4835_niche_cutover.json` and
`2026-09-08_bab4835_niche_postcheck.json`.

For an authorized restart, inspect the job/PID, then use
`launchctl kickstart -k gui/501/com.curlit.paper.event-pipeline`; never start
a second instance through `daemons.sh`. On a fault, stop this pipeline job
with `launchctl bootout gui/501/com.curlit.paper.event-pipeline` and inspect
before resuming. Preserve the Git release and records; use a reviewed fix or
explicit rollback, not a worktree reset. Reverting to `e97ecdf` would restore
the old permissive research semantics, so it is not an automatic fallback.
No database rollback is required by this additive assessment-JSON change.
Existing FX source-drift warnings and execution/accounting hardening gates
remain unchanged.

### Follow-up: first-cycle failure and recovery changes (CL-i3js / CL-ep4q)

The startup-only checkpoint above did **not** establish sustained health.
PID 45317 failed its first cycle at 13:26:54 local time and repeatedly logged
`Too many open files` from 13:30:46. It had all numeric descriptors 0–255
occupied: 102 regular files (mostly Yahoo timezone-cache SQLite/WAL files),
100 pipes and 50 IPv4 descriptors, plus three other sockets and stdin.
The failure began in the threaded Yahoo volume scan, before niche research.
No new evidence report was persisted. CL-338q remains the end-to-end canary.

CL-i3js changes both event-pipeline Yahoo batch download paths (RVOL and
price enrichment) to `threads=False`, reusing the caller's thread-local
cache instead of creating per-symbol workers. This trades scan concurrency
for bounded resources without changing prices, scoring, limits or order policy.
An offline subprocess regression exercises real yfinance dispatch and SQLite
cache with 80 symbols across eight batches, cyclic GC disabled and a 256-FD
limit. It checks exact fixture prices and bounded descriptor usage. The local
service manifest explicitly retains the same 256-FD soft limit; no limit
increase substitutes for the code fix.

CL-ep4q preloads each Claude prompt into a private 0600 temporary file before
spawning the CLI. Prompts remain off argv, files are per-call and close on
success/error/timeout. Tests inspect the descriptor at spawn and use real local
child processes to verify concurrent UTF-8 delivery. This removes dependence
on a live pipe writer; it does not prove every historical transport failure
was caused by a scheduling race or eliminate external provider failures.
Existing stdin tests intentionally retain content/argv assertions while
switching their transport expectation from `input=` to file-backed `stdin=`.

CL-hzrb makes the niche summary count the actual post-review eligible set,
matching the already-enforced persistence gate. Its regression checks that an
unavailable review retains a source-backed lead but reports zero surfaced.
The evidence fixtures use a fixed clock so future CI dates cannot change their
freshness oracle. Implementation tests alone are not a completed production cycle.

#### Verified recovery, 2026-09-08 21:43 UTC

Source release **`900b6f1`** passed hosted
[CI](https://github.com/jackctj117/curLit/actions/runs/34277123233) and
[Security Scan](https://github.com/jackctj117/curLit/actions/runs/34277123111).
Local and Linux suites each passed **3,717 tests**, with three existing skips
and the existing sklearn warning. Ruff, typechecking, Bandit and staged secret
scanning passed. An installed-Claude smoke returned the exact marker through
file-backed stdin. Its reported model ID differed from the requested model;
serving-model attribution remains unverified under CL-h7c1. No runtime model
selection changed.

After hosted checks passed, the exhausted job was stopped and PID 45317
verified gone before replacement **51644** started at **20:55 UTC**. The job
retains `--ingest --assess --loop 900`, `KeepAlive=false`, and explicitly sets
the same **256-FD soft limit**. No limit increase or duplicate pipeline was
used. FX **28922**, options **21444**, and equities **21457** stayed running.

Two normal cycles completed at **21:19:28 UTC** and **21:42:24 UTC**, with no
top-level cycle failure. Both Yahoo scans processed **73/73 tickers**, in
about 8.8 and 8.2 seconds. Each cycle processed 20 events (12 then 15 assessed;
8 then 5 dismissed). Across 64 samples about 45 seconds apart, descriptors
ranged from **5 to 44**, with **28** at final verification. This is a sampled
maximum, not every transient peak. The second scan dropped usage from 44 to
21 rather than accumulating another batch's resources. There were **29 Claude
invocations started**, zero logged stdin-deadline errors, zero logged transport
failures, and zero FD-exhaustion errors.

This was **not an error-free research-yield run**. GDELT rate-limit warnings
continued. Of 12 niche discoveries, five exhausted the tool-call budget and
seven returned incomplete generations; none supplied parsed or eligible niche
ideas. Nine reports persisted with explicit status and zero eligible count.
Three events (48530, 48550, 48551) became CONFIRMED before the conditional
assessment update, so the safe merge guard rejected those writes and logged
the missing audit outcomes. No incomplete niche idea entered the assessment's
trading feed. Base-impact ideas continued through their ordinary ledger path;
this does not establish accounting accuracy or broker reconciliation.

Remaining findings: **CL-uofe** evaluates Kimi completion within explicit
budgets; **CL-27s0** preserves audit outcomes across event-status races;
**CL-h7c1** covers CLI budget/cost/model provenance. Evidence is in ignored
`data/deployments/2026-09-08_900b6f1_pipeline_recovery.json` and
`data/deployments/2026-09-08_900b6f1_pipeline_verified.json`. The single-job
restart/stop procedure above still applies. No migration, risk-limit change,
account reset, dependency upgrade or broker operation was performed.

### September 8 follow-up: development fixes, not yet deployed

**CL-27s0** is implemented with migration 020 (`niche_research_audit`): independent
audit commits survive later event-status/assessment races, while niche merges
require unchanged ASSESSED state. The three historical log-only reports have
not been reconstructed. **CL-uofe** has native-loop fixes for K3 reasoning
history, explicit low reasoning effort, cached duplicate lookups, reserved
finalization, and bounded truncation recovery within the existing total output
token envelope. It remains open for captured-input evaluation; no claim of
improved live research yield or current account balance is made.

Validation: 3,730 unit tests passed (three existing skips), then the final
focused suite passed 209 checks including four against a disposable PostgreSQL
16 database. Ruff, source mypy, and medium/high Bandit passed. The disposable
database had fixture credentials, no operational mounts, and was removed after
testing. No operational migration or daemon restart occurred. **CL-9lrx** tracks
operator-approved migration, single-writer pipeline rollout, and measured Kimi
canary. Apply migration 020 before starting the new pipeline code; a missing
audit table blocks niche merges. Details: `NICHE_RESEARCH_EVIDENCE.md`.

### September 8 deployment attempt: migration applied, restart blocked

The operator authorized deployment of **`0b84647`**. Hosted
[CI](https://github.com/jackctj117/curLit/actions/runs/34284199742) and
[Security Scan](https://github.com/jackctj117/curLit/actions/runs/34284199731)
passed. A redacted runtime baseline and an encrypted full database snapshot
were captured. The 33 MB AES-256-encrypted archive fully decrypted in memory
and its 3,319-entry restore inventory validated. No plaintext dump was written.
This is not a completed full restore drill. The Linux backup-key path was absent
on this Mac; a dedicated 0600 local recovery key is retained separately under
ignored `data/deployment_keys/2026-09-08_0b84647_backup.key`.

Migration **020 applied at 22:15:37 UTC**, creating the independent audit table
with zero initial rows. Transactional DDL rollback and repeat application were
verified on the operational PostgreSQL 15 instance. No historical assessment,
trade, or accounting row was rewritten. The additive table remains compatible
with the old pipeline and is retained.

**CL-9lrx is blocked by CL-lu3d; no daemon was restarted.** A one-event native
Kimi comparison used captured event 48489, the same finite source collection,
and predeclared bounds: eight calls, 24 tools, 32,768 requested output tokens,
100,000 serialized prompt characters per call. Baseline exhausted tools after
five requests (54,162 input / 4,930 output tokens, 121.3 seconds). The revision
made four requests (49,961 input / 3,310 output, 86.0 seconds); its next prompt
grew to **131,225 characters**, so the local canary wrapper refused request five
before network transmission. This was **not a Moonshot balance error or an API
rejection**. Neither arm supplied parsed candidates; billing remains unknown.
The failure was retained, not bypassed by raising the input limit.

Original pipeline **51644** remains on running release **`900b6f1`**; its local
manifest's release marker was restored to that value. FX **28922**, options
**21444**, and equities **21457** were untouched. Fix native context/finalization
budget handling and rerun the bounded canary before replacing the pipeline.
Ignored evidence prefix: `data/deployments/2026-09-08_0b84647_` (preflight,
encrypted backup and verification, captured research/comparison, migration,
and blocked rollout record). Do not interpret the successful migration as a
completed deployment or the one-event experiment as a model-quality benchmark.

## 4. Data layer

| Store | Source | Refresh | Notes |
|---|---|---|---|
| `prices` (daily) | yfinance via Airflow `fx_daily_pipeline` (23:00 UTC weekdays) | daily, runs ~2-3 d behind on weekends (normal) | daily-vol baselines; do NOT write intraday rows here |
| `prices` source `oanda_daily` | OANDA daily candles (CL-lb03) | daily via intraday pricer | vol baselines for the 10 unmapped event instruments |
| `intraday_quotes` (mig 012) | OANDA pricing, 19 instruments | 120 s, 24 h rolling | confluence Gate B reference + current price |
| `macro_data` foreign/derived (CL-gr8o) | Bundesbank SDMX (DE2Y), ECB (€STR 3M), FRED (SOFR 90d), own G10 realized-vol CVIX proxy | daily via `refresh_rates.py` | series ids the strategies expect: `DE2Y`, `US2Y_MINUS_DE2Y`, `USD_3M_OIS`, `EUR_3M_ESTR_OIS`, `CVIX` (documented proxies) |
| `symbols` (mig 010/011) | NASDAQ Trader files + SEC EDGAR names/CIK | `scripts/refresh_symbols.py` | 13k US-listed; niche verification + CIK → 10-K tools |
| `options_activity` (mig 015) | yfinance chains, 4 nearest expiries | daily daemon | P/C, OI, ATM IV; self-building volume baseline (needs ≥5 snapshots) |
| `idea_outcomes` (mig 013) | own scorer | daily | forward returns, MFE/MAE, win/loss/flat |
| `alpaca_option_orders` (mig 014) | executor | per order | dedup + audit |

`scripts/data_health.py` prints the coverage report and exits non-zero on
starvation (cron/CI-able).

**Research tools.** Obsidian knowledge-graph vault (CL-uuy0): `make knowledge-vault` regenerates an offline Obsidian vault of theme↔instrument↔region coverage from the playbook YAML; research-only, NOT part of the live fleet; Coverage.md surfaces under-wired/watch-only themes. `make knowledge-vault-live` (CL-w0ox) additionally adds a `Discovered/` layer of the net-new tickers the niche agent has surfaced from LIVE events (green nodes branching off their themes) — DB-sourced, opt-in, a snapshot regenerated on each run (research-only, NOT live state; a DB blip degrades to the pure vault).

---

## 5. LLM stack & cost posture

| Role | Model | Billing |
|---|---|---|
| Triage | claude-haiku-4-5 (one batched call/cycle) | subscription |
| Impact agent | claude-sonnet-4-6 | subscription |
| Niche discovery | **kimi-k3 agentic tool-loop** | **Moonshot API (paid)** — `NICHE_TOOL_AGENT_ENABLED=0` selects Claude cycles; CLI invocation does not establish zero cost |
| Red-team critic | claude-sonnet-4-6 (one batched call/event) | subscription |
| Reflective review | claude-sonnet-4-6 (weekly) | subscription |
| Research pipeline | claude-fable-5 | subscription |

Grok: no subscription-billed API path exists; deliberately not used.
Cost controls: triage relevance gate, niche urgency ≥ 7 gate,
`KIMI_MAX_ITERATIONS`, red-team batching, per-day Alpaca caps.
The subscription labels describe the existing authentication path, not verified
per-call billing. CLI budget/usage accounting limitations remain CL-h7c1.

---

## 6. Monitoring & manual controls

- Web API (engine-embedded, port 8200): `/health` (open), `/docs`, and
  authenticated `/api/{account,positions,pnl,signals,system}` — auth is the
  `X-API-Key: <WEB_API_SECRET>` HEADER only (the legacy `?secret=` query
  param was removed, CL-pu7i), e.g.
  `curl -H "X-API-Key: $WEB_API_SECRET" http://127.0.0.1:8200/api/system`;
  `POST /api/system/halt` + `/resume` for an emergency stop (resume also
  re-arms the kill-switch daily dedup when the manager is wired —
  check `kill_switches_rearmed` in the response, CL-8lv6).
- OANDA practice dashboard: fxTrade Practice login shows positions/history.
- Alpaca paper dashboard: app.alpaca.markets (paper) shows option positions.
- Telegram: digests (grounded trade cards, niche 🎯 tags, red-team bear
  cases, Polymarket shift alerts), gate approvals (`approve <id>`), `ideas`.
- Grafana/Prometheus/Loki via `docker compose --profile full`.

---

## 7. Known gaps, honest limits, and blocked items

- **Reddit monitoring (CL-okww, BLOCKED)**: fully built (tiered watchlist →
  theme match → geo_events, OAuth client ready); Reddit's Responsible
  Builder policy requires an approved application. Keyless JSON *and* RSS
  are 403-blocked (verified live). Set `REDDIT_CLIENT_ID/SECRET` once
  approved and launch `scripts/reddit_monitor.py --loop 300`.
- **ENTSO-E (CL-526a, BLOCKED)**: needs the operator to register at
  transparency.entsoe.eu and set `ENTSOE_API_TOKEN`.
- **Options flow is aggregate, not tick-level**: the free scanner sees P/C
  skew + volume-vs-baseline, NOT sweeps/aggressor side. Upgrade path
  (Unusual Whales ~$50/mo) swaps the source; table + consumers stay.
- **rate_diff dormancy is honest**: R² gate, not a bug. carry OIS legs are
  documented backward-looking proxies, not dealer quotes. CVIX is a
  realized-vol proxy (real CVIX is proprietary).
- **pyexpat footgun (CL-169t)**: python@3.14 is brew-pinned; do NOT run
  `brew cleanup` until upstream fixes the 3.14.6 bottle (the venv pins the
  working 3.14.3_1 binary). Boot canary logs CRITICAL if it regresses.
- **War-gaming (CL-i0tk)** deliberately gated until the outcome tracker
  proves base-pipeline edge.
- **Paper-phase relaxations to re-tighten before real money**: Alpaca
  confidence 0.45 + gates off; event Gate A 0.60. The reflective loop is
  expected to propose the re-tightening from data.

---

## 8. Hardening posture (2026-07-22 review remediation)

All P0/P1/P2 items from the July code review + ultrareview are closed
(CL-xdnh, CL-qyav, CL-e6lx; only repo-wide ruff/format debt remains,
filed separately). Operationally visible changes:

- **Broker mode never lies**: missing `OANDA_API_KEY/ACCOUNT_ID` with an
  oanda-* mode now CRASHES the engine at boot (`BrokerCredentialsError`)
  instead of silently paper-trading. `ALLOW_PAPER_FALLBACK=1` is the
  explicit opt-in (CRITICAL log; status reports say "paper").
- **Slippage is enforced, not just journaled**: intents' `max_slippage_bps`
  becomes a direction-aware FOK `priceBound` on OANDA orders (venue
  rejects fills beyond tolerance); PaperBroker simulates the same check
  (`SLIPPAGE_EXCEEDED` rejection). Pricing-fetch failure is fail-open by
  design — that path carries kill-switch flattens.
- **No sync broker I/O on the engine's event loop**: OMS submits, health
  ticks, and alignment checks run via `asyncio.to_thread` /
  `submit_intent_async`.
- **Vault**: socket accepts same-UID peers only (kernel-verified;
  activates on next vault-agent restart); new seals require ≥12 chars /
  ≥60 bits; the live passphrase is KNOWN-WEAK (~31 bits) — rotate with
  `python -m scripts.rotate_secrets --rotate-passphrase` at a maintenance
  window (re-run recovery setup after; the old printed document only
  covers the .bak files).
- **Stress tests can't fabricate zeros**: missing data skips the scenario
  with a named WARNING or raises if nothing is priceable.
- **Structure**: event book state lives in `src/strategies/event_book.py`
  (reconciler contract unchanged); impact-agent assessments are typed
  (`Assessment` dataclass, byte-identical persisted format).

Review #2 (2026-07-22, second audit — CL-8lv6) closed all 4 P0 + 15 P1:
multi-strategy target memory in the coordinator (one strategy's exit can
no longer flatten a shared symbol; seeded from books on restart);
`self_sized` strategies pass through unscaled (root of the boot-time
size_mismatch); OMS halt passes risk-REDUCING intents, `_pending` no
longer poisons shutdown, halt is race-free; paper price stream never
fabricates (unpriced symbols don't tick); slippage bound fails closed
except for emergency flatten orders; `/api/system/resume` re-arms kill
switches and control endpoints are honest (503 when unwired, validated
manual trades, X-API-Key only); Alpaca live needs a dual gate
(`ALPACA_LIVE_UNLOCK=1` + `--confirm-live`); vault deploy bring-up works
end-to-end with secrets never touching disk; strategy ticks run off the
event loop. Residual by design: entry-side book state still advances at intent
time (phantom pruner covers it; full fill-lifecycle remains CL-hqyj);
the EXIT side is confirmed-only.

Review #3 (2026-07-22, third audit — CL-8cw1) closed its P0 + all
in-scope P1s: aggregation keys canonically (mixed EURUSD/EUR_USD can't
double-count), cross-tick memory stores post-constraint values,
remove_strategy preserves co-holders, kill-switch triggers are NOT
spent when the flatten couldn't enumerate positions (halt + re-fire
until it works), event-book exits are two-phase (pending until broker
confirms flat, re-emitting every tick — a rejected exit self-heals),
DB URLs percent-safe + password-redacting everywhere, vault interactive
writes atomic, dashboard auth hash-then-compare, liquidity sizing fails
closed — NOTE: adjust_for_liquidity currently has NO production call
sites (verification finding); the fail-closed behavior is latent until
it is wired into the sizing path (tracked in CL-9dhg follow-up). The httpx thread-safety finding was verified FALSE (Client is
documented thread-safe; usage audit clean, comment added).
