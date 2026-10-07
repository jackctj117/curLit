# CURRENT OPERATIONS — what curLit is practicing right now

**Latest recovery verification — September 9, 2026, 21:54 UTC:** code release
`634e1d2` is deployed. Alpaca remains **close-only** (options PID 93628, equities
93630); FX PID 93632 remains **entry-paused**, with unchanged aggressive risk
configuration and all 259 startup file hashes matching. Pipeline PID 93634
completed its first assessment cycle. Six additionally approved September 18
option allocations retain their original dates under exit management. Historical
replay now has 177 allocations with zero unresolved projections; 14 aggregate
cash attributions and 350 fee dispositions are preserved, with incomplete net
costs still unknown. Kimi again reported `insufficient_balance` at 21:43 UTC;
the four-event/eight-arm shadow comparison is inconclusive about useful quality.
Full new-entry gates and research validation remain open under CL-koeg/CL-uofe.
See [the recovery runbook](ALPACA_RECOVERY.md) for actual backup, repair, tests,
process verification and remaining limitations; older dated sections below
describe earlier operational states.

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
- **Account-currency sizing and P&L (CL-vfw7)**: stop risk, concentration
  notionals and realized P&L are converted from the pair's QUOTE currency
  to the ACCOUNT currency (`CURLIT_ACCOUNT_CURRENCY`, default USD, logged at
  boot) using a fresh (≤ 15 min) live-tick mid — direct pair or one cross
  via USD (`src/risk/currency.py`). No fresh rate → the entry is SKIPPED
  (`conversion_unavailable`); there is no fallback rate. Before this fix
  every non-USD-quoted leg was mis-sized (a 2026-08-03 USD_JPY short was
  318 units, ~150x too small) and **all pre-fix event P&L figures are
  mixed-currency** — e.g. that leg's "+44.29" was ¥44.29 ≈ $0.28. The
  event book state (`data/event_book_state.json`, now version 2) keeps the
  old aggregate verbatim as `legacy_mixed_currency_pnl` and restarts
  `realized_pnl` (account currency) at 0 on first load (the v2 file is
  written immediately at that load). The legacy
  figure's USD value is UNKNOWN, so **new event entries are BLOCKED
  (`legacy_pnl_unreconciled`) after deploy until the operator reconciles
  it**: with the engine stopped, set `legacy_reconciled_account_pnl` in
  the state file to the account-currency realized P&L of all pre-migration
  event trades (from OANDA transaction history). Loss-cap consumed is then
  `-realized_pnl - legacy_reconciled_account_pnl` (the migration does not
  reset the budget). An empty v1 history (0 trades, 0.0) does not block.
  A close whose exit rate is unavailable, or whose broker-flat
  confirmation came > 15 min after the trigger (fill-time rate unknown),
  is held in `unconverted_closes` (quote amount
  only) and, if a loss, blocks new entries (`unconverted_realized_loss`)
  until the operator sets that row's `reconciled_pnl_account`; a later
  rate is never back-filled. Exits are never blocked. Exit P&L is still
  priced at the trigger-time mid (`exit_price_basis:
  trigger_price_estimate`), not the broker fill. Rollback note:
  pre-CL-vfw7 code reading a v2 file would treat the account-currency
  `realized_pnl` as its aggregate and ignore the legacy figure.
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
- **Emergency-order fences, persistence and recovery (CL-o9sq / CL-pksi —
  NOT YET DEPLOYED; needs migration 025 applied BEFORE the engine restarts on
  this code).** Kill switches count a flatten/reduce leg complete only on a
  verified outcome (the OMS returns a typed `SubmissionResult`), keep a FIXED
  per-leg target across retries, and never resend a close whose outcome is
  unknown:
  - **Persistence.** Every emergency order is written to
    `fx_emergency_attempts` (migration 025) as `SUBMITTING` *before* the
    broker call, keyed by the intent id that is also the OANDA client id
    (`clientExtensions.id`), with the original position and the fixed target.
    Every status change is written back. If the row cannot be written the
    order is NOT sent and the sticky cause `external:emergency_attempts_unavailable`
    halts entries.
  - **Fences.** `SUBMITTING`/`WORKING`/`UNKNOWN`/`PARTIAL_TERMINAL` attempts
    fence their symbol in the kill-switch manager AND in the OMS: every
    writer (strategies, reconciler, `/api/trade`, the other kill switch) gets
    `BLOCKED` on that instrument, and the sticky cause
    `external:unresolved_emergency_orders` holds the entry halt.
  - **Resolution = broker evidence only.** A streamed `ORDER_FILL` carrying
    the attempt's client id (via the OMS's transaction-id-deduped `on_fill`)
    or an order lookup (`GET /v3/accounts/{id}/orders/@<clientID>` plus the
    filling transaction, every health tick and at startup). A flat position
    snapshot never clears a fence. Fills are summed per venue transaction id
    (duplicates count once, also across restarts); a synchronous fill counts
    only the units OANDA reports in `orderFillTransaction`. A verified
    zero-fill rejection/cancel allows ONE retry per symbol per tick against
    the ORIGINAL target. A partial fill (streamed, or a short synchronous
    fill) followed by a terminal state is `PARTIAL_TERMINAL`: the cumulative
    fill is recorded, nothing is resent, the fence stays. OANDA "no such
    order" (HTTP 404) never resolves an attempt — OANDA also returns it for
    executed orders that aged out — so such an attempt waits for a fill or an
    operator release.
  - **Restart.** Each action's FIXED per-leg targets are persisted before its
    first order (`fx_emergency_episodes`, OPEN until the daily re-arm /
    operator resume drops them), so a restart never reduces an already
    completed leg again (a failed write sends nothing that tick). Startup
    restores open episodes AND their attempts' verified fills; a leg whose
    close is verified filled stays fenced for every writer (cold-start
    reconciler included) until the position feed shows it
    (`derisk_fences.awaiting_position_confirmation`), so a lagging feed
    cannot trigger a second close. It then re-loads every
    unresolved attempt, re-fences it (before the cold-start reconciler can
    submit), records `external:unresolved_emergency_orders`, then asks OANDA.
    A restart never resolves anything. If the tables are unreadable, EVERY OMS
    submission is blocked (any writer could duplicate an unknown order) and
    entries halt (`external:emergency_attempts_unavailable`); the health tick
    retries recovery and lifts the block once it succeeds.
  - **Operator.** `/api/system` → `derisk_fences` (`count`, `symbols`, and per
    attempt status / client id / cumulative fill). `/api/system/resume` returns
    **409** while any attempt is unresolved (auto-resume can never lift
    `unresolved_emergency_orders`). After reconciling a fence the evidence
    cannot settle (e.g. `PARTIAL_TERMINAL`) at the broker:
    `POST /api/system/derisk-fences/release` with
    `{"symbol", "reason", "changed_by"}` (attributed, persisted as
    `OPERATOR_RELEASED`), then resume. Inspect rows with
    `SELECT * FROM fx_emergency_attempts WHERE status IN
    ('SUBMITTING','WORKING','UNKNOWN','PARTIAL_TERMINAL');`.
  - Limits: an emergency order whose request never reached OANDA (404 on
    lookup) needs an operator release before the leg is retried; the paper
    broker has no order lookup (its orders resolve synchronously).
- **Sporadic practice 401s (CL-wrsa)**: OANDA practice intermittently
  returns `401 Unauthorized` on `GET accounts/<id>/summary` / `/positions`
  (and on price-stream connects). Observed 2026-07-21..10-01: 15 of 16 REST
  401s and 5 of 5 stream 401s landed within ~60 s after a :00/:30
  wall-clock boundary (one REST 401 at +180 s). Where sleep/wake history
  exists (2026-09-26 onward) none was within 23 min of a wake; most were
  hours into continuous uptime. The token is never mutated in-process and
  no further broker-read failure was logged in the 10 min after any REST
  401. Strongest supported
  hypothesis: a periodic, short OANDA-side auth rejection — UNPROVEN (no
  body/RequestID was logged before this fix). Now: every 401/403 logs one `OANDA auth rejection:` line
  with `request_id`, `server_date`, `clock_skew_s` and the redacted body —
  quote the `request_id` to OANDA support if it recurs. Summary/positions
  READS are retried exactly once after 2 s; orders, cancels and the
  order-path `/pricing` GET are NEVER retried. A `clock_skew_s` of more than
  a few seconds is only a clue toward the host clock (response latency also
  inflates it) — corroborate before acting on it.
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
  - **what the niche/red-team gates mean (CL-7kuu, development change, not
    yet deployed)**: when `ALPACA_{OPT,EQ}_REQUIRE_NICHE` or `_RED_TEAM` is
    on, an idea is a candidate only if it has a write-once
    `idea_research_status` row (migration 025) with `research_eligible` true
    and the recorded statuses — niche: discovery `completed`, evidence
    `source_backed`, liquidity `sufficient`; red-team: review `supported`.
    The row is written by the idea ledger in the same transaction as the
    idea, only for ideas the niche merge created from a research-eligible
    NicheIdea; note text, an LLM-emitted `niche`/`research` key, or an idea
    persisted earlier never gains one. Executors only read it; a Postgres
    rules discard any UPDATE or DELETE. Ideas persisted before migration 025 have no
    row, so they are NOT executable under the gates;
  - with both gates **off** (the current paper `.env`), the candidate query
    is byte-identical to before CL-7kuu — no behavior change;
  - `ALPACA_LEGACY_NOTE_MATCH=1` (default 0, read by both daemons) restores
    the old `notes LIKE '%niche%' / '%red-team%'` matching as an explicit
    transition shim and logs a WARNING at config load and every cycle;
    no effect when both gates are off. Deploy order: apply migration 025
    BEFORE the new event pipeline: without the table, the first event with a
    merged niche idea raises in the ledger (logged, fail-soft) and that
    cycle's idea writes from that event onward are lost;
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
| Fleet watchdog | `scripts/health_watch.py --loop 300` | 5 min — Telegram page on: (1) daemon up→down transitions (once per onset + recovery notice; born from the 34-min silent x_monitor gap); (2) X-ingest staleness (newest x-sourced event > `X_INGEST_STALE_HOURS`, default 3h); (3) **engine halt** — OMS new-trades halt going active (via `/api/system` `oms_halted`, enriched with the triggering kill-switch), once per onset + a recovery notice, so a halt (bug OR legit VIX/drawdown/desync trip) reaches Telegram in ≤5 min instead of waiting for someone to ask; (4) **host suspend / unobserved gaps** (CL-cmg9 — deployed 2026-09-23): each cycle compares wall-clock vs monotonic elapsed time (macOS monotonic stops while asleep), pages once when the host slept or the watchdog was down, marks readiness `stale_after_gap` until an awake cycle has every daemon up + engine state readable + X ingest fresh, and keeps `observation` totals that never count sleeping hours as observed (`readiness=`/`observed=` in the cycle log; `host_gaps` in `data/health_watch_state.json`). Slow-but-awake cycles are logged as active-runtime latency, not sleep. Nothing watches the watchdog by design — simplest process in the fleet (CL-fmqp) | `logs/health_watch.log` |
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
Process discovery is bounded and identity-verified (CL-wv3v, via
`src/runtime/proc_discovery.py`): each argv scan runs in a worker abandoned
after 10 s (5 s for `status`). The old writer is tracked by PID plus process
start time, so a recycled PID is never treated as the old writer. Only
re-verified identities are signalled. Restart has a TOTAL deadline of
`DAEMONS_RESTART_DEADLINE_S` (default 60 s: 30 s SIGTERM grace, then SIGKILL,
then start/verify). A failure prints
`RESTART FAILED [phase=discover|stop-wait|start|verify]` with the last known
writer state. Two matching processes are refused and reported. A scan that
cannot confirm "none running" never launches. That includes a stalled scan
and a scan that hit an unreadable process running as our user. Either way, a
third writer is never started. SIGKILL is re-verified against a fresh scan,
and launch verification checks the launched PID's start time. `status` lines
are unchanged. A duplicate appends `— DUPLICATE: N matching processes`. A
stalled scan prints `? <name> UNKNOWN`. The watchdog neither pages that as
down nor counts it as "every daemon up" for readiness. Launch is unchanged:
`nohup`, same process group as the caller, SIGINT/SIGQUIT ignored like bash
`cmd &`. It does not cover the launchd-owned `com.curlit.paper.*` jobs.
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

### Context blocker resolved; rollout revalidation (CL-lu3d)

Native `:kimi-budget-v3` now checks the 100,000-character serialized message cap
before every request. When history exceeds it, a fresh tools-disabled request
carries all captured sources exactly once and every tool-result fact/error,
with no document truncation or omission. If the complete packet cannot fit,
research stops explicitly with `budget_exhausted/prompt_char_limit`. The original
model-call, tool-call and total requested-output-token ceilings are unchanged.

The captured failure replay reproduced the first four original request bodies
exactly, then reduced the next request from 131,225 to **47,673 characters**,
retaining all nine source records. Four new regression cases first failed on
the old implementation and pass with the guard, including JSON Unicode escaping
and cases where no bounded finalization is possible. The downstream evidence,
liquidity and review gates also pass with context finalization exercised.

A repeat paid comparison on the identical frozen event completed within bounds
for both v2 and v3: v2 used four calls, 14,896 input / 1,483 output tokens and
41.4 seconds; v3 used four calls, 5,961 input / 1,640 output and 44.0 seconds.
Neither produced source-backed eligible candidates. This small stochastic case
does not prove improved research quality. A separate exact-boundary check replayed
four historical responses offline and made **one new paid finalization call**:
16,163 input tokens and 1,310 output tokens. The call finished normally with all nine sources retained,
two parsed but ineligible research leads. No broker or ledger path was used.
Costs remain unknown. Private evidence is retained in
`data/deployments/2026-09-08_context_v3_captured_canary.json` and
`2026-09-08_context_v3_boundary_canary.json`.

**CL-lu3d is resolved; CL-9lrx resumes deployment verification.** This subsection
records pre-cutover evidence, not a completed daemon restart. CL-uofe remains
open for useful evidence/completion evaluation; no research approval gate has
been relaxed to create a successful-looking canary.

Local validation: the first full suite passed 3,735 tests (three existing skips).
The final rerun passed 3,735 but exposed one pre-existing floating-point property
failure, **CL-1vqh**: unchanged sizing code returns 6,577,397.812500001 versus the
test's bound of 6,577,397.8125. The identical counterexample reproduces on prior
release `791381c`; neither the risk function nor that test was changed or skipped.
The final niche-focused suite passed 75 tests, including source/critic/liquidity
gates through context finalization. Source mypy (225 files), Ruff, and medium/high
Bandit passed. This is not an assertion that the final local full suite was green.

### Context-v3 pipeline cutover (CL-9lrx)

After CL-lu3d was resolved, source commit `0c824b5` passed hosted CI
(`34287038969`: unit tests, lint, typecheck) and Security Scan (`34287039084`).
The paid canaries' source hashes match this release. The encrypted backup and
additive migration 020 were rechecked before cutover.

At **2026-09-08 22:46:31 UTC**, the completed-cycle guard stopped only
`gui/501/com.curlit.paper.event-pipeline`, confirmed PID **51644** exited, and
bootstrapped its existing manifest with the new source marker. The single new
pipeline writer is PID **58060**, source **`0c824b5`**, still
`--ingest --assess --loop 900`, with the existing 256-file descriptor limit.
FX/options/equities PIDs **28922 / 21444 / 21457** were not restarted. No broker
orders, account resets, trading-limit changes or dependency upgrades were made
by this rollout procedure.

Two-cycle observation and read-only audit/eligibility verification **passed**;
CL-9lrx is closed. The first completed cycle,
at **2026-09-09 02:25:05 UTC**, had four qualifying discoveries and four matching
independent audit records with verified payload hashes. Events **48603, 48602,
48593** became EXPIRED during processing: their audits survived and their
assessment merges were correctly skipped. Event **45679** retained its research
projection. All four discoveries completed within the original resource caps;
all eight candidates failed evidence/review eligibility and no niche ledger
rows were created. Six candidates had malformed claim records; the other two
contained unknown source IDs or nonmatching passages. **CL-uofe** retains this
output-contract/citation-quality follow-up; completion is not evidence quality.

The Mac slept during this first observation, confirmed by its power-management
log and large sample gaps. The unchanged pipeline PID resumed after waking.
Those sleeping hours are **not** uninterrupted healthy uptime; **CL-cmg9** tracks
host availability and suspend/resume readiness. No power settings were changed.
GDELT's pre-existing 429 backoffs also delay ingestion (628 seconds in the first
new-code cycle), tracked separately in **CL-jk7i**, not attributed to Kimi
credits or context handling.
Private evidence is under ignored `data/deployments/2026-09-08_context_v3_*`.

The second cycle completed at **2026-09-09 02:52:48 UTC**. Across both cycles:
**15 qualifying discoveries = 15 unique hash-verified audits = 15 commit logs**;
all three changed-event audits survived, and 12 assessment projections were
present. Every discovery completed, with no abstentions, partial/budget/error
outcomes in this observed sample. **All 23 candidates remained ineligible** for
insufficient evidence; all their reviews were `insufficient_evidence`, while
their liquidity measurements were sufficient. No new niche ledger rows or
approval-gate violations were found. The critic's legacy "survived" log label
also includes incomplete research leads; **CL-00c7** tracks replacing that
misleading summary with explicit status counts.

The 90 logged Kimi requests matched the persisted response traces, using
516,903 input and 36,234 output tokens in total. Each invocation stayed within
8 calls, 24 tools, 32,768 requested output tokens and 100,000 message characters;
the largest observed prompt was 82,159 characters. No live invocation needed
context compaction in these two cycles: the overflow path was proven by the
separate exact-boundary paid canary above. Discovery latency was 52.7–184.6
seconds; costs remain unknown, not zero. The unchanged pipeline PID was verified
after both cycles, with 66 open descriptors and a sampled peak of 75 against
the existing 256 limit. This short observation is not a long-duration leak test.
FX/options/equities process identities remained unchanged. Verification artifact:
`data/deployments/2026-09-08_context_v3_verified.json`.

This completes the scoped pipeline rollout, **not** CL-uofe's useful-evidence
evaluation or the broader CL-0deu operational/live-trading readiness gates.

### September 9 recovery investigation (historical, before cutover)

Read-only broker/DB capture at 15:50:47–50 UTC verified that the six internally
pending options exit orders are already filled, with matching individual fill
quantities, while same-contract net positions remain. Seven unmatched equity
holdings trace to original curLit entries marked `closed_external`. They now
have proposed restoration targets, not applied corrections. See
[the recovery evidence and handoff](ALPACA_RECOVERY.md) and **CL-cojs/CL-0deu.3**.

Development fixes cover the shared-symbol exit-loop defect (**CL-qz3f**), broker
asset preflight (**CL-wtl6**), bounded resumable GDELT ingestion (**CL-jk7i**),
opt-in passage-reference research (**CL-uofe**) and explicit critic status logs
(**CL-00c7**). Migration 021 was tested in a disposable PostgreSQL instance only.
The running checkout, operational rows and daemon identities were unchanged;
FX remains PID 28922 with its July 31 start time. **CL-koeg** tracks separately
approved restoration and per-path rollout. Tests passing is not account repair
or proof that the research revision produces better ideas.

### September 9 authorized close-only recovery cutover

As of 19:04 UTC, **`adddbe8` is pushed and deployed** to the Alpaca execution
paths and a fresh FX process. Hosted CI and security passed. A fresh encrypted
database backup was fully restored in isolation before migrations 021/022 and
the audited 13-row repair. The six original pending option exits are now
fill-backed closed; all seven approved equity allocations have original-date
management restored. The repair itself submitted no orders. FRO subsequently
closed its verified 22-share allocation, with the actual broker fill reflected
in the ledger. See [the operational evidence](ALPACA_RECOVERY.md).

Options PID **5004** and equities PID **5006** run
`ALPACA_LEDGER_CLOSE_ONLY=1`: **no new Alpaca entries**, verified reductions only,
one writer per path and an account-wide advisory fence. Do not turn this off
to bypass remaining new-entry lifecycle/reservation work. Six further current
option allocations with legacy `closed_external` state remain unmanaged pending
explicit restoration approval (CL-jr3z); historical aggregate overcloses remain
unresolved (CL-sweu). New reports separate fill-based gross P&L from unknown
costs/net instead of reusing quote estimates.

FX PID **5365** replaces July 31 PID 28922. OANDA practice was flat with no
pending orders; startup reconciliation is clean, the aggressive profile is
unchanged, and the fresh-process manifest's 256 source/config hashes match.
`CURLIT_START_ENTRY_PAUSED=1` protects startup; monitoring/reconciliation continue
while entries are paused. The morning digest was separately restarted with
its daily dedup state preserved. Pipeline PID **6071** replaced PID 58060 only
after the old cycle completed and its model children exited. The new process
has release marker `adddbe8`. Its first real GDELT 429 was durably deferred and
ingestion yielded in **13 seconds**, versus **615 seconds** in the old cycle;
assessment began at 19:09:40 UTC after the unchanged volume scan. All 16
incomplete themes remain explicitly deferred, not successfully covered. This
proves the scheduling path progressed despite upstream throttling; it does not
prove improved news availability or research quality.

The paid same-capture Kimi baseline/passage comparison both abstained within
the original bounds. This is an inconclusive research-quality result, not a
reason to loosen evidence gates; passage references remain opt-in. CL-koeg,
CL-uofe and the remaining CL-0deu dependencies stay open.

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
per-call billing.

**claude-code budget and cost provenance (CL-h7c1 — in code; running daemons pick
it up only when the operator next restarts them).** Checked against the installed
CLI (Claude Code 2.1.287), not assumed:

- `max_tokens` enforcement is GATED, default OFF (`CURLIT_CLAUDE_ENFORCE_OUTPUT_CAP`):
  when enabled the driver sets `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (the CLI has no
  `--max-tokens` flag) to each caller's `max_tokens`; the cap includes thinking
  tokens and is clamped to the model limit. Caller caps (triage 900, impact 2000,
  ...) were never calibrated as total output+thinking budgets, so enforcing them
  could fail previously-successful calls. While OFF the env is untouched (calls
  run under the CLI default, as before) and `max_tokens` is listed in
  `LLMResponse.unenforced_params`. Enable only after calibrating caller budgets.
- `temperature` cannot be enforced (the CLI has no sampling control). Each
  response records it in `LLMResponse.unenforced_params`, and the driver logs it
  once per driver at WARNING.
- `usd_cost` is `None` with `cost_provenance="subscription_unmetered"`. It is no
  longer `0.0`, because nothing meters per-call USD on a subscription. The CLI's
  `total_cost_usd` is kept only as `nominal_usd_cost`, an API-equivalent
  ESTIMATE that is never summed as spend. Spend totals (`LLMClient.usage_summary`,
  `DebateResult.total_cost_usd`, the transcript footer, `scripts/scorecard.py`'s
  research-spend metric, `scripts/compare_llm_providers.py`) report UNKNOWN when
  any call in the total is unmetered. The weekly scorecard therefore shows
  research spend as UNKNOWN (AMBER) while research runs on claude-code. Ledger
  rows that older code wrote as `provider=claude-code, usd_cost=0.0` are also
  treated as unknown.
- Tokens come from the CLI `usage` block. A missing or malformed count is `None`
  (unknown), never 0.
- `model` is the serving model only when the payload's `modelUsage` has an entry
  matching the requested model that produced output. Otherwise it is
  `"unverified"`, because `modelUsage` also lists Claude Code's internal Haiku
  utility calls. The request is always kept in `requested_model`.

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
- **Account-wide entry halt (CL-0deu.2, migration 024 — DEPLOYED 2026-09-23 at
  `4b6cc05`; all three paths acknowledged `PAUSE_ENTRIES` v1).** One durable record
  (`trading_halt_state`) is consulted immediately before new exposure by the
  FX OMS (per non-reducing intent, incl. manual `/api/trade`) and by both
  Alpaca entry executors (top of every cycle AND right before each submit).
  Unreadable/missing state blocks entries (fail closed); risk-reducing FX
  intents and the Alpaca exit managers are unaffected. A restart re-reads the
  record, so it can never clear a halt.
  - **Deploy consequence:** migration 024 seeds `PAUSE_ENTRIES`. After
    applying it, NO path opens new exposure until an explicit resume.
  - Halt: `POST /api/system/halt` with optional JSON
    `{"reason": "...", "changed_by": "...", "mode": "PAUSE_ENTRIES"|"CLOSE_ONLY"}`
    (bodyless still works; the FX OMS is braked locally first, then the
    durable record is written — a failed write returns 503 and says the
    Alpaca books are NOT covered). `EMERGENCY_FLATTEN` is refused until a
    path implements it.
  - Resume: `POST /api/system/resume` REQUIRES
    `{"reason": "...", "changed_by": "..."}` once the store is wired (400
    otherwise, still halted). Every change is versioned and appended to
    `trading_halt_events`.
  - Application: `GET /api/system/halt-status` (also `account_halt` in
    `/api/system`) lists each path as `applied` / `lagging` / `missing` /
    `unavailable`; the halt is only `applied: true` once every path has
    acknowledged the current version at a quiescent point (Alpaca daemons:
    top of cycle; FX: 60 s health tick, deferred while an entry placement is
    in flight) or reported itself unavailable.
  - Not yet covered (follow-up beads): cancelling opening orders still
    working at the broker when a halt lands, verified CLOSE_ONLY reductions,
    and EMERGENCY_FLATTEN. While `ALPACA_LEDGER_CLOSE_ONLY` is set the
    Alpaca daemons open no entries at all and acknowledge any halt at the
    top of each ledger cycle.
- **Broker/account-read interlock (CL-oqos — development, not yet deployed).**
  An unknown broker snapshot is never treated as flat:
  - Cold start: if `get_positions()` fails or returns a malformed snapshot
    (not a list, nonfinite/bool quantity, empty/blank/non-string symbol, two rows for one
    canonical symbol such as `USD_CAD` + `USDCAD`) the reconciler raises
    `SnapshotUnavailableError` BEFORE confirming/rejecting any pending event
    entry or flattening an "orphan"; the engine halts entries with the
    sticky cause `external:cold_start_snapshot_unavailable`. A book that
    fails to confirm its pending entries aborts the reconcile the same way
    (`external:cold_start_reconciliation_failed`) instead of flattening a
    real fill as an orphan.
  - Periodic alignment (300 s) treats the same failures as UNKNOWN (no
    mismatch streak, no `reconciliation_failure` trip); the event book's
    per-tick reconcile applies the same whole-snapshot validation, so it
    never prunes, promotes or rejects a pending leg against a malformed
    snapshot. If the submit-time entry baseline cannot be read (broker
    raises or snapshot malformed) the event strategy opens NO new entries
    that tick and leaves the assessed events un-transitioned for retry, so a
    co-holder's position can never be claimed as our fill.
  - Health tick (60 s): three consecutive `get_account()` failures (or
    nonfinite equity) record the sticky cause
    `external:account_snapshot_unavailable` and halt entries; every further
    failed tick re-applies it. A successful read resets the counter but
    does NOT lift the halt — auto-resume never clears `external:*`
    causes; only `POST /api/system/resume` does (CL-d7ex). Resuming while
    reads still fail re-halts on the next tick.
  - Where to see why: `GET /api/system` now also returns `halt_causes`
    (active kill-switch + `external:*` causes), `account_read_failures`
    (consecutive count) and `last_reconciliation`
    (`{source: cold_start|alignment, status: ok|mismatch|unavailable|failed,
    reason, at}`); each is `null` when unwired. Existing fields are
    unchanged. The fleet watchdog's ENGINE HALTED page quotes these causes
    when present, falling back to the engine-log kill-switch scrape.
  - Restart rule: a restart does not fix an account-read outage — the cold
    start re-halts if the snapshot is still unavailable. Verify reads
    recovered (`account_read_failures: 0`, `last_reconciliation.status: ok`)
    before an attributed resume.
- OANDA practice dashboard: fxTrade Practice login shows positions/history.
- Alpaca paper dashboard: app.alpaca.markets (paper) shows option positions.
- Telegram: digests (grounded trade cards, niche 🎯 tags, red-team bear
  cases, Polymarket shift alerts), gate approvals (`approve <id>`), `ideas`.
  - **Approver allowlist (CL-esh6):** `approve` / `reject` / `skip` are
    accepted only from Telegram USER ids listed in `TELEGRAM_APPROVER_IDS`
    (comma-separated, e.g. `TELEGRAM_APPROVER_IDS=123456789`; find your id
    from the `from.id` of any message the bot receives, or the refusal log
    line). **Unset = default deny**: the bot logs one WARNING at startup that
    approvals are disabled and refuses every mutating command (use
    `scripts/research_approve.py` meanwhile). A malformed value (non-integer,
    zero, or a negative chat id) makes the bot exit 2 at startup. Refused
    attempts log `refusing '<verb>' from unauthorized sender id=<id>` and
    change nothing. `help` / `pending` / `ideas` / `idea` stay open to anyone
    in `TELEGRAM_CHAT_ID` (read-only). **Deploy consequence:** set
    `TELEGRAM_APPROVER_IDS` before restarting `telegram_approval_bot`, or
    Telegram approvals stop working.
- Grafana/Prometheus/Loki via `docker compose --profile full`.
- **Prometheus `/metrics` bind (CL-esh6):** the engine's metrics endpoint
  (`CURLIT_METRICS_PORT`, default 8099) binds `METRICS_BIND_ADDR`, default
  `127.0.0.1` (it was every interface). Local checks (`curl
  localhost:8099/metrics`, `health_watch`, `soak_monitor`) are unaffected. To
  expose it deliberately, set `METRICS_BIND_ADDR` (e.g. `0.0.0.0`) — only behind a
  firewall or a loopback-published container port; `docker-compose.app.yml`
  sets `0.0.0.0` inside the engine container for exactly that reason. The
  engine logs a WARNING at startup whenever the bind is non-loopback.
  Prometheus-in-Docker scrapes the native engine via `host.docker.internal:8099`;
  on Docker Desktop (macOS) that reaches the host's loopback, but on a Linux
  host (`host-gateway`) a loopback bind is NOT reachable from the container —
  set `METRICS_BIND_ADDR` to the bridge gateway address there. Verify the
  `fx_live_engine` target is UP in Prometheus after the first restart.
- **claude CLI child environment (CL-esh6):** research/event LLM calls spawn
  `claude -p` with an explicit environment allowlist (PATH, HOME, USER,
  LOGNAME, SHELL, TERM, TMPDIR, LANG, LC_*, TZ, XDG_CONFIG_HOME,
  CLAUDE_CONFIG_DIR, CLAUDE_SECURESTORAGE_CONFIG_DIR, CLAUDE_CODE_OAUTH_TOKEN,
  CLAUDE_CODE_MAX_OUTPUT_TOKENS, proxy/CA variables). Broker, DB, Telegram,
  Moonshot and web-API secrets — and `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` —
  never reach it. If a host needs another variable for the CLI, add it to
  `_CLI_ENV_ALLOWLIST` in `src/research/llm/claude_code.py` deliberately.
- **Strategy registry (CL-nix8):** `configs/live_portfolio.yaml` entries are
  routed by their `class:` path (or, without one, by the four canonical ids)
  through `STRATEGY_CLASS_REGISTRY` / `STRATEGY_ID_REGISTRY` in
  `src/runtime/run_engine.py` — no longer by id substring. An enabled entry
  with an unregistered class, an unknown id without `class:`, a duplicate id,
  or an id/class mismatch now **fails engine startup** with
  `StrategyConfigError` (previously misrouted or silently skipped). The live
  four resolve exactly as before. A PROMOTE-registered paper-shadow entry
  (`src/research/promote.py`) needs a registry entry + builder before it is
  enabled, or set `enabled: false` to park it.

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
