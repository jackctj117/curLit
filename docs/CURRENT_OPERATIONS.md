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
- **Other engine strategies**: rate_diff is DATA-FED (n≈252 daily rows) but
  SIGNAL-GATED — it refuses to trade while rolling R² < 0.25 (currently far
  below; refits weekly). carry_vol has its OIS/CVIX inputs and rebalances
  monthly. cb_sentiment awaits a detected CB shift. This is fail-closed by
  design, not breakage — see `scripts/data_health.py`.

### 1b. Alpaca paper — machine-bought options on advisory ideas
- **Account**: Alpaca paper (`ALPACA_API_KEY/SECRET`), $100,000, options
  level 3. Executor: `scripts/execute_options.py --loop 300` (market-hours
  aware; a clean no-op when closed).
- **Policy (paper phase — deliberately widened to build a sample, CL-ldd2)**:
  - pool: pending `buy_calls`/`buy_puts` ideas, confidence ≥ **0.45**
    (`ALPACA_OPT_MIN_CONFIDENCE`); niche/red-team gates OFF
    (`ALPACA_OPT_REQUIRE_NICHE/RED_TEAM=0`) — re-tighten before any
    real-money move;
  - each day it takes the TOP-5 by confidence (`ALPACA_OPT_MAX_PER_DAY`),
    1 contract each (`ALPACA_OPT_QTY`), skipping any contract whose premium
    exceeds $500 (`ALPACA_OPT_MAX_PREMIUM`);
  - contract selection parses the idea's moneyness band + DTE window
    ("slightly OTM ~5%, 3-5 weeks") into a target strike/expiry and picks
    the nearest LISTED contract (nearest expiry, then strike);
  - **technical-alignment gate** (CL-3xoj): computed price structure that
    scores below `ALPACA_OPT_MIN_ALIGNMENT` (−0.4) against the thesis
    skips the idea (no calls into a confirmed downtrend at the lows);
    fail-open when no history is computable;
  - dedup via `alpaca_option_orders` (migration 014) keyed by idea_id;
    terminal decisions recorded, transient misses retried.
- **Sizing philosophy**: 1 contract, breadth over size — edge measurement is
  size-independent; more distinct positions = more information, bigger
  positions = only more variance.

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
                    check_ticker / resolve_company / profile / SEC 10-K
                    tools mid-generation (API-billed, Moonshot)  [CL-ddzt]
                    → SymbolUniverse verification (13k US-listed + SEC
                      names; unverified tickers DROPPED)       [CL-tzug/9xha]
                    → asymmetry scoring + $-volume liquidity floor
                    → RED-TEAM critic (claude-code) attacks every
                      survivor; refuted ideas die, survivors carry the
                      bear case + lowered confidence            [CL-3v56]
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

Grounding injected into the niche pass per candidate (CL-2czc/3xoj/mtum):
SEC 10-K risk/business excerpt (TOC-skipping) · yfinance profile ·
**computed technicals** (trend vs 20/50d SMA, swing S/R in dollars,
breakout state incl. approaching_*, level-test counts, volume ratio — never
pattern names; a test bans "flag/head/shoulders/wedge") · **options
activity** (P/C volume skew, volume vs self-built baseline, ATM IV).

---

## 3. The nine daemons

| Daemon | Command | Cadence | Log |
|---|---|---|---|
| Trading engine | `CURLIT_RISK_PROFILE=aggressive .venv/bin/python -m src.runtime.run_engine --broker oanda-practice` | async loop | `logs/engine_stdout.log`, `logs/live_engine.jsonl` |
| Event pipeline | `scripts/event_pipeline.py --ingest --assess --loop 900` | 15 min | `logs/event_pipeline.log` |
| Intraday pricer | `scripts/intraday_pricer.py --loop 120` | 2 min + daily candle backfill | `logs/intraday_pricer.log` |
| Foreign rates | `scripts/refresh_rates.py --loop 86400` | daily | `logs/refresh_rates.log` |
| X monitor | `scripts/x_monitor.py --loop 300` | 5 min (tiered priorities) | `logs/x_monitor.log` |
| Options executor | `scripts/execute_options.py --loop 300` | 5 min (market-hours aware) | `logs/execute_options.log` |
| Options activity | `scripts/options_activity.py --loop 86400` | daily chain snapshots | `logs/options_activity.log` |
| Outcome scorer | `scripts/score_outcomes.py --loop 86400` | daily | `logs/score_outcomes.log` |
| Telegram bot | `scripts/telegram_approval_bot.py` | long-poll | `logs/` |

All launch as `nohup … & disown` from the repo root. Every entrypoint runs
the interpreter-health canary (pyexpat, CL-169t) and the engine additionally
runs the DATA-HEALTH preflight (CL-q4n1) at boot — a starved series is a
loud WARN banner, never a silent dormant strategy.

**Restart rules**
- Engine: safe with open positions since CL-8s1e; still check
  `/api/positions` first out of caution. Always preserve
  `CURLIT_RISK_PROFILE=aggressive`.
- Pipeline/monitors: restart freely; state (since_ids, event book, research
  state) persists in files/DB.

---

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

---

## 5. LLM stack & cost posture

| Role | Model | Billing |
|---|---|---|
| Triage | claude-haiku-4-5 (one batched call/cycle) | subscription |
| Impact agent | claude-sonnet-4-6 | subscription |
| Niche discovery | **kimi-k3 agentic tool-loop** (~13-15k tok/event, a few cents) | **Moonshot API (paid)** — `NICHE_TOOL_AGENT_ENABLED=0` reverts to free claude-code cycles |
| Red-team critic | claude-sonnet-4-6 (one batched call/event) | subscription |
| Reflective review | claude-sonnet-4-6 (weekly) | subscription |
| Research pipeline | claude-fable-5 | subscription |

Grok: no subscription-billed API path exists; deliberately not used.
Cost controls: triage relevance gate, niche urgency ≥ 7 gate,
`KIMI_MAX_ITERATIONS`, red-team batching, per-day Alpaca caps.

---

## 6. Monitoring & manual controls

- Web API (engine-embedded, port 8200): `/health` (open), `/docs`, and
  authenticated `/api/{account,positions,pnl,signals,system}?secret=<WEB_API_SECRET>`;
  `POST /api/system/halt` + `/resume` for an emergency stop.
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
