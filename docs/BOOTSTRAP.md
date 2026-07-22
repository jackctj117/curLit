# BOOTSTRAP — fresh-device bring-up, from `git clone` to trading

_A new machine (or a new agent) should get from zero to the full paper-trading
fleet with this checklist. Companion docs: `CURRENT_OPERATIONS.md` (what runs
and why), `BOOT.md` (engine-centric details), `.env.example` (every knob)._

## 0. Prerequisites
- macOS/Linux, **Python ≥ 3.11** (Homebrew `python@3.14` works; run
  `python3 -c "import pyexpat"` first — a broken brew bottle here crashes XML
  ingest, see CL-169t), Docker, git.
- **Claude Code CLI logged in** (`claude` on PATH) — every LLM role
  (triage/impact/niche-cycles/red-team/research) rides the operator's
  subscription headlessly. No `claude` login = no assessments.
- Accounts/keys you'll need for full function (each degrades gracefully if
  absent): OANDA practice, Alpaca paper, FRED key, Telegram bot,
  Moonshot/Kimi key (optional — niche falls back to claude-code),
  Reddit app (blocked on their approval), ENTSO-E token (optional).

## 1. Clone + install
```bash
git clone <repo> && cd curLit
make install                 # creates .venv + deps
.venv/bin/pytest tests/unit -q   # must be green BEFORE any config
```

## 2. Environment
```bash
cp .env.example .env         # then fill:
```
**Required for core operation**: `POSTGRES_*` (or `DATABASE_URL`),
`OANDA_API_KEY` + `OANDA_ACCOUNT_ID` + `OANDA_PRACTICE=true`,
`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, `FRED_API_KEY`,
`WEB_API_SECRET` (any random string), `CURLIT_RISK_PROFILE=aggressive`
(soak convention).

**Feature gates as currently practiced** (see CURRENT_OPERATIONS §1/§5):
`EVENT_TRIAGE_ENABLED=1`, `NICHE_MAX_CYCLES=2`, `NICHE_TOOLS_ENABLED=1`,
`NICHE_CRITIC_ENABLED=1`, `NICHE_TOOL_AGENT_ENABLED=1` + `MOONSHOT_API_KEY`
(paid; omit both to stay subscription-only), `ALPACA_API_KEY/SECRET` +
`ALPACA_OPTIONS_ENABLED=1` + the `ALPACA_OPT_*` paper-phase knobs,
`SEC_EDGAR_USER_AGENT` (browser-shaped, with your contact).

**Machine-specific / optional**: the X monitor's `cli` backend needs the
`bird` CLI + X cookies on the device (ToS/ban risk — burner account only);
without it set `X_MONITOR_BACKEND=api` + a paid `TWITTER_BEARER_TOKEN`, or
leave unset and the monitor idles (GDELT still feeds events).
`REDDIT_CLIENT_ID/SECRET` only after Reddit approval (CL-okww).

## 3. Database + schema
```bash
docker compose up -d postgres        # or point POSTGRES_* at an existing PG
.venv/bin/python -m migrations.run   # 15 migrations, idempotent
```

## 4. Backfills (order matters; all idempotent, all re-runnable)
```bash
.venv/bin/python -m scripts.refresh_symbols     # 13k US-listed + SEC names
.venv/bin/python scripts/refresh_rates.py --once    # DE2Y/OIS/CVIX (~380d)
.venv/bin/python scripts/intraday_pricer.py --once  # quotes + daily candles
# daily prices/macro history: enable the Airflow fx_daily DAG (compose
# --profile full) or run it manually — the engine tolerates a short gap.
.venv/bin/python scripts/data_health.py         # MUST end "no data-starved"
```

## 5. Verify before launching
```bash
.venv/bin/pytest tests/unit -q      # green
.venv/bin/python scripts/data_health.py   # exit 0
# one live event cycle end-to-end:
.venv/bin/python scripts/event_pipeline.py --ingest --assess --once
```

## 6. Launch the fleet
```bash
./scripts/daemons.sh start    # idempotent; logs to logs/<name>.log
./scripts/daemons.sh status   # all ✓ (reddit stays commented until approved)
curl -s localhost:8200/health # {"status":"ok"}
```
Engine boot log must show the DATA-HEALTH banner and
`risk profile active: aggressive`.

## 7. Ongoing
- `bd ready` for open work; `docs/CURRENT_OPERATIONS.md` for policies/knobs.
- Restarts: `./scripts/daemons.sh stop && ./scripts/daemons.sh start`
  (engine restarts are position-safe since CL-8s1e).
- State that stays on the device: `.env` (secrets), `data/*.json`
  (event book, kill-switch, research state), the Postgres volume, `logs/`.
  Migrating devices = copy `.env` + pg_dump the DB (or just re-backfill —
  only `idea_outcomes`/`options_activity` history is genuinely lost).
