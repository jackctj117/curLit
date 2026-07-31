# BOOTSTRAP — fresh-device bring-up, from `git clone` to trading

_A new machine (or a new agent) should get from zero to the full paper-trading
fleet with this checklist. Companion docs: `CURRENT_OPERATIONS.md` (what runs
and why), `BOOT.md` (engine-centric details), `.env.example` (every knob)._

**Fast path:** the deterministic, key-independent parts of steps 1 + 3 are one
command — `make bootstrap` (venv + deps + spaCy + Postgres on `127.0.0.1:5432`
+ migrations), which then prints the key-dependent next steps. After you fill
`.env` (step 2), `make backfill` runs step 4. The steps below are the manual
equivalents / the full detail — read them to understand what those targets do
and for the parts they don't cover (keys, Claude login, the fleet).

## 0. Prerequisites
- **bd (beads)** issue tracker on PATH — CLAUDE.md mandates it for ALL task
  tracking (`bd prime` for workflow). Install per its README, then `bd ready`.
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
python3 -m venv .venv                    # Makefile does NOT create the venv
.venv/bin/pip install -e ".[dev]"        # core + test deps
.venv/bin/python -m spacy download en_core_web_sm   # NLP preprocessing model
                                         # (CB-sentiment / scraper paths need it)
.venv/bin/pytest tests/unit -q           # must be green BEFORE any config
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
# The native daemons connect to 127.0.0.1:5432, so the DB must be published to
# the host loopback. Two equivalent options — run exactly ONE (they share 5432):
#   (a) compose: `docker compose up -d postgres` (now publishes 127.0.0.1:5432)
#   (b) standalone soak container (what the original box runs):
docker run -d --name curlit-postgres-soak -p 127.0.0.1:5432:5432 \
  -e POSTGRES_USER=fx -e POSTGRES_PASSWORD=<your POSTGRES_PASSWORD> \
  -e POSTGRES_DB=fx -v curlit_pgdata:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg15
# (or point POSTGRES_*/DATABASE_URL at any existing TimescaleDB)
.venv/bin/python -m migrations.run   # 17 migrations, idempotent
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

**Boot persistence (CL-rmvt):** a reboot (macOS update, power loss) erases
the nohup fleet — install the launchd agent so it self-heals at login:
```bash
cp deploy/launchd/com.curlit.fleet.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.curlit.fleet.plist
```
It runs `scripts/boot_recovery.sh` (wait for Docker → ensure the DB
container → idempotent `daemons.sh start`; log: `logs/boot_recovery.log`).
Manual trigger: `launchctl kickstart gui/$(id -u)/com.curlit.fleet`. Also
give the DB container a restart policy once:
`docker update --restart unless-stopped curlit-postgres-soak`.

**No OANDA yet?** The engine's `--broker oanda-practice` fail-fasts without
credentials (by design — no silent fake broker). For a keys-less smoke, run the
engine directly on the paper broker instead of the fleet:
`.venv/bin/python -m src.runtime.run_engine --broker paper` (or, to keep the
`oanda-practice` line but downgrade when creds are absent, set
`ALLOW_PAPER_FALLBACK=1`). The other daemons idle cleanly until their keys
exist — GDELT still feeds events without X/Reddit, etc.

## 7. Ongoing
- `bd ready` for open work; `docs/CURRENT_OPERATIONS.md` for policies/knobs.
- Restarts: `./scripts/daemons.sh restart` — or `restart <name>` for one
  daemon (verifies the pid actually changed, CL-obgy). Engine restarts are
  position-safe since CL-8s1e.
- State that stays on the device: `.env` (secrets), `data/*.json`
  (event book, kill-switch, research state), the Postgres volume, `logs/`.
  Migrating devices = copy `.env` + pg_dump the DB (or just re-backfill —
  only `idea_outcomes`/`options_activity` history is genuinely lost).
