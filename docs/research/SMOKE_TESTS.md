# Research Pipeline — Real-Network Smoke Tests

The unit + integration suites verify the pipeline's wiring with mocked
LLMs, mocked HTTP, and synthetic data. They prove the wires connect to
each other; they do **not** prove the wires connect to anything real.

These three smoke tests close that gap. Each requires real credentials
+ network and is run **manually** by the operator before the loop is
trusted in production. None of them are on CI; none of them are on
cron.

## Preflight: `--dry-run` (no creds, no network)

Before running any of the smokes below, run the dry-run preflight
to confirm the pipeline wires up cleanly:

```bash
.venv/bin/python -m scripts.research_loop --dry-run --auto-approve
```

Expected output:

```
DRY RUN: stubbed LLM + HTTP + backtest; outputs under /tmp/research-dry-run-XXXX
Run done — extracts_new=1 ideas_proposed=1 declined=0 implemented=1
  impl_rejected=0 debates=1 PROMOTE=1 REJECT=0 ESCALATE=0 errors=0
```

The dry-run uses `DryRunDriver` (canned LLM responses by prompt
fingerprint), `stub_http_get` (canned arXiv Atom XML), and
`stub_backtest_runner` (synthetic metrics that pass every gate).
Outputs land under a tmp dir; the production tree (`src/strategies/`,
`configs/live_portfolio.yaml`, etc.) is never touched. Use this any
time you change agent prompts, the loop's wiring, or sub-component
interfaces.

If the dry-run errors on a wiring bug, fix it before spending real
LLM tokens on `CL-77rq`.

## CL-77rq — full pipeline against live arXiv + real LLMs

**What it proves:** the loop survives real arXiv Atom XML, real
LLM responses with their actual phrasing under temperature=0, and
real DataProvider data fetches against Postgres.

**Required env vars:**

```bash
# At least one of (Claude is the strongest default for code gen)
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=sk-...
# Optional (only if you flip an agent to grok in research_agents.yaml)
export XAI_API_KEY=xai-...

# Postgres (the real backtest_runner needs DataProvider; see
# src/runtime/run_engine.py:_build_db_engine for the fallback chain)
export POSTGRES_PASSWORD=...           # if using the default user/host
# OR
export DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
```

**Optional (skip the gates so it runs single-pass):**

```bash
# --auto-approve flips PENDING → APPROVED automatically. SAFE here
# because skip_git=True + skip_pr=True keep nothing from being
# pushed; the registrar runs locally, mutates a tmp portfolio.yaml,
# and exits.
```

**Command:**

```bash
.venv/bin/python -m scripts.research_loop \
    --auto-approve \
    --backtest-start 2018-01-01 --backtest-end 2024-12-31 \
    --state /tmp/research-smoke/state.json \
    --runs-dir /tmp/research-smoke/runs \
    --hypothesis-dir /tmp/research-smoke/hypotheses \
    --candidate-dir /tmp/research-smoke/candidates
```

**What to look for:**

1. **Ingest phase**: `ingest_summary.papers_extracted >= 1`. If 0,
   the arXiv Atom parser hit an edge case — check the run log under
   `data/research/runs/{ts}.json`.
2. **Idea phase**: at least one extract converted to PROPOSED. If
   everything DECLINED, the idea-generator prompt's regex requirements
   (rule-IDs, prediction sub-fields) may be too strict for real LLM
   output — open a hypothesis brief in `/tmp/research-smoke/hypotheses/`
   and inspect.
3. **Implementer**: at least one IMPLEMENTED. REJECTED with
   "syntax gate failed" means the LLM emitted code that doesn't
   compile; "no python code block" means the prompt didn't constrain
   format strictly enough.
4. **Debate**: bull/bear should declare PROMOTE/REJECT/ABSTAIN
   cleanly. If `parse_position` defaults to ABSTAIN, the regex didn't
   match — check the transcript at
   `docs/research/debates/{slug}/transcript.md` and grep for
   `**FINAL_POSITION**`.
5. **Verdict**: at least one of {PROMOTE, REJECT, ESCALATE} should
   fire. Most-likely outcome on a first run: ESCALATE (real metrics
   often fail one of the threshold rules).

**Cost guardrail:** worst-case the loop processes ~50 arXiv
abstracts × 5 agents × ~2k tokens × Claude rates = ~$5–10. Set
`max_results=5` in `configs/paper_streams.yaml` for a cheaper first
run.

**Cleanup:** `rm -rf /tmp/research-smoke/`. The production tree
stays clean since `--auto-approve` doesn't push and we routed
outputs to `/tmp/`.

## CL-2uns — verify `gh pr create` URL extraction

**What it proves:** `PromoteRegistrar._open_pr` correctly extracts
the PR URL from real `gh pr create` stdout. Today the code assumes
the URL is on the last line; this is typical but unverified.

**Required:**

- `gh` CLI installed + authenticated (`gh auth login`)
- A throwaway test branch + repo (don't run on main)

**Manual procedure:**

```bash
# 1. Create a throwaway branch with a trivial change
git checkout -b experiment/smoke-test-pr-extraction
echo "# smoke" >> /tmp/smoke.md
git add /tmp/smoke.md  # or any test file
git commit -m "smoke: test PR URL extraction"
git push -u origin experiment/smoke-test-pr-extraction

# 2. Capture gh stdout exactly as the registrar would see it
gh pr create \
    --title "smoke: PR URL extraction test" \
    --body "Smoke test for CL-2uns. Close immediately." \
    --head experiment/smoke-test-pr-extraction \
    > /tmp/gh-output.txt 2>&1
cat /tmp/gh-output.txt

# 3. Apply the registrar's extraction logic to the captured stdout
.venv/bin/python -c "
out = open('/tmp/gh-output.txt').read()
url = out.strip().splitlines()[-1].strip()
print(f'Extracted: {url!r}')
assert url.startswith('https://github.com/') and '/pull/' in url, (
    f'extraction broken: {url!r}'
)
print('OK')
"

# 4. Close the throwaway PR + delete the branch
gh pr close --delete-branch experiment/smoke-test-pr-extraction
```

**What to file if it fails:** if step 3 fails, file a bead under
CL-2uns saying "switch from last-line to regex match for
`https://github.com/.*/pull/\d+`" and patch
`src/research/promote.py:_open_pr`.

## CL-b7i3 — Telegram + Telegram delivery smoke

**What it proves:** `notify_operator` actually delivers messages on
both channels and the Markdown formatting renders correctly when
the message body contains underscores in slugs (a known Telegram
parse-mode footgun).

**Required env vars:**

```bash
export TELEGRAM_BOT_TOKEN=...
export PUSHOVER_USER_KEY=...
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
```

**Command:**

```bash
.venv/bin/python -c "
from src.research.notifications import notify_operator
result = notify_operator(
    title='SMOKE: GATE 1 alert preview',
    message=(
        'Slug: regime_carry_underscores_in_slug\n'
        'Hypothesis: docs/research/hypotheses/regime_carry.md\n\n'
        'Approve via:\n'
        '  python -m scripts.research_approve --slug regime_carry --action GO'
    ),
    priority=0,
)
print(f'Telegram attempted={result.pushover_attempted} '
      f'succeeded={result.pushover_succeeded} '
      f'error={result.pushover_error!r}')
print(f'Telegram attempted={result.telegram_attempted} '
      f'succeeded={result.telegram_succeeded} '
      f'error={result.telegram_error!r}')

# Priority=1 (deploy decision wake-up)
result = notify_operator(
    title='SMOKE: GATE 2 alert preview',
    message='Strategy ready for paper-shadow at allocation=0.',
    priority=1,
)
print(f'priority=1: pushover={result.pushover_succeeded} '
      f'telegram={result.telegram_succeeded}')
"
```

**What to verify on the device:**

1. Both messages arrive on Telegram.
2. Both messages arrive on Telegram.
3. **Underscores in `regime_carry_underscores_in_slug` render as
   underscores, not as italic-toggle marks.** Telegram's Markdown
   `parse_mode` interprets `_text_` as italic. If the message looks
   weirdly formatted, the dispatcher needs to switch to MarkdownV2
   (with backslash-escaped underscores) or HTML mode. File under
   CL-b7i3.
4. Priority=1 message wakes the device (Telegram's "high priority"
   bypasses quiet hours) — confirm by running the smoke during a
   normally-quiet time.
5. Telegram bot's chat shows both messages even if you've muted the
   chat.

**Cost:** Telegram free tier is 10k messages/month. Telegram is
free. No real cost.

## After all three smokes pass

You're ready for paper trading: real LLM responses parse cleanly,
real arXiv data shapes don't break the ingester, the registrar can
open real PRs, and operator alerts actually arrive. At that point
re-read `RUNBOOK.md` section 8 (Safety boundaries) — `allocation=0`
+ G9 + manual allocation flip remain the firewall against real-money
risk.
