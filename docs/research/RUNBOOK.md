# Research Pipeline — Operator Runbook

This runbook is what an operator reads to actually run, monitor, and
extend the multi-agent research pipeline (`CL-h986`). It is the
companion to the technical doctrine docs:

- [`docs/research/EVIDENCE_FIRST.md`](EVIDENCE_FIRST.md) — operating
  principle every agent inherits
- [`docs/research/REVIEW_RULES.md`](REVIEW_RULES.md) — the gates the
  verdict engine evaluates
- [`docs/research/SMART_QUESTIONS.md`](SMART_QUESTIONS.md) — the format
  agents must use for blocker-resolving questions
- [`docs/research/SMOKE_TESTS.md`](SMOKE_TESTS.md) — real-network
  smokes (manual, operator-driven) before trusting the loop in
  production

Where this doc says "the loop", that's
`src/research/loop.py:ResearchLoop` invoked via
`python -m scripts.research_loop`.

## 1. What the loop does and where each phase produces output

```
                                                    GATE 1 (operator)
                                                            │
   feeds → ingest → extract.md → idea.md  ─────────► hypothesis pending
   data/research/extracts/    docs/research/hypotheses/        │
                                                               │ approved
                                                               ▼
                                                          implementer
                                                               │
                                                ┌──────────────┼──────────────┐
                                                ▼              ▼              ▼
                                       _experimental/  reports/candidates/  candidate report
                                                               │
                                                            debate
                                                               │
                                              docs/research/debates/{slug}/transcript.md
                                                               │
                                                          verdict engine
                                                               │
                                          ┌────────────────────┼────────────────────┐
                                          ▼                    ▼                    ▼
                                     PROMOTE              REJECT               ESCALATE
                                          │                    │                    │
                                  GATE 2 (operator)       archived           pushover/telegram
                                          │
                                  paper-shadow registrar
                                          │
                              src/strategies/{slug}.py + PR
                              configs/live_portfolio.yaml (allocation=0)
```

| Phase | Input | Output | Module |
|-------|-------|--------|--------|
| Ingest | `configs/paper_streams.yaml` (feeds list) — arXiv (Atom), substacks/blogs (RSS), Polymarket (JSON) | `data/research/extracts/{paper_hash}.md` | `src/research/ingest.py` + `polymarket.py` |
| Idea | extracts/`*.md` | `docs/research/hypotheses/{slug}.md` | `src/research/agents/idea.py` |
| GATE 1 | hypothesis brief | state flip APPROVED/SKIPPED | `src/research/loop.py` |
| Implementer | hypothesis brief | `src/strategies/_experimental/{slug}.py` + `reports/candidates/{slug}.json` | `src/research/agents/implementer.py` |
| Backtest | strategy file + DataProvider | metrics dict (oos_metrics + sharpe_ci_95 + ...) | `src/research/backtest_runner.py` |
| Debate | candidate report + REVIEW_RULES.md | `docs/research/debates/{slug}/transcript.md` + `transcript.jsonl` | `src/research/orchestrator.py` |
| Verdict | candidate report + agent positions | PROMOTE / REJECT / ESCALATE | `src/research/verdict.py` |
| GATE 2 | PROMOTE verdict | state flip DEPLOY_APPROVED/REJECTED | `src/research/loop.py` |
| Paper-shadow | DEPLOY_APPROVED entry | `src/strategies/{slug}.py` move + `configs/live_portfolio.yaml` entry + GH PR | `src/research/promote.py` |

State for cross-run idempotency: `data/research/state.json`. Per-run
summary: `data/research/runs/{ts}.json`.

### Currently-configured feeds (9)

| Adapter | Feed | What it pulls |
|---|---|---|
| arxiv | `arxiv_qfin_pm` | Portfolio-management preprints |
| arxiv | `arxiv_qfin_tr` | Trading & microstructure preprints |
| rss | `alpha_architect` | Quant-research blog summaries |
| rss | `newfound_research` | Flirting with Models — regime / trend research |
| rss | `doomberg` | Energy / macro contrarian substack |
| rss | `the_diff` | Byrne Hobart's capital-markets / tech essays |
| rss | `concoda` | Funding-markets / macro substack |
| rss | `macro_hive` | Institutional FX/rates research |
| polymarket | `polymarket_macro` | Active prediction markets, filtered to FX/macro keywords |

Idea agent will DECLINE most blog/substack content — that's
correct, conversational commentary mostly isn't a falsifiable
thesis. Curation principle: small, signal-dense, FX-relevant.

## 2. How to read a debate transcript

Every debate produces two files under `docs/research/debates/{slug}/`:

- **`transcript.md`** — human-readable. Header lists participants and
  rounds; one block per agent invocation with timestamp, model, token
  cost, and the agent's full markdown response. Footer records each
  reviewer's final position + total cost + open questions.
- **`transcript.jsonl`** — one JSON line per invocation. The verdict
  engine + the dashboard parse this; it is the audit trail of record.

Reading order:

1. Skip to the **footer summary** to see the bottom line — Bull and
   Bear positions and any unresolved smart-questions.
2. Read **Round 1** (`initial_positions`) — both agents' independent
   PROMOTE_CASE / REJECT_CASE markdown. Each rule cited by ID with the
   metric value or code line.
3. Read **Round 2** (`smart_questions`) — what unknowns each agent
   surfaced. Look for `routing: code_tool` (resolver ran a tool) vs
   `routing: human` (operator must answer).
4. Read **Round 3** (`rebuttal`) — each agent engages with the other's
   case. This is where genuine disagreement vs theatre disagreement
   shows up.
5. Read **Round 4** (`final_position`) — one-line `**FINAL_POSITION**:
   PROMOTE | REJECT | ABSTAIN`. The verdict engine reads exactly this
   keyword.

Common transcript anti-patterns to flag:

- **Rule citations without line numbers**: "look-ahead bias" without a
  file:line cite is a fabrication; bear should not pass on this.
- **Agreement without engagement**: if Round 3 is just both agents
  agreeing without contesting any cited rule, the asymmetric prompts
  failed; treat the verdict as ESCALATE in your own head even if the
  engine says PROMOTE.
- **Open questions still pending in Round 4**: the verdict engine
  ESCALATEs in this case, but read the questions yourself — they
  often reveal what the operator should look at next.

## 3. Interpreting an ESCALATE alert

The loop fires a Pushover priority-1 + Telegram alert when the verdict
engine returns ESCALATE. Body contains: slug, verdict reason, bull/bear
positions, problem rules with their detail strings, OOS metrics blurb,
and links to the transcript + candidate report.

**Most common causes**:

1. **Required metric missing or malformed** (`reason: "Required
   metric(s) missing or malformed in candidate report"`). The
   implementer's report omitted a field the rule expects. Fix: re-run
   implementer with a richer backtest_runner, or amend
   `REVIEW_RULES.md` to drop / loosen the rule.
2. **Mixed agent positions** (`reason: "Mixed agent positions: bull=X,
   bear=Y"`). Both agents engaged with the candidate but landed on
   different verdicts despite all gates passing. This is the highest-
   value escalation for operator judgment — read the rebuttal round.
3. **Unresolved smart-questions**. The agents flagged blockers the
   resolver couldn't dispatch (typically `routing: human`). Open the
   transcript to see what they need.
4. **Threshold rule passed but agents both ABSTAIN**. The verdict
   engine treats this as REJECT, not ESCALATE — but if you see it,
   the agents found qualitative concerns the rules don't capture. Read
   the rebuttal carefully before re-running.

**What to do when you get one**:

```bash
# See what's pending operator action across both gates
python -m scripts.research_approve --gate=2 --list

# Open the transcript at the path in the alert
open docs/research/debates/{slug}/transcript.md

# Either re-promote (if you read the transcript and you're satisfied):
python -m scripts.research_approve --gate=2 --slug {slug} --action GO

# Or skip with a reason:
python -m scripts.research_approve --gate=2 --slug {slug} --action SKIP \
    --reason "regime concentration concern not in rules"
```

ESCALATE alerts dedupe automatically — a slug only enters the debate
loop once across runs, so you won't get spammed if you defer the
decision.

## 4. Adding a new agent

YAML edit only. No Python changes anywhere else.

1. Drop a system-prompt markdown file at
   `configs/research_prompts/{role}.md`. Read existing prompts
   (`bull_reviewer.md`, `bear_reviewer.md`,
   `implementer.md`, `idea_generator.md`,
   `paper_extractor.md`) to match the format. Always reference
   `docs/research/EVIDENCE_FIRST.md` in a "Foundational principle"
   section — agents that skip evidence-first produce vibes, and the
   operator gets to debug the resulting bad strategy.
2. Add an entry under `agents:` in `configs/research_agents.yaml`:

   ```yaml
   agents:
     my_new_agent:
       provider: claude       # must match a registered driver
       role: my_new_role      # free-form label
       prompt_path: configs/research_prompts/my_new_agent.md
       max_tokens: 4096
       temperature: 0.0
   ```
3. If the agent participates in a debate, add it to that debate's
   `participants:` list.
4. Smoke-test by loading the config:

   ```bash
   .venv/bin/python -c "
   from src.research.config import load_config
   cfg = load_config('configs/research_agents.yaml')
   print(cfg.agents['my_new_agent'].provider)
   "
   ```

If the new agent needs custom post-processing (e.g. parsing a position
keyword, validating output structure), write a subclass of
`src.research.agents.base.Agent` next to the existing reviewer/idea/
implementer subclasses. Most agents won't need this.

## 4b. Strategy contract for the Implementer

Strategies generated by the Implementer agent run through
`WalkForwardRunner` via `src/research/backtest_runner.py`. The
contract (CL-40n2 v1):

```python
class MyStrategy:
    # Class-level, NOT @property. Backtest harness reads it
    # without instantiating.
    symbols: list[str] = ['EURUSD', 'USDJPY', 'DXY']

    # Optional. Defaults to symbols[0]. The pair whose price
    # returns drive P&L. Use this when you read multiple
    # symbols but trade ONE.
    execution_symbol: str = 'EURUSD'

    def __init__(self, config=None):
        # Zero-arg constructable — walk-forward does
        # strategy_factory() with no args. Do NOT take a
        # DataProvider, broker, db engine, or any curLit
        # internal as a constructor argument.
        ...

    def fit(self, train_data: pd.DataFrame) -> None:
        # train_data: wide DataFrame with one column per declared
        # symbol PLUS a 'close' alias = train_data[execution_symbol].
        # DatetimeIndex, forward-filled across timestamp gaps.
        # Fit only on this in-sample slice. No look-ahead.
        ...

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Return a numeric position series indexed by data.index.
        # Walk-forward shifts(1) so signal[t] applies to return[t+1].
        # Must be non-zero on a meaningful fraction (≥5-10%) of
        # bars — strategies that don't trade get REJECTED at the
        # backtest gate.
        ...
```

**What the harness can run:**

- Single-asset time-series strategies (momentum, mean-reversion,
  regime-switching, vol-targeting). `generate_signals` returns
  `pd.Series`, position interpreted as weight on `execution_symbol`.
- Cross-symbol signal strategies (read DXY + rates; trade EURUSD).
- **Joint multi-asset / cross-sectional strategies (CL-40n2 v2):**
  `generate_signals` returns a `pd.DataFrame` with one column per
  tradeable pair holding position weights; walk-forward computes
  per-symbol returns and aggregates to portfolio P&L. Use for
  long-short cross-sectional, equal-risk allocation across pairs,
  hierarchical clustering selection.
- **Prediction-market-feature strategies (CL-3t4j v2):** declare
  ``symbols=['EURUSD', 'POLY:fed-cut-jun-2026']`` to read implied
  probabilities alongside FX prices. The market list is operator-
  curated in ``configs/polymarket_markets.yaml`` and seeded into
  the ``prices`` table via ``python -m scripts.seed_polymarket_history``.
  POLY columns are read-only feature inputs; can be combined with
  either Series or DataFrame signal output.

**What the harness can't yet run** (idea agent DECLINEs at brief stage):

- Anything needing OHLCV beyond close, vix, options IV, sentiment,
  news embedding, or other alt-data not in DataProvider.
- Polymarket markets NOT in ``configs/polymarket_markets.yaml`` —
  operator must add the market entry + run the seed script before
  a strategy can use it.
- Continuous rebalancing semantics with realistic per-pair costs —
  the v2 portfolio P&L uses one cost_per_turn for all pairs; a v3
  refinement could route per-pair via cost_model.get_cost_per_turn(symbol).

The full contract spec lives in
`configs/research_prompts/implementer.md`.

## 5. Adding a new debate type

Also YAML-only. The orchestrator (`src/research/orchestrator.py`)
loads the named debate config and runs its rounds.

```yaml
debates:
  cost_only_review:
    participants:
      - cost_reviewer
    rules_path: docs/research/COST_REVIEW_RULES.md
    verdict_engine: rule_based
    rounds:
      - name: initial_positions
        type: parallel
      - name: final_position
        type: parallel
```

Round types: `parallel` / `sequential` / `per_agent_async` (smart
questions). Hard caps still apply: 3 rebuttal rounds, 8 questions per
agent, 30s per code-tool resolution.

To run a different debate from the loop, override `debate_name` when
constructing the orchestrator — for now this means editing the loop
or running the orchestrator standalone.

## 6. Adding a new paper source

Three adapter types ship today: **arxiv** (Atom 1.0), **rss** (RSS
2.0 — substacks, quant blogs, news feeds), and **polymarket** (JSON
via Gamma API). Other sources plug in via the registry pattern in
`src/research/ingest.py:_FETCHER_REGISTRY`.

### Adding an RSS feed (most common — substack / blog)

YAML edit only. The generic RSSFetcher handles RSS 2.0 with optional
`<dc:creator>`, `<content:encoded>`, and `<pubDate>` namespaces:

```yaml
feeds:
  my_substack:
    adapter: rss
    query_url: https://newsletter.example.com/feed
    source_label: "Example Newsletter"
```

Then dry-run to verify:
```bash
.venv/bin/python -m scripts.ingest_papers --feed=my_substack --dry-run
```

### Adding an arXiv-style Atom feed

Same pattern, `adapter: arxiv`. Many academic preprint servers
(SSRN, NBER, RePEc) expose Atom-compatible endpoints.

### Adding a JSON / GraphQL source (e.g. another prediction market)

Write a fetcher class that implements `fetch(feed) -> list[Paper]`,
mirror `src/research/polymarket.py:PolymarketFetcher`. Use the
injected `http_get` callback for testability. Register in
`_FETCHER_REGISTRY`:

```python
_FETCHER_REGISTRY: dict[str, Callable[[HttpGet], Any]] = {
    "arxiv": lambda http_get: ArxivFetcher(http_get=http_get),
    "rss":   lambda http_get: RSSFetcher(http_get=http_get),
    "polymarket": lambda http_get: PolymarketFetcher(http_get=http_get),
    "kalshi": lambda http_get: KalshiFetcher(http_get=http_get),  # new
}
```

If the new module self-registers (like polymarket does via a
top-level `register()` call), import it from
`src/research/__init__.py` so the registration runs at package load.

Tracked follow-up beads: `CL-kxcs` (SSRN/NBER/Fed/BIS academic
preprints), `CL-sy32` (full-text PDF extraction beyond abstracts),
Reddit / Twitter (filed separately if pursued — both have auth +
curation overhead worth their own session).

## 7. Cost monitoring

Each LLM invocation logs `usd_cost` and token counts via the LLM
client (`src/research/llm/client.py`). Costs surface in three places:

- **Per-call** in the debate `transcript.jsonl`:
  ```json
  {"agent_name": "bull_reviewer", "model": "claude-opus-4-7",
   "input_tokens": 1234, "output_tokens": 567, "usd_cost": 0.0612, ...}
  ```
- **Per-debate** in the `DebateResult.total_cost_usd` field returned
  by the orchestrator (also persisted at the bottom of `transcript.md`).
- **Per-run** in `data/research/runs/{ts}.json` — sum across all
  agent invocations for the run (this isn't a built-in field today;
  inspect the transcripts or `jq` the per-run summary).

Provider pricing lives in `src/research/llm/client.py:_PRICING_USD_PER_MTOK`.
When a provider re-prices, edit that table; unknown models log $0
(silent — watch for new model IDs in the transcripts).

To estimate spend per run:

```bash
# Tally costs across today's runs
.venv/bin/python -c "
import json, glob
total = 0.0
for f in glob.glob('docs/research/debates/*/transcript.jsonl'):
    for line in open(f):
        total += json.loads(line).get('usd_cost', 0)
print(f'\${total:.2f} across {len(glob.glob(\"docs/research/debates/*/\"))} debates')
"
```

The default agent assignments mix providers by cost-sensitivity:
DeepSeek for high-volume cheap tasks (paper extraction, idea
generation, smart-question routing), Claude for code-generation +
adversarial reviewing. Switching one agent to a different provider is
a one-line YAML edit.

## 8. Safety boundaries — paper-shadow vs live; allocation=0; G9 still required

The pipeline is **autonomous up to paper-shadow**, never autonomous
to real money. Three independent gates protect this:

### GATE 1 — pre-research operator approval (CL-0hr3)

Catches obviously-bad ideas BEFORE the loop spends Implementer +
Backtest + Debate budget. Default 7-day timeout → auto-SKIP. This is
mostly a token-economics safeguard, not a financial one.

### GATE 2 — pre-deploy operator confirmation (CL-yta6)

Catches PROMOTE-verdict candidates the operator would still reject on
qualitative grounds. Default 7-day timeout → **auto-REJECT** (safer-
default; opposite of GATE 1, because silence on a deploy decision
should never read as "yes"). Still operator-awareness, not financial-
loss prevention.

### Allocation = 0 invariant

The paper-shadow registrar (`src/research/promote.py`) writes the new
strategy to `configs/live_portfolio.yaml` with
`initial_weights[slug] = 0.0`. The coordinator's allocation-based
scaling zeroes out every intent the strategy emits. The strategy runs
in production but emits no real-money trades.

**Broker modes (CL-920k):** the engine entrypoint accepts three
broker selections via `--broker`:

```bash
python -m src.runtime.run_engine --broker paper           # default
python -m src.runtime.run_engine --broker oanda-practice  # real OANDA practice
python -m src.runtime.run_engine --broker oanda-live --confirm-live   # REAL MONEY
```

| Mode | What it does | Money at risk |
|---|---|---|
| `paper` (default) | In-process `PaperBroker`, $100k simulated capital, mid-price fills, no network | None — pure simulation |
| `oanda-practice` | `OandaBroker` against `api-fxpractice.oanda.com`. Real spreads, real fills, practice account state survives engine restarts | None — practice account |
| `oanda-live` | `OandaBroker` against `api-fxtrade.oanda.com`. Requires `--confirm-live` flag | **Real money** |

`paper` is the default because it works without network or credentials.
`oanda-practice` is what you want for realistic paper-trading: real
broker semantics without real-money exposure. `oanda-live` is gated
behind an explicit `--confirm-live` flag to prevent accidents.

The allocation=0 invariant holds regardless of broker mode: a
research-pipeline-promoted strategy still emits intents the
coordinator scales to zero until you raise the allocation manually.

**Promoting to real allocation is a separate manual step**: the
operator edits `initial_weights[slug]` to a positive number in a
follow-up commit. There is intentionally no automation for this — see
the failure mode the bead description for CL-yta6 calls out:
"silence = no". Real-money decisions stay manual.

### G9 still required

The `src/edge_testing/` G1-G9 gate suite (CL-4lp era) is unchanged
and still required before any allocation > 0. Specifically G9 (live-
trading readiness) blocks promotion to non-zero allocation regardless
of what the research pipeline does. The research pipeline produces
candidates; G9 + operator soak time + manual allocation flip turns
candidates into capital.

## Quickstart

```bash
# Preflight: walk every phase with stubbed LLM + HTTP + backtest, no
# creds, no network, no production-tree pollution. Outputs land under
# /tmp/research-dry-run-XXXX. Run this any time you change agent
# prompts or sub-component interfaces — see SMOKE_TESTS.md for what
# to look for.
.venv/bin/python -m scripts.research_loop --dry-run --auto-approve

# One full pipeline pass (cron-friendly, idempotent)
.venv/bin/python -m scripts.research_loop

# Skip the real backtest (verdict engine will then ESCALATE
# everything for missing-metric — useful only for wiring smokes,
# not for finding edge):
.venv/bin/python -m scripts.research_loop --no-backtest

# Adjust backtest window (default 2018-01-01 → 2024-12-31):
.venv/bin/python -m scripts.research_loop \
    --backtest-start 2015-01-01 --backtest-end 2025-06-30

# Inspect what's pending operator action
.venv/bin/python -m scripts.research_approve --list           # GATE 1
.venv/bin/python -m scripts.research_approve --gate=2 --list  # GATE 2

# Approve / skip
.venv/bin/python -m scripts.research_approve --slug X --action GO
.venv/bin/python -m scripts.research_approve --gate=2 --slug X --action SKIP \
    --reason "regime concentration concern not in rules"

# Read latest run summary
ls -t data/research/runs/ | head -1 | xargs -I{} cat data/research/runs/{}
```

## Cron entry (currently installed)

Daily at 3 AM local, three steps chained sequentially:

```
(1) discover_polymarket_markets  → refresh polymarket_markets.yaml
(2) seed_polymarket_history       → pull probability history (rolling 2y)
(3) research_loop                 → ingest → idea → debate pipeline
```

Verify on this machine with `crontab -l`. The auto-loader at
`src/dotenv_bootstrap.py` reads `.env` from the project root so the
cron job inherits credentials without sourcing.

### What each step does

**(1) discover_polymarket_markets** (no LLM cost)
- Hits Polymarket Gamma `/markets`, filters to FX/macro keywords +
  min volume threshold, ranks by volume
- Updates `configs/polymarket_markets.yaml`: replaces `PLACEHOLDER_*`
  tokens with real CLOB token_ids; operator-edited entries are preserved
- Output: `logs/polymarket_discover_cron.log`

**(2) seed_polymarket_history** (no LLM cost)
- Pulls per-market probability history via Polymarket CLOB API for
  every market in the YAML, upserts to Postgres `prices` table with
  symbols `POLY:<slug>`
- `--verify` flag prints per-symbol counts after seeding
- Output: `logs/polymarket_seed_cron.log`

**(3) research_loop** (DeepSeek tokens)
- Hits all 9 feeds (arXiv + substacks + Polymarket research extracts),
  dedupes against `data/research/extracts/`. New items cost ~$0.01
  each via DeepSeek extractor.
- Idea agent ideates new extracts via DeepSeek. Most DECLINE; PROPOSED
  ones fire **Pushover/Telegram alert** and hold at GATE 1.
- Output: `logs/research_loop_cron.log`

### What does NOT fire automatically

- Implementer + Bull/Bear debate (Claude tokens) — only run after
  operator approves at GATE 1.
- Paper-shadow registrar — only after operator approves at GATE 2.

**Per-day automatic spend cap:** ≈ $0.50 worst case (50 new items ×
extract + idea). Implementer + debate spend (~$0.20–0.50/candidate)
is bounded by your engagement. Polymarket discovery + seed cost $0
(both APIs are free).

The loop is idempotent — extra runs are cheap (each phase short-
circuits on already-processed work) and safe (state writes are
atomic per phase).

## When something goes wrong

- **Loop crashes mid-pipeline** — state has already been written
  through whichever phase last completed. The next run picks up from
  there. Read the traceback in `data/research/runs/{ts}.json:errors`.
- **Implementer ships REJECTED with `syntax gate failed`** — the
  generated code didn't parse. The code is preserved at
  `src/strategies/_experimental/{slug}.py` for forensic inspection;
  no candidate report was written. Either fix the code manually or
  re-run with a stronger model on the implementer agent.
- **Debate transcript is empty** — likely an LLM provider error. Check
  `data/research/runs/{ts}.json:errors` for the underlying exception.
- **GATE 2 alert says `auto-REJECTED`** — the entry sat too long; the
  PROMOTE verdict is still in the state file under
  `debates_completed[slug].verdict`. Re-promote by manually flipping
  `deploy_status` back to `PENDING_DEPLOY_CONFIRMATION` and
  re-approving, OR just delete the entry to re-run from the debate
  phase.
- **Notification channels silent** — env-var gating: set
  `PUSHOVER_API_TOKEN` + `PUSHOVER_USER_KEY` and/or
  `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`. Either channel works
  alone.

## Changelog

- 2026-04-29 — initial version (CL-yhue), covers the pipeline as of
  CL-h986 closure.
- 2026-05-01 — feed expansion + harness v2:
  - **Ingest**: added RSSFetcher (substacks, quant blogs) and
    PolymarketFetcher (prediction markets, FX/macro filtered).
    Feed list grew 2 → 9.
  - **Backtest harness (CL-40n2 v1)**: strategies can declare
    multiple `symbols` + optional `execution_symbol`. The wide
    DataFrame contract is documented in section 4b above. Joint
    multi-asset positions still NOT supported (CL-40n2 v2).
  - **Cron installed**: daily 3 AM auto-fire, output to
    `logs/research_loop_cron.log`. Spend cap ≈ $0.50/day worst
    case before operator engagement.
  - **Slim debate context**: `_slim_candidate_report` +
    `_slim_transcript` cut round-4 debate calls from ~11k to
    ~5k input tokens to fit Anthropic tier-1's 10k/min cap.
  - **`.env` auto-loader**: `src/dotenv_bootstrap.py` wired into
    every CLI entrypoint. No more `source .env` step.
  - **Polymarket as strategy feature (CL-3t4j v2)**: new
    ``src/data/polymarket.py:PolymarketHistoryIngester`` pulls
    per-market probability history via Polymarket's CLOB
    prices-history API and upserts to the existing ``prices``
    table with synthetic ``POLY:<slug>`` symbols. Strategies
    declare these symbols alongside FX pairs and use them as
    feature inputs (probabilities in [0, 1]). Curated market list
    at ``configs/polymarket_markets.yaml``; seed via
    ``python -m scripts.seed_polymarket_history``.
  - **Bug fixes from real-network smokes**: Telegram parse_mode
    plain-text (was 400-magnet on slugs with underscores), bot-
    token scrubbing in error logs, httpx INFO suppression
    (CL-wmn4), implementer prompt constraints against `src.*`
    imports (CL-zjcx), Claude opus-4-7 temperature deprecation
    (default model pinned to sonnet-4-6).
  - **Provider config**: DeepSeek `deepseek-v4-pro`, Claude
    `claude-sonnet-4-6`. Multi-provider routing restored
    (implementer + Bull + Bear on Claude, ingest agents on
    DeepSeek). CL-wkil closed.
