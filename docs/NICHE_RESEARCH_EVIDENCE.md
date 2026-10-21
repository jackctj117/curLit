# Evidence-grounded niche research

Implemented under Beads epic **CL-eh28**, with status/persistence (.1),
source claims (.2), criticism/scoring (.3), and shadow comparison (.4).
Beads remains the only task tracker. This document describes the code, not a
completed provider benchmark or a deployment approval.

## Subscription-only Claude pipeline (CL-zhkg)

Implemented in `scripts/event_pipeline.py`; the targeted October 20 deployment
and its limitations are recorded in [current operations](CURRENT_OPERATIONS.md).
Code alone is **not deployment or research-quality evidence**. The pipeline
now chooses the existing Claude Code transport explicitly, regardless of a
legacy `NICHE_TOOL_AGENT_ENABLED=1`. It makes at most two discovery calls:
provisional leads, then revision using captured filings for up to two verified
tickers. The evidence critic is required and uses the same subscription-only
transport policy. All identity/evidence/liquidity/review gates remain intact.
Two calls are not two relationship hops; `hop_count` remains descriptive.

The second pass may replace an earlier company's unsupported hypothesis with
claims tied to the retrieved documents; it cannot create a duplicate candidate
by repeating that company. Raw responses remain in the audit trace. Source
documents precede optional profile text in the bounded prompt. Seed hypotheses
have no trading eligibility merely because a ticker exists. Retrieval failure,
unsupported claims, unknown liquidity and failed review still prevent merging.
The prompt version is `niche-claude-refinement-v2`. CL-a7xg makes the canonical
`passage` JSON field explicit with a parser-tested example; `exact_passage`
remains invalid. Provenance and semantic evidence requirements are unchanged.

### Critic/extraction revision (CL-5hj5 / CL-4knt)

Deployment status is recorded in `CURRENT_OPERATIONS.md` under CL-2m1c;
the checks below do not by themselves establish deployment. Subscription criticism requests
the JSON Schema generated from `ReviewBatch`, then validates the response using
that same strict model. Required fields, types, finite confidence, and an exact
one-verdict-per-requested-ticker set are checked locally. There is no substring
salvage, quote rewriting, favorable duplicate selection, or plain-text fallback
when the CLI omits `structured_output`. Unknown sources and changed quotations
still fail the existing exact-source checks; valid JSON is not thesis approval.

The installed CLI 2.1.304 exposes `--json-schema`; its result field is documented
in [programmatic structured output](https://code.claude.com/docs/en/headless#get-structured-output).
The driver reads the dedicated `structured_output` object and retains the raw
envelope. The critic records the failed response and actual model where
available. Transport errors without a response stay explicitly unknown. Review
provenance records `niche-review-v2` and the schema hash. Queue retries caused by
unavailable review now say `review_unavailable`, not discovery `completed`.

One curLit critic invocation remains one CLI generation subprocess. Schema calls
set `--max-turns 2` and `MAX_STRUCTURED_OUTPUT_RETRIES=1` (recognized by installed
CLI 2.1.304), retain the existing wall-clock ceiling, disable external tools,
and retain subscription authentication and paid-fallback restrictions. These
are turn/retry limits, **not guaranteed token or dollar limits**. Missing output
fails closed. Do not enable API auth or external tools to make structured output
succeed.

The authorized no-order canary completed at **20:37 UTC October 20** on CLI
2.1.304 with serving model `claude-sonnet-4-6`. Both calls returned structured
objects in two agent turns with no permission denials. The captured AVAV/PLTR
evidence was judged insufficient; a clearly supported synthetic claim was
recognized, while the source-free control remained insufficient. All four
candidates remained ineligible because liquidity was unknown. Calls took
55.2/27.0 seconds and reported 2,879/1,581 output tokens; the requested 1,200
token cap remains explicitly unenforced. Cost was subscription-unmetered, not
reported as zero. This validates the output path, not useful live research.

The first canary attempt encountered an exclusive-file-write error in the
private audit helper after its model call. That attempt was preserved as failed,
not counted as a pass. Unique result filenames fixed the helper; the replacement
was bounded to two calls. Private artifacts live in
`data/deployments/2026-10-20_critic_schema_v2/`; no source retrieval, database,
broker, order, or trading-service action was part of these canaries.

Filing normalization now removes inline-XBRL headers/hidden metadata, contexts,
units, and explicitly hidden HTML nodes before choosing excerpts. Visible inline
facts remain. New locators refer to the cleaned, whitespace-normalized text;
historical source hashes/passages are not rewritten. This is not a full CSS
renderer and does not solve primary-document truncation, missing exhibit
retrieval, irrelevant annual-meeting notices, or semantic evidence coverage.

The proposed job policy is `claude-subscription-review-schema-v3`. A rollout can
create a new bounded attempt series for still-recent events under that policy;
review its quota impact before activation. Older audits keep their prior policy.
Useful-source evaluation, slow refinement calls and completed-result persistence
behind slow peer jobs remain in CL-uofe. Cleaner excerpts and passing mocks do
not demonstrate that the live pipeline will produce eligible research.

Development verification on October 20, 2026 used a network-denied macOS
sandbox that denied home/credential reads and checkout writes. The shell
environment was cleared with `env -i`; Python bytecode was disabled and pytest,
Hypothesis and mypy caches used a disposable `/tmp` directory. Commands inside
that sandbox were:

```sh
.venv/bin/python -m pytest \
  tests/unit/test_claude_code_driver.py tests/unit/test_niche_jobs.py \
  tests/unit/test_morning_digest.py tests/unit/test_niche_evidence.py \
  tests/unit/test_event_niche_agent.py tests/unit/test_adversarial_critic.py \
  tests/unit/test_idea_ledger.py tests/unit/test_research_status.py \
  tests/unit/test_niche_audit.py tests/unit/test_event_digest.py \
  tests/unit/test_niche_quality_export.py tests/unit/test_niche_hopgraph_shadow.py \
  tests/unit/test_critic_structured_output.py tests/unit/test_research_tools.py \
  -o addopts='' -q -p no:cacheprovider
.venv/bin/ruff check src/ tests/ scripts/event_pipeline.py scripts/morning_digest.py --no-cache
make typecheck
.venv/bin/bandit -q -ll src/events/critic_contract.py src/events/adversarial_critic.py \
  src/events/research_tools.py src/events/niche_jobs.py src/research/llm/claude_code.py
```

All exited **0**: **450 tests passed**, Ruff clean, mypy checked **252 source
files**, and no medium/high Bandit findings in the listed files. This was not
a full-suite run. Pytest emitted the existing `Unknown config option: timeout`
warning: the host environment lacks pytest-timeout. `-o addopts=''` made that
existing configuration mismatch non-fatal; it did not install or verify the
timeout plugin. Full isolated-environment limitations remain tracked in CL-i6ke.
Mypy also reported existing unused override sections. The new regressions were
observed failing before their code fixes: 16 critic/transport cases, one queue
reason case, and two extraction cases. No live model or trading service was used.

### Billing safeguards and operator activation

This is cloud inference through a subscription CLI, not a free local model.
Before activation, the operator must verify included-plan access and disable
paid overflow/usage credits and automatic top-ups in the Claude account.
Only after that verification set `CURLIT_CLAUDE_INCLUDED_USAGE_CONFIRMED=1` in
the pipeline's environment. This flag records an **operator attestation**, not
an automated billing-settings check. Without it, no auth or model subprocess
is invoked for niche discovery (`unavailable/included_usage_not_confirmed`).
The development patch did not alter account settings. The October 20 authorized
rollout set the attestation and disabled the legacy Kimi selector; the operator,
not this flag, confirmed the account billing settings.

Each strict invocation checks `claude auth status` with the same sanitized
environment and setting-source restrictions used for generation. Only
`loggedIn=true, authMethod=claude.ai` passes. API keys/helpers, third-party
providers, unknown/malformed status and opaque `oauth_token` auth are blocked;
use a normal Claude subscription login for this path. No login, key extraction,
credit purchase or fallback happens automatically. User/project/local settings
are excluded; hooks and built-in/MCP tools are disabled for generation.
Managed policy still applies: the operational host must not inject a competing
provider. CLI version/flag incompatibility fails closed and needs operator review.
The strict path pins the existing `claude-sonnet-4-6` baseline: other model
overrides, including a credits-only tier selected through `NICHE_CRITIC_MODEL`,
are blocked as `subscription_model_not_approved` before any subprocess. Adding
another model needs review and confirmation of included-plan eligibility.

Quota failures are `subscription_quota_exhausted`, never abstention or approval.
A failed second call retains the first pass only as partial research; no critic
call follows a subscription failure. There are no curLit retries or metered
fallbacks within that invocation. CL-ut1d adds the bounded durable retry queue
described below; it does not replay weeks-old events or bypass the billing gate.
Requested token limits retain the driver's existing enforcement policy; the
two-call bound is not a guaranteed token or dollar ceiling. Cost stays unknown,
not fabricated as zero.

Operator rollout requires reviewing the patch, verifying billing/login and CLI
flags on the intended account, then a captured-input/no-order canary that checks
source-backed refinement, failures, and critic behavior. Only then restart the
event pipeline through its approved runbook. Preserve trading pauses and do not
restart executors or FX as part of this switch. Rolling back to older pipeline
code can re-enable Kimi if the old flag/key remain present: keep
`NICHE_TOOL_AGENT_ENABLED=0` during rollback. Standalone Kimi/shadow tools remain
available only for separately authorized experiments; they are not fallbacks.

This changes the discovery workflow, not just the model, and makes no claim
that Claude is more accurate or profitable. Research-quality evaluation remains
separate from transport tests. Authentication and settings behavior follow the
[Claude CLI reference](https://code.claude.com/docs/en/cli-reference) and
[subscription billing guidance](https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan).

## Durable research and honest Telegram freshness (CL-ut1d / CL-ayd8)

The October 20 targeted rollout applied additive migration `029_niche_jobs.sql`
before restarting the revised event pipeline and morning digest. See the dated
operational record for backup, process and runtime verification. No migration
or operational record was changed during the preceding development phase.

The pipeline seeds jobs from the database's recent, still-`ASSESSED` events,
independently of this cycle's newly assessed results. This closes the crash
gap between assessment and niche scheduling. Immutable event/input/policy hashes
deduplicate jobs; failed research does not reset events to `NEW`.

Research policy is deliberately bounded:

- Only events seen in the preceding **24 hours** qualify for automatic research
  or publication. This is a new-research freshness limit, not a trading parameter.
  Historical backfill needs its own reviewed plan; weeks-old headlines are not
  turned into fresh trading ideas.
- At most **three attempts per input/policy version**, with delays of 15 minutes
  and then one hour. Quota/auth/transport/partial-refinement and unavailable-review
  outcomes retry. Completed abstentions, unsupported evidence, and malformed output
  do not repeatedly consume quota. Exhaustion remains explicitly failed.
- Each cycle claims one bounded concurrency batch with a **one-hour lease**.
  A crashed worker can be reclaimed after that lease, but its late completion
  cannot publish. No database transaction remains open during model/retrieval calls.
- Before claiming and again before committing, current event status and input
  must match. Changed/confirmed/dismissed events supersede the old job; an age
  limit expires it. Research evidence survives independently in the audit.
- The event merge, fresh niche idea insertion, write-once research status and
  job completion commit in **one transaction**. A ledger failure rolls all
  of those back. Audit persistence occurs first in its own transaction.
  Retriable incomplete batches publish no partial subset; all gates still apply.
- Niche rows are now inserted before optional downstream price enrichment.
  Missing signal marks remain unknown; no current quote is fabricated. Ordinary
  impact ideas retain their existing enrichment path.

Telegram's morning ideas message distinguishes verified niche ideas created in
the last 24 hours, other fresh pending ideas, older pending backlog, and unknown
dates. This is a **display grouping**, not an expiry or execution policy.
Verified niche labels come from `idea_research_status`, never note substrings.
Shown rows carry idea/event dates and IDs; niche rows also carry invocation IDs.
Duplicate ticker records retain references within their freshness category.
Counts include the full pending pool even when display rows are capped.

The message separately shows recent audit outcomes, review statuses, queue state
and the number of ideas actually published from each invocation. Unavailable
queries are labeled unavailable, not empty/successful. An unchanged backlog or
current report timestamp no longer implies a new successful discovery. The
hardcoded automatic-buy promise is removed; the digest never authorizes orders.

Activation still requires the operator's included-usage billing confirmation,
verified Claude subscription login, a captured-input no-order canary, and an
approved migration/targeted pipeline + morning-digest rollout. Keep all trading
halts and executor/FX processes unchanged. Observe a new audit/job/ledger result
and rendered Telegram output before declaring niche restored. Apply migration
first; rollback leaves the additive queue/audit tables intact and must not
re-enable a metered provider.

### Development verification, October 20

The following pytest command ran in the credential-free test image
`sha256:31caaf834c784ae039588cf3031bcf752ee63a3e313b73eb45a4623b7a24f35c`,
against a source-only snapshot and disposable PostgreSQL 16: **416 passed,
exit 0**. The database had no host ports or mounts and `network=none`; the
test container shared only that isolated network namespace. No operational
database was connected. Both disposable containers were removed after testing.

```sh
python -m pytest \
  tests/unit/test_niche_jobs.py tests/unit/test_morning_digest.py \
  tests/unit/test_event_digest.py tests/unit/test_idea_ledger.py \
  tests/unit/test_research_status.py tests/unit/test_niche_evidence.py \
  tests/unit/test_claude_code_driver.py tests/unit/test_event_niche_agent.py \
  tests/unit/test_adversarial_critic.py tests/unit/test_niche_audit.py \
  tests/unit/test_niche_quality_export.py tests/unit/test_niche_hopgraph_shadow.py \
  tests/integration/test_niche_jobs_postgres.py \
  tests/integration/test_niche_audit_postgres.py -q -o addopts= --tb=short
```

Additional commands ran with cleared environments and filesystem/network
sandboxing; all exited 0:

```sh
.venv/bin/mypy src/
.venv/bin/ruff check src/ tests/ scripts/event_pipeline.py scripts/morning_digest.py --no-cache
.venv/bin/python -m bandit -r src/events/niche_jobs.py src/events/idea_ledger.py \
  src/monitoring/morning_digest.py -ll -q
git diff --check
```

Mypy checked 251 source files. The initial four freshness regressions failed
against the old implementation before the fix. Two new SQL fixture binding
mistakes were corrected during development; no test was weakened or skipped.
This is targeted verification, not a new full-repository green claim
(the separate full-suite environment work remains CL-i6ke), and not evidence
that the operator's Claude account or running pipeline has been activated.

### Deployment preflight, October 20 (CL-cdov / CL-foln / CL-a7xg)

The installed CLI was 2.1.304. Its variadic `--mcp-config` option originally
consumed the following `auth status` tokens. Placing the existing boolean
`--strict-mcp-config` after the MCP JSON fixed the boundary without removing
settings, hook or tool isolation. Actual auth returned `claude.ai` / Max.
An initial cleared-environment canary omitted `USER`, which macOS Keychain
lookup needs, and correctly failed `subscription_auth_unverified`; restoring
the non-secret OS username allowed the subscription calls. No login or
credential copying was performed.

Two three-call captured-input runs used historical event 14500, nine original
filing excerpts, a finite captured identity universe, and deliberately missing
current market observations. The second run used the explicit claim example:
four claims parsed and matched their captured passages. This is provenance
matching, not semantic approval. Its critic emitted invalid JSON, so both leads
remained `review_unavailable`; no ideas were published. Raw reports and failures
were preserved privately. Low-information filing excerpts and robust review
output remain CL-uofe; neither run is a provider-quality benchmark.

After both regression fixes, this command passed **412 tests**, exit 0, under
the cleared-environment filesystem/network sandbox:

```sh
.venv/bin/python -m pytest \
  tests/unit/test_claude_code_driver.py tests/unit/test_niche_jobs.py \
  tests/unit/test_morning_digest.py tests/unit/test_niche_evidence.py \
  tests/unit/test_event_niche_agent.py tests/unit/test_adversarial_critic.py \
  tests/unit/test_idea_ledger.py tests/unit/test_research_status.py \
  tests/unit/test_niche_audit.py tests/unit/test_event_digest.py \
  tests/unit/test_niche_quality_export.py tests/unit/test_niche_hopgraph_shadow.py \
  -o addopts='' -q -p no:cacheprovider
```

The two new tests each failed against their pre-fix implementation. The native
environment still warns that its pytest timeout option is unknown; it was not
silenced or worked around by skipping tests. `make typecheck`, Ruff, targeted
Bandit and diff/secret scans passed. The source release is `3962f56`; the
authorized rollout applied only migration 029 and restarted only the launchd
pipeline/digest jobs. See current operations for actual post-cutover results.

## Outcomes, leads, and the trading feed

`NicheAgent.run_report()` returns an invocation-local discovery outcome and all
parsed research candidates. There is no shared last-result field across worker
threads. With migration 020, `scripts/event_pipeline.py` first commits the report
to `niche_research_audit`, independent of event status (CL-27s0). Each invocation
has an ID allocated before discovery, an input snapshot, report, and payload
hash. Identical replays insert once; a conflicting payload under the same ID
raises instead of overwriting evidence. The application only inserts these
records; this is not a claim of database-level tamper-proof storage.

The legacy `assessment.niche_research` projection and trade-idea merge still use
a private copy and reach in-memory ledger inputs only after a committed update.
That update requires both `status = ASSESSED` and an unchanged original JSONB
assessment. A status or assessment race leaves the independent audit committed
but skips the merge. An audit failure blocks the niche merge. DB exceptions and
commit failures cannot leak uncommitted niche ideas into the later ledger step.
The original audit change did not add retries or reconstruct the three lost
reports. CL-ut1d now adds the bounded recent-event queue described above;
historical missing reports remain unreconstructed.

Discovery distinguishes `completed`, `abstained`, `partial`, `invalid_output`,
`unavailable`, and `budget_exhausted`. Empty valid JSON is abstention; malformed
JSON or an API/billing outage is not. Missing credentials for an explicitly
enabled Kimi path produce `unavailable`, not an implicit provider switch.
Provider errors are classified without copying account/key identifiers to logs.

## Native Kimi completion budgets (CL-uofe)

The funded failures observed on September 8 were local tool-budget exhaustion
and incomplete generations, not insufficient-balance responses. Account funding
and per-invocation research limits are separate. We have not rechecked the
account balance as part of this development patch.

Native K3 requests now explicitly use `reasoning_effort=low` and retain the
complete assistant message, including interleaved reasoning. Moonshot documents
K3 as always thinking, with `max` effort by default, and requires complete
assistant history for tool conversations:
[K3 API guide](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart).
This is a versioned research-policy revision (`:kimi-budget-v2`), not a model
replacement or evidence that lower effort yields better investment ideas.
Other model names do not receive K3-only request parameters.

The loop reserves its last model call for finalization with tools disabled.
Reaching the 24-tool ceiling answers remaining requests with explicit budget
errors and requests a final answer from collected evidence. Identical lookups
reuse invocation-local results (including unavailable results); they still count
against the requested-tool ceiling. Sources are deduplicated by their hashes.
A length-truncated response is audited but never executed as tool instructions;
one tools-disabled finalization attempt may use up to twice the ordinary response
allocation. A second truncation stays `partial`; fabricated empty output is
never substituted by code. Valid provider-authored empty output is abstention.

No more than eight model calls or 32,768 **requested maximum output tokens**
under the defaults: a larger final allocation comes from that same envelope,
not an increased total. Per-call finish reason, usage, reasoning tokens when
reported, effort, finalization flag, and requested limit are captured. Hidden
SDK retries are disabled, with a 120-second request timeout. These constraints
do not guarantee a dollar ceiling: input tokens and provider billing also
matter. Reported costs remain unknown unless independently measured.

The synthetic captured-source regression proves that a final answer can retain
its source identities and still must pass evidence, liquidity, and critic gates.
It is not a paid baseline-versus-revision experiment. CL-uofe remains open until
an operator-approved, captured-input canary measures completion, useful evidence,
abstention, latency, and usage without orders. The equivalent-tool Kimi/Claude
experiment below is unchanged and must not be conflated with this native-loop
revision.

Migration 020 must be applied and verified before the updated pipeline is
started; a missing audit table safely prevents niche merges. The migration is
additive. An older pipeline can ignore the retained table on rollback. This
development patch does not authorize migration or service restart.

### Context-bounded finalization (CL-lu3d)

Native policy `:kimi-budget-v3` enforces the same **100,000-character serialized
message limit** as the captured-input canary before every provider call. This
counts `json.dumps(messages)`, including escaping and embedded tool JSON, not
just human-visible character length. The eight-call, 24-tool and 32,768 requested
output-token defaults are unchanged. Each response trace records prompt size
and its cap.

If accumulated history would exceed the cap, the loop starts an independent,
tools-disabled finalization request inside the remaining call/token allowance.
It carries the original task, every captured source (exact bytes and hashes,
once each), and all tool-result facts, errors and contrary observations.
Identical tool results are deduplicated; source references point into the shared
packet. No source is shortened or discarded to fit. Model-generated reasoning
remains in the audit, not the new evidence packet. Ordinary continued tool
conversations still retain complete assistant/reasoning fields; there are no
orphaned tool messages in the fresh request.

If even that complete packet cannot fit, the result is explicitly
`budget_exhausted/prompt_char_limit`, with no over-cap provider request. The code
does not invent an empty result or treat discarded evidence as approval. The
existing evidence, liquidity and independent critic gates remain mandatory.

The exact private event-48489 failure was replayed offline: its first four
request bodies matched the captured run, and its next 131,225-character history
became a 47,673-character finalization request with all nine sources preserved.
The final answer in that offline test was synthetic; it proves transport and
provenance behavior, not research quality. Public regression fixtures cover
repeated multi-source batches, contrary facts, Unicode escaping, oversized
initial events, packets that cannot fit, and downstream eligibility gates.
Paid canary results and actual rollout state are recorded separately in
`CURRENT_OPERATIONS.md` and Beads; passing these tests is not deployment proof.

Disposable database regression (never use the operational database):

```sh
CURLIT_AUDIT_TEST_DB_URL='postgresql+psycopg2://USER:PASSWORD@127.0.0.1:PORT/curlit_test_audit' \
  .venv/bin/pytest tests/integration/test_niche_audit_postgres.py -q
```

The integration tests require an explicit loopback `curlit_test_*` database,
create isolated random schemas, apply migration 020 twice, and remove only their
own fixture schemas. They exercise actual pipeline status/assessment races and
concurrent replay. Existing unit fixtures were extended to recognize the new
audit INSERT/SELECT transaction; their trade-merge assertions were preserved.

## Review and trading eligibility

Review distinguishes `supported`, `contradicted`, `insufficient_evidence`,
`review_unavailable`, and `not_requested`. Missing, duplicate or malformed
verdicts cannot preserve an earlier approval. A disabled critic leaves research
leads visible but does not approve them. Unsupported objections also retain the
lead; a grounded contradiction is recorded rather than erasing the candidate.
The original response, citations, requested/actual reviewer model where known,
prompt hash, review time and separate evidence cutoff accompany the review.

`run()` and `merge_into_assessment()` permit only candidates with completed
discovery, verified ticker identity, source-backed critical claims, sufficient
dated liquidity, and a supported review. Only these receive the legacy
`red_team_verdict=confirmed` compatibility marker. A failure cannot enter the
existing order-consumed `trade_ideas` feed just because the execution policy
does not require red-team tags. Existing non-niche ideas, historical ledger
rows and open positions are not rewritten by this patch.

CL-7kuu: execution eligibility is now persisted structurally. When the niche
merge adds an eligible candidate, the idea ledger writes a write-once
`idea_research_status` row (statuses, eligibility, source hashes, score version,
invocation id) with the `trade_ideas` row. With their niche/red-team policy flags
on, the Alpaca executors select on that row instead of note substrings; the
legacy note match survives only behind `ALPACA_LEGACY_NOTE_MATCH=1`.

**Intentional behavior change:** research failures preserve leads, not trading
eligibility. This may substantially reduce new niche entries, including when
the critic is disabled or the configured provider is unfunded. The initial
implementation did not restart services. The operator subsequently authorized
the targeted CL-294s rollout; see `CURRENT_OPERATIONS.md` for deployment evidence
and the distinction between startup checks and a completed research cycle.

## What source-backed means

A captured document carries ticker, HTTPS URL, publication and retrieval times,
the exact normalized passage, source location and a SHA-256 identity covering
that record. Claims separately label `documented_fact` versus `inference` and
their role: relationship, economic exposure, catalyst or disconfirming evidence.
The model cannot promote itself by emitting verification/status fields.

Critical relationship, exposure and catalyst facts must cite captured source
IDs with passages actually present in those records. Future, undated, stale,
wrong-ticker and fabricated citations fail. All submitted claims, including
inferences, need a captured factual basis. The evidence reviewer must still
judge whether a passage supports the proposition and whether the inferred
economic benefit follows. **Substring matching and model agreement do not
prove truth.** Human evaluation remains necessary.

Research retrieval (CL-mkqq) selects the latest annual (10-K/20-F/40-F),
quarterly (10-Q), and current (8-K/6-K) primary documents, up to the existing
three-document limit. Missing categories use the newest remaining unique
filings. Current-report bursts cannot crowd out annual business coverage.
Bounded candidate claims/rationale supply unverified search hints; deterministic
multi-term relevance and prose ranking locate exact normalized-text slices.
Offsets and accession identify the passage. This is retrieval relevance, not
proof of the claim, and includes current-report announcements, not an exhaustive
search of investor-relations releases, exhibits or every historical filing.
Day-only SEC publication dates are conservatively represented at end of day;
a same-day document can remain unavailable until that cutoff passes.

The research freshness defaults are explicit engineering policies: two annual
filing cycles (730 days) for documents, seven calendar days for market
observations. They are not empirically optimized trading thresholds.
Liquidity requires finite nonnegative average dollar volume plus observation
and receipt timestamps; invalid/missing/stale data is `unknown`, not zero or
passed. A known value below the configured floor is `insufficient`.
The critic receives the observed average volume, market cap, last close when
available, timestamps and the evidence packet; absent measurements stay null.

Active scoring is equal-weight coverage of the four documented roles, bounded
in [0,1]. The critical-role and liquidity requirements apply independently of
the score threshold. This is an evidence-completeness diagnostic, **not an
expected return or calibrated opportunity score**. Relationship hops, market
smallness and leverage keywords earn no active bonus. Legacy numerical helpers
remain for historical analysis and identify their score version explicitly.

## Equivalent-tool shadow comparison

`src/events/niche_shadow.py` accepts a frozen JSON capture containing:

- `captured_at`: timezone-aware cutoff;
- `event`: captured event information, ideally including publication/receipt times;
- `symbols`: captured identity/profile records keyed by ticker;
- `sources`: serialized `SourceDocument.to_dict()` records;
- `market_data`: captured observations keyed by ticker.

Source hashes and availability cutoffs are validated. The harness owns canonical
bytes, so changing a caller's mutable object cannot change a trial. The operator
must establish the capture's provenance: a hash proves integrity, not that an
operator-supplied document is authentic. Missing event provenance means this is
not proof of point-in-time research correctness.

Both models use the same JSON tool-request protocol and captured collection for
identity checks, company resolution, profiles and targeted filing retrieval.
Unknown tools are rejected. There is no arbitrary URL/file/SQL tool, live-data
fallback, production database, ledger merge or broker submission in the harness.
Use a credential-isolated environment regardless: a Python allowlist is not an
OS sandbox against arbitrary modifications or malicious code.

Both trials receive the same caps: eight model calls, 24 tool calls, 4,096
requested output tokens per call, 32,768 cumulative output tokens and 100,000
serialized prompt characters. These are workload controls, not a guaranteed
dollar ceiling. Provider/tokenizer differences and internal model computation
still differ. An exceeded cap, incomplete generation or actual-model substitution
remains visible; it is not silently included as a comparable successful trial.

The Claude challenger uses the **Anthropic API**, not Claude Code. Inspection
found that the CLI driver accepted `max_tokens` but did not enforce it. CL-h7c1
can pass it to the CLI as `CLAUDE_CODE_MAX_OUTPUT_TOKENS` behind a default-off rollout gate, but the CLI still
cannot enforce `temperature`, has no metered per-call USD cost, and its usage
and serving-model reports differ from the API's (see
`docs/CURRENT_OPERATIONS.md` §5), so the API path remains the comparable arm.
The same API-based critic is applied to both arms. Runtime Kimi
discovery remains its existing native-tool baseline; the shadow experiment
standardizes the protocol for *both* providers rather than claiming their
existing different production methods are equivalent.

To run one captured event deliberately, in a research-only environment with
separate API credentials already provided through the environment:

```sh
.venv/bin/python scripts/compare_niche_research.py \
  --snapshot /path/to/captured-event.json \
  --output-dir /path/to/new-experiment-directory \
  --kimi-model YOUR_AVAILABLE_KIMI_MODEL \
  --claude-model YOUR_AVAILABLE_CLAUDE_MODEL \
  --critic-model YOUR_AVAILABLE_CRITIC_MODEL \
  --allow-model-calls
```

The command does not load the production `.env`. It requires both
`MOONSHOT_API_KEY` and `ANTHROPIC_API_KEY`; calls may be billed. It refuses to
overwrite an existing output directory. Without `--allow-model-calls`, it
validates the capture and refuses provider calls.

`private_report.json` retains the input, hashes, raw transcripts/tool results,
requested/actual models, usage, latency, raw pre-critic candidates, post-critic
candidates and basic coverage counts. Costs remain null unless independently
measured; invocation through a CLI is not proof of free usage.
`blinded_candidates.json` omits explicit provider identity and critic results;
style/content may still reveal clues. Keep the private mapping away from raters.

For an initial engineering evaluation, predeclare a cohort of about 100 distinct
events, alternate provider order with `--claude-first`, and retain failures and empty
outputs. Independently rate claim accuracy, useful non-obvious exposure and
restraint on raw discoveries before considering the common critic. Record
human judgments separately from automatic citation-coverage counts. Compare
reliability, latency and verified billing without selecting on a few winning
trades. Later forward evaluation needs fixed horizons, execution costs and an
untouched period; this harness does not establish profitability or choose a
production model automatically.

## Regression intent

Existing tests that treated missing research as eligible or an unsourced
refutation as final were updated intentionally. Their discovery, deduplication,
identity, concurrency, merge-cap and serialization assertions remain; new tests
require incomplete candidates to remain research-only and failure reports to
persist. Legacy score-vector tests remain unchanged for historical readers.
New deterministic and property-based tests cover fabricated/altered sources,
fact-versus-inference requirements, stale/nonfinite liquidity, no hop/keyword
bonus, review failures and duplicate verdicts, sourced positive/negative review,
full evidence-to-merge flow, immutable inputs, equivalent tool access and
bounded shadow loops. No paid comparison or production migration is implied by
these tests.

## Hop-graph discovery (CL-ynuh, shadow only)

**Status: shadow only.** Nothing in the hop graph is wired into
`niche_agent`, `scripts/event_pipeline.py`, `trade_ideas`, a ledger or an
order path. Promotion is a separate operator decision after the
pre-registered comparison below. The evidence gate, liquidity rule, critic,
`research_eligible` and `merge_into_assessment` are unchanged.

Motivation (30-day logs): 192 Kimi runs gave 307 candidates, all identity
verified, but only 4 passed the evidence gate and 2 were eligible. One
agentic loop did discovery *and* verification; tools could only fetch the
filing of a ticker the model had already guessed; and a relationship is
usually disclosed in the *other* party's filing (a supplier's
customer-concentration note names its customer). Every event started cold.

The hop graph separates those concerns:

1. **Traversal** (`src/events/hop_graph.py`). A typed graph of nodes
   (`commodity`, `route`, `company`, `product`, `country`) and edges
   (`supplies`, `buys_from`, `competes_with`, `substitutes_for`,
   `depends_on_route`, `hedged_by`, `priced_off`), each with a falsifiable
   claim and a where-to-look pointer (ticker, section, keywords). Hop 0 is the
   playbook's equity/FX instruments, tickers already named by the impact
   assessment, and optional captured `hop_seeds`. A text-only model
   (`no_tools=True`) proposes at most **4 edges per frontier node**. It is not
   asked to hold evidence. Hard caps, which the constructor refuses to raise:
   **3 hops, 12 frontier nodes per hop, 6 traversal model calls per event**
   (6 nodes per call). Frontier order uses unseen tickers first, then the
   legacy `torque_from_reason` and market-cap smallness heuristics, then the
   node id. That order is search priority only and never counts as
   evidence. Given the same model and verifier replies, the result is
   deterministic.
2. **Edge verification** (`src/events/hop_graph_verify.py`). For each edge, the
   verifier retrieves filings for **both endpoints with tickers**: pointer
   first, then the other. The budget is **at most 2 retrievals per edge**,
   through `ResearchTools.filing_documents(cik, symbol, query=...)` (live) or
   the capture's frozen collection (shadow). A candidate passage is an exact
   sentence of a usable captured source that names the counterparty. A
   Haiku-class entailment check (`no_tools=True`) is given the structured
   statement, for example "Acme Marine Coatings Inc (ACME) supplies Frontline
   Ltd (FRO)", plus the model's claim, so the relation and both endpoints must
   be stated, not just the claim prose. It answers only yes/no/unclear:
   - yes and no "no": `sourced`, with a `documented_fact` /
     `relationship` claim;
   - no and no "yes": `contradicted`;
   - anything else (unclear, mixed, no passage, transport error):
     `unverifiable`.

   Hedging, contract-expiry, termination and "<5% / not material"
   sentences in the same documents become `disconfirming` claims for the
   critic. They are always kept whole. Sentences are never clipped and
   limitations are never dropped. If a relevant limiting sentence exceeds
   2,000 characters, or a target has more than four limiting passages, the
   target is `unverifiable` (`limiting_context_incomplete`), never `sourced`
   on a fragment. Edges run on a bounded thread pool with a per-thread
   `claude-code` client.
3. **Path assembly.** Only simple paths from a seed whose **every** edge is
   sourced become `NicheIdea` candidates: status `sourced` plus exact
   passages backed by usable captured sources (`Edge.is_sourced`). The
   terminal must be a listed company that is not a seed and has a
   bullish/bearish direction. `hop_count` is the path length. The rationale
   is the chain with a source id, locator and passage for each hop. The
   terminal's own exposure and catalyst claims are verified the same way, with
   8-K documents preferred for catalysts.
4. **Unchanged gates.** Candidates go through `verify_ideas` →
   `evidence_score` → `AdversarialCritic.apply`.

**Documented deviation (evidence binding).** The unchanged gate binds every
claim and critic citation to the **candidate's own** filings, because
`RelationshipClaim.backed` and the critic check `source.symbol`. A
candidate therefore carries only claims sourced from its own filings. That
includes the edge into it when the edge is disclosed there, which is the
customer-concentration case. Upstream hops sourced from other companies'
filings remain in the rationale and the path record, where the hop-graph
verifier enforced them. Upstream limiting disclosures (hedges, expiries,
immaterial shares) are appended to the rationale as `LIMITING DISCLOSURES`,
so the critic sees them even though they are not candidate claims. If the
only relationship evidence sits in the near node's filing, the unchanged gate
marks the candidate `insufficient_evidence`. The gate was not relaxed to
accept it.

**Memory** (`src/events/hop_graph_memory.py`, migration
`026_niche_edges.sql`, additive). Sourced and contradicted edges are stored
with the primary source hash, exact passage, locator and source record, plus
an evidence bundle holding every supporting and limiting claim and its
source (both endpoints' filings). Each row also has the theme, `as_of` and
`expires_at = as_of + 730 days`, the same two-filing-cycle policy as
documents. Before traversal:
- Non-expired sourced edges for the theme, recorded at or before the cutoff,
  pre-seed the graph. Every bundled source is re-hashed and every claim
  passage re-checked; each edge must still be usable at the cutoff. Reuse
  respects the per-node edge cap.
- Non-expired contradicted edges from any theme veto both re-proposals and
  remembered sourced edges with the same key. They are listed to the model as
  `do_not_propose`. Vetoes are loaded first and stay in force if loading
  sourced memory later fails; that failure means only "no reuse". If the
  vetoes themselves cannot be loaded, no cached edge is reused, because a
  cached edge cannot be checked against contradictions.

Every frontier node is still sent to the traversal model, because directions
and terminal facts are specific to the event and are never stored. Memory
therefore saves retrieval and entailment work on known edges, not traversal
calls.

Unverifiable edges carry no evidence and are not stored. A memory outage or
corrupt row means "no memory", never evidence. The memory never supplies a
trade direction.

**Operator CLI models (CL-cnwz):** traversal, entailment, criticism and the
optional baseline all use strict subscription-only `claude-sonnet-4-6`, with
tools disabled. API baselines and other model overrides are rejected before
calls. Component-level injectable defaults do not override this CLI policy.
Authentication and included-usage attestation remain mandatory; no paid
fallback is enabled. USD cost is unknown (`subscription_unmetered`), never $0.

Every role shares one thread-safe per-event CLI invocation budget, default
three. An operator can explicitly set a bounded experiment allowance. Quota,
authentication, transport and budget failures stop subsequent reservations and
invalidate eligibility for that event; already in-flight calls can finish.
The cap counts CLI generations, not internal turns, tokens or dollars. Graph
work runs first, so a shared cap is **not equal allocation between arms** and
does not establish an equivalent-budget comparison.

**Shadow comparison** (`scripts/niche_hopgraph_shadow.py`,
`src/events/hop_graph_shadow.py`). The script replays captured events (the
`CapturedInput` format above) through the hop graph and, optionally, the
existing equivalent-tool baseline. Both arms get the same identity universe,
market data, cutoff and critic instance. For each event and in total, the
report covers:
- candidates, evidence-gate passes and eligible ideas;
- hop-depth distribution;
- traversal, entailment and tool calls;
- tokens and cost provenance.

Unit tests run it end to end on fakes only. An operator runs it like this:

```sh
.venv/bin/python scripts/niche_hopgraph_shadow.py \
  --captures /path/to/captures/ --output /path/to/new-report.json \
  --baseline claude-subscription --max-model-calls-per-event 3 \
  --memory-sqlite /path/to/hopgraph-memory.sqlite --allow-model-calls
```

Without `--allow-model-calls` the script validates the captures and stops.
The script refuses to overwrite a report. Edge memory goes only to a local
sqlite file. The production `.env` and database are never read.

The original CL-ynuh acceptance called for at least 20 captured events and a
Kimi comparison. That API arm is no longer allowed by the subscription-only
operator policy. A Claude baseline can support engineering evaluation, but
does not fulfill the original model comparison. Quality evaluation still needs
adequate independent captured events, comparable budgets and human evidence
review, measuring:
- eligible ideas per event;
- hop-depth distribution;
- gate pass rate;
- tool calls and cost.

Development produced **no live measurement**; only the code and tests exist.
Substring provenance plus a cheap entailment answer does not prove truth. Human
evaluation of the eligible ideas remains necessary.
