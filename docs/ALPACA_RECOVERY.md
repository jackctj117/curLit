# Controlled recovery: evidence before account repair

The approved **close-only recovery release `adddbe8` is deployed** to Alpaca
and FX as of September 9, 19:04 UTC. The initial investigation below is retained
as historical evidence; the operational cutover section records what actually
changed. CL-koeg remains open for staged verification, and CL-0deu.3 is not a
claim of complete accounting or new-entry readiness.

## Initial September 9 inventory (before the approved repair)

GET/SELECT capture at 15:50:47–15:50:50 UTC exhausted the API-visible order and
activity histories: 317 orders over two requests and 781 activities over nine.
Inventory and internal submitted-order rows were stable across capture. This
does not establish lifetime completeness or rule out delayed paper activities.

All six internally pending option exit orders were broker `filled`, with their
cumulative quantities matching individual FILL activities. Each contract still
has a one-contract net broker position. Do not resend those six exits: reconcile
the closed allocation separately from later inventory and preserve original
estimates before any fill-based accounting correction.

The seven unmatched holdings have proposed original-entry restoration targets:

| Holding | Signed quantity | Original idea |
| --- | ---: | --- |
| DAC | 6 | `33d512286c956720` |
| DHT | 51 | `9186ee136d981087` |
| FRO | 22 | `0bce0493c5f79790` |
| LMT | 1 | `8bdf777aa4b1f505` |
| MGA | -15 | `b5fe21bbcf7fe951` |
| RTX | 4 | `40b4809a38dbeab1` |
| STNG | 12 | `ab7a80954f30733f` |

These proposals require stored broker ID, original client ID, broker asset ID,
matching individual fills, net quantity conservation, existing idea identity,
and no later/other activity in captured history. They are not automatic adoption
instructions. The original rows are marked `closed_external`; original dates
must be retained, so restoring management can trigger overdue exits immediately.

Private artifacts on the operational host are under
`data/deployments/2026-09-09_alpaca_recovery_plan/`. Snapshot canonical SHA-256:
`770205555bd78f57df62677dcbdf3367d107ca370dbf79e5f8dc2037fe67ace7`.
The initial incomplete export remains preserved separately; its order-ID cursor
returned recent/changed records while crossing archived history. The revised
collector uses timestamp paging and explicitly refuses to claim complete
coverage across a full-page timestamp boundary it cannot prove covered.

## Read-only tooling

Offline replay loads no environment file, broker client or operational database:

```bash
python -m scripts.alpaca_recovery_report \
  --snapshot /private/evidence/snapshot.json \
  --output-dir /private/evidence/new-replay
```

Only an explicitly authorized operator should capture account evidence:

```bash
python -m scripts.alpaca_recovery_report --capture-paper \
  --env-file /operator/approved/.env \
  --output-dir /private/evidence/new-capture
```

The output parent must already exist. New directories are mode 0700 and files
0600, exclusively created. The broker adapter uses only GET against the fixed
paper host; SQL runs in read-only repeatable-read transactions. Incomplete
history exits nonzero while retaining available evidence. The report contains
row/snapshot fingerprints, coverage limitations, and proposals, **not executable
SQL or order commands**. An applier must freshly verify row, order, position,
account and configuration preconditions; a saved fingerprint is not a lock.

Broker references: [order history](https://docs.alpaca.markets/us/reference/getallorders-1),
[individual activities](https://docs.alpaca.markets/us/reference/getaccountactivities-2),
[original client-order identity](https://docs.alpaca.markets/us/reference/getorderbyclientorderid),
[paper options activity timing](https://docs.alpaca.markets/us/docs/options-trading).

## Development changes and boundaries

CL-qz3f prevents both exit loops from treating a position consumed by another
row as absent. Multiple active rows sharing one symbol now remain unchanged,
with explicit `reconciliation_required` counters and `ambiguous_allocation`
logs. This is containment, not an allocation ledger; it deliberately does not
invent which idea owns a net position. Single-row disappearance inference,
aggregate/external ownership, uncertain submissions, canceled exits and realized
cost allocation still require CL-0deu.3 before historical restoration.

CL-wtl6 adds exact broker asset preflight. Unknown/malformed metadata and
non-404 failures block entry. Shorts require explicit shortability and
easy-to-borrow support; this code has no locate workflow. A confirmed asset 404
becomes `skipped_unsupported_asset` only after the original client-order lookup
also confirms absence. The research idea is retained, no foreign suffix is
removed, and no order POST occurs. That expression stays skipped until an
operator reviews a new expression; do not delete its skip row to force a retry.
Existing timeout/duplicate-ID lifecycle debt is not declared solved by preflight.

CL-jk7i adds migration 021 and a bounded, durable GDELT slice in the existing
pipeline. It retains request spacing, honors source-wide Retry-After, persists
pending windows before requests, and advances coverage only after persistence.
A source lease fences concurrent slices; no second assessment worker is created.
Fair rotation and shrinking capped windows allow later resumption. Non-JSON,
malformed articles and missing dates are failures, not successful empty batches.
The default 60-second budget covers network calls and waits; database availability
is also required and is not implied by that network budget. Missing migration or
ingestion errors are logged and do not skip the existing-event assessment phase.
The older standalone `GdeltIngester.run()` API remains unchanged; the pipeline
uses `run_slice()` and must not run an additional legacy ingestion writer.

CL-uofe adds opt-in `niche-passages-v1` to native Kimi and both arms of the
equivalent-tool shadow harness. Use `passage_references=True` explicitly when
constructing `KimiToolAgent` or calling `compare_captured`. Defaults and running
research are unchanged. One generated schema controls prompts and validation;
model claims reference invocation-local labels, while code supplies exact
captured text/hashes. Unknown references and malformed claims remain invalid
output. The original response stays in the audit trace. Source provenance is
not semantic support: company/date, liquidity, critic and eligibility gates
remain intact. No paid comparison or improved model-quality claim is included.

CL-00c7 replaces ambiguous critic “survived” summaries with explicit supported,
contradicted, insufficient-evidence, unavailable and unreviewed counts. Retained
research leads and trading-eligible candidates are labeled separately.

## Validation evidence

Development checks on September 9, using the existing Python 3.14 interpreter
with credential environment variables removed:

- `python -m pytest tests/unit -q -o addopts=''`: **3817 passed, 3 existing skips**,
  exit 0. No existing test assertions were weakened. The equity fake client was
  extended at the new asset-metadata boundary; shared-allocation regressions
  first failed on the original exit loops before the guard was implemented.
- `python -m pytest tests/integration/test_recovery_postgres.py
  tests/integration/test_niche_audit_postgres.py -q -o addopts=''`: **7 passed**,
  exit 0, against a new loopback-only disposable PostgreSQL instance. Migration
  idempotence, actual read-only transactions, concurrent leases, crash/replay and
  existing independent audit persistence were exercised. The test instance was
  removed afterward; operational PostgreSQL was not used for these tests.
- `ruff check src/ tests/ scripts/alpaca_recovery_report.py scripts/event_pipeline.py`:
  exit 0. `make typecheck`: exit 0, 229 source files. Strict mypy also passed on
  the five new production/report modules.
- CI-equivalent `bandit -r src/ -ll`: exit 0, zero findings and parser errors.
  Working-tree Gitleaks: exit 0, no detected leaks. This is not a new dependency
  audit or an independently approved code review.

Validation artifacts are under the development worktree's ignored
`data/validation/`. The earlier flaky sizing oracle CL-1vqh was not modified;
passing this sample does not establish that unrelated issue is resolved.

## Operational handoff

Apply no corrections before operator approval, a fresh backup, preserved broker
evidence, entry pause, allocation/fill-ledger implementation, and compare-and-swap
validation. Do not reset paper accounts or run old/new order writers together.
Migration 021 was exercised only in a disposable PostgreSQL instance; operational
application and before/after GDELT cycle measurements remain outstanding.

FX PID 28922 still has a July 31 start time. Its watcher stores baseline hashes
only in memory and covers a curated subset of files; current checkout HEAD is
not evidence of the entire loaded release. This patch does not restart FX or
suppress drift. Baseline/cutover and recovery gates CL-0deu.14/18 remain required.

## Authorized recovery implementation, September 9

The operator subsequently authorized the remaining work, including restoration
of the seven original equity allocations and their original holding clocks.
Authorization does not waive fresh evidence, backup, writer fencing or tests.

Migration 022 adds immutable broker activities/fills, durable order intents and
attempts, decimal per-idea allocations, unknown-cost status, and an original-value
repair audit. Gross realized cashflows come from individual executions using
average-cost allocation; net remains NULL when costs are unknown. Contract sizes
come from captured broker metadata, not an assumed multiplier or a guessed OCC
root. The morning digest's closed results now read this ledger, not submitted
orders or differences between estimated premiums. Other legacy consumers must
not treat old quote fields as independently verified realized accounting.

`ALPACA_LEDGER_CLOSE_ONLY=1` selects the upgraded exit path in the existing two
daemons. It reconciles before considering any order, holds an account-wide
PostgreSQL session advisory lock without a long SQL transaction, checks net
inventory conservation and other working orders, and limits each exit to its
proven allocation. One submission per fresh cycle prevents later actions using
the same stale snapshot. There are at most three exit attempts; a new attempt
requires confirmed terminal prior orders, refreshed fills and verified residual
inventory. Unknown submission outcomes retain their original client identity
and never trigger a blind resend. Options explicitly use sell-to-close intent;
equity reductions preserve signed fractional quantities. Missing evidence blocks
management, and no new entry path runs in this mode. Legacy writers must be
stopped before cutover; the advisory lock cannot fence a manual external broker
order or an old process that does not participate.

This is a **close-only recovery rollout**, not completion of full new-entry
reservations, cross-venue halt acknowledgements, price-bounded entries or live
readiness. Those remain within the existing CL-0deu dependencies. Do not disable
the persistent close-only switch to bypass incomplete entry integration.

The fresh 18:09 UTC capture still has the six pending exits and seven unmatched
equities, with complete API-visible history and broker contract sizes. It also
exposes four historical option and two equity aggregate exits exceeding their
own idea allocations (CL-sweu). They remain unresolved rather than assigning
excess fills to arbitrary ideas. Separately evidenced current entries after a
broker execution stream returns to flat can be managed; prior accounting gaps
are not silently erased. Non-trade activities and external inventory remain
explicit blockers where ownership cannot be established.

`scripts/recovery_backup.py` produced an AES-256-encrypted current database
snapshot and fully restored it into a network-isolated disposable TimescaleDB
container. All 34 public-table counts matched the same exported source snapshot;
the archive round-trip hash matched. Counts are not row-by-row equivalence. The
private backup and manifest are in `data/deployments/2026-09-09_recovery_restore/`.
The test container was removed. Two disposable rehearsals applied all 13
proposed corrections and then replayed all 13 without further mutation. No
operational corrections are implied by these rehearsal results.

`scripts/apply_alpaca_recovery.py` requires explicit paper-repair authorization,
an intact fully restored backup, stopped writer processes, a persistent close-only
switch and individually approved equity IDs. It captures fresh broker evidence,
refuses working/unknown orders and changed rows, and stores original values before
corrections. New accounting fields never overwrite missing fills with quotes.

FX startup now halts new entries when cold-start reconciliation is missing,
fails or reports mismatches; monitoring remains available. The explicit
`CURLIT_START_ENTRY_PAUSED=1` cutover setting survives successful reconciliation
until separately resumed. A fresh-process manifest records commit, full source
hashes, resolved risk/strategy configuration, executable and Python version;
unavailable lock/migration evidence remains labeled unknown.

Native Kimi baseline versus passage references was run with the same frozen
event 48489, tools, source collection and resource ceilings. Both abstained,
with no candidates or gate violations. This engineering result is inconclusive
about evidence-quality improvement and does not justify a profitability claim.
Passage references remain opt-in; no model/provider switch is included.
Private trial output: `data/deployments/2026-09-09_passage_quality/`.

Current local checks: **3,837 unit tests passed, three existing skips**, and
**14 disposable PostgreSQL integration tests passed**. Source typechecking,
strict checking of nine new ledger/operator modules, Ruff, workflow actionlint,
medium/high Bandit and secret scanning pass. CI now also runs the disposable
recovery integration suite. Publishing, migration application and each daemon's
verified operational cutover are separate evidence recorded under CL-koeg.

## Authorized operational cutover, September 9

Release `adddbe809453d39fbf504d54f8a8ab48f31ad8b4` was committed, pushed and
fast-forwarded into the operational checkout after hosted
[CI 34390896602](https://github.com/jackctj117/curLit/actions/runs/34390896602)
and [security 34390900171](https://github.com/jackctj117/curLit/actions/runs/34390900171)
passed. These are actual results, not just added workflow definitions.

Both old Alpaca writers were stopped and verified absent. A second fresh full
backup was encrypted, restored into a disposable isolated TimescaleDB instance,
and checked against all 34 table counts from the same exported source snapshot.
The archive and restore manifest are private under
`data/deployments/2026-09-09_recovery_cutover_backup/`. The pre-cutover `.env`,
configuration and persisted strategy/risk clocks also have an encrypted,
round-trip-verified archive. No paper account was reset.

Migrations 021 and 022 committed together at 18:51 UTC. Fresh broker evidence
and locked compare-and-swap repair applied **13 corrections**: all six original
pending option exits now have fill-backed closed dispositions, and all seven
approved equity allocations have management restored using their original
dates. Original values remain in `alpaca_ledger_repairs`. The repair submitted
no orders. Snapshot and result are private under
`data/deployments/2026-09-09_recovery_applied/`; snapshot hash is
`c58da7e35f24a99fc29ce2dd9a478124cad837a83a71adbd48ce0d856136d8bd`.
Closed legacy `pnl_pct` is unknown rather than an estimated realized return.

The replacement options and equity workers are PIDs **5004** and **5006**,
respectively, using the existing 300-second cadence and persistent
`ALPACA_LEDGER_CLOSE_ONLY=1`. They reconcile on each cycle and serialize through
the account advisory fence. New entries remain disabled. FRO's restored
22-share allocation sold at a broker-confirmed average **$47.39** at
18:57:50 UTC; its actual fill was subsequently ingested into the ledger.
This is a verified management/fill cycle, not a claim that all seven holdings
have already exited. Gross accounting is available; missing costs/net remain
explicitly unknown.

By 19:12 UTC, DAC's six shares and RTX's four shares also have broker-filled
orders and zero residual allocations in the ledger. All seven approved
allocations remain management-enabled; DHT 51, LMT 1, MGA -15 and STNG 12 still
remain under the original exit policy. A read-only audit confirmed all 13 repair
records preserved `submitted_at`. No cadence acceleration or forced liquidation
was used to make the verification finish sooner.

The old July 31 FX writer PID 28922 exited after an explicit entry halt and a
fresh OANDA-practice check of zero positions, open trades and pending orders.
Replacement PID **5365** runs the approved release with the unchanged aggressive
risk profile. Its cold-start reconciliation found zero entries/mismatches;
all **256** recorded source/configuration hashes match the fresh-process
manifest `data/runtime_manifests/engine_5365.json`. It starts entry-paused;
resumption requires separate successful runtime verification. Persisted strategy
holding clocks were not rewritten. The morning digest was restarted separately,
retaining today's sent-state, to load fill-ledger reporting without a duplicate
message.

Further reconciliation identified **six additional current option allocations**
incorrectly marked `closed_external`: FRO $47, DHT $20/$21, STNG $85 and RTX
$230/$240 calls, all September 18 expiry, one contract each. These are different
allocations from the six already-filled exits. Their automatic management is
**not enabled** pending explicit policy approval and fresh repair safeguards
(CL-jr3z). Four historical option and two equity aggregate overcloses also
remain unresolved under CL-sweu. No arbitrary ownership or zero-cost assumption
was used to make historical reporting appear complete.

Pipeline PID **6071** replaced PID 58060 after the old cycle's completion marker
and verification that no model child remained. The old cycle spent **614,933 ms**
on ingestion and **1,214,867 ms** overall. The new process's first GDELT request
received a real 429, persisted the retry window/status without advancing covered
history, released its lease and yielded after approximately **13 seconds**.
It recorded all 16 themes deferred; the unchanged volume scan then finished and
assessment began at **19:09:40 UTC**. The configured ingestion ceiling is 60
seconds. This is evidence of reduced scheduling blockage, not increased source
availability or causal improvement in model output.

Captured account activities include 350 fees: 220 carry execution references
matching captured fill-ID suffixes; 130 have no such reference. All raw records
are preserved. CL-z97c tracks validated fee links and visible account-level
unallocated costs; fully verified per-trade net remains unknown until applicable
costs can be attributed without arbitrary assumptions. Restoring account
management and completing historical accounting remain distinct milestones.
