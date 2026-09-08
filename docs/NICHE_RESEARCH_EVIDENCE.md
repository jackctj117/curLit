# Evidence-grounded niche research

Implemented under Beads epic **CL-eh28**, with status/persistence (.1),
source claims (.2), criticism/scoring (.3), and shadow comparison (.4).
Beads remains the only task tracker. This document describes the code, not a
completed provider benchmark or a deployment approval.

## Outcomes, leads, and the trading feed

`NicheAgent.run_report()` returns an invocation-local discovery outcome and all
parsed research candidates. There is no shared last-result field across worker
threads. `scripts/event_pipeline.py` stores this under
`assessment.niche_research`, including failures and valid abstentions with zero
trade ideas. Merges use a private copy and reach the in-memory ledger inputs
only after a committed one-row assessment update. DB exceptions, commit failures
and changed event status cannot leak new niche ideas into the later ledger step
(CL-eh28.1.1). A persistence failure is logged; this change does not introduce a
durable retry queue for failed assessment writes.

Discovery distinguishes `completed`, `abstained`, `partial`, `invalid_output`,
`unavailable`, and `budget_exhausted`. Empty valid JSON is abstention; malformed
JSON or an API/billing outage is not. Missing credentials for an explicitly
enabled Kimi path produce `unavailable`, not an implicit provider switch.
Provider errors are classified without copying account/key identifiers to logs.

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

**Intentional behavior change:** research failures preserve leads, not trading
eligibility. This may substantially reduce new niche entries, including when
the critic is disabled or the configured provider is unfunded. No daemon has
been restarted or model configuration changed by this implementation.

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

Research retrieval selects up to three recent 10-K/10-Q/8-K primary documents
by filing date, instead of always preferring an annual report. An optional
query locates relevant text; normalized-text offsets and accession identify
the passage. This includes current-report announcements, not an exhaustive
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

The Claude challenger uses the **Anthropic API**, not Claude Code: inspection
found that the existing CLI driver accepts but does not enforce `max_tokens`
(CL-h7c1). The same API-based critic is applied to both arms. Runtime Kimi
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
