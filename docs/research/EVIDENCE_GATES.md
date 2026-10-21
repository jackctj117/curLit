# Research evidence contract (CL-0deu.6 / CL-z4fu)

The CL-7ys8 patch was completed in the proposal-preview worktree, but was not
present in this checkout. CL-z4fu integrates its inspected backtest/verdict
changes without changing the other worktree or importing its unrelated work.

Missing columns, too few folds, all-zero or malformed/nonfinite Sharpes produce
unknown diagnostics (`null`), not favorable defaults. Available fold proxies
retain `_metric_evidence` with status, method, observation count and reason.
They cannot satisfy measured-evidence gates. Legacy provenance labelled PROXY
is also inconclusive. Failed bootstrap confidence intervals remain unknown.
Null, wrong-type and nonfinite values cannot pass inequality or negative-
membership rules. Adequately measured failures remain distinct from unknowns.

Instrument declarations must be a nonempty list or tuple of distinct, exact
uppercase data identifiers. Missing declarations and failed construction never
substitute EURUSD. Multiple inputs require an explicit execution symbol; the
single-input case permits inference. An explicit invalid/empty target never
falls back. This runner supports a single target, not a declared execution
basket. Signal inputs may include macro series, but the execution target must
be a supported trading instrument in the existing backtest metadata registry.
Registry membership is not proof of current venue eligibility or historical
liquidity. Invalid contracts fail before data retrieval.

The old favorable-default tests intentionally changed to assert unknowns or
contract errors. Independent numerical vectors, property tests and strict JSON
roundtrips cover the new requirements; tests are not skipped to hide defects.

CL-0deu.6 now requires `_metric_evidence` for every quantitative threshold in
promotion decisions, including legacy reports. Records require measured status,
a positive integer observation count and a nonempty calculation method. A
field-specific record overrides its parent record: the runner's position-change
count remains a proxy for completed trade lifecycles, even though its other OOS
return diagnostics are measured. Missing metadata is inconclusive, never a pass.
The standalone arithmetic evaluator remains useful for diagnostic legacy reads;
`compute_verdict` always requires evidence.
An empty quantitative rule set, or an A/B rule whose threshold is absent,
is inconclusive rather than a vacuous pass. C/D qualitative rules retain
their existing reviewer-driven contract.

The registrar reloads the report and current rules and repeats the quantitative
gates before moving code, modifying portfolio configuration, or publishing. A
missing/malformed report, empty rules document, proxy, or failed gate prevents
all registration side effects. Report strategy identity must match the requested
strategy. Gate 2 records the exact debated report SHA-256 and refuses changed
reports or legacy approvals without that identity before calling the registrar.
The registrar hashes the exact bytes it parses against that expected identity,
so replacing the report between the loop's check and registration is rejected.
This is necessary but not sufficient approval: binding approval to exact
code/config/data hashes and eliminating concurrent file-change races, plus
isolated generated-code execution remain CL-0deu.11. The manifest describes
evidence; it cannot authenticate a dishonest producer.

Dry-run synthetic diagnostics now intentionally ESCALATE rather than PROMOTE.
They must not be relabelled measured merely to exercise the registrar. Unit
fixtures separately test valid measured decision tables and rejected legacy,
partial, proxy and synthetic evidence. No strategy was promoted or evaluated
against operational data for this patch.
