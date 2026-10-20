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

This is not full promotion or research-isolation completion. CL-0deu.6 remains
open for a mandatory complete measured-evidence manifest on every promotion
path. Bare legacy scalar reports still follow the threshold contract. Artifact-
bound approvals and isolated generated-code execution remain CL-0deu.11.
No strategy was promoted or evaluated against operational data for this patch.
