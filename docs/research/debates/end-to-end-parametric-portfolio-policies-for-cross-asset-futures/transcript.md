# Debate transcript — promotion_review — end-to-end-parametric-portfolio-policies-for-cross-asset-futures

Started: 2026-07-20T16:11:32+00:00
Participants: bull_reviewer, bear_reviewer
Rounds: initial_positions, smart_questions, rebuttal, final_position

## Round: initial_positions — agent: bull_reviewer (bull)
*ts=2026-07-20T16:13:41+00:00 model=claude-fable-5 in/out_tokens=168084/7815 cost=$0.0000 elapsed=112.62s*

I now have complete information about what evidence exists: the candidate report and REVIEW_RULES.md are in context, but the strategy source file, per-fold breakdown, and test/mypy outputs are **not available** in this environment, and the `knowledge_search` archive tool is not loadable here (ToolSearch resolves only web/search tools). Per my standing instructions, I will cite what exists and explicitly ABSTAIN where evidence is absent rather than fabricate line numbers or archive quotes.

---

# PROMOTE_CASE for end-to-end-parametric-portfolio-policies-for-cross-asset-futures

*Bull Reviewer — Round 1 (initial position). All metric citations are from `reports/candidates/end-to-end-parametric-portfolio-policies-for-cross-asset-futures.json` (candidate report, schema_version 1). The strategy source at `src/strategies/_experimental/end-to-end-parametric-portfolio-policies-for-cross-asset-futures.py` was not provided in this review context and could not be read from disk; I abstain on every rule requiring `file:line` evidence rather than invent citations. The knowledge archive (`knowledge_search`) is unavailable in this session; no historical-precedent quotes are offered rather than fabricated.*

## Section A — Statistical reality

**A.1 (Sharpe ≥ 0.50): FAIL.** Evidence: `oos_metrics.sharpe = 0.18697` (candidate report). Below the 0.50 floor by a wide margin. Bull mitigation for the record, not the verdict: 0.187 falls **inside the pre-registered prediction band `[0.0, 0.6]`** from `prediction.predicted_sharpe_range`, which the report explicitly down-scoped from the brief's [0.3, 0.8] because this is a ~1.1k-weight SPSA-trained miniature of the paper's transformer with gross leverage capped at 1.0 (`prediction["Note on the deltas vs the brief"]`). The result is honest and expected — but the rule reads the metric, and the metric fails.

**A.2 (CI lower > 0): FAIL.** Evidence: `sharpe_ci_95.low = -0.9319` (candidate report). Zero is well inside the interval `[-0.9319, +1.3682]`. This is the rule REVIEW_RULES.md calls "the single most important gate," and it fails decisively — notably **despite** a well-powered sample (see A.3), so this is not a small-n artifact. I cannot construct a good-faith counter to this.

**A.3 (n_trades ≥ 30): PASS.** Evidence: `oos_metrics.n_trades = 819` — 27× the floor. The bootstrap CI in A.2 is therefore well-estimated and interpretable; the sample size itself is a genuine strength of this candidate's evaluation.

**A.4 (hit rate ≥ 0.45): PASS.** Evidence: `oos_metrics.hit_rate = 0.50721`, above the 0.45 floor and above 0.50, so it also clears the 40–50% sign-flip-review band entirely. Additionally within the pre-registered range `[0.50, 0.57]` (`prediction.predicted_hit_rate_range`) — the direction of the trade is not inverted.

**A.5 (max drawdown ≥ -0.25): PASS.** Evidence: `oos_metrics.max_drawdown = -0.05797`. At -5.8%, roughly a quarter of the catastrophic floor — the leverage-capped policy is structurally compatible with the engine's kill switches with large margin.

**A.6 (profit factor ≥ 1.10): FAIL.** Evidence: `oos_metrics.profit_factor = 1.03709`. Wins exceed losses by only ~3.7% in dollar magnitude versus the required 10%. Consistent with the low-but-positive Sharpe (`total_return = 0.01931`, `sortino = 0.23106`): real but thin.

**A.7 (IS/OOS ratio ≤ 2.5): FAIL — borderline.** Evidence: `is_oos_sharpe_ratio = 2.73569`, 9.4% above the 2.5 threshold. The rules invite a per-fold check to see whether the divergence is driven by a single anomalous fold, but **no per-fold breakdown is present in the candidate report**, so the borderline fail stands unrescued. Note the ratio's denominator is a small OOS Sharpe (0.187), which mechanically inflates this ratio; still, the threshold is the threshold.

## Section B — Edge structure

**B.1 (Edge concentration ≤ 0.60): PASS — strong.** Evidence: `edge_concentration = 0.20325` (G5 layer). Barely a third of the allowed ceiling; no single feature dominates. This is exactly the robustness profile a portfolio-level attention policy over nine pairs should show, and it did.

**B.2 (Regime diversified): PASS.** Evidence: `regime_diversified = true` (G6 layer) — edge is positive in ≥ 2 of 5 market regimes, matching the brief's "regime-agnostic" prediction (`prediction.regime_dependence`).

**B.3 (No active decay): PASS.** Evidence: `decay_severity = "NONE"` (G7 layer) — not in ['STRONG', 'MODERATE']. Whatever edge exists is not eroding on the OOS sample. The pre-registered abandon condition (rolling 126-day OOS Sharpe < 0, `prediction.abandon_threshold`) has also not triggered: OOS Sharpe is positive.

## Section C — Code integrity

**C.1 (No future-dated lookups): ABSTAIN.** NO EVIDENCE — the strategy source file was not provided in this review context and does not exist at the report's `code_path` in this environment; I cannot cite lines I have not read. `gates.syntax = "passed"` and `gates.backtest = "ran"` confirm the harness executed it, but say nothing about time-ordering.

**C.2 (≤ 5 free parameters): ABSTAIN.** NO EVIDENCE — parameter count requires reading the config/source. The report describes "~1.1k weights" trained by SPSA; learned weights are typically not "free tunable knobs" under this rule, but the hyperparameter count (e.g., the 0.15 L1 no-trade band, leverage cap, block size) is unverifiable without the file.

**C.3 (Realistic cost model): ABSTAIN.** NO EVIDENCE — no spread/slippage figures appear in the candidate report. Material to this candidate: the universe includes exotics (USDNOK, USDSEK — 1.5/1.0 bps floors) and the profit factor is only 1.037, so cost-model realism is decisive here. This should be surfaced as a Round 2 smart-question.

**C.4 (No PaperBroker constants): ABSTAIN.** NO EVIDENCE — source not available.

**C.5 (Standard Strategy protocol): ABSTAIN — weak supporting indication only.** The report's design description (portfolio weight-vector rebalances across 9 pairs) is consistent with the `generate_intents()` portfolio-level protocol, and `data_sources_consumed` documents harness-supplied wide-DataFrame consumption with the `close` alias explicitly dropped via `reindex(columns=self.symbols)` — a quoted code behavior indicating no engine-side patching was needed. But without the file I cannot confirm no `LiveEngine.run()` modifications.

## Section D — Operational fitness

**D.1 (Tests pass): ABSTAIN.** NO EVIDENCE — no `make test` output or test inventory in the candidate report.

**D.2 (mypy --strict clean): ABSTAIN.** NO EVIDENCE — no mypy output provided.

**D.3 (Property tests for numerical helpers): ABSTAIN.** NO EVIDENCE — the L1 no-trade band and SPSA machinery are numerical helpers that would require hypothesis tests; none are evidenced.

**D.4 (FeatureSnapshot emission): ABSTAIN.** NO EVIDENCE — no reference to `_emit_snapshot` (CL-xpw9) in the report.

## Evidence-first note (data consumption)

The brief-vs-implementation data check **passes cleanly**: `data_sources_consumed` lists all nine pairs from the brief's data-requirements section, explicitly notes "no external datasets, no FRED series, no `POLY:` features," and documents that the unused harness alias column is dropped rather than silently consumed. No listed-but-unconsumed dataset gap exists. This is the evidence-first discipline done right.

## Final position

**REJECT**

The strongest promote-side facts are real and I want them on the record: an exceptionally robust edge *structure* (B.1 edge concentration 0.203, B.2 regime-diversified, B.3 zero decay), a well-powered evaluation (A.3, n=819), benign risk behavior (A.5, -5.8% drawdown), and unusual research integrity — the realized Sharpe 0.187 and hit rate 0.507 both landed inside their pre-registered prediction bands, and the abandon threshold was honestly specified and not triggered. But the Bull's job is to cite, not to wish: four Section A rules fail on hard thresholds — A.1 (0.187 < 0.50), A.2 (CI low -0.932, zero comfortably inside despite 819 trades), A.6 (1.037 < 1.10), and A.7 (2.736 > 2.5, borderline but unrescued absent per-fold data). A.2 failing at this sample size means the edge is statistically indistinguishable from luck, and no structural virtue overrides that gate. The honest disposition is REJECT-and-iterate: the down-scoped 1.1k-weight SPSA implementation performed exactly as its own note predicted, and the right next step is a fuller-capacity implementation per the original brief (predicted Sharpe [0.3, 0.8]) resubmitted through this gate — not promotion of this version. I additionally flag for Round 2: the cost-model inputs for the exotic pairs (C.3) and the per-fold IS/OOS breakdown (A.7) are missing and material.

---

## Round: initial_positions — agent: bear_reviewer (bear)
*ts=2026-07-20T16:13:41+00:00 model=claude-fable-5 in/out_tokens=137342/9278 cost=$0.0000 elapsed=129.10s*

Before writing my case, I attempted to audit the strategy source (`src/strategies/_experimental/end-to-end-parametric-portfolio-policies-for-cross-asset-futures.py`) and to query the knowledge archive. The strategy file is not present in this review sandbox, and no `knowledge_search` tool is connected in this session. Per my severe-claim threshold, I therefore make **no** line-number-dependent claims (C.1, C.4) and quote **no** archive chunks — the candidate report and REVIEW_RULES.md alone are sufficient to decide this one, and Section C/D gaps are logged for the Round 2 smart-question stage.

---

# REJECT_CASE for end-to-end-parametric-portfolio-policies-for-cross-asset-futures

## Section A — Statistical reality (failures found)

**A.1 (Annualized OOS Sharpe): FAIL** — `oos_metrics.sharpe = 0.18697`, threshold `>= 0.50`. The candidate delivers **37% of the minimum floor**. This is not a marginal miss; the strategy would need to nearly triple its OOS Sharpe to clear the lowest bar we set. Note also the implied IS Sharpe (0.18697 × 2.7357 ≈ **0.51**) would itself only just scrape past A.1 — even before OOS degradation, this strategy was borderline.

**A.2 (95% bootstrap-CI lower bound > 0): FAIL** — `sharpe_ci_95.low = -0.9319`, threshold `> 0.0`. This is the single most important gate in the rulebook, and the candidate fails it emphatically: zero is not merely inside the interval, it sits **0.93 Sharpe units above the lower bound**, near the middle of a CI spanning [-0.93, +1.37] — a width of 2.30 Sharpe units. Under a normal approximation of that bootstrap interval, there is roughly a **one-in-three probability the true Sharpe is negative**. Evidence-first: this is precisely where a real edge would be publicly visible in the candidate's own data, and it is not there. The edge is statistically indistinguishable from luck — on 819 trades, which is not a small sample. That the CI is this wide with this much data means the per-trade signal-to-noise is close to nil.

**A.6 (Profit factor): FAIL** — `oos_metrics.profit_factor = 1.0371`, threshold `>= 1.10`. Gross wins exceed gross losses by only 3.7%, against a required 10%. Combined with a hit rate of 50.72% (see note below), the P&L structure is a coin flip with a rounding error.

**A.7 (IS/OOS divergence): FAIL** — `is_oos_sharpe_ratio = 2.7357`, threshold `<= 2.5`. The strategy retains barely a third of its in-sample performance out of sample. This is the canonical overfitting signature, and it has a mechanism: the candidate report itself states the policy is "a deliberately tiny single-block attention policy (**~1.1k weights**) trained by SPSA." Eleven hundred trained parameters fit against daily FX closes, with a 2.74× IS/OOS collapse, is textbook capacity-driven overfit. Per-fold breakdown is not included in the report, so the possibility that a single anomalous fold drives this cannot even be checked — that absence is itself a gap (Round 2 question Q1).

**A.4 — supporting note (not a FAIL):** `hit_rate = 0.5072` clears the 0.45 floor, but with n = 819 the standard error on the hit rate is ≈ 1.75 percentage points, so 50.72% is **0.4 standard errors above a fair coin**. This does not violate A.4; I cite it as corroboration of A.2 — there is no detectable directional skill.

## Section B — Edge structure (failures found)

**No B-rule violations on the reported values.** `edge_concentration = 0.203` (≤ 0.60, pass), `regime_diversified = true` (pass), `decay_severity = "NONE"` (pass).

One caveat for the record, not a rule violation: with a Sharpe whose 95% CI straddles zero by ±1, the G6 finding of "positive edge in ≥2 of 5 regimes" is a decomposition of statistical noise. Regime-diversified zero is still zero.

## Section C — Code integrity (failures found)

**No citable violations — code not auditable this round.** The strategy file at `src/strategies/_experimental/end-to-end-parametric-portfolio-policies-for-cross-asset-futures.py` is not present in the review sandbox. Per the severe-claim threshold, I assert no C.1 (look-ahead) or C.4 (PaperBroker constant) violations without line citations. However, two SUSPECT flags from the report's own text, escalated to Round 2:

- **C.2 (parameter discipline) — SUSPECT.** The report states ~1.1k trained weights plus, at minimum, these visible hyperparameters: L1 no-trade band (0.15), gross leverage cap (1.0), attention block dimensions, and SPSA training schedule parameters. Whether "config knobs" exceed 5 cannot be counted without the file, but a 1.1k-parameter policy is exactly the effective-degrees-of-freedom regime C.2 exists to control, and A.7's failure is the empirical confirmation. **Q2 for Round 2: exact count of free config knobs; ablation study status.**
- **C.3 (realistic cost model) — UNVERIFIED.** The universe includes **USDNOK, USDSEK, and NZDUSD — FX exotics carrying a 1.5 bps spread + 1.0 bps slippage floor**, higher than the majors. The report nowhere states the cost assumptions used, only that the abandon threshold is on "net" Sharpe. With a profit factor of 1.037, even a fraction of a basis point of understated cost on the exotic legs would push this strategy below water. **Q3 for Round 2: cost parameters per pair, and whether exotic floors were applied.**
- **Bootstrap unit-of-resampling — SUSPECT.** `n_trades = 819` appears to count per-pair fills (~60 rebalances/yr × up to 9 pairs). Fills within one rebalance are cross-sectionally correlated; if the stationary bootstrap resampled at fill level rather than rebalance level, the already-failing CI in A.2 is *optimistically narrow*. **Q4 for Round 2.**

## Section D — Operational fitness (failures found)

**No affirmative violations found — but no affirmative evidence exists either.** The report's `gates` block shows only `syntax: passed` and `backtest: ran`. There is no evidence for D.1 (`make test` + 3 strategy-specific tests), D.2 (`mypy --strict` clean), D.3 (property tests for the sizing/smoothing helpers an attention-weight policy necessarily contains), or D.4 (`_emit_snapshot` / FeatureSnapshot emission). Under the verdict mapping, missing required evidence routes to ESCALATE — but it never gets that far here, because Section A already fails four times.

## Historical precedent

The knowledge archive is not reachable from this session, so I quote no chunks and cite no page-level precedent — consistent with my fabrication constraints. I will note only what is already embedded in REVIEW_RULES.md itself: rule A.7 exists precisely because high-capacity models that retain <40% of IS performance OOS are the standard backtest-overfitting profile, and rule A.2 exists because a CI containing zero means the strategy has not distinguished itself from the null. This candidate fails both simultaneously. Archive-backed precedent (e.g., convergence-trade and ML-overfit case studies) can be attached in Round 3 if the tool becomes available.

## Final position

**FINAL_POSITION**: REJECT

The candidate fails four independent Section A gates, any one of which is deterministically fatal under the verdict mapping. The strongest three: **A.2** — the bootstrap CI [-0.93, +1.37] contains zero near its center, so on 819 trades the edge is indistinguishable from luck, with roughly a one-in-three chance the true Sharpe is negative; **A.1** — OOS Sharpe of 0.187 is barely a third of the 0.50 floor; **A.7** — a 2.74× IS/OOS collapse with ~1.1k trained weights is capacity-driven overfitting with the mechanism stated in the candidate's own report. I add that the researcher's own pre-registered prediction band was [0.0, 0.6] — a band that includes "no edge at all" — and the realized result landed at the bottom of it, exactly as a null strategy would. Nothing here requires Bear aggression; the candidate's own numbers reject it. To Bull's credit, the report is honest and the B-layer diagnostics are clean — this looks like a faithfully executed experiment whose answer is "no." The right disposition is REJECT to `data/research/rejected/`, with the Round 2 questions (per-fold breakdown, knob count, exotic-pair cost floors, bootstrap resampling unit) recorded for any future re-submission at higher capacity.

REJECT

---

## Round: smart_questions — agent: bull_reviewer (bull)
*ts=2026-07-20T16:17:26+00:00 model=claude-fable-5 in/out_tokens=111061/8248 cost=$0.0000 elapsed=118.10s*

The repository artifacts are not reachable from this sandbox: the working directory is an empty scratch dir, `docs/research/SMART_QUESTIONS.md` does not exist at any reachable path, and the shell itself is broken (dyld failure on every command). This matches the Bear's Round-1 experience. Since I cannot read the spec file, I emit my questions with six explicitly-named fields — `id, question, rule_ids, why_it_matters, route, resolution_criteria` — and flag the spec-access failure itself to the orchestrator below.

---

# SMART_QUESTIONS — Round 2 — Bull Reviewer
**Candidate:** `end-to-end-parametric-portfolio-policies-for-cross-asset-futures`

**Honest framing before the questions.** Four Section A thresholds fail deterministically on the candidate report as given — A.1 (`oos_metrics.sharpe = 0.18697` < 0.50), A.2 (`sharpe_ci_95.low = -0.93191` ≤ 0), A.6 (`oos_metrics.profit_factor = 1.03709` < 1.10), A.7 (`is_oos_sharpe_ratio = 2.73569` > 2.5). Per the verdict mapping, any single A-rule failure → REJECT regardless of agent positions. **No answer to any question below can flip this verdict.** These questions exist because (a) Section C/D evidence is entirely missing, which the audit record must reflect accurately, and (b) A.7 imposes an explicit per-fold confirmation duty on agents that I currently cannot discharge. They are questions for the record and for resubmission guidance, not attempts to relitigate thresholds.

```yaml
id: BULL-Q1
question: >
  The strategy source at
  src/strategies/_experimental/end-to-end-parametric-portfolio-policies-for-cross-asset-futures.py
  is not accessible from the review sandbox (file absent; shell broken with a
  dyld/libllhttp load failure on every command). Can the orchestrator inject
  the file contents (or a working read path) so rules C.1, C.2, C.3, C.4, and
  C.5 can be assessed with real file:line citations?
rule_ids: [C.1, C.2, C.3, C.4, C.5]
why_it_matters: >
  REVIEW_RULES.md Section C requires file:line evidence — "Bear cannot
  vague-reject; Bull cannot vague-defend." Without the source, both agents
  must ABSTAIN on all of Section C, and specifically C.3 cannot be checked
  even though the strategy trades three exotic pairs (USDNOK, USDSEK, NZDUSD
  per data_sources_consumed) that carry the higher 1.5 bps spread / 1.0 bps
  slippage floors. Cannot flip the Section-A-driven REJECT, but leaving C
  unassessed on a report whose sandbox note ("numpy-only sandbox", SPSA
  training) suggests unusual implementation constraints would be an
  incomplete audit record.
route: code_tool  # fallback: human escalation if the orchestrator cannot inject the file
resolution_criteria: >
  Full file contents (with line numbers) provided in Round 3 context, OR the
  orchestrator confirms the artifact is unavailable, in which case both
  agents' Section C entries are recorded as ABSTAIN-for-missing-artifact and
  the missing-artifact fact itself is logged with the rejection.
```

```yaml
id: BULL-Q2
question: >
  The candidate report contains the aggregate is_oos_sharpe_ratio = 2.735694
  but no per-fold breakdown. Can the harness supply per-fold IS and OOS
  Sharpe values for the walk-forward folds?
rule_ids: [A.7]
why_it_matters: >
  A.7's rule text explicitly instructs: "Agents should look at per-fold
  metrics to confirm the divergence isn't driven by a single anomalous
  fold." The ratio fails the 2.5 threshold by a relative margin of ~9.4%
  (2.7357 vs 2.5). The threshold fails either way — this cannot rescue the
  verdict — but whether the divergence is broad-based or single-fold-driven
  materially changes the rejection reason recorded to
  data/research/rejected/{slug}/ and whether a resubmission with a longer
  OOS window is worth the research loop's time. I have a duty under A.7 that
  I currently cannot discharge; I want that on record rather than silently
  skipped.
route: code_tool  # per-fold metrics from the walk-forward harness output
resolution_criteria: >
  Per-fold (fold_id, is_sharpe, oos_sharpe) table provided, OR harness
  confirms per-fold retention was not enabled for this run, in which case
  A.7's confirmation step is recorded as not-performable.
```

```yaml
id: BULL-Q3
question: >
  What is the unit of oos_metrics.n_trades = 819, and what is the resampling
  unit of the stationary bootstrap behind sharpe_ci_95? The report predicts
  ~60 rebalances/year and notes "each touches up to 9 pairs" — 819 is
  consistent with counting per-pair legs of ~91 rebalance events. If the
  bootstrap resamples per-leg returns, the up-to-9 legs of a single
  rebalance are cross-sectionally correlated draws, and the effective sample
  is far smaller than 819.
rule_ids: [A.2, A.3]
why_it_matters: >
  A.3 passes at face value (819 >= 30), but if the true independent unit is
  ~91 rebalance events the sample is thinner than reported — still passing
  A.3, but the CI interpretation changes. Note this question can only make
  the candidate look worse or equal: the CI already spans zero
  (-0.932, +1.368), so resolving it cannot flip A.2. I raise it because as
  Bull I would otherwise be tempted to cite "n_trades = 819, a comfortable
  27x the floor" as a strength, and I will not cite that number as strong
  evidence until its unit is confirmed. Also relevant to future candidates
  from the same multi-pair harness — if legs are being counted as trades,
  A.3 is systematically overstated for portfolio strategies.
route: code_tool  # bootstrap implementation + trade-counting logic in the harness; fallback: other_agent if Bear has already audited this
resolution_criteria: >
  Harness answer specifying (a) trade unit = leg or rebalance-event, and
  (b) bootstrap resampling unit = per-leg, per-event, or daily portfolio
  return. Either answer resolves the question; no threshold changes hinge
  on it in this debate.
```

```yaml
id: BULL-Q4
question: >
  No Section D artifacts are present in the candidate report or sandbox: no
  `make test` output, no mypy --strict exit code for the strategy module, no
  evidence of hypothesis property tests, and no evidence of FeatureSnapshot
  emission via _emit_snapshot (CL-xpw9). The gates block shows only
  {syntax: passed, backtest: ran}. Were D.1–D.4 checks run, and can their
  outputs be attached?
rule_ids: [D.1, D.2, D.3, D.4]
why_it_matters: >
  All four D rules are currently NO EVIDENCE — ABSTAIN for both agents. The
  verdict is already REJECT on Section A, but the rejection record should
  distinguish "D not run" from "D run and failed": if this strategy family
  is resubmitted after more OOS history accrues (its CI upper bound of
  +1.368 and clean B-section — edge_concentration 0.203, regime_diversified
  true, decay NONE — make resubmission plausible), knowing whether the
  operational work is done changes the resubmission cost estimate.
route: code_tool  # CI artifacts; fallback: human if the pipeline never ran D-gates for _experimental candidates
resolution_criteria: >
  Attached outputs (test summary, mypy exit code, snapshot-emission grep or
  journal sample), OR confirmation that D-gates are not executed at the
  _experimental stage, in which case D.1–D.4 are recorded as
  not-applicable-at-this-stage rather than ABSTAIN.
```

---

**Process note to the orchestrator (not a rule question):** `docs/research/SMART_QUESTIONS.md` itself was unreachable from this sandbox, so the six fields above (`id, question, rule_ids, why_it_matters, route, resolution_criteria`) are my best-faith reconstruction of the required schema. If the canonical spec names different fields, please re-key these blocks rather than dropping them — the content stands.

**No verdict-flipping blockers exist.** I state explicitly: even with all four questions resolved in the candidate's favor, A.1, A.2, A.6, and A.7 fail on values the verdict engine reads literally from the candidate report. My questions are for audit completeness and resubmission guidance only. Round 3 can proceed once these are resolved or formally escalated.

---

## Round: smart_questions — agent: bear_reviewer (bear)
*ts=2026-07-20T16:17:26+00:00 model=claude-fable-5 in/out_tokens=136028/7112 cost=$0.0000 elapsed=106.00s*

# Bear Reviewer — Round 2: Smart Questions
**Strategy:** `end-to-end-parametric-portfolio-policies-for-cross-asset-futures`

## Preliminary note on evidence access (stated per EVIDENCE_FIRST discipline)

I attempted to read `docs/research/SMART_QUESTIONS.md`, the strategy source at the report's `code_path`, and the working directory before writing this round. **None are accessible from my sandbox** (the file does not exist at the docs path; shell access to the repo aborts). I therefore cannot quote the canonical six-field schema. Rather than fabricate a claim of having read it, I state explicitly: the field set below is my declared schema — `id`, `question`, `why_it_matters`, `evidence_required`, `proposed_venue`, `resolution_criteria` — and the resolver should remap field names if the canonical doc differs. Every question is genuine; none is manufactured.

**Standing context for the resolver:** four Section A rules already FAIL on face-value report metrics (A.1: `sharpe=0.187 < 0.50`; A.2: `sharpe_ci_95.low=-0.932 ≤ 0`; A.6: `profit_factor=1.037 < 1.10`; A.7: `is_oos_sharpe_ratio=2.736 > 2.5`), so per the verdict mapping the outcome is deterministically REJECT regardless of how these questions resolve. I raise them because the protocol requires unknowns to be resolved or escalated before Round 3, and because Q1/Q3/Q4 affect whether the *record* of this rejection is complete and whether the research loop's next iteration inherits hidden defects.

---

```yaml
id: BEAR-Q1
question: >
  Can the orchestrator provide the actual strategy source file at
  src/strategies/_experimental/end-to-end-parametric-portfolio-policies-for-cross-asset-futures.py?
  It is unreadable from my sandbox, which blocks the entire Section C audit
  (C.1 look-ahead, C.2 knob count, C.4 PaperBroker constants, C.5 protocol
  conformance).
why_it_matters: >
  REVIEW_RULES Section C requires file:line citations in both directions —
  "Bear cannot vague-reject; Bull cannot vague-defend." Without the source I
  can neither assert nor clear C.1–C.5, and the severe-claim threshold in my
  own charter forbids alleging look-ahead without a line number. Bull's
  Round-1 PASS claims on Section C are equally uncitable if Bull had the same
  access failure.
evidence_required: >
  The full strategy file contents (or read access to it), plus the walk-forward
  harness invocation showing how eval timestamps are passed to the policy.
proposed_venue: code/data tool (orchestrator file provision); escalate to human if the file cannot be produced
resolution_criteria: >
  Resolved when the file text is in debate context and both agents can cite
  lines. If unproducible, Section C evidence is "missing required" and the
  C-section record must be marked UNVERIFIED, not PASS.
```

```yaml
id: BEAR-Q2
question: >
  What is the per-fold IS/OOS breakdown behind is_oos_sharpe_ratio = 2.7357?
  Specifically: fold count, per-fold IS Sharpe, per-fold OOS Sharpe, and
  per-fold variance.
why_it_matters: >
  A.7 fails on face value (2.736 > 2.5), but the rule text directs agents to
  confirm whether divergence is broad or driven by a single anomalous fold.
  This distinction doesn't change the verdict, but it changes the rejection
  reason logged to data/research/rejected/{slug}/ — "systematically overfit"
  vs "one bad fold" — which determines whether a resubmission with more OOS
  data is worth the loop's compute.
evidence_required: >
  The per-fold metrics table from the walk-forward harness output for this
  candidate run; it is absent from reports/candidates/{slug}.json.
proposed_venue: code/data tool (harness artifact retrieval)
resolution_criteria: >
  Resolved when per-fold Sharpe values are quoted in context. If the harness
  did not persist per-fold metrics, resolved-as-escalated with a ticket to add
  per-fold persistence — A.7 is otherwise unauditable beyond the headline ratio.
```

```yaml
id: BEAR-Q3
question: >
  What does n_trades = 819 count, and what is the bootstrap resampling unit
  for sharpe_ci_95? The prediction block says ~60 weight-vector rebalances per
  year, each touching up to 9 pairs. Is 819 counting per-pair legs of
  simultaneous rebalances — and if so, does the stationary bootstrap treat 9
  cross-sectionally correlated same-day legs as independent draws?
why_it_matters: >
  If 819 = legs, the effective independent sample is roughly 819/9 ≈ 91
  rebalance events, and a bootstrap over correlated legs understates CI width.
  The reported CI (-0.932, +1.368) ALREADY straddles zero and fails A.2; the
  true CI would be even wider. This also bears on A.3: 91 events passes the
  30-trade floor, but the report's headline "819 trades" materially overstates
  the evidence base, and that overstatement pattern should be flagged to the
  research loop before it recurs in future candidate reports.
evidence_required: >
  The trade-counting and bootstrap-unit definitions in the harness code, or a
  statement from the candidate report generator; recomputed CI at the
  rebalance-event level if the unit is per-leg.
proposed_venue: code/data tool (harness inspection); secondary — bull_reviewer, who cited n_trades=819 as PASS evidence on A.3
resolution_criteria: >
  Resolved when the resampling unit is stated and, if per-leg, an event-level
  CI is produced or the A.3 PASS is annotated as "819 legs ≈ 91 independent
  events."
```

```yaml
id: BEAR-Q4
question: >
  What per-pair spread and slippage assumptions were applied in the backtest
  cost model? The universe includes USDNOK, USDSEK, and NZDUSD, which fall
  under the C.3 "FX exotic" floors (min 1.5 bps spread, 1.0 bps slippage) —
  the candidate report contains no cost-model disclosure at all.
why_it_matters: >
  With profit_factor at 1.037, the strategy is within rounding distance of
  breakeven; even a 1 bps/turn cost misstatement across ~819 legs plausibly
  flips total_return (+1.93%) negative. If exotic pairs were costed at
  FX-major floors (0.3/0.3 bps), that is a C.3 violation in its own right and
  the reported OOS metrics are all optimistic. Where would this be visible if
  real? In the cost-model config or harness constants — it has not been pulled.
evidence_required: >
  The cost-model configuration used for this run: per-pair spread and slippage
  values, quoted with source (config file path or harness default).
proposed_venue: code/data tool (harness/config inspection)
resolution_criteria: >
  Resolved when per-pair cost values are quoted and compared line-by-line
  against the C.3 table. If costs for NOK/SEK/NZD legs are below exotic
  floors, C.3 is recorded as FAIL, not merely the A-section failures.
```

```yaml
id: BEAR-Q5
question: >
  Where is the Section D evidence? The candidate report contains no `make
  test` output (D.1), no mypy --strict exit status (D.2), no hypothesis
  property tests for the SPSA/attention numerical helpers (D.3), and no
  demonstration of FeatureSnapshot emission via _emit_snapshot (D.4).
why_it_matters: >
  Per the verdict mapping, "any required metric missing from the candidate
  report → ESCALATE." Section D is entirely absent — the gates block reports
  only "syntax: passed" and "backtest: ran." If the loop's report generator is
  silently omitting Section D artifacts, every future candidate will arrive
  with the same hole, and D.4 in particular (snapshot-per-intent auditability,
  CL-xpw9) is the difference between a reconstructible trade record and an
  unauditable one.
evidence_required: >
  Test run output showing >=3 strategy-specific tests (entry, exit, edge
  case), mypy --strict exit code for the strategy file, property-test file
  paths, and one example trade_journal_events row carrying a snapshot
  reference.
proposed_venue: code/data tool (test/type-check execution); escalate to human if the pipeline never captured these
resolution_criteria: >
  Resolved when all four artifacts are in context, or formally escalated as
  "Section D evidence missing" — in which case D.1–D.4 are recorded
  UNVERIFIED and the report-generator gap gets its own ticket.
```

---

## Summary for the question resolver

Five genuine blockers, none manufactured, none capable of rescuing the candidate: **A.1, A.2, A.6, and A.7 fail deterministically on the report's own numbers**, so the verdict is REJECT before any of these resolve. Their purpose is (a) completing the Section C/D audit record honestly — I will not claim C-section or D-section failures I cannot cite by line, and equally will not let uncited PASSes stand — and (b) surfacing two pipeline defects (per-leg trade counting inflating the evidence base, Section D artifacts missing from candidate reports) that will contaminate every future debate if left unresolved.

One further note for the orchestrator: my sandbox also could not reach the knowledge archive tools (`knowledge_search` is not available in this environment), so no historical-precedent citations were possible this round — consistent with my charter, I am flagging the absence rather than quoting from memory.

---

## Round: rebuttal — agent: bull_reviewer (bull)
*ts=2026-07-20T16:20:16+00:00 model=claude-fable-5 in/out_tokens=58511/6560 cost=$0.0000 elapsed=90.93s*

Evidence-access check complete: the `knowledge_search` archive tool is still not reachable (ToolSearch surfaced only unrelated web/mail/LSP tools), and the working directory remains an empty scratch dir — `src/strategies/_experimental/` and `reports/candidates/` do not exist here. All citations below are therefore from the in-context candidate report JSON and REVIEW_RULES.md only; no archive quotes will be used, per the no-fabrication constraint.

---

# REBUTTAL — Round 3 — Bull Reviewer

**Candidate:** `end-to-end-parametric-portfolio-policies-for-cross-asset-futures`

**Evidence-access note (per EVIDENCE_FIRST):** Re-verified this round: strategy source at `code_path` unreachable (sandbox cwd is an empty scratch dir), knowledge archive tool unavailable. I engage with the Bear's claims as recorded in the debate transcript; where the Bear's full REJECT_CASE prose was summarized, I engage with the rule-level claims, which are fully reconstructible from the candidate report.

---

## Bear's claim: "A.1 (Annualized OOS Sharpe): FAIL — `oos_metrics.sharpe` = 0.187 < 0.50"
**Engagement**: agree

The report states `oos_metrics.sharpe = 0.18697151517123545`. The threshold is literal (`oos_metrics.sharpe >= 0.50`) and agents may not override it. 0.187 < 0.50. This is a clean deterministic FAIL and I concede it without qualification. The only context I add — as context, not as a defense against the rule — is that the report's own `predicted_sharpe_range` was `[0.0, 0.6]`, deliberately lowered from the brief's [0.3, 0.8] because this is "a deliberately tiny single-block attention policy (~1.1k weights) trained by SPSA... with gross leverage capped at 1.0" (report, `prediction` block). The realized 0.187 sits inside the pre-registered band. That speaks to research-process integrity, not to promotability. **Conceded: FAIL.**

## Bear's claim: "A.2 (CI lower bound > 0): FAIL — `sharpe_ci_95.low` = -0.932"
**Engagement**: agree

`sharpe_ci_95 = {low: -0.9319104222755284, high: 1.3682141833317643}`. Zero is deep inside the interval. REVIEW_RULES.md calls A.2 "the single most important gate": if zero is in the CI, "the edge is statistically indistinguishable from luck on the available sample." This is the Bear's strongest claim and it is dispositive. With 819 trades (a healthy sample per A.3), a CI this wide around a 0.187 point estimate means the data simply do not support an edge claim yet. I cannot and do not contest it. **Conceded: FAIL.**

## Bear's claim: "A.6 (Profit factor ≥ 1.10): FAIL — 1.037"
**Engagement**: agree

`oos_metrics.profit_factor = 1.0370881316090637 < 1.10`. Wins exceed losses by only ~3.7% in dollar magnitude versus the required 10%. Consistent with the marginal Sharpe and `total_return = 0.0193`. No counter-evidence exists in the report. **Conceded: FAIL.**

## Bear's claim: "A.7 (IS/OOS divergence ≤ 2.5): FAIL — `is_oos_sharpe_ratio` = 2.736"
**Engagement**: partially agree

`is_oos_sharpe_ratio = 2.735694262118248 > 2.5` — the threshold line is literal, so this is a FAIL and I concede the rule outcome. Two honest qualifications, neither of which changes the verdict: (1) the breach is marginal (~9.4% over threshold), and with an OOS Sharpe this close to zero the ratio is numerically unstable — a denominator of 0.187 amplifies small IS differences; (2) A.7's own text directs agents to "look at per-fold metrics to confirm the divergence isn't driven by a single anomalous fold," and the per-fold breakdown is absent from the report, so that confirmation step cannot be performed by either agent. This is a missing-evidence observation for the orchestrator's records, not a rescue: the deterministic check fails on the number as reported. **Conceded: FAIL (with the per-fold artifact gap noted for the record).**

## Bear's claim: "Section C is unverifiable — the strategy source file is unavailable (BEAR-Q1)"
**Engagement**: partially agree

I agree the source is unreachable — I re-verified this round (`src/strategies/_experimental/` does not exist in the sandbox; cwd is an empty temp directory). I therefore ABSTAIN on C.1–C.5 and D.1–D.4, exactly as I did in Round 1: no line numbers exist to cite, so neither violations nor compliance can be affirmed.

Where I push back is on characterization: an unreadable source file is **absence of evidence, not evidence of violation**. Under the verdict mapping, missing required artifacts route to ESCALATE, not to an affirmative C-rule FAIL — the Bear cannot vague-reject on Section C any more than I can vague-defend it (Section C preamble: "Each must be cited with `file:line` evidence"). In this instance the distinction is moot — Section A fails on the report's own numbers regardless — but for the record: if Section A had passed, the correct disposition for the source-access failure would be ESCALATE, not REJECT.

## Bear's claim: "Five genuine blockers... none capable of rescuing the candidate"
**Engagement**: partially agree

On the verdict mechanics, I fully agree: `compute_verdict` rejects on any Section A failure "regardless of agent positions," and four A-rules (A.1, A.2, A.6, A.7) fail on the report's own numbers. Nothing I argued in Round 1 or could argue now changes that, and I will not pretend otherwise.

I partially disagree only with any implied conclusion that the candidate is structurally rotten rather than under-powered. The rules the strategy *does* pass are the fragility screens, and it passes them cleanly on cited values:

- **A.3 PASS**: `n_trades = 819` ≥ 30 — the CI is wide despite a large sample, not because of a small one.
- **A.4 PASS**: `hit_rate = 0.5072` ≥ 0.45, above 0.50 and inside the predicted `[0.50, 0.57]` band — no sign-flip flag warranted.
- **A.5 PASS**: `max_drawdown = -0.058` ≥ -0.25 — far from the kill-switch floor.
- **B.1 PASS**: `edge_concentration = 0.2033` ≤ 0.60 — not a one-feature strategy.
- **B.2 PASS**: `regime_diversified = true`.
- **B.3 PASS**: `decay_severity = "NONE"` — no evidence of pre-deployment edge erosion.
- The report's pre-registered abandon condition (rolling 126-day OOS Sharpe < 0) is not triggered on the reported aggregate (`total_return = +0.0193`, Sharpe positive).

So the honest synthesis is: **weak edge, honestly measured, cleanly structured — but statistically indistinguishable from luck at current capacity (A.2)**. My recommendation to the human reviewer accompanying the rejection: this is a re-submission candidate (larger policy capacity and/or longer OOS sample per the report's own note on the SPSA/1.1k-weight constraint), not a blacklist candidate. That recommendation has no bearing on today's verdict.

---

## Revised final position

The Bear's A-section case is correct on the report's own numbers, and rule failures are non-overridable. Continuing to argue PROMOTE would be advocacy against arithmetic, which EVIDENCE_FIRST and my own charter ("honest abstention beats fabrication") forbid.

**REJECT**

*Rationale: A.2 is dispositive — `sharpe_ci_95 = [-0.932, +1.368]` contains zero on an 819-trade sample, so the edge is statistically indistinguishable from luck; A.1 (0.187 < 0.50), A.6 (1.037 < 1.10), and A.7 (2.736 > 2.5) fail independently. The candidate's clean B-section profile and honest pre-registration make it a good re-submission candidate at higher capacity, but not a promotable one today.*

---

## Round: rebuttal — agent: bear_reviewer (bear)
*ts=2026-07-20T16:20:16+00:00 model=claude-fable-5 in/out_tokens=55902/5917 cost=$0.0000 elapsed=79.43s*

Evidence-access re-check for this round is complete: the working directory remains an empty scratch dir (no `src/`, `docs/`, or `reports/` reachable), and the knowledge archive tool is still absent — ToolSearch surfaced only an unrelated web-search tool, not `knowledge_search`. Consequences for this rebuttal, stated up front per EVIDENCE_FIRST: (1) I make **no** Section C claims requiring `file:line` citations — the severe-claim threshold binds me exactly as it binds Bull; (2) I quote **no** historical-precedent chunks, since I cannot call the archive and will not fabricate quotes from memory. Everything below rests on the candidate report's own numbers, which are in context and sufficient.

---

# REBUTTAL — Round 3 — Bear Reviewer

**Candidate:** `end-to-end-parametric-portfolio-policies-for-cross-asset-futures`

**Evidence basis:** `reports/candidates/{slug}.json` (in context), `REVIEW_RULES.md` (in context). Strategy source, per-fold breakdown, test/mypy outputs, and knowledge archive: unreachable this round (re-verified above, consistent with both agents' Round 2 findings).

---

## Bull's claim: A.3 (Sample size) — PASS, "`oos_metrics.n_trades = 819 >= 30`"

**Engagement**: accept, with one caveat.

819 ≥ 30. The threshold passes literally and I concede it. Caveat carried over from my Round 2 question (BEAR series): the report predicts `expected_n_trades_per_year: 60` *rebalances*, each touching up to 9 pairs, while `n_trades = 819` appears to count per-pair fills (819 ≈ 91 rebalances × 9 pairs). If the bootstrap CI in A.2 was computed on 819 per-pair fills, it treats up-to-9-way simultaneous, correlated legs of one portfolio decision as independent draws — which would make the reported CI *optimistically narrow*. This caveat cannot flip A.3, and note its direction: correcting it would make the candidate look **worse** on A.2, not better.

## Bull's claim: A.4 (Hit rate) — PASS, "`hit_rate = 0.5072 >= 0.45`"

**Engagement**: accept.

0.5072 ≥ 0.45, and it sits just above the 40–50% sign-flip review band, so no sign-flip flag is warranted. Also consistent with the candidate's own predicted range [0.50, 0.57] — at its extreme bottom edge. Passes.

## Bull's claim: A.5 (Max drawdown) — PASS, "`max_drawdown = -0.058 >= -0.25`"

**Engagement**: partially accept.

The threshold passes: −0.058 is comfortably inside the −0.25 floor. But I contest the *inference* Bull draws from it. A shallow drawdown on a strategy with gross leverage capped at 1.0, total OOS return of 1.93%, and Sharpe 0.187 is not evidence of risk control — it is evidence the strategy barely moves. A near-flat P&L stream cannot draw down much. A.5 passing is a property of low activity, not of edge. It counts toward the gate; it should count for nothing in the promotion narrative.

## Bull's claim: B.1 (Edge concentration) — PASS, "`edge_concentration = 0.203 <= 0.60`"

**Engagement**: accept.

0.203 is genuinely well-diversified across features. Concede without reservation.

## Bull's claim: B.2 (Regime diversification) — PASS, "`regime_diversified == True`"

**Engagement**: partially accept.

The G6 flag is `true`; the threshold passes literally and I concede the rule. I contest the weight: with `sharpe_ci_95 = [-0.932, +1.368]`, the total edge is statistically indistinguishable from zero (A.2). Decomposing a statistically-zero edge across regimes and finding it positive in ≥2 of 5 is diversification *of noise*. B.2 answers "is the edge concentrated?"; it cannot answer "is there an edge?" — that is A.2's job, and A.2 fails.

## Bull's claim: B.3 (No active decay) — PASS, "`decay_severity = NONE`"

**Engagement**: accept. An edge near zero has little to decay from, but the rule passes as written. Conceded.

## Bull's claims on A.1 / A.2 / A.6 / A.7 — mitigation via the prediction band ("realized 0.187 sits inside the pre-registered [0.0, 0.6] band; capacity was deliberately reduced — tiny SPSA-trained policy, numpy sandbox, leverage cap 1.0")

**Engagement**: contest — and note Bull has, to their credit, already conceded this in their own Round 3.

Three points:

1. **Rules are non-overridable by construction.** REVIEW_RULES.md, Section A preamble: "The verdict engine reads the `THRESHOLD:` line literally — agents may NOT override these." The four failures are arithmetic, not judgment:
   - **A.1**: `oos_metrics.sharpe = 0.18697 < 0.50` — FAIL.
   - **A.2**: `sharpe_ci_95.low = -0.9319 ≤ 0.0` — FAIL. Zero is not merely inside the interval; it is nowhere near the boundary. This is the rule the rules document itself calls "the single most important gate."
   - **A.6**: `profit_factor = 1.0371 < 1.10` — FAIL. Wins exceed losses by 3.7% in dollar magnitude against a required 10%.
   - **A.7**: `is_oos_sharpe_ratio = 2.7357 > 2.5` — FAIL. IS/OOS divergence above the overfit line — and this is the one Section C-adjacent failure that needs **no code access** to establish: the harness measured it.

2. **The prediction band is an admission, not a defense.** A pre-registered band of [0.0, 0.6] whose lower bound is *zero edge* concedes ex ante that the capacity-reduced implementation might have no edge at all. The realized outcome — Sharpe 0.187 with a CI spanning [−0.93, +1.37] — is precisely the "we cannot distinguish this from zero" scenario the band anticipated. Correct forecasting of a weak result is good research hygiene and I commend it; it is not grounds for promotion. The honest disposition for "faithful implementation, insufficient capacity, edge unproven" is REJECT with a documented reason, and if the researcher believes the full-capacity version would clear the gates, that is a **new candidate**, not an exemption for this one.

3. **Verdict mapping is dispositive regardless of positions.** Per Section E: "if any rule in (A, B, C, D) fails on the candidate metrics → REJECT (regardless of agent positions)." Four A-rules fail. Nothing either agent writes can produce PROMOTE here.

## Bull's claims on Sections C and D (if claimed as PASS)

**Engagement**: contest as unverifiable — symmetrically.

Neither agent could read the strategy source, the tests, or the mypy output this round (re-verified above). Bull cannot affirm C.1–C.5 or D.1–D.4 as PASS without evidence, and — symmetrically — I make no C-section FAIL claims either: per my severe-claim threshold, alleging look-ahead or fixture constants without a line citation would itself be a fabrication. I therefore **ABSTAIN on C.1, C.4, C.5 and D.1–D.4** individually. Two C-adjacent observations that require no source access: (a) A.7's overfit signal (2.736 > 2.5) is already captured quantitatively above; (b) C.2 parameter discipline is *prima facie* strained by "~1.1k weights" in a trained attention policy — whether "free tunable knobs" means hyperparameters (likely ≤5) or trained weights is a definitional question for the orchestrator, not a citable violation, and I do not claim it as one. Missing Section D evidence is ESCALATE-shaped under the rules, but it is mooted: deterministic A-failures dominate and force REJECT regardless.

## On Bull's Round 3 concession

**Engagement**: accept.

Bull's revised position — "continuing to argue PROMOTE would be advocacy against arithmetic" — is the correct reading of the rules. I register that this convergence is evidence the debate worked as designed: no claim I contested above required manufacturing an objection, and the claims I conceded (A.3, A.4, A.5, B.1, B.2, B.3) I conceded cleanly.

---

## Revised final position

**FINAL_POSITION**: REJECT

The three strongest failures, in order: **A.2** — the bootstrap CI lower bound is −0.932, so the edge is statistically indistinguishable from luck on 819 fills, and the rules name this the single most important gate; **A.7** — IS Sharpe is 2.736× the OOS Sharpe, above the 2.5 overfit threshold, measured by the harness itself with no code access required; **A.1/A.6 jointly** — Sharpe 0.187 and profit factor 1.037 describe a strategy that, even taken at face value, earns too little per unit of risk and per dollar of loss to justify book risk after costs. The candidate's own pre-registered prediction band bottomed at zero edge, and zero edge is what the sample cannot exclude. No fabricated objections were needed: the report's own numbers close the case, and per the verdict mapping the deterministic A-failures force REJECT regardless of agent positions.

**REJECT**

---

## Round: final_position — agent: bull_reviewer (bull)
*ts=2026-07-20T16:20:25+00:00 model=claude-fable-5 in/out_tokens=25603/469 cost=$0.0000 elapsed=7.96s*

**FINAL_POSITION**: REJECT

The candidate fails four Section A gates deterministically on its own reported numbers — A.1 (Sharpe 0.187 < 0.50), A.2 (CI lower bound −0.932 ≤ 0, the edge is indistinguishable from luck), A.6 (profit factor 1.037 < 1.10), and A.7 (IS/OOS ratio 2.736 > 2.5) — and since rule failures are non-overridable regardless of Section B strengths (edge concentration 0.203, regime-diversified, no decay), the strongest honest Bull position is REJECT rather than advocacy against arithmetic.

---

## Round: final_position — agent: bear_reviewer (bear)
*ts=2026-07-20T16:20:25+00:00 model=claude-fable-5 in/out_tokens=26370/584 cost=$0.0000 elapsed=8.71s*

# FINAL_POSITION — Round 4 — Bear Reviewer

**Candidate:** `end-to-end-parametric-portfolio-policies-for-cross-asset-futures`

**FINAL_POSITION**: REJECT

Four Section A rules fail deterministically on the candidate report's own numbers — A.1 (`oos_metrics.sharpe = 0.187 < 0.50`), A.2 (`sharpe_ci_95.low = −0.932 ≤ 0`, edge indistinguishable from luck on 819 fills), A.6 (`profit_factor = 1.037 < 1.10`), and A.7 (`is_oos_sharpe_ratio = 2.736 > 2.5`) — with the strategy source, per-fold breakdown, and test/mypy artifacts additionally unreachable, so the Section C/D audit could not even be completed; Bull conceded the arithmetic in Round 3, and rule failures are non-overridable.

REJECT

---

## Debate summary
- **bull_reviewer**: REJECT
- **bear_reviewer**: REJECT

**Total cost**: $0.0000
