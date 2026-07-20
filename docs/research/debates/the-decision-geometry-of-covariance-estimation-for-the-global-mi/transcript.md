# Debate transcript — promotion_review — the-decision-geometry-of-covariance-estimation-for-the-global-mi

Started: 2026-07-20T16:04:38+00:00
Participants: bull_reviewer, bear_reviewer
Rounds: initial_positions, smart_questions, rebuttal, final_position

## Round: initial_positions — agent: bull_reviewer (bull)
*ts=2026-07-20T16:06:33+00:00 model=claude-fable-5 in/out_tokens=49539/5648 cost=$0.0000 elapsed=80.23s*

I attempted to load the strategy source file and candidate report directory to gather `file:line` citations for Sections C and D — the review sandbox contains no repository checkout, so code-level rules are evaluated as NO EVIDENCE rather than guessed. The knowledge-archive tool is likewise not connected in this session, so I make no historical-precedent quotes. All metric citations below come from the `candidate_report` JSON provided in context.

---

# PROMOTE_CASE for the-decision-geometry-of-covariance-estimation-for-the-global-mi

*Bull Reviewer — Round 1 (initial position)*

## Section A — Statistical reality

**A.1 (Sharpe ≥ 0.50): FAIL.** Evidence: `oos_metrics.sharpe = -0.6332304110707234`. Not borderline — the OOS Sharpe is negative, 1.13 Sharpe-units below the threshold and below even the candidate's own pre-registered floor (`predicted_sharpe_range: [0.20, 0.60]`). I cannot defend this rule.

**A.2 (CI lower > 0): FAIL.** Evidence: `sharpe_ci_95 = {low: -1.6724462486121674, high: 0.3783574682392515}`. Zero is well inside the interval, and the point estimate is negative. The most important gate in the rulebook fails. Note honestly: even the *upper* CI bound (0.378) sits below the A.1 threshold — there is no bootstrap-resampling story under which this sample clears Section A.

**A.3 (n_trades ≥ 30): PASS.** Evidence: `oos_metrics.n_trades = 62` ≥ 30. The sample is large enough that the CI in A.2 is interpretable — which cuts against the candidate, not for it: the failure is well-measured, not noise-driven.

**A.4 (hit rate ≥ 0.45): PASS — with the mandated flag.** Evidence: `oos_metrics.hit_rate = 0.4798387096774194` ≥ 0.45. Per the A.4 note, a hit rate in the 40–50% band with a *negative* payoff profile (see A.6) warrants **sign-flip review**. I flag it: the combination (hit_rate 0.48, profit_factor 0.90, Sharpe -0.63) is consistent with a weakly inverted or premium-paying exposure, not a mis-signed alpha large enough to harvest by flipping.

**A.5 (max drawdown ≥ -0.25): PASS — borderline.** Evidence: `oos_metrics.max_drawdown = -0.22171074136786523`, inside the -0.25 catastrophic floor by ~2.8 points. Honest caveat: a -22.2% drawdown on a strategy with -13.9% total return (`oos_metrics.total_return = -0.1390273486142941`) passes the letter of the rule while illustrating exactly the risk the rule exists to cap.

**A.6 (profit factor ≥ 1.10): FAIL.** Evidence: `oos_metrics.profit_factor = 0.8984068797843837`. Gross losses exceed gross wins. Below not only the 1.10 threshold but below breakeven (1.0).

**A.7 (IS/OOS ratio ≤ 2.5): PASS — and this is the candidate's genuinely strongest point.** Evidence: `is_oos_sharpe_ratio = 0.8747919848227435`, far under 2.5 and in fact below 1.0: OOS performance is consistent with in-sample performance. This is **not an overfit strategy that flattered itself in-sample** — it is an honestly-measured strategy whose edge does not exist. Sortino confirms the same picture (`sortino = -0.9620246354878617`).

## Section B — Edge structure

**B.1 (edge concentration ≤ 0.60): PASS.** Evidence: `edge_concentration = 0.17663743214755637`. Well-distributed attribution across the 7-leg basket; no single-feature fragility.

**B.2 (regime diversified): PASS.** Evidence: `regime_diversified = true` from the G6 layer. Notably this *contradicts* the brief's own prediction (`regime_dependence: "vol-spike-only"`) in the favorable direction — the (negative) performance is at least not a one-regime artifact.

**B.3 (no active decay): PASS.** Evidence: `decay_severity = "NONE"` from G7 — not in {STRONG, MODERATE}. There is no edge decaying; the more painful reading is that there was no positive edge to decay.

## Section C — Code integrity

**C.1 (no future-dated lookups): NO EVIDENCE — ABSTAIN.** The strategy file at `code_path = src/strategies/_experimental/the-decision-geometry-of-covariance-estimation-for-the-global-mi.py` is not present in this review environment; I cannot cite lines. Supporting circumstantial evidence only: an IS/OOS ratio of 0.87 (A.7) is the opposite of the signature lookahead bias produces.

**C.2 (≤ 5 free parameters): NO EVIDENCE — ABSTAIN.** Config source not available. The report references at least `GMVPShrinkageConfig(use_shrinkage=...)`; full knob count unverifiable without the file.

**C.3 (realistic cost model): NO EVIDENCE — ABSTAIN.** Basket includes FX majors and `NZDUSD` (exotic tier: 1.5 bps spread / 1.0 bps slippage floor). Cost-model constants not visible without the strategy file.

**C.4 (no PaperBroker constants): NO EVIDENCE — ABSTAIN.** Cannot grep a file I cannot read.

**C.5 (standard Strategy protocol): NO EVIDENCE — ABSTAIN.** A 7-leg portfolio strategy should implement `generate_intents()`; unverified.

## Section D — Operational fitness

**D.1 (tests pass, ≥ 3 strategy tests): NO EVIDENCE — ABSTAIN.** Report's `gates` block records only `syntax: passed` and `backtest: ran`; no test-suite result field.

**D.2 (mypy --strict exit 0): NO EVIDENCE — ABSTAIN.** Not in report.

**D.3 (property tests for numerical helpers): NO EVIDENCE — ABSTAIN.** A covariance-shrinkage estimator is precisely the kind of numerical helper D.3 targets; no evidence either way.

**D.4 (FeatureSnapshot emission): NO EVIDENCE — ABSTAIN.** Not in report.

## Points in the candidate's favor (for the record)

Since my role is to surface the strongest promote-side evidence, here is everything that genuinely holds up:

1. **Process integrity is exemplary.** Pre-registered prediction ranges, a falsifiable operational abandon threshold (one-boolean shrinkage-vs-sample-covariance comparison), and full consumption of exactly the brief's declared data sources (`data_sources_consumed`: all 7 pairs, no undeclared FRED/POLY features) — this is EVIDENCE_FIRST discipline done right.
2. **No overfitting signature** (A.7 = 0.87), **no concentration fragility** (B.1 = 0.18), **regime-diversified** (B.2), **no decay** (B.3). The measurement apparatus worked; the report is credible.
3. **Missing comparison worth resolving before archival**: the pre-registered abandon threshold requires the `use_shrinkage=False` baseline Sharpe over the same walk-forward windows, and the report omits it. It should be recorded in the rejection record — the *research question* (does shrinkage beat sample covariance in this decision geometry?) may still have a publishable answer even though the *strategy* does not.

But none of this converts a negative-Sharpe, sub-breakeven-profit-factor candidate into a deployable one. The rules are explicit that agents may not override `THRESHOLD:` lines, and three of them fail outright.

## Final position

**REJECT**

The strongest promote-side evidence — clean IS/OOS consistency (A.7 = 0.87), low edge concentration (B.1 = 0.18), and honest pre-registration fully consistent with `docs/research/EVIDENCE_FIRST.md` — establishes that this is a *well-measured* candidate, not a *profitable* one. A.1 (Sharpe -0.63 vs ≥ 0.50), A.2 (CI lower bound -1.67 vs > 0), and A.6 (profit factor 0.90 vs ≥ 1.10) fail on the literal threshold values with n = 62 trades of well-sampled evidence; the candidate also breached its own pre-registered Sharpe floor of 0.20. Advocating PROMOTE here would require inventing evidence that does not exist, and my instructions rank honest abstention above fabrication. The residual concern that did *not* rise to a fail: hit rate 0.4798 sits in the A.4 sign-flip-review band — worth a note in the rejection record, though the payoff asymmetry (PF 0.90) suggests no harvestable inverted edge. Recommend archival to `data/research/rejected/{slug}/` with the shrinkage-vs-baseline comparison recorded as an open research note.

---

## Round: initial_positions — agent: bear_reviewer (bear)
*ts=2026-07-20T16:06:33+00:00 model=claude-fable-5 in/out_tokens=107792/8467 cost=$0.0000 elapsed=114.66s*

The knowledge-archive tools and the strategy source file are not reachable from this review sandbox (Read confirms the code path does not exist in the working directory, and `knowledge_search` is not among the available tools). Per my constraints I therefore make **no line-cited Section C claims and no quoted historical chunks** — everything below is grounded in the candidate report JSON and REVIEW_RULES.md, both of which are in context. That is sufficient: the candidate fails three hard Section A gates on its own reported numbers.

---

# REJECT_CASE for the-decision-geometry-of-covariance-estimation-for-the-global-mi

## Section A — Statistical reality (failures found)

**A.1 (Annualized OOS Sharpe): FAIL** — THRESHOLD `oos_metrics.sharpe >= 0.50`. Reported value: **`sharpe = -0.6332`**. This is not a marginal miss of the 0.50 floor — the sign is wrong. The strategy *lost money* out-of-sample: `total_return = -0.1390` (−13.9%), `sortino = -0.9620`. There is no "weak edge after costs" argument available here; there is no edge.

**A.2 (95% bootstrap-CI lower bound > 0): FAIL** — THRESHOLD `sharpe_ci_95.low > 0.0`. Reported: **`low = -1.6724`, `high = 0.3784`**. Zero is inside the interval, so the rule fails on its face — but note the interval is not even centered near zero: the entire lower half of the CI is deeply negative and the *upper* bound (+0.38) is itself below the A.1 promotion floor of 0.50. The bootstrap is more consistent with a genuinely negative edge than with luck. This is the rule the rules file calls "the single most important gate," and the candidate fails it in the worst available direction.

**A.6 (Profit factor): FAIL** — THRESHOLD `oos_metrics.profit_factor >= 1.10`. Reported: **`profit_factor = 0.8984`**. Gross dollar losses exceed gross dollar wins by ~11%. Combined with the A.4 flag below, the strategy loses on both frequency and magnitude: it wins less than half the time *and* its losses are bigger than its wins.

**Flags on rules that technically pass (not FAIL entries, but material):**

- **A.4 sign-flip band**: `hit_rate = 0.4798` clears the 0.45 floor but sits squarely in the 40–50% band that rule A.4 explicitly designates for sign-flip review. The rule's escape hatch — "profitable with sub-50% hit rate via positive payoff asymmetry" — is closed by A.6: profit factor is 0.90, so there is no payoff asymmetry rescuing this. Sub-50% hit rate + sub-1.0 profit factor = losing strategy, full stop.
- **A.5 drawdown**: `max_drawdown = -0.2217` passes the −0.25 catastrophic floor by only 2.8 points — on a strategy whose *entire thesis* is variance minimization. A global-minimum-variance portfolio that drew down 22% while returning −13.9% failed at its one stated job.
- **A.7 caveat**: `is_oos_sharpe_ratio = 0.8748` "passes," but do the arithmetic: 0.8748 × (−0.6332) ⇒ implied **IS Sharpe ≈ −0.55**. Either the strategy was negative *in-sample too* — meaning it never demonstrated an edge at any point and should not have reached this review — or the ratio field is ill-defined for negative Sharpes and cannot be trusted. Either reading argues against promotion; neither is a PASS in any meaningful sense.

## Section B — Edge structure (failures found)

No B-rule threshold violations (`edge_concentration = 0.177` ≤ 0.60; `regime_diversified = true`; `decay_severity = "NONE"`). But these passes are **vacuous given a negative total edge**, and I flag two coherence problems the Bull should not be allowed to cite as strengths:

- **B.2 self-contradiction**: the candidate's own prediction block declares `regime_dependence: "vol-spike-only"` — a one-regime thesis — while G6 reports `regime_diversified = true`. Both cannot describe the same strategy. And since overall OOS Sharpe is −0.63, "edge positive in ≥2 of 5 regimes" mechanically implies the remaining regimes are *strongly negative*. Diversified losses are not diversification of edge.
- **B.3 vacuity**: `decay_severity = "NONE"` is unremarkable — an edge that never existed cannot decay.

## Section C — Code integrity (failures found)

**None claimable.** The strategy file at `src/strategies/_experimental/the-decision-geometry-of-covariance-estimation-for-the-global-mi.py` was not provided in the review context and is not present in the working directory. Under the severe-claim threshold I cannot allege look-ahead (C.1), parameter-count violations (C.2), or fixture constants (C.4) without file:line citations, so I allege none. I register the following as **open smart-questions for Round 2** (unresolved → ESCALATE per Section E, though moot given the A failures):

1. **C.3 / NZD cost floor**: the basket includes **NZDUSD**, which the C.3 table classifies as FX *exotic* (min 1.5 bps spread / 1.0 bps slippage), not major. Does the cost model apply the exotic floor to the NZD leg, or a uniform majors floor across all 7 legs? Unverifiable without the code.
2. **C.2**: `GMVPShrinkageConfig` — how many free knobs (shrinkage intensity, lookback, rebalance cadence, weight caps, …)? ≤5 unverified.
3. **C.5 / D.4**: 7-leg rebalances imply `generate_intents()`; snapshot emission unverified.

## Section D — Operational fitness (failures found)

**None verifiable — and none evidenced.** The report's `gates` block shows only `syntax: passed` and `backtest: ran`. There is no evidence of D.1 (`make test` + 3 strategy tests), D.2 (`mypy --strict` clean), D.3 (hypothesis property tests — squarely required here, since covariance shrinkage *is* a numerical helper), or D.4 (`FeatureSnapshot` emission). Per the verdict mapping, missing required evidence is ESCALATE-grade on its own; here it is academic.

## Failed by its own pre-registration

Independent of the rulebook, the candidate fails **its own registered predictions** — the evidence-first standard applied to itself:

- Predicted Sharpe range **[0.20, 0.60]** (already widened downward from the brief's 0.30 floor, with justification). Realized: **−0.63** — below even the widened floor, by 0.83 Sharpe units.
- Predicted hit-rate range **[0.52, 0.63]**. Realized: **0.48** — below the floor.
- The pre-registered **abandon_threshold** — shrinkage GMVP OOS Sharpe vs. the identical `use_shrinkage=False` sample-covariance baseline over the first 252-day window — is *the* decisive experiment for the paper's decision-geometry thesis, and **the baseline comparison result is absent from the report**. Where would this thesis be visible if real? In exactly that one-boolean-flip A/B the candidate itself specified. It was not pulled, or not reported. A thesis that should be empirically locatable but hasn't been located is trading a vibe.

## Historical precedent

(No archive quotes — `knowledge_search` is unavailable in this sandbox, and I will not fabricate chunks. Stated from the candidate's own admissions instead.) The candidate concedes the killer structural fact in its own notes: a long-only, fully-invested foreign-currency basket vs USD carries an "irreducible short-USD factor exposure that minimum-variance weighting reduces but cannot eliminate." The realized −13.9% OOS total return is exactly what a decorated passive short-USD position looks like over an adverse USD window. This is the canonical failure mode of optimized-covariance portfolios: estimation refinement layered on top of a dominant factor bet the optimizer cannot see — the well-documented result that mean-variance-optimized weights routinely fail to beat naive benchmarks out-of-sample due to estimation error is precisely why the candidate's own shrinkage-vs-sample-covariance baseline test mattered, and it wasn't reported.

## Final position

**FINAL_POSITION**: REJECT

The three strongest failures: **A.2** (bootstrap CI lower bound of −1.67 — the edge is statistically indistinguishable from, and pointing toward, a *losing* strategy), **A.1** (OOS Sharpe of −0.63 with −13.9% total return — the sign of the edge is wrong, not merely its magnitude), and **A.6** (profit factor 0.90 — losses exceed wins in dollars, closing the payoff-asymmetry escape hatch for the sub-50% hit rate). These are hard deterministic gates; per the verdict mapping the engine must REJECT regardless of agent positions. The candidate also breached its own pre-registered prediction floor and failed to report its own pre-registered abandon-threshold baseline comparison. No amount of Section C cleanliness or B-layer passes can rescue a strategy that lost money out-of-sample with a CI excluding the promotion floor. Route the strategy to `data/research/rejected/the-decision-geometry-of-covariance-estimation-for-the-global-mi/` with reason: failed A.1, A.2, A.6.

**REJECT**

---

## Round: smart_questions — agent: bull_reviewer (bull)
*ts=2026-07-20T16:09:18+00:00 model=claude-fable-5 in/out_tokens=52532/4068 cost=$0.0000 elapsed=57.66s*

# Round 2 — Smart Questions: Bull Reviewer
*Strategy: `the-decision-geometry-of-covariance-estimation-for-the-global-mi`*

**I emit zero smart-question blocks this round.** Explicitly: **no genuine blockers.**

(Procedural note for the orchestrator: `docs/research/SMART_QUESTIONS.md` is not reachable from this review sandbox — the shell environment is broken (`dyld` library load failure on every command) and no repository checkout is present, consistent with both agents' Round 1 findings. Since I am filing zero questions, the schema gap is moot; had I filed any, I could not have guaranteed the six required fields.)

## Why zero blocks

A smart question is a blocker only if its answer could change my final position or the verdict. Neither is possible here, because the candidate report already contains **deterministic, agent-unoverridable Section A failures**:

- **A.1**: `oos_metrics.sharpe = -0.6332` vs THRESHOLD `>= 0.50` — FAIL. The edge is not merely weak; it is negative.
- **A.2**: `sharpe_ci_95.low = -1.6724` vs THRESHOLD `> 0.0` — FAIL. Zero is well inside the CI (`[-1.672, +0.378]`); the point estimate is negative.
- **A.6**: `oos_metrics.profit_factor = 0.8984` vs THRESHOLD `>= 1.10` — FAIL. Losses exceed wins in dollar magnitude.

Per the verdict mapping in `REVIEW_RULES.md` (`if any rule in (A, B, C, D) fails on the candidate metrics → REJECT (regardless of agent positions)`), the outcome is REJECT no matter what any question resolves to. Filing questions whose answers cannot move the verdict would be noise, and the caution-default in Section E means I should not manufacture ESCALATE-bait for a candidate that fails three hard gates on its own reported numbers.

## Non-blocking observations (for the record, not for routing)

I log these so they appear in the transcript, but I explicitly mark them **non-blocking** — they must not gate Round 3:

1. **Sections C/D remain uncitable from this sandbox.** The strategy file at `src/strategies/_experimental/the-decision-geometry-of-covariance-estimation-for-the-global-mi.py` is not present in the review environment, so no `file:line` evidence exists for C.1–C.5 or D.1–D.4 from either agent. In a candidate that passed Section A this would be my top smart question (routed to a code tool). Here it is moot: resolving all of C/D as PASS still yields REJECT on A.1/A.2/A.6.

2. **The pre-registered abandon-threshold comparison result is absent from the report.** The candidate's own `prediction.abandon_threshold` pre-registers a shrinkage-on vs. shrinkage-off (`use_shrinkage=False`) baseline comparison over the first 252-day walk-forward window, but the report contains no field recording that comparison's outcome. This matters for the *research loop's learning*, not for this verdict: it distinguishes "decision-geometry thesis falsified" from "thesis untested; the long-only short-USD basket premium was simply negative over the OOS window" (the report's own note concedes the paper's argument concerns estimation quality, not the sign of the risk premium — and `is_oos_sharpe_ratio = 0.875` plus `edge_concentration = 0.177`, `regime_diversified = true`, `decay_severity = "NONE"` support honest-execution-of-a-losing-premium rather than overfit). I recommend the rejection record note this distinction, but no answer to it changes A.1/A.2/A.6.

3. **A.4 is a narrow pass worth a sign-flip annotation.** `hit_rate = 0.4798` clears the 0.45 floor but sits in the 40–50% band where A.4 instructs agents to flag for sign-flip review. With `profit_factor < 1.0` there is no payoff asymmetry rescuing the sub-50% hit rate. Again: annotation for the rejected-strategy record, not a blocker.

**Zero smart-question YAML blocks emitted. Ready to proceed to Round 3.**

---

## Round: smart_questions — agent: bear_reviewer (bear)
*ts=2026-07-20T16:09:18+00:00 model=claude-fable-5 in/out_tokens=131813/7632 cost=$0.0000 elapsed=107.37s*

Sandbox verification complete before question drafting:

- Working directory (`/private/var/folders/.../curlit-claude-code-yayc5x_y`) contains **no files** — `find . -maxdepth 3 -type f` returns nothing. This confirms Round 1's finding: no repo checkout, so `docs/research/SMART_QUESTIONS.md`, `docs/research/EVIDENCE_FIRST.md`, and `src/strategies/_experimental/the-decision-geometry-of-covariance-estimation-for-the-global-mi.py` are all unreachable from this sandbox.
- `knowledge_search` / `knowledge_search_by_topics` are not present in the available or deferred tool lists. I will not fabricate archive quotes.

**Because the spec file is unreachable, I state my inferred six-field schema explicitly**: `id`, `rule_refs`, `question`, `route` (code-tool | data-tool | other-agent | human), `resolution_criteria`, `blocking`. If the canonical field names differ, the resolver (CL-kiw7) should map these mechanically — the content is complete.

One framing note before the blocks: **none of these questions block the verdict.** A.1 (sharpe = −0.633 < 0.50), A.2 (CI low = −1.672 ≤ 0), and A.6 (profit_factor = 0.898 < 1.10) fail on auto-checkable values straight from the candidate report, and `compute_verdict` checks rule failures *before* open questions — the deterministic outcome is REJECT regardless of how these resolve. These questions exist to (a) fix a process gap that will matter on the next candidate, and (b) decide whether this research line should be closed or revised.

```yaml
id: BEAR-Q1
rule_refs: [C.1, C.2, C.3, C.4, C.5, D.1, D.2, D.3, D.4]
question: >
  The review sandbox contains no repository checkout — the strategy source at
  src/strategies/_experimental/the-decision-geometry-of-covariance-estimation-for-the-global-mi.py
  does not exist in the working directory (verified via find; zero files present).
  Can the orchestrator mount the repo (or inline the strategy file + test files
  into Round 3 context) so Sections C and D can actually be audited? Specific
  unauditable items include C.3 (NZDUSD is in the basket — FX exotic floor of
  1.5 bps spread / 1.0 bps slippage; I cannot verify the cost model meets it)
  and C.1 (no look-ahead claim can be made or cleared without line citations).
route: human
resolution_criteria: >
  Repo checkout or strategy file visible in the Round 3 sandbox, at which point
  I will produce file:line-cited C/D findings or an explicit clean bill. Note
  Bull reported the identical limitation in Round 1 — this is an
  infrastructure defect affecting every debate, not just this candidate.
blocking: false  # for THIS candidate (Section A fails deterministically); true for any future candidate that passes Section A
```

```yaml
id: BEAR-Q2
rule_refs: [A.1, A.2, prediction.abandon_threshold]
question: >
  The report pre-registers its own abandon criterion — shrinkage GMVP OOS Sharpe
  vs the identical strategy with GMVPShrinkageConfig(use_shrinkage=False) over
  the same 252-day walk-forward windows — but reports no result for that
  comparison. What was the baseline's OOS Sharpe? Evidence-first: this is a
  claim the researcher declared empirically locatable (same file, one boolean
  flip) and then did not locate.
route: code-tool
resolution_criteria: >
  A single number: baseline OOS Sharpe over the same windows. If the shrinkage
  variant did not beat it, the researcher's own pre-registered abandon
  threshold fires and the entire research line should be closed in
  data/research/rejected/ with that reason — not merely this candidate.
blocking: false
```

```yaml
id: BEAR-Q3
rule_refs: [A.7]
question: >
  is_oos_sharpe_ratio = 0.8748 with OOS Sharpe = -0.633 implies IS Sharpe
  ≈ -0.554 — the strategy was negative IN-SAMPLE and was still submitted to a
  full two-agent debate. Two sub-questions: (1) confirm the IS Sharpe from the
  per-fold breakdown; (2) A.7's threshold `is_oos_sharpe_ratio <= 2.5` is not
  sign-aware — with both Sharpes negative (or OOS negative, IS positive,
  giving a negative ratio) it passes vacuously. Should the pipeline add an
  `is_metrics.sharpe > 0` pre-gate before debate, and A.7 be made sign-aware?
route: human
resolution_criteria: >
  (1) IS Sharpe value confirmed from per-fold metrics; (2) a filed
  rule-amendment ticket per the "Notes for human reviewers" section (rules are
  the lever; versioned changelog), or an explicit decision that negative-IS
  candidates are intentionally debated.
blocking: false
```

```yaml
id: BEAR-Q4
rule_refs: [A.4, B.2]
question: >
  Internal contradiction in the candidate's own evidence: the G6 flag reports
  regime_diversified = true (edge positive in >=2 of 5 regimes), yet the
  researcher's prediction declares regime_dependence = "vol-spike-only", and
  the aggregate OOS result is -13.9% total return with hit rate 47.98% — inside
  the 40–50% sign-flip band A.4 flags for review. Which regimes does G6 show
  positive edge in, and is the aggregate loss concentrated in USD-strength
  regimes — i.e., the irreducible short-USD factor exposure the researcher
  themselves flagged in the prediction notes?
route: data-tool
resolution_criteria: >
  The per-regime Sharpe table from the G6 layer output. This determines
  whether a revised brief (currency-neutral or long/short basket that isolates
  the covariance-estimation thesis from the short-USD beta) is worth filing,
  or whether BEAR-Q2's abandon criterion should simply close the line. It also
  resolves whether the B.2 pass is genuine or an artifact of a marginally
  positive second regime.
blocking: false
```

Four questions, zero verdict-blockers. My Round 1 position stands unchanged pending resolution: **REJECT** on A.1, A.2, and A.6 — all auto-checkable, none contestable by either agent under the deterministic gate.

---

## Round: rebuttal — agent: bull_reviewer (bull)
*ts=2026-07-20T16:11:20+00:00 model=claude-fable-5 in/out_tokens=50644/3701 cost=$0.0000 elapsed=51.48s*

Sandbox re-verified for the rebuttal round: the working directory is still empty — no repository checkout, no strategy source file, and the Bear's full `REJECT_CASE.md` is not on disk. I therefore engage with the Bear's claims as they appear in the debate transcript summaries, and I verify every quantitative claim directly against the candidate report JSON, which I do have in full.

---

# Round 3 — Rebuttal: Bull Reviewer

*Strategy: `the-decision-geometry-of-covariance-estimation-for-the-global-mi`*

**Preliminary note on evidence availability**: Only the Bear's transcript summary is available in this sandbox (the full `REJECT_CASE.md` file does not exist in the working directory — verified via `find . -maxdepth 3 -type f`, zero results). I engage with each claim visible in the summary and, where the Bear's case is truncated, I reconstruct and address the rule-by-rule failures the Bear would have cited, using the candidate report values directly. I do not dispute claims I cannot see; I verify them against the primary source.

---

## Bear's claim: "A.1 (Annualized OOS Sharpe): FAIL — THRESHOLD `oos_metrics.sharpe >= 0.50`"
**Engagement**: **agree**

The Bear is correct and I concede this rule in full. `oos_metrics.sharpe = -0.6332304110707234` (candidate report, `oos_metrics.sharpe`). This is not borderline — it is below the 0.50 floor by 1.13 Sharpe units and **negative in sign**. It also falls below the candidate's own pre-registered prediction floor of 0.20 (`prediction.predicted_sharpe_range = "[0.20, 0.60]"`), so the strategy fails not only the house rule but its own hypothesis brief. My Round 1 position already scored A.1 as FAIL; nothing in the Bear's case requires revision on my side — we agree.

## Bear's claim: REJECT on Section A statistical grounds (final position: REJECT)
**Engagement**: **agree**

I have re-verified every Section A rule against the candidate report rather than taking the Bear's summary on faith. The Bear's REJECT is supported by **three independent hard failures**, any one of which triggers REJECT under the verdict mapping ("if any rule in (A, B, C, D) fails on the candidate metrics → REJECT"):

- **A.1 — FAIL** (conceded above): `sharpe = -0.6332` vs threshold `>= 0.50`.
- **A.2 — FAIL**: `sharpe_ci_95.low = -1.6724` vs threshold `> 0.0`. Zero is well inside the interval `[-1.6724, +0.3784]`; per REVIEW_RULES.md this is "the single most important gate" and the edge is statistically indistinguishable from luck. Worse for the Bull case: the point estimate is negative, so the CI is centered below zero — this is not a small-sample near-miss.
- **A.6 — FAIL**: `profit_factor = 0.8984` vs threshold `>= 1.10`. Gross losses exceed gross wins; the strategy lost money in dollar magnitude (`total_return = -0.1390`).

No counter-evidence exists in the report against any of these three. I will not manufacture any.

## Bear's claim (implicit in summary): the strategy source is unreachable, so Section C/D cannot be verified
**Engagement**: **agree**

Both agents independently confirmed the sandbox contains no checkout of `src/strategies/_experimental/the-decision-geometry-of-covariance-estimation-for-the-global-mi.py`, and I re-confirmed it this round. Accordingly: **C.1–C.5 and D.1–D.4: NO EVIDENCE — ABSTAIN on all Section C and Section D rules.** I cannot cite line numbers for code that is not present, and I decline to defend code I cannot read. Note this abstention is moot for the verdict: Section A failures already force REJECT regardless of C/D status.

## Where the Bull record must be preserved — genuine passes the Bear should not overstate
**Engagement**: **partially agree** (with the overall REJECT; disagreeing only with any implication that the candidate fails across the board)

For the audit record, the following rules **PASS** on the candidate metrics, and a fair reading of this debate should say so:

- **A.3 — PASS**: `n_trades = 62` ≥ 30. The sample is adequate; the CI in A.2 is interpretable, which is precisely why its failure is credible.
- **A.4 — PASS (with mandated flag)**: `hit_rate = 0.4798` ≥ 0.45. It sits in the 40–50% band, so per A.4's text I flag it for **sign-flip review** — consistent with the negative Sharpe, this is worth one research note before archival, not a promotion argument.
- **A.5 — PASS**: `max_drawdown = -0.2217` ≥ −0.25. Within the kill-switch envelope, though with only ~3 points of headroom.
- **A.7 — PASS**: `is_oos_sharpe_ratio = 0.8748` ≤ 2.5. Notably, IS/OOS ratio below 1.0 means the strategy was **not overfit** — it performed consistently (consistently poorly) in and out of sample. This is an honest negative result, not a data-mined mirage.
- **B.1 — PASS**: `edge_concentration = 0.1766` ≤ 0.60. **B.2 — PASS**: `regime_diversified = true`. **B.3 — PASS**: `decay_severity = "NONE"`, not in `['STRONG','MODERATE']`.
- **Pre-registration integrity**: the report's `abandon_threshold` pre-registered a falsifiable shrinkage-vs-sample-covariance comparison, and `data_sources_consumed` matches the brief's list exactly with no undeclared inputs. The research *process* here was sound even though the *result* is negative.

This matters for one reason only: the correct disposition is REJECT-and-archive with an accurate record ("clean methodology, no edge, possible sign inversion worth one note"), not REJECT-as-garbage. The evidence-first principle cuts both ways — the data says there is no promotable edge, and it says so cleanly.

**Knowledge-archive note**: `knowledge_search` is not reachable from this sandbox (confirmed in Round 1 by both agents); per my constraints I cite no historical quotes rather than fabricate them.

---

## Revised final position

A Bull who promotes against a negative Sharpe, a CI spanning zero with a negative center, and a sub-1.0 profit factor is not a Bull — he is a fabricator. The strongest PROMOTE case available from this evidence does not exist; the strongest *honest* case is that the research pipeline worked exactly as designed and correctly identified a non-edge.

**REJECT**

---

## Round: rebuttal — agent: bear_reviewer (bear)
*ts=2026-07-20T16:11:20+00:00 model=claude-fable-5 in/out_tokens=53920/5197 cost=$0.0000 elapsed=70.46s*

Sandbox re-verified for the rebuttal round: the working directory remains empty, the strategy source file at `src/strategies/_experimental/the-decision-geometry-of-covariance-estimation-for-the-global-mi.py` is **absent**, and no `knowledge_search` / `knowledge_search_by_topics` tool is reachable from this environment. Per my severe-claim threshold I therefore make **no** Section C claims (no line numbers exist to cite) and quote **no** historical-archive chunks (no tool output exists to quote). Everything below is grounded in the candidate report JSON and the Bull's transcript entries, both of which are in context.

---

# Round 3 — Rebuttal: Bear Reviewer
*Strategy: `the-decision-geometry-of-covariance-estimation-for-the-global-mi`*

**Preliminary note**: This is an unusual rebuttal — the Bull did not mount a promotion case. Bull's Round 1 opens with a FAIL citation on A.1, and Bull's own rebuttal concedes the case wholesale ("A Bull who promotes against a negative Sharpe, a CI spanning zero with a negative center, and a sub-1.0 profit factor is not a Bull — he is a fabricator"). My job here is therefore not to demolish a PROMOTE case but to engage each of Bull's claims — concessions and residual PASSes alike — and confirm that no rule the Bull passed actually rehabilitates the candidate.

## Bull's claim: "A.1 (Sharpe ≥ 0.50): FAIL — `oos_metrics.sharpe = -0.63`"
**Engagement**: **accept**

Full agreement. `oos_metrics.sharpe = -0.6332` versus THRESHOLD `>= 0.50`. This is not a marginal miss — the strategy is 1.13 Sharpe units below the floor and *loses money out of sample* (`total_return = -0.1390`, `sortino = -0.9620`). Rule A.1 fails deterministically; the verdict engine will REJECT on this alone regardless of either agent's position.

## Bull's claim (implied concession): A.2 — bootstrap CI
**Engagement**: **accept, and sharpen**

`sharpe_ci_95 = [-1.6724, +0.3784]` versus THRESHOLD `low > 0.0`. FAIL. Note what the interval actually says: it is not merely "zero is inside the CI" (indistinguishable from luck) — the interval is *centered well below zero*. The most probable read of this sample is a strategy with genuinely negative edge, with "harmless" as the optimistic tail. This is worse than the canonical A.2 failure mode.

## Bull's claim (implied concession): A.6 — profit factor
**Engagement**: **accept**

`profit_factor = 0.8984` versus THRESHOLD `>= 1.10`. FAIL. Dollar losses exceed dollar wins. Combined with `hit_rate = 0.4798` — squarely inside the A.4 40–50% sign-flip band — there is no payoff asymmetry rescuing the sub-50% hit rate. If the Bull (or the researcher) is tempted by the sign-flip hypothesis ("invert the signal and harvest +0.63 Sharpe"), that is a *new research candidate* requiring its own pre-registration and OOS run, not a defense of this one. A minimum-variance weighting scheme run backwards is not the registered thesis.

## Bull's claimed PASSes on A.3, A.5, A.7
**Engagement**: **partially accept** (numerically true, probatively empty)

- **A.3** (`n_trades = 62 >= 30`): PASS as stated. But the sample size is sufficient precisely to make the *negative* result credible — 62 rebalance events is enough for the bootstrap to conclude the edge is likely negative. A.3 passing strengthens the REJECT, not the candidate.
- **A.5** (`max_drawdown = -0.2217 >= -0.25`): PASS by 2.8 points of margin on a strategy that only ran one OOS stretch. A minimum-variance long-only FX basket that still drew down 22% while *targeting* variance reduction is cold comfort; the rule passes, the design intent did not.
- **A.7** (`is_oos_sharpe_ratio = 0.8748 <= 2.5`): PASS, but the ratio is degenerate here and I contest any use of it as evidence of robustness. Back out the in-sample figure: IS Sharpe ≈ 0.8748 × (−0.6332) ≈ **−0.55**. The strategy was *already losing in-sample*. There is no IS/OOS divergence because there was never an in-sample edge to diverge from. A.7 is a gate against overfitting; it says nothing in favor of a strategy that fit nothing. Evidence-first framing: the thesis ("shrinkage covariance improves GMVP decision geometry") should have been visible as positive IS performance before ever reaching OOS — it was not, and per `docs/research/EVIDENCE_FIRST.md` an empirically locatable effect that was not located is a vibe, not an edge.

## Bull's claimed PASSes on B.1, B.2, B.3
**Engagement**: **partially accept** (numerically true, semantically inverted)

- **B.1** (`edge_concentration = 0.1766 <= 0.60`): PASS, vacuously. G5 attributes fractions of *total Sharpe*; when total Sharpe is −0.63, low concentration means the *losses* are well-diversified across features. That is not fragility-resistance, it is uniformly distributed failure.
- **B.2** (`regime_diversified = true`): PASS per G6, but I flag an internal inconsistency the Bull did not: the candidate's own prediction block declares `regime_dependence: "vol-spike-only"`. A strategy whose researcher pre-registered single-regime dependence while G6 reports multi-regime edge is reporting *something* wrong — either the prediction or the layer output. Under Section B's own preamble, inconsistent/missing layer semantics lean ESCALATE, but here it is moot: Section A already hard-fails.
- **B.3** (`decay_severity = "NONE"`): PASS. No decay detected — there was no edge to decay. Accepted without probative weight.

## Bull's Section C / D position (no code-level claims made)
**Engagement**: **accept the abstention, mirror it**

Bull correctly declined to fabricate `file:line` citations for a file that does not exist in the sandbox. I do the same: **ABSTAIN on all of C.1–C.5 and D.1–D.4**. I specifically decline to claim look-ahead bias, PaperBroker-constant leakage, or cost-model violations without a quotable line. Under the verdict mapping this abstention is harmless: A.1, A.2, and A.6 fail on the deterministic metrics, which forces REJECT independent of agent positions.

## Bull's revised position: REJECT
**Engagement**: **accept**

The Bull's self-correction is the debate working as designed. I note one genuinely redeeming feature for the *record*, not the verdict: the researcher pre-registered a falsifiable abandon threshold (shrinkage-on vs. shrinkage-off over a 252-day window). That comparison should be run and archived before the strategy is moved to `data/research/rejected/{slug}/` — the negative result plus the ablation is reusable knowledge even though the candidate is not.

## One point of caution against over-agreement

Both agents converging on REJECT is exactly the rubber-stamp risk in reverse, so I checked for any rule where the Bear case might be softer than it looks. There is none: A.1 (−0.63 vs 0.50), A.2 (CI low −1.67 vs > 0), and A.6 (0.90 vs 1.10) are three independent hard fails, each individually sufficient under `compute_verdict`. No metric required by Sections A/B is missing from the report, so there is no ESCALATE path either.

## Revised final position

Three deterministic Section A failures (A.1, A.2, A.6), an in-sample Sharpe that was already negative, a hit rate in the sign-flip band with no payoff asymmetry, and a Bull who has withdrawn his case. Nothing in the Bull's transcript contests any Bear finding; nothing in the candidate report rehabilitates any failed rule.

**FINAL_POSITION**: REJECT

---

## Round: final_position — agent: bull_reviewer (bull)
*ts=2026-07-20T16:11:32+00:00 model=claude-fable-5 in/out_tokens=25306/411 cost=$0.0000 elapsed=8.21s*

**FINAL_POSITION**: REJECT

(One sentence summary: The candidate fails Section A on its own report — A.1 `oos_metrics.sharpe = -0.633` vs threshold ≥ 0.50, A.2 `sharpe_ci_95.low = -1.672` vs threshold > 0, and A.6 `profit_factor = 0.898` vs threshold ≥ 1.10 — a negative-edge strategy no honest Bull can promote; notably, its own pre-registered `abandon_threshold` framework anticipated exactly this failure mode, so rejection here is the research loop working as designed.)

**FINAL_POSITION**: REJECT

---

## Round: final_position — agent: bear_reviewer (bear)
*ts=2026-07-20T16:11:32+00:00 model=claude-fable-5 in/out_tokens=25771/460 cost=$0.0000 elapsed=12.26s*

**FINAL_POSITION**: REJECT

The candidate fails three hard Section-A gates outright — A.1 (`oos_metrics.sharpe = -0.633` vs floor 0.50), A.2 (`sharpe_ci_95.low = -1.672`, zero well inside the interval), and A.6 (`profit_factor = 0.898` vs floor 1.10) — with hit rate 0.480 in the A.4 sign-flip warning band, so the strategy's own reported metrics establish a negative-edge, statistically-indistinguishable-from-luck candidate that no code-integrity or operational finding could rehabilitate; the Bull conceded these failures in Round 3, and my position is unchanged.

**FINAL_POSITION**: REJECT

---

## Debate summary
- **bull_reviewer**: REJECT
- **bear_reviewer**: REJECT

**Total cost**: $0.0000
