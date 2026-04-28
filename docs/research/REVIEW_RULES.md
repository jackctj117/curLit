# Strategy Promotion Review Rules

This document is the **deterministic gate** that decides whether a candidate
strategy proposed by the AI research loop is promoted from research to paper-
shadow deployment. Both review agents (Bull, Bear) read this file as system
context; the verdict engine (`src/research/verdict.py`) parses the threshold
expressions to convert agent evidence + raw metrics into a binary outcome.

The agents do not decide. The agents surface evidence — citing rule IDs and
specific values, code paths, or transcript references. The rules decide.

## Outcomes

A strategy receives ONE verdict per debate:

| Verdict | When |
| --- | --- |
| **PROMOTE** | Every Section A/B/C/D rule passes against the candidate metrics AND both review agents end Round 4 with PROMOTE. Strategy registers in paper-shadow with `allocation = 0` (CL-3xn1). |
| **REJECT** | Any single Section A/B/C/D rule fails. Or both agents end with REJECT/ABSTAIN. Strategy moved to `data/research/rejected/{slug}/` with reason. |
| **ESCALATE** | Agents disagree (one PROMOTE, one REJECT). Or any required metric is missing. Or any unresolved smart-question remains after Round 2. Pushover/Telegram alert with full transcript link (CL-o2vb). |

---

## Section A — Statistical reality

Auto-checkable from the candidate report at `reports/candidates/{slug}.json`.
The verdict engine reads the `THRESHOLD:` line literally — agents may NOT
override these.

### A.1 — Annualized OOS Sharpe

THRESHOLD: `oos_metrics.sharpe >= 0.50`

Below 0.50, edge is too weak after slippage and operational overhead to
justify book risk. The architectural target for the existing strategies is
0.5–1.0; we set the floor at the bottom of that band.

### A.2 — 95% bootstrap-CI lower bound > 0

THRESHOLD: `sharpe_ci_95.low > 0.0`

The single most important gate. If zero is in the confidence interval the
edge is statistically indistinguishable from luck on the available sample.
This rule — combined with stationary-bootstrap on per-trade returns — is
what catches most overfit strategies.

### A.3 — Sample size

THRESHOLD: `oos_metrics.n_trades >= 30`

Below 30 trades the bootstrap CI is itself too noisy to interpret. For
event-driven strategies this means waiting for more events; for daily
strategies this is roughly two months of OOS history.

### A.4 — Hit rate floor

THRESHOLD: `oos_metrics.hit_rate >= 0.45`

A profitable strategy can have a hit rate below 50% with positive payoff
asymmetry, but anything under 45% is a strong signal the trade direction is
inverted. Agents should flag for sign-flip review when hit rate is in the
40-50% band.

### A.5 — Max drawdown floor

THRESHOLD: `oos_metrics.max_drawdown >= -0.25`

Catastrophic floor. A 25% drawdown is the operational limit at which the
existing kill switches would force an unwind anyway, so anything worse
indicates the strategy is structurally incompatible with the engine's risk
controls.

### A.6 — Profit factor

THRESHOLD: `oos_metrics.profit_factor >= 1.10`

Wins must exceed losses by at least 10% in dollar magnitude — not just
barely positive.

### A.7 — In-sample / out-of-sample divergence

THRESHOLD: `is_oos_sharpe_ratio <= 2.5`

If the IS Sharpe is more than 2.5x the OOS Sharpe, the strategy almost
certainly overfit. Agents should look at per-fold metrics to confirm the
divergence isn't driven by a single anomalous fold.

---

## Section B — Edge structure

Requires the G5 / G6 / G7 layer outputs to be present in the candidate
report. If any of these layers is missing → ESCALATE (not REJECT).

### B.1 — Edge concentration (G5)

THRESHOLD: `edge_concentration <= 0.60`

The G5 feature-attribution layer measures what fraction of total Sharpe
comes from a single top feature. Above 60% means the strategy is fragile —
if that one feature regime-shifts, the whole edge evaporates.

### B.2 — Regime diversification (G6)

THRESHOLD: `regime_diversified == True`

The G6 regime decomposition layer flags `regime_diversified=True` when edge
is positive in at least 2 of the 5 market regimes. Strategies that profit
only in one regime (e.g., only in low-vol-trending) are too fragile for
production.

### B.3 — No active decay

THRESHOLD: `decay_severity not in ['STRONG', 'MODERATE']`

The G7 decay-detection layer reports decay severity by Mann-Kendall +
Mann-Whitney + rolling slope. Strategies showing moderate or strong decay
on the OOS sample are already losing edge before deployment.

---

## Section C — Code integrity

Bear's primary hunting ground. Each must be cited with `file:line` evidence.
Bear cannot vague-reject; Bull cannot vague-defend.

### C.1 — No future-dated lookups

The walk-forward harness must respect time-order. Any data access where
`ts > current_eval_ts` is a fatal lookahead bias. Bear must cite the
specific line if claiming a violation.

### C.2 — Parameter discipline

The strategy config has at most 5 free tunable knobs. Each additional knob
quadruples the search space and the overfitting risk. Above 5, the
strategy needs an ablation study (separate ticket).

### C.3 — Realistic cost model

| Asset class | Min spread | Min slippage |
| --- | --- | --- |
| FX major (EUR/USD, USD/JPY, GBP/USD) | 0.3 bps | 0.3 bps |
| FX exotic (NOK, NZD, SEK, EM) | 1.5 bps | 1.0 bps |
| Crypto | 5 bps | 5 bps |

Below these floors → cost model unrealistic → REJECT.

### C.4 — No PaperBroker constants

Strategy must not hardcode the `PaperBroker.stream_prices` constant
`(1.1000, 1.1002)` or any value derived from the test fixtures. This is a
common bug when porting backtest code to live strategy.

### C.5 — Standard Strategy protocol

The strategy implements either `fit()` + `generate_signals()` (single-pair)
OR `generate_intents()` (portfolio-level), with no patches to
`LiveEngine.run()` or `PortfolioCoordinator.process_intents()`. Any
engine-side change is a separate concern and triggers a different review.

---

## Section D — Operational fitness

### D.1 — Test coverage

`make test` passes. The strategy file has at least 3 strategy-specific tests
covering: a known entry case, a known exit case, an edge case (zero data,
broken model fit, etc.).

### D.2 — Type checking

`.venv/bin/python -m mypy --strict src/strategies/{slug}.py` exits with
code 0. Pre-existing mypy errors elsewhere are not the strategy's concern.

### D.3 — Property tests for numerical helpers

For any numerical helper (e.g., custom sizing, signal smoothing), there are
hypothesis property tests covering: monotonicity (where applicable),
boundedness (output stays in its declared range), no-NaN-on-valid-input.

### D.4 — Feature snapshots per intent

The strategy emits `FeatureSnapshot` via the `_emit_snapshot` helper
established in CL-xpw9. This is what makes the trade auditable — every
intent in `trade_journal_events` carries a snapshot reference enabling
exact reconstruction of the values that produced it.

---

## Section E — Discussion protocol

### Round 1 — Initial positions (parallel)

Bull writes `PROMOTE_CASE.md` citing each A.N / B.N / C.N / D.N rule with
specific evidence — line numbers from the strategy file, metric values from
the candidate report, fold IDs from the per-fold breakdown.

Bear writes `REJECT_CASE.md` citing failures by the same scheme. If Bear
finds none, Bear must write so explicitly: "no rule violations found at
Round 1." (Fabricating violations is a worse failure mode than admitting
none.)

### Round 2 — Smart-question stage (per-agent async)

Each agent surfaces unknowns per `docs/research/SMART_QUESTIONS.md`. The
question resolver (CL-kiw7) routes each to the right venue: a code/data
tool, the other agent, or human escalation. Round 3 starts only after all
questions are resolved or escalated.

### Round 3 — Rebuttal (sequential)

Each agent must engage with the OTHER agent's strongest claim from Rounds 1
and 2. Refusing to engage with a cited claim — saying "I disagree" without
citing why — counts as ABSTAIN on that claim. The orchestrator records the
specific claims that were not engaged with.

### Round 4 — Final position (parallel)

Each agent declares its final verdict: `PROMOTE`, `REJECT`, or `ABSTAIN`.
ABSTAIN counts as REJECT (caution-default — if you can't argue for it,
don't promote it).

### Verdict mapping (deterministic, in `src/research/verdict.py`)

```
verdict = compute_verdict(rules_pass, bull_position, bear_position, open_questions)

  if any rule in (A, B, C, D) fails on the candidate metrics:
      → REJECT (regardless of agent positions)
  elif any required metric is missing from the candidate report:
      → ESCALATE
  elif any open smart-question remains unresolved after Round 2:
      → ESCALATE
  elif bull_position == 'PROMOTE' and bear_position == 'PROMOTE':
      → PROMOTE
  elif bull_position in ('REJECT', 'ABSTAIN') and bear_position in ('REJECT', 'ABSTAIN'):
      → REJECT
  else:
      → ESCALATE  # mixed positions
```

The verdict engine is pure Python — no LLM call. Test-covered. Audit-able.
This is the safety net that prevents two LLMs from rubber-stamping each
other on a strategy that fails a quantitative gate.

---

## Notes for human reviewers

If the loop produces too many REJECTs or too many ESCALATEs, the rules
themselves are the lever — tighten or loosen them deliberately. Don't
relax thresholds to get a strategy through; that defeats the purpose. If
the candidate is genuinely good but the rules block it, the rule is
probably wrong for this asset class — file a ticket to amend it.

This file is the contract between the researcher and the engine. Changes
to thresholds should go through PR review and a versioned changelog at
the bottom of this file.

## Changelog

- 2026-04-27 (CL-kr82) — initial version, baselines drawn from architecture
  doc expectations and existing G1–G9 layer outputs.
