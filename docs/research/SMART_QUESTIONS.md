# Smart Questions in the Multi-Agent Debate

Adapted from Eric S. Raymond's [How To Ask Questions The Smart
Way](http://www.catb.org/~esr/faqs/smart-questions.html). The principles
transfer directly from human-to-human Q&A to agent-to-agent and
agent-to-tool questions: the asker must have done their homework, the
question must be specific enough to answer concretely, and the answer
must be verifiable against the original ask.

The question resolver (`src/research/agents/resolver.py`, CL-kiw7)
enforces this format mechanically. Vague or unmotivated questions get
auto-bounced with a "reformulate" note BEFORE any other agent or tool is
consulted — wasted-cycle prevention.

## When to ask vs decide

Round 1 of the debate (initial positions) is for stating your case using
the evidence already in the candidate report. If you reach the end of
Round 1 with a specific gap that prevents you from committing to a
verdict in Round 4, that gap becomes a smart-question candidate.

A gap is a smart-question candidate only when:

1. **Resolving the unknown WOULD change your verdict.** ("If I knew the
   regime decomposition included regime X, I'd shift from REJECT to
   PROMOTE on rule B.2.") — not a fishing expedition.
2. **The unknown is specific** — a metric value, a code path, a regime
   check. Not vague ("I'm uncertain about the strategy").
3. **You've already tried at least two ways** to resolve it from
   available evidence before asking.

If your "question" is really "the candidate report doesn't tell me
something I need" — the right move is `routing: code_tool` so the
resolver runs the analysis and adds it to the report. Don't ask another
agent something you can compute.

## Required format

Every smart question is YAML with all six fields. The resolver auto-
rejects any question missing a field — this is the format check, not a
content check (content quality is judged separately).

```yaml
question_id: q-{round}-{n}              # auto-assigned by orchestrator
asker: bull_reviewer                    # which agent is asking

decision_blocker: |
  Specific verdict-affecting decision this question would resolve.
  Must reference a rule ID from REVIEW_RULES.md.
  Example:
    "Whether to abstain on rule B.1 because the candidate report's
    edge_concentration field is computed across only the OOS window
    and may not reflect the full-history concentration."

what_i_tried_first: |
  At least 2 specific things attempted before asking, with the result
  of each attempt. Vague effort = bounce.
  Example:
    - "Re-read reports/candidates/{slug}.json — edge_concentration is a
       single scalar without a time-window annotation."
    - "Read src/edge_testing/feature_attribution.py:60-115 to confirm
       what window is used; the docstring says 'on the supplied returns'
       but doesn't specify whether the runner passes IS or OOS."

specific_evidence_needed: |
  Exactly what answer would unblock the verdict. Be precise.
  Example:
    "A direct line reference in the implementer's strategy code or the
    backtest runner showing whether edge_concentration is computed on
    is_returns vs oos_returns."

routing: code_tool                      # code_tool | other_agent | human

acceptance: |
  What answer is acceptable. NOT a re-statement of the question.
  Example:
    "A specific line from src/research/agents/implementer.py or
    scripts/backtest_*.py with the call site for FeatureEdgeAttributor.
    Acceptable answers: 'IS' (then I'd ESCALATE because B.1 is on wrong
    window), 'OOS' (then B.1 evaluates correctly), 'both with a
    breakdown' (then I PROMOTE if both pass)."
```

## The six fields, in detail

### `decision_blocker`

The most important field. This is what filters fishing expeditions from
real blockers. It must:

- Reference a specific rule from `REVIEW_RULES.md` by ID
- Describe a binary or small-N decision the answer would resolve
- Be falsifiable (you could imagine an answer that doesn't help)

Bad: "I'm not sure if this strategy is good."
Good: "I cannot decide rule A.2 because the report's `sharpe_ci_95.low`
is reported but I don't know if it's the 2.5%ile (correct for 95% CI
two-sided) or the 5%ile (would inflate confidence)."

### `what_i_tried_first`

ESR's principle. Show your work. Two minimum, each with the specific
artifact you consulted and what you found there.

Bad:
- "Looked at the code."
- "Couldn't figure it out."

Good:
- "Read `src/backtest/bootstrap.py:55-67` — the function takes
   `confidence=0.95` and the alpha calculation is `(1 - confidence) / 2`
   = 0.025, suggesting 2.5%ile."
- "Cross-checked against the per-fold metrics in the report — the per-
   fold Sharpes vary [-0.4, +1.1] and the reported CI low is -0.073,
   which is only consistent with the 2.5%ile interpretation."

If those two efforts already answered the question, no question needed —
you have your answer.

### `specific_evidence_needed`

What concrete artifact would resolve the question. Not "an explanation"
— a thing that can be cited.

Bad: "Help me understand the cost model."
Good: "The CostModel.cost_per_turn value used by this backtest run, as
a number. Acceptable: a value or a chain of code references that
computes it."

### `routing`

| Value | When to use | Resolver behavior |
| --- | --- | --- |
| `code_tool` | Answer is in code or data — runnable / queryable | Resolver runs grep, re-runs backtest with adjusted param, queries Postgres. Returns answer in transcript. No other agent. |
| `other_agent` | Answer requires the other reviewer's reasoning | Resolver POSTs the question to the named agent, waits for response, returns to transcript. |
| `human` | Answer requires operator judgment NOT in code/data | Triggers ESCALATE notification immediately. Strategy stays pending until operator answers via dashboard or CLI. |

If you route to `human` and the answer was actually obtainable from the
repo — the resolver bounces it back ("did you grep for X first?") with
a reformulate request.

### `acceptance`

The criterion that says "yes, this answer unblocks me." Different from
`specific_evidence_needed` — that's WHAT, this is HOW you'd know the
WHAT is correct. Two ends of the same epistemic operation.

Bad acceptance: "I just need to know."
Good acceptance: "An answer of '0.0008 per turn' tied to the asset
class lookup in `src/backtest/cost_model.py:30` for the relevant pair.
Anything else I'd treat as untrustworthy."

## Routing rules (cheatsheet)

```
+-----------------+----------------+----------------------+
| Question is...  | Routing        | Example              |
+-----------------+----------------+----------------------+
| In source code  | code_tool      | "what window does    |
|                 |                |  G5 use?"            |
+-----------------+----------------+----------------------+
| In a DB row     | code_tool      | "is fred series      |
|                 |                |  IRLTLT01DEM156N     |
|                 |                |  populated for       |
|                 |                |  2018?"              |
+-----------------+----------------+----------------------+
| Other agent's   | other_agent    | Bear: "Bull, did you |
| reasoning       |                |  consider that 2020  |
|                 |                |  COVID skews the     |
|                 |                |  IS Sharpe?"         |
+-----------------+----------------+----------------------+
| Operator-only   | human          | "should we trade     |
| context         |                |  this strategy on    |
|                 |                |  the OANDA practice  |
|                 |                |  account?"           |
+-----------------+----------------+----------------------+
```

## Resolver bounce conditions

The resolver returns "reformulate" instead of an answer when:

- `decision_blocker` is vague ("not sure", "don't know if good", "feels
  off") OR doesn't reference a rule ID from REVIEW_RULES.md
- `what_i_tried_first` has fewer than 2 specific attempts (each with a
  concrete artifact consulted)
- `specific_evidence_needed` is a re-statement of the question rather
  than a description of an answer
- `acceptance` is a yes/no without verification criterion
- `routing` is wrong for the question content (e.g., asking the human
  for a code path that's right there in the repo)

The asker gets the bounce with reasons attached and may re-submit
within the same round. If three consecutive bounces on the same
question, the orchestrator forces ESCALATE on the underlying decision
blocker — that's a signal the agent can't formulate the question
clearly enough for any answer to unblock it.

## Caps

- Max 8 questions per agent per debate. Above this, the agent is
  fishing.
- Max 30 seconds wall time per `code_tool` resolution. Above this,
  ESCALATE.
- Max 3 reformulate rounds per question. Above this, ESCALATE the
  underlying decision.

## Why this format matters

Without this discipline, two-LLM debates collapse into either:

- **Ping-pong agreement** — both agents nod and converge to "looks
  fine" because the harder path is to surface specific concerns.
- **Theatre disagreement** — agents adopt their assigned roles
  performatively without engaging with the actual evidence.

Smart questions force the agent to do the homework before disagreeing,
which means the disagreement that survives is genuinely about evidence
the resolver cannot dispatch deterministically. That residual is the
ESCALATE bucket — exactly what humans should look at, where
their judgment adds value beyond the rules.

## Changelog

- 2026-04-27 (CL-kr82) — initial version, derived from ESR's
  smart-questions methodology.
