"""Unit tests for the question resolver (CL-kiw7).

Exercises every code path: YAML parsing, validation against the
SMART_QUESTIONS.md format, the 3 routing dispatchers, all caps. Pure
Python — no LLM or DB dependency, fully deterministic.
"""

from __future__ import annotations

from typing import Any

from src.research.agents.resolver import (
    MAX_QUESTIONS_PER_AGENT,
    MAX_REFORMULATES_PER_QUESTION,
    QuestionResolver,
    ResolutionStatus,
    Routing,
    SmartQuestion,
    parse_questions,
    validate_question,
)

# =============================================================================
# Fixtures
# =============================================================================


def _well_formed_question(
    qid: str = "q-1-1",
    asker: str = "bull_reviewer",
    routing: Routing = Routing.CODE_TOOL,
) -> SmartQuestion:
    """A question that passes every validation rule. Tests mutate this
    to fail one rule at a time."""
    return SmartQuestion(
        question_id=qid,
        asker=asker,
        decision_blocker=(
            "Whether to abstain on rule B.1 (edge concentration) because "
            "the candidate report's edge_concentration field doesn't "
            "specify the time window used."
        ),
        what_i_tried_first=(
            "- Read reports/candidates/test.json — edge_concentration is a "
            "single scalar with no window annotation\n"
            "- Read src/edge_testing/feature_attribution.py:60-115 — the "
            "docstring says 'on the supplied returns' but doesn't specify "
            "whether the runner passes IS or OOS"
        ),
        specific_evidence_needed=(
            "A specific line reference in the implementer's strategy code or "
            "the backtest runner showing whether edge_concentration is "
            "computed on is_returns or oos_returns."
        ),
        routing=routing,
        acceptance=(
            "A line citation from src/research/agents/implementer.py or "
            "scripts/backtest_*.py with the call site for "
            "FeatureEdgeAttributor; 'IS' answer → ESCALATE, 'OOS' → PROMOTE"
        ),
    )


# =============================================================================
# parse_questions — extract YAML blocks from agent text
# =============================================================================


class TestParseQuestions:
    def test_single_question_parsed(self) -> None:
        text = """\
Some preamble.

```yaml
question_id: q-1-1
asker: bull_reviewer
decision_blocker: rule A.1 needs metric value clarified
what_i_tried_first: |
  - searched report
  - read code
specific_evidence_needed: the actual sharpe value as a number
routing: code_tool
acceptance: a number between -2 and 3
```

Rest of response.
"""
        questions = parse_questions(text)
        assert len(questions) == 1
        q = questions[0]
        assert q.question_id == "q-1-1"
        assert q.asker == "bull_reviewer"
        assert q.routing == Routing.CODE_TOOL

    def test_multiple_questions(self) -> None:
        text = """\
```yaml
question_id: q-1-1
asker: bull
decision_blocker: rule A.1
what_i_tried_first: a
specific_evidence_needed: b
routing: code_tool
acceptance: c
```

```yaml
question_id: q-1-2
asker: bear
decision_blocker: rule A.2
what_i_tried_first: d
specific_evidence_needed: e
routing: human
acceptance: f
```
"""
        questions = parse_questions(text)
        assert len(questions) == 2
        assert questions[0].routing == Routing.CODE_TOOL
        assert questions[1].routing == Routing.HUMAN

    def test_yaml_block_without_required_fields_is_skipped(self) -> None:
        """Code samples in YAML format that aren't questions get
        ignored (don't have all 6 required fields)."""
        text = """\
```yaml
foo: bar
baz: qux
```

```yaml
question_id: q-real
asker: bull
decision_blocker: rule A.1
what_i_tried_first: a
specific_evidence_needed: b
routing: code_tool
acceptance: c
```
"""
        questions = parse_questions(text)
        # Only the real question; the foo:bar block is skipped silently
        assert len(questions) == 1
        assert questions[0].question_id == "q-real"

    def test_unfenced_yaml_not_parsed(self) -> None:
        """Yaml-looking content without ``` fences is NOT a question —
        avoids false positives on agent narrative."""
        text = "question_id: q-1-1\nasker: bull\n(no fence above)"
        assert parse_questions(text) == []

    def test_malformed_yaml_skipped_silently(self) -> None:
        text = """\
```yaml
question_id: q-1-1
asker: [unclosed bracket
```
"""
        # Malformed YAML inside a fence — skipped, not crashing
        assert parse_questions(text) == []

    def test_invalid_routing_keeps_question_for_validator(self) -> None:
        """Routing field with an unknown value still produces a
        SmartQuestion (with HUMAN placeholder) so validate_question
        can surface a precise reformulate reason."""
        text = """\
```yaml
question_id: q-1-1
asker: bull
decision_blocker: rule A.1
what_i_tried_first: a
specific_evidence_needed: b
routing: rocket_launch
acceptance: c
```
"""
        questions = parse_questions(text)
        # Block IS captured (placeholder routing) — validator handles
        assert len(questions) == 1


# =============================================================================
# validate_question — SMART_QUESTIONS.md bounce conditions
# =============================================================================


class TestValidation:
    def test_well_formed_passes(self) -> None:
        q = _well_formed_question()
        assert validate_question(q) == []

    def test_decision_blocker_without_rule_id_fails(self) -> None:
        q = _well_formed_question()
        q.decision_blocker = "I'm not sure if the strategy is good."
        issues = validate_question(q)
        assert any("rule ID" in i for i in issues)

    def test_decision_blocker_vague_phrase_fails(self) -> None:
        q = _well_formed_question()
        q.decision_blocker = "Not sure about A.1"
        issues = validate_question(q)
        # Both: vague phrase AND short. Should at least flag vagueness.
        assert any("vague" in i.lower() for i in issues)

    def test_what_i_tried_first_one_attempt_fails(self) -> None:
        q = _well_formed_question()
        q.what_i_tried_first = "- Just looked at the report once"
        issues = validate_question(q)
        assert any("at least 2" in i for i in issues)

    def test_what_i_tried_first_no_list_fails(self) -> None:
        q = _well_formed_question()
        q.what_i_tried_first = "looked at things, didn't find them"
        issues = validate_question(q)
        assert any("at least 2" in i for i in issues)

    def test_specific_evidence_restating_question_fails(self) -> None:
        q = _well_formed_question()
        # Make specific_evidence_needed a copy of decision_blocker
        q.specific_evidence_needed = q.decision_blocker
        issues = validate_question(q)
        assert any("restates" in i for i in issues)

    def test_acceptance_bare_yes_fails(self) -> None:
        q = _well_formed_question()
        q.acceptance = "Yes"
        issues = validate_question(q)
        assert any("yes/no" in i for i in issues)

    def test_acceptance_substantive_passes(self) -> None:
        q = _well_formed_question()
        # Substantive answer with verification criterion is fine
        q.acceptance = (
            "An answer of '0.0008 per turn' tied to the asset class "
            "lookup in src/backtest/cost_model.py:30"
        )
        assert validate_question(q) == []

    def test_empty_field_caught(self) -> None:
        q = _well_formed_question()
        q.decision_blocker = ""
        issues = validate_question(q)
        assert any("decision_blocker is empty" in i for i in issues)

    def test_multiple_issues_all_returned(self) -> None:
        q = _well_formed_question()
        q.decision_blocker = "vague"  # no rule ID + likely vague
        q.what_i_tried_first = "single thing"  # no list
        q.acceptance = "yes"  # bare
        issues = validate_question(q)
        # Should surface all three
        assert len(issues) >= 3


# =============================================================================
# QuestionResolver — code_tool routing
# =============================================================================


class TestCodeToolRouting:
    def test_code_tool_dispatch_returns_resolved(self) -> None:
        captured: dict[str, Any] = {}

        def grep_tool(name: str, args: dict[str, Any]) -> str:
            captured.update({"name": name, "args": dict(args)})
            return "found at line 142"

        resolver = QuestionResolver(code_tools={"grep": grep_tool})
        q = _well_formed_question(routing=Routing.CODE_TOOL)
        result = resolver.resolve(
            q,
            tool_name="grep",
            tool_args={"pattern": "future_dated"},
        )
        assert result.status == ResolutionStatus.RESOLVED
        assert result.answer == "found at line 142"
        assert captured == {"name": "grep", "args": {"pattern": "future_dated"}}

    def test_unknown_tool_fails_loud(self) -> None:
        resolver = QuestionResolver(code_tools={"grep": lambda n, a: "x"})
        q = _well_formed_question(routing=Routing.CODE_TOOL)
        result = resolver.resolve(q, tool_name="bogus_tool")
        assert result.status == ResolutionStatus.FAILED
        assert "unknown code tool" in result.escalate_reason

    def test_no_tool_name_fails_loud(self) -> None:
        resolver = QuestionResolver(code_tools={"grep": lambda n, a: "x"})
        q = _well_formed_question(routing=Routing.CODE_TOOL)
        result = resolver.resolve(q)  # no tool_name
        assert result.status == ResolutionStatus.FAILED

    def test_tool_exception_returns_failed(self) -> None:
        def boom(name: str, args: dict[str, Any]) -> str:
            msg = "subprocess died"
            raise RuntimeError(msg)

        resolver = QuestionResolver(code_tools={"grep": boom})
        q = _well_formed_question(routing=Routing.CODE_TOOL)
        result = resolver.resolve(q, tool_name="grep")
        assert result.status == ResolutionStatus.FAILED
        assert "subprocess died" in result.escalate_reason


# =============================================================================
# QuestionResolver — other_agent routing
# =============================================================================


class TestOtherAgentRouting:
    def test_other_agent_dispatch_returns_resolved(self) -> None:
        captured: dict[str, str] = {}

        def dispatch(target: str, prompt: str) -> str:
            captured["target"] = target
            captured["prompt"] = prompt
            return "Bear says: agreed, that line is 142, look-ahead."

        resolver = QuestionResolver(agent_dispatch=dispatch)
        q = _well_formed_question(routing=Routing.OTHER_AGENT)
        result = resolver.resolve(q, target_agent="bear_reviewer")
        assert result.status == ResolutionStatus.RESOLVED
        assert "Bear says" in result.answer
        assert captured["target"] == "bear_reviewer"
        assert "decision_blocker" in captured["prompt"].lower() or "rule B.1" in captured["prompt"]

    def test_no_target_agent_fails_loud(self) -> None:
        resolver = QuestionResolver(agent_dispatch=lambda t, p: "x")
        q = _well_formed_question(routing=Routing.OTHER_AGENT)
        result = resolver.resolve(q)  # no target_agent
        assert result.status == ResolutionStatus.FAILED

    def test_no_dispatch_callback_fails_loud(self) -> None:
        resolver = QuestionResolver()  # no agent_dispatch
        q = _well_formed_question(routing=Routing.OTHER_AGENT)
        result = resolver.resolve(q, target_agent="bear")
        assert result.status == ResolutionStatus.FAILED


# =============================================================================
# QuestionResolver — human routing
# =============================================================================


class TestHumanRouting:
    def test_human_routes_to_escalate_immediately(self) -> None:
        resolver = QuestionResolver()
        q = _well_formed_question(routing=Routing.HUMAN)
        result = resolver.resolve(q)
        assert result.status == ResolutionStatus.ESCALATE
        assert "human" in result.escalate_reason.lower()


# =============================================================================
# Validation → reformulate
# =============================================================================


class TestReformulate:
    def test_invalid_question_returns_reformulate(self) -> None:
        resolver = QuestionResolver()
        q = _well_formed_question()
        q.decision_blocker = "vague — no rule"  # fails validation
        result = resolver.resolve(q, tool_name="grep")
        assert result.status == ResolutionStatus.REFORMULATE
        assert len(result.reformulate_reasons) >= 1
        # The escalation must NOT have been triggered for a malformed
        # question that hasn't hit the reformulate cap yet
        assert result.escalate_reason == ""


# =============================================================================
# Caps
# =============================================================================


class TestCaps:
    def test_questions_per_agent_cap(self) -> None:
        resolver = QuestionResolver(code_tools={"grep": lambda n, a: "ok"})
        # Submit MAX_QUESTIONS_PER_AGENT + 1 questions from same asker
        for i in range(MAX_QUESTIONS_PER_AGENT):
            q = _well_formed_question(qid=f"q-1-{i}", routing=Routing.CODE_TOOL)
            r = resolver.resolve(q, tool_name="grep")
            assert r.status == ResolutionStatus.RESOLVED, f"failed at i={i}"
        # Next one over cap
        q_over = _well_formed_question(qid="q-1-over", routing=Routing.CODE_TOOL)
        r = resolver.resolve(q_over, tool_name="grep")
        assert r.status == ResolutionStatus.ESCALATE
        assert "exceeded" in r.escalate_reason

    def test_reformulate_cap_forces_escalate(self) -> None:
        resolver = QuestionResolver()
        # Submit same question_id with malformed content N times.
        # After MAX_REFORMULATES_PER_QUESTION bounces, the next attempt
        # auto-escalates.
        statuses: list[ResolutionStatus] = []
        for _ in range(MAX_REFORMULATES_PER_QUESTION + 1):
            q = _well_formed_question(qid="q-stuck")
            q.decision_blocker = "vague"  # always fails
            r = resolver.resolve(q, tool_name="anything")
            statuses.append(r.status)
        # First MAX_REFORMULATES are REFORMULATE
        assert all(
            s == ResolutionStatus.REFORMULATE for s in statuses[:MAX_REFORMULATES_PER_QUESTION]
        )
        # The (cap+1)-th attempt escalates the underlying decision
        assert statuses[-1] == ResolutionStatus.ESCALATE


# =============================================================================
# Ledger inspection
# =============================================================================


class TestLedger:
    def test_ledger_records_questions_per_agent(self) -> None:
        resolver = QuestionResolver(code_tools={"grep": lambda n, a: "x"})
        for i in range(3):
            q = _well_formed_question(qid=f"q-1-{i}", routing=Routing.CODE_TOOL)
            resolver.resolve(q, tool_name="grep")
        assert resolver.ledger("bull_reviewer").questions_asked == 3
        assert resolver.ledger("nonexistent_agent").questions_asked == 0
