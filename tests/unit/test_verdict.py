"""Unit tests for the verdict engine (CL-ath2).

Validates the deterministic outcome mapping. This is the safety net
against two-LLM rubber-stamping — the rules + the engine ultimately
decide PROMOTE / REJECT / ESCALATE, not the agents.

Tests cover:
  * Threshold expression parser (literals, comparators, paths)
  * Rule extraction from REVIEW_RULES.md
  * Each verdict outcome with its trigger condition
  * Missing metrics → ESCALATE
  * Mixed positions → ESCALATE
  * Open questions → ESCALATE
  * Rule failures override agent positions (gate is hard)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.research.agents.reviewer import Position
from src.research.verdict import (
    ParsedRule,
    Verdict,
    _parse_literal,
    _resolve_path,
    compute_verdict,
    evaluate_threshold,
    parse_rules,
)

# =============================================================================
# Path resolution
# =============================================================================


class TestResolvePath:
    def test_top_level(self) -> None:
        report = {"sharpe": 0.7}
        v, found = _resolve_path(report, "sharpe")
        assert found
        assert v == 0.7

    def test_nested(self) -> None:
        report = {"oos_metrics": {"sharpe": 0.7, "n_trades": 35}}
        v, found = _resolve_path(report, "oos_metrics.n_trades")
        assert found
        assert v == 35

    def test_missing(self) -> None:
        report = {"oos_metrics": {"sharpe": 0.7}}
        v, found = _resolve_path(report, "oos_metrics.missing_key")
        assert not found
        assert v is None

    def test_partial_path_missing(self) -> None:
        report = {"top": {"a": 1}}
        v, found = _resolve_path(report, "top.b.c")  # b doesn't exist
        assert not found


# =============================================================================
# Literal parsing
# =============================================================================


class TestParseLiteral:
    def test_int(self) -> None:
        assert _parse_literal("30") == 30

    def test_float(self) -> None:
        assert _parse_literal("0.5") == 0.5
        assert _parse_literal("-0.25") == -0.25
        assert _parse_literal("1e-3") == 1e-3

    def test_bool(self) -> None:
        assert _parse_literal("True") is True
        assert _parse_literal("False") is False

    def test_string_list(self) -> None:
        assert _parse_literal("('A', 'B')") == ["A", "B"]
        assert _parse_literal("['STRONG', 'MODERATE']") == ["STRONG", "MODERATE"]

    def test_empty_list(self) -> None:
        assert _parse_literal("[]") == []


# =============================================================================
# Threshold evaluation
# =============================================================================


class TestEvaluateThreshold:
    def test_pass_gte(self) -> None:
        report = {"oos_metrics": {"sharpe": 0.7}}
        passed, missing, actual = evaluate_threshold(
            "oos_metrics.sharpe >= 0.50", report,
        )
        assert passed
        assert not missing
        assert actual == 0.7

    def test_fail_gte(self) -> None:
        report = {"oos_metrics": {"sharpe": 0.3}}
        passed, missing, _ = evaluate_threshold(
            "oos_metrics.sharpe >= 0.50", report,
        )
        assert not passed
        assert not missing

    def test_strict_gt(self) -> None:
        # Boundary: equal does NOT pass on >
        report = {"sharpe_ci_95": {"low": 0.0}}
        passed, _, _ = evaluate_threshold(
            "sharpe_ci_95.low > 0.0", report,
        )
        assert not passed

    def test_missing_path(self) -> None:
        report = {"sharpe": 0.7}
        passed, missing, actual = evaluate_threshold(
            "oos_metrics.sharpe >= 0.50", report,
        )
        assert not passed
        assert missing
        assert actual is None

    def test_not_in(self) -> None:
        report = {"decay_severity": "NO"}
        passed, _, _ = evaluate_threshold(
            "decay_severity not in ['STRONG', 'MODERATE']", report,
        )
        assert passed
        report["decay_severity"] = "MODERATE"
        passed2, _, _ = evaluate_threshold(
            "decay_severity not in ['STRONG', 'MODERATE']", report,
        )
        assert not passed2

    def test_eq_bool(self) -> None:
        report = {"regime_diversified": True}
        passed, _, _ = evaluate_threshold(
            "regime_diversified == True", report,
        )
        assert passed

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ValueError, match="unparseable"):
            evaluate_threshold("just some random text", {})

    def test_type_mismatch_treated_as_missing(self) -> None:
        # actual is a string, threshold expects numeric — should be
        # treated as missing/malformed → ESCALATE upstream.
        report = {"sharpe": "not a number"}
        passed, missing, _ = evaluate_threshold("sharpe >= 0.5", report)
        assert not passed
        assert missing


# =============================================================================
# Rule parsing from REVIEW_RULES.md
# =============================================================================


class TestParseRules:
    def test_real_review_rules_md(self) -> None:
        """The committed REVIEW_RULES.md must produce parseable rules
        — that's the contract between the doctrine doc and the engine."""
        rules = parse_rules(Path("docs/research/REVIEW_RULES.md"))
        # Section A.1 through A.7, B.1 through B.3, plus C/D rules
        rule_ids = [r.rule_id for r in rules]
        # Must have at least the Section A statistical-reality rules
        for rid in ("A.1", "A.2", "A.3", "A.4", "A.5", "A.6", "A.7"):
            assert rid in rule_ids, f"missing rule {rid}"
        # And Section B
        for rid in ("B.1", "B.2", "B.3"):
            assert rid in rule_ids, f"missing rule {rid}"

    def test_threshold_expressions_extracted(self) -> None:
        """Section A/B rules must have THRESHOLD lines parsed."""
        rules = parse_rules(Path("docs/research/REVIEW_RULES.md"))
        a1 = next(r for r in rules if r.rule_id == "A.1")
        assert a1.threshold_expr is not None
        assert "sharpe" in a1.threshold_expr.lower()
        a2 = next(r for r in rules if r.rule_id == "A.2")
        assert a2.threshold_expr is not None
        assert ">" in a2.threshold_expr

    def test_section_c_rules_have_no_threshold(self) -> None:
        """Section C (code integrity) rules are agent-evidence-driven —
        no THRESHOLD: line, threshold_expr is None."""
        rules = parse_rules(Path("docs/research/REVIEW_RULES.md"))
        c1 = next((r for r in rules if r.rule_id == "C.1"), None)
        assert c1 is not None
        assert c1.threshold_expr is None

    def test_inline_rules_yaml(self, tmp_path: Path) -> None:
        """Test the parser against a tiny inline rules file."""
        rules_md = tmp_path / "rules.md"
        rules_md.write_text("""
### A.1 — Test rule

Description here.

THRESHOLD: `oos_metrics.sharpe >= 0.50`

### A.2 — Another

Description.

THRESHOLD: `n_trades > 30`

### C.1 — Code rule

No threshold here, agent-evidence only.
""")
        rules = parse_rules(rules_md)
        rule_ids = [r.rule_id for r in rules]
        assert rule_ids == ["A.1", "A.2", "C.1"]
        assert rules[0].threshold_expr == "oos_metrics.sharpe >= 0.50"
        assert rules[1].threshold_expr == "n_trades > 30"
        assert rules[2].threshold_expr is None


# =============================================================================
# Verdict outcomes
# =============================================================================


def _make_passing_report() -> dict[str, Any]:
    """Candidate report that passes every Section A/B threshold."""
    return {
        "oos_metrics": {
            "sharpe": 0.65,
            "n_trades": 45,
            "hit_rate": 0.55,
            "max_drawdown": -0.15,
            "profit_factor": 1.30,
        },
        "sharpe_ci_95": {"low": 0.20, "high": 1.10},
        "is_oos_sharpe_ratio": 1.4,
        "edge_concentration": 0.45,
        "regime_diversified": True,
        "decay_severity": "NO",
    }


def _make_rules() -> list[ParsedRule]:
    return parse_rules(Path("docs/research/REVIEW_RULES.md"))


class TestVerdictOutcomes:
    def test_promote_when_all_pass_and_both_agree(self) -> None:
        result = compute_verdict(
            candidate_report=_make_passing_report(),
            rules=_make_rules(),
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        assert result.verdict == Verdict.PROMOTE

    def test_reject_when_a1_fails(self) -> None:
        report = _make_passing_report()
        report["oos_metrics"]["sharpe"] = 0.3  # below A.1 threshold
        result = compute_verdict(
            candidate_report=report,
            rules=_make_rules(),
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        assert result.verdict == Verdict.REJECT
        # Even with both agents promoting, gate failure overrides
        assert "REVIEW_RULES gates failed" in result.reason

    def test_reject_when_ci_includes_zero(self) -> None:
        """The most important gate — A.2 sharpe_ci_95.low > 0."""
        report = _make_passing_report()
        report["sharpe_ci_95"]["low"] = -0.1
        result = compute_verdict(
            candidate_report=report,
            rules=_make_rules(),
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        assert result.verdict == Verdict.REJECT

    def test_escalate_when_metric_missing(self) -> None:
        report = _make_passing_report()
        del report["oos_metrics"]["sharpe"]  # required for A.1
        result = compute_verdict(
            candidate_report=report,
            rules=_make_rules(),
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        assert result.verdict == Verdict.ESCALATE
        assert "missing" in result.reason.lower()

    def test_escalate_when_open_questions(self) -> None:
        result = compute_verdict(
            candidate_report=_make_passing_report(),
            rules=_make_rules(),
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
            open_questions=["q-2-1: cost model basis unclear"],
        )
        assert result.verdict == Verdict.ESCALATE
        assert "smart-question" in result.reason.lower()

    def test_reject_when_both_agents_reject(self) -> None:
        """Even if all gates pass, both agents rejecting → REJECT
        (caution-default; agents found qualitative concerns)."""
        result = compute_verdict(
            candidate_report=_make_passing_report(),
            rules=_make_rules(),
            bull_position=Position.REJECT,
            bear_position=Position.REJECT,
        )
        assert result.verdict == Verdict.REJECT

    def test_reject_when_both_agents_abstain(self) -> None:
        # Abstain == reject in this caution-default semantics
        result = compute_verdict(
            candidate_report=_make_passing_report(),
            rules=_make_rules(),
            bull_position=Position.ABSTAIN,
            bear_position=Position.ABSTAIN,
        )
        assert result.verdict == Verdict.REJECT

    def test_escalate_when_mixed_positions(self) -> None:
        result = compute_verdict(
            candidate_report=_make_passing_report(),
            rules=_make_rules(),
            bull_position=Position.PROMOTE,
            bear_position=Position.REJECT,
        )
        assert result.verdict == Verdict.ESCALATE
        assert "mixed" in result.reason.lower()

    def test_evaluations_record_each_rule(self) -> None:
        result = compute_verdict(
            candidate_report=_make_passing_report(),
            rules=_make_rules(),
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        # All Section A rules should appear in evaluations
        evaluated_ids = [e.rule_id for e in result.rule_evaluations]
        for rid in ("A.1", "A.2", "A.3"):
            assert rid in evaluated_ids

    def test_passing_evaluation_carries_actual_value(self) -> None:
        result = compute_verdict(
            candidate_report=_make_passing_report(),
            rules=_make_rules(),
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        a1 = next(e for e in result.rule_evaluations if e.rule_id == "A.1")
        assert a1.passed
        assert a1.actual_value == 0.65


# =============================================================================
# Determinism
# =============================================================================


class TestDeterminism:
    def test_same_inputs_same_verdict(self) -> None:
        """Verdict engine is pure — same inputs always produce the same
        output. This is what makes it auditable."""
        rules = _make_rules()
        report = _make_passing_report()
        result1 = compute_verdict(
            candidate_report=report,
            rules=rules,
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        result2 = compute_verdict(
            candidate_report=report,
            rules=rules,
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        assert result1.verdict == result2.verdict
        assert result1.reason == result2.reason
