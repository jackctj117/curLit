"""Verdict engine (CL-ath2) — deterministic mapping of agent evidence
to PROMOTE / REJECT / ESCALATE.

This is the safety net. Two LLMs from the same family agree by default;
without a deterministic gate, a debate can rubber-stamp a candidate
that fails a quantitative threshold. This engine is pure Python — no
LLM call, no inference, just rule lookup against the candidate report
metrics + the agents' final positions.

The rules are parsed at runtime from ``docs/research/REVIEW_RULES.md``
so the operator can amend them via PR review without touching code.
Each Section A/B rule has a ``THRESHOLD: <expr>`` line that this engine
parses into a small AST and evaluates against the candidate report.

Section C and D rules (code integrity, operational fitness) require
agent-cited evidence to fail — they pass by default unless an agent
explicitly cites a violation. This is intentional: the verdict engine
can't grep code or run mypy itself; it trusts the agents to surface
those violations during the debate, and the rebuttal round forces
engagement with each cited claim.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.research.agents.reviewer import Position

logger = logging.getLogger(__name__)


class Verdict(StrEnum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


@dataclass
class RuleEvaluation:
    """One rule's pass/fail/missing decision."""

    rule_id: str         # 'A.1', 'B.2', etc.
    description: str     # short text from the rule heading
    threshold_expr: str  # 'oos_metrics.sharpe >= 0.50'
    passed: bool         # True if rule passed (or no evidence to fail it)
    missing: bool = False  # True if a required metric was absent
    actual_value: Any = None
    detail: str = ""


@dataclass
class VerdictResult:
    """Output of compute_verdict — fully auditable."""

    verdict: Verdict
    reason: str
    rule_evaluations: list[RuleEvaluation] = field(default_factory=list)
    bull_position: Position | None = None
    bear_position: Position | None = None
    open_questions: list[str] = field(default_factory=list)


# =============================================================================
# Rule parsing
# =============================================================================


# Match a rule heading like "### A.1 — Annualized OOS Sharpe" — the
# rule ID is captured for cross-referencing in agent citations.
_RULE_HEADING_PATTERN = re.compile(r"^###\s+([A-D]\.\d+)\s*[—\-:]\s*(.+)$", re.MULTILINE)

# Match a THRESHOLD: line. The expression is intentionally a tiny
# subset (field path + comparator + literal) — we eval it ourselves
# rather than risk running arbitrary user-supplied Python.
_THRESHOLD_PATTERN = re.compile(
    r"^THRESHOLD:\s*`?(.+?)`?\s*$",
    re.MULTILINE,
)


@dataclass
class ParsedRule:
    """One rule extracted from REVIEW_RULES.md."""

    rule_id: str
    description: str
    threshold_expr: str | None  # None for code-integrity / op-fitness rules


def parse_rules(rules_md_path: Path | str) -> list[ParsedRule]:
    """Parse REVIEW_RULES.md into a list of rules. Sections A and B have
    THRESHOLD: lines we evaluate; C and D rules are agent-evidence-driven."""
    text = Path(rules_md_path).read_text()
    rules: list[ParsedRule] = []

    # Walk the document by rule heading, then look ahead a few lines for
    # an immediately-following THRESHOLD line (allow blank lines + a
    # description paragraph between heading and threshold).
    headings = list(_RULE_HEADING_PATTERN.finditer(text))
    for i, m in enumerate(headings):
        rule_id = m.group(1)
        description = m.group(2).strip()
        # Slice from this heading to the next, search for the THRESHOLD.
        section_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[m.end():section_end]
        thr = _THRESHOLD_PATTERN.search(section)
        threshold_expr = thr.group(1).strip() if thr else None
        rules.append(ParsedRule(
            rule_id=rule_id,
            description=description,
            threshold_expr=threshold_expr,
        ))
    return rules


# =============================================================================
# Threshold evaluation — tiny safe expression evaluator
# =============================================================================


_COMPARATOR_PATTERN = re.compile(
    r"^\s*([\w.\[\]'\"]+)\s*(>=|<=|>|<|==|!=|not in|in)\s*(.+?)\s*$",
)


def _resolve_path(report: dict[str, Any], path: str) -> tuple[Any, bool]:
    """Walk a dotted path like 'oos_metrics.sharpe' through the report.

    Returns (value, found). found=False if any segment is missing —
    the verdict engine then marks the rule as ESCALATE (missing metric).
    """
    cur: Any = report
    for segment in path.split("."):
        if isinstance(cur, dict) and segment in cur:
            cur = cur[segment]
        else:
            return None, False
    return cur, True


def _parse_literal(s: str) -> Any:
    """Parse a literal: number, list of strings, True/False, None.

    Tiny custom parser — we don't want to eval() arbitrary code."""
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("none", "null"):
        return None
    # Numeric (int or float, possibly negative)
    try:
        if "." in s or "e" in s or "E" in s:
            return float(s)
        return int(s)
    except ValueError:
        pass
    # List of strings: ('A', 'B') or ['A', 'B']
    if (s.startswith("(") and s.endswith(")")) or (
        s.startswith("[") and s.endswith("]")
    ):
        inner = s[1:-1].strip()
        if not inner:
            return []
        # Split on commas, strip each, drop quotes
        items: list[str] = []
        for raw in inner.split(","):
            stripped = raw.strip().strip("'\"")
            items.append(stripped)
        return items
    # Fallback: bare string (unlikely in our threshold language)
    return s.strip("'\"")


def evaluate_threshold(
    expr: str, report: dict[str, Any],
) -> tuple[bool, bool, Any]:
    """Evaluate a threshold expression against the candidate report.

    Returns (passed, missing, actual_value). If the path doesn't exist
    in the report, returns (False, True, None) so the verdict engine
    knows to ESCALATE rather than assume PASS or FAIL.

    Raises ValueError for malformed expressions — fail loud at debate
    time, not silently in some downstream consumer.
    """
    m = _COMPARATOR_PATTERN.match(expr)
    if not m:
        msg = f"unparseable threshold expression: {expr!r}"
        raise ValueError(msg)
    path = m.group(1)
    op = m.group(2)
    rhs_literal = _parse_literal(m.group(3))

    actual, found = _resolve_path(report, path)
    if not found:
        return False, True, None

    try:
        if op == ">=":
            passed = actual >= rhs_literal
        elif op == "<=":
            passed = actual <= rhs_literal
        elif op == ">":
            passed = actual > rhs_literal
        elif op == "<":
            passed = actual < rhs_literal
        elif op == "==":
            passed = actual == rhs_literal
        elif op == "!=":
            passed = actual != rhs_literal
        elif op == "in":
            passed = actual in rhs_literal
        elif op == "not in":
            passed = actual not in rhs_literal
        else:
            msg = f"unsupported comparator: {op}"
            raise ValueError(msg)
    except TypeError:
        # actual was wrong type for comparator (e.g. comparing str to
        # number) — treat as missing/malformed metric, ESCALATE
        return False, True, actual

    return bool(passed), False, actual


# =============================================================================
# Main verdict computation
# =============================================================================


def compute_verdict(
    candidate_report: dict[str, Any],
    rules: list[ParsedRule],
    bull_position: Position,
    bear_position: Position,
    open_questions: list[str] | None = None,
) -> VerdictResult:
    """Apply the rules to the candidate report + the agent positions and
    produce a verdict. See REVIEW_RULES.md Section E for the mapping.

    The function is order-independent and stateless — given the same
    inputs it always produces the same output.
    """
    open_questions = open_questions or []

    # Evaluate each rule that has a parseable threshold expression.
    # Rules without one (Section C and D) default to passed=True; they
    # only fail if an agent's debate output cited a violation, which
    # the verdict engine sees via a separate evidence channel (TBD —
    # this v1 trusts the agents' final positions for those rules).
    evaluations: list[RuleEvaluation] = []
    any_failed = False
    any_missing = False
    for rule in rules:
        if rule.threshold_expr is None:
            evaluations.append(RuleEvaluation(
                rule_id=rule.rule_id,
                description=rule.description,
                threshold_expr="(agent-evidence-driven)",
                passed=True,
                detail="no parseable threshold; agent positions decide",
            ))
            continue
        try:
            passed, missing, actual = evaluate_threshold(
                rule.threshold_expr, candidate_report,
            )
        except ValueError as exc:
            logger.exception("rule %s threshold malformed", rule.rule_id)
            evaluations.append(RuleEvaluation(
                rule_id=rule.rule_id,
                description=rule.description,
                threshold_expr=rule.threshold_expr,
                passed=False,
                missing=True,
                detail=f"malformed threshold: {exc}",
            ))
            any_missing = True
            continue
        evaluations.append(RuleEvaluation(
            rule_id=rule.rule_id,
            description=rule.description,
            threshold_expr=rule.threshold_expr,
            passed=passed,
            missing=missing,
            actual_value=actual,
            detail=(
                "missing metric at path"
                if missing else f"actual={actual!r}"
            ),
        ))
        if missing:
            any_missing = True
        elif not passed:
            any_failed = True

    # Verdict mapping per REVIEW_RULES.md Section E:
    #   - any rule fails on a present metric → REJECT
    #   - any required metric missing → ESCALATE
    #   - any unresolved smart-question → ESCALATE
    #   - both agents PROMOTE + all rules pass → PROMOTE
    #   - both agents REJECT/ABSTAIN → REJECT
    #   - mixed agent positions → ESCALATE
    if any_failed:
        return VerdictResult(
            verdict=Verdict.REJECT,
            reason="One or more REVIEW_RULES gates failed against the metrics",
            rule_evaluations=evaluations,
            bull_position=bull_position,
            bear_position=bear_position,
            open_questions=open_questions,
        )
    if any_missing:
        return VerdictResult(
            verdict=Verdict.ESCALATE,
            reason="Required metric(s) missing or malformed in candidate report",
            rule_evaluations=evaluations,
            bull_position=bull_position,
            bear_position=bear_position,
            open_questions=open_questions,
        )
    if open_questions:
        return VerdictResult(
            verdict=Verdict.ESCALATE,
            reason=f"{len(open_questions)} unresolved smart-question(s)",
            rule_evaluations=evaluations,
            bull_position=bull_position,
            bear_position=bear_position,
            open_questions=open_questions,
        )

    # All rules pass + no missing metrics + no open questions → agent
    # positions decide.
    if bull_position == Position.PROMOTE and bear_position == Position.PROMOTE:
        return VerdictResult(
            verdict=Verdict.PROMOTE,
            reason="All gates pass; both reviewers PROMOTE",
            rule_evaluations=evaluations,
            bull_position=bull_position,
            bear_position=bear_position,
        )
    rejecting = {Position.REJECT, Position.ABSTAIN}
    if bull_position in rejecting and bear_position in rejecting:
        return VerdictResult(
            verdict=Verdict.REJECT,
            reason=(
                "Both reviewers REJECT or ABSTAIN despite gates passing — "
                "agents found qualitative concerns the rules don't capture"
            ),
            rule_evaluations=evaluations,
            bull_position=bull_position,
            bear_position=bear_position,
        )
    # Mixed positions
    return VerdictResult(
        verdict=Verdict.ESCALATE,
        reason=f"Mixed agent positions: bull={bull_position}, bear={bear_position}",
        rule_evaluations=evaluations,
        bull_position=bull_position,
        bear_position=bear_position,
    )
