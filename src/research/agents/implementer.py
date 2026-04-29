"""Implementer agent (CL-2pww) — converts a pre-registered hypothesis
into a runnable strategy file plus a prediction the debate reviewers
will compare against the actual backtest result.

Pipeline owned by this agent:

  1. Read hypothesis brief from ``docs/research/hypotheses/{slug}.md``.
  2. Run the LLM with the implementer system prompt + the brief as
     context.
  3. Extract the Python code block, the prediction block, and the
     position keyword (IMPLEMENTED / REJECTED) from the LLM's response.
  4. If REJECTED — return immediately; the orchestrator routes the
     refusal to the operator.
  5. If IMPLEMENTED — write the strategy code to
     ``src/strategies/_experimental/{slug}.py`` and run a syntax gate
     (``compile()``) on it. Failure here = REJECTED with a precise
     reason; we don't ship broken code.
  6. Optionally run a backtest via a caller-supplied callback; the
     agent itself has no DB or backtest knowledge.
  7. Compile a candidate report at ``reports/candidates/{slug}.json``
     containing: code path, prediction, backtest metrics (when run),
     gate pass/fail, provenance (model, cost, elapsed).

The agent is deliberately scoped: it owns codegen + persistence + the
syntax gate + report assembly. It does NOT own backtest execution —
that's a callback the caller supplies. Keeps the agent unit-testable
without a Postgres / WalkForwardRunner setup, and makes the same agent
usable from the autonomous loop, from a CLI, or from a notebook.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.research.agents.base import Agent, AgentResponse

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


class ImplementerStatus(StrEnum):
    IMPLEMENTED = "IMPLEMENTED"
    REJECTED = "REJECTED"


@dataclass
class ImplementerResult:
    """Output of one implement() call. Fully auditable."""

    status: ImplementerStatus
    strategy_slug: str
    response: AgentResponse
    raw_text: str
    code_path: Path | None = None
    report_path: Path | None = None
    reason: str = ""                               # set when REJECTED
    candidate_report: dict[str, Any] = field(default_factory=dict)


# Caller-supplied backtest hook. Takes the strategy code path, returns a
# dict that will be merged into the candidate report under
# ``backtest_metrics``. Raise on failure — the implementer turns the
# exception into a REJECTED with the reason.
BacktestRunner = Callable[[Path], dict[str, Any]]


# --------------------------------------------------------------------------- #
# Parsing — extract code + prediction + position from the LLM output
# --------------------------------------------------------------------------- #


# A fenced ```python ... ``` block. We take the LARGEST one — agents
# sometimes include short illustrative snippets before the main code.
_PYTHON_FENCE_PATTERN = re.compile(
    r"```python\s*\n(.*?)\n```", re.DOTALL,
)


# Position keyword anywhere in the text. Tolerant of bold formatting
# and the explicit FINAL_POSITION: prefix (matching the prompt).
_POSITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\*\*FINAL_POSITION\*\*\s*:\s*(IMPLEMENTED|REJECTED)",
        re.IGNORECASE,
    ),
    re.compile(
        r"FINAL[_\s]POSITION\s*:\s*\*?\*?(IMPLEMENTED|REJECTED)\*?\*?",
        re.IGNORECASE,
    ),
    re.compile(r"\*\*(IMPLEMENTED|REJECTED)\*\*", re.IGNORECASE),
)


def parse_position(text: str) -> ImplementerStatus:
    """Extract the implementer's declared position. The LAST match wins
    so an earlier exploratory mention doesn't override the final
    declaration. Defaults to REJECTED if no keyword is found — caution-
    default; we never ship code without an explicit IMPLEMENTED."""
    for pattern in _POSITION_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return ImplementerStatus(matches[-1].group(1).upper())
    logger.warning(
        "No position keyword found in implementer output — defaulting to REJECTED",
    )
    return ImplementerStatus.REJECTED


def extract_code(text: str) -> str | None:
    """Extract the largest ```python block from the LLM output. Returns
    None if no python-fenced block is present — the caller treats that
    as a malformed response and REJECTS the implementation."""
    matches = list(_PYTHON_FENCE_PATTERN.finditer(text))
    if not matches:
        return None
    largest = max(matches, key=lambda m: len(m.group(1)))
    return largest.group(1)


# Match the `## Prediction` section between this heading and the next
# `##`. Forgiving — agents may use bullet points or YAML-style lines.
_PREDICTION_SECTION = re.compile(
    r"##\s+Prediction\s*\n(.+?)(?=\n##\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)
# Match `## Data sources actually consumed`.
_DATA_SOURCES_SECTION = re.compile(
    r"##\s+Data sources actually consumed\s*\n(.+?)(?=\n##\s|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def extract_prediction(text: str) -> dict[str, str]:
    """Extract the prediction section as a name → raw-line map. We don't
    parse value types here — the verdict engine reads the candidate
    report's structured fields, which are written by the caller-supplied
    backtest_runner. This block is agent-stated *predicted* metrics and
    we preserve them verbatim for the auditor."""
    m = _PREDICTION_SECTION.search(text)
    if not m:
        return {}
    section = m.group(1).strip()
    out: dict[str, str] = {}
    # Lines like "- `predicted_sharpe_range`: [0.5, 0.9]"
    for line in section.splitlines():
        line = line.strip().lstrip("-").lstrip("*").strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().strip("`").strip()
        val = val.strip()
        if key:
            out[key] = val
    return out


def extract_data_sources(text: str) -> list[str]:
    """Extract the data-sources-actually-consumed list. Same forgiving
    parser — one item per non-empty line stripped of bullet prefixes."""
    m = _DATA_SOURCES_SECTION.search(text)
    if not m:
        return []
    out: list[str] = []
    for line in m.group(1).splitlines():
        line = line.strip().lstrip("-").lstrip("*").strip().strip("`")
        if line and not line.startswith("##"):
            out.append(line)
    return out


# --------------------------------------------------------------------------- #
# Syntax gate
# --------------------------------------------------------------------------- #


def syntax_check(code: str, slug: str) -> tuple[bool, str]:
    """Compile-check the generated code. Returns (passed, detail)."""
    filename = f"<implementer:{slug}>"
    try:
        compile(code, filename, "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"
    except ValueError as exc:
        # Source contains null bytes etc.
        return False, f"ValueError compiling source: {exc}"
    return True, "compile() ok"


# --------------------------------------------------------------------------- #
# Implementer agent
# --------------------------------------------------------------------------- #


class Implementer(Agent):
    """Codegen agent. Inherits Agent's ``run()`` for LLM dispatch; adds
    the implement() entry point that wires hypothesis-loading, code
    extraction, syntax gate, optional backtest, and report writing.

    Construct via ``Implementer.from_config(name="implementer",
    research_config=cfg)`` — the YAML wires the system prompt
    (``configs/research_prompts/implementer.md``) and provider.
    """

    def implement(
        self,
        hypothesis_path: Path | str,
        strategy_slug: str,
        backtest_runner: BacktestRunner | None = None,
        code_dir: Path | str = "src/strategies/_experimental",
        report_dir: Path | str = "reports/candidates",
    ) -> ImplementerResult:
        """Run the full pipeline against one hypothesis brief.

        ``backtest_runner`` is optional — if None, the implementer stops
        at the syntax gate (still produces an IMPLEMENTED result with
        the code path and prediction, just no backtest_metrics in the
        report). Useful for unit tests + for the operator-approval
        gate (CL-0hr3) where we want to surface the implementation
        before consuming the backtest budget.
        """
        hypothesis_path = Path(hypothesis_path)
        if not hypothesis_path.exists():
            msg = f"hypothesis brief not found at {hypothesis_path}"
            raise FileNotFoundError(msg)
        brief_text = hypothesis_path.read_text()

        # 1) Run the LLM
        resp = self.run(
            user_prompt=(
                f"Implement the strategy described in the hypothesis "
                f"brief. The strategy slug is {strategy_slug!r}. Produce "
                f"the markdown document specified by your system prompt."
            ),
            context_files={
                f"hypothesis_brief:{hypothesis_path.name}": brief_text,
            },
        )
        # 2) Parse position
        position = parse_position(resp.text)
        if position == ImplementerStatus.REJECTED:
            return ImplementerResult(
                status=ImplementerStatus.REJECTED,
                strategy_slug=strategy_slug,
                response=resp,
                raw_text=resp.text,
                reason=self._extract_rejection_reason(resp.text),
            )

        # 3) Extract code
        code = extract_code(resp.text)
        if code is None:
            return ImplementerResult(
                status=ImplementerStatus.REJECTED,
                strategy_slug=strategy_slug,
                response=resp,
                raw_text=resp.text,
                reason=(
                    "implementer claimed IMPLEMENTED but no ```python code "
                    "block was present in the response"
                ),
            )

        # 4) Persist code
        code_path = Path(code_dir) / f"{strategy_slug}.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code)

        # 5) Syntax gate
        ok, detail = syntax_check(code, strategy_slug)
        if not ok:
            return ImplementerResult(
                status=ImplementerStatus.REJECTED,
                strategy_slug=strategy_slug,
                response=resp,
                raw_text=resp.text,
                code_path=code_path,
                reason=f"syntax gate failed: {detail}",
            )

        # 6) Optional backtest
        backtest_metrics: dict[str, Any] = {}
        if backtest_runner is not None:
            try:
                backtest_metrics = backtest_runner(code_path)
            except Exception as exc:
                logger.exception(
                    "backtest_runner failed for %s", strategy_slug,
                )
                return ImplementerResult(
                    status=ImplementerStatus.REJECTED,
                    strategy_slug=strategy_slug,
                    response=resp,
                    raw_text=resp.text,
                    code_path=code_path,
                    reason=f"backtest failed: {type(exc).__name__}: {exc}",
                )

        # 7) Compile + persist report
        candidate_report = self._build_report(
            slug=strategy_slug,
            code_path=code_path,
            hypothesis_path=hypothesis_path,
            response=resp,
            raw_text=resp.text,
            backtest_metrics=backtest_metrics,
        )
        report_path = Path(report_dir) / f"{strategy_slug}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(candidate_report, indent=2, default=str))

        return ImplementerResult(
            status=ImplementerStatus.IMPLEMENTED,
            strategy_slug=strategy_slug,
            response=resp,
            raw_text=resp.text,
            code_path=code_path,
            report_path=report_path,
            candidate_report=candidate_report,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_rejection_reason(text: str) -> str:
        """Pull the rationale paragraph that follows REJECTED. Best-
        effort — the prompt asks for one paragraph after the keyword."""
        m = re.search(
            r"\*\*FINAL_POSITION\*\*\s*:\s*REJECTED\s*\n+(.+?)(?:\n\n|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return "(no rationale extracted from REJECTED response)"

    def _build_report(
        self,
        slug: str,
        code_path: Path,
        hypothesis_path: Path,
        response: AgentResponse,
        raw_text: str,
        backtest_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        # Backtest metrics are merged at top level so the verdict
        # engine's threshold paths (e.g. ``oos_metrics.sharpe``) resolve
        # against the candidate report directly. The nested
        # ``backtest_metrics`` copy is kept for audit / backwards-
        # compatible consumers.
        report: dict[str, Any] = {
            "schema_version": 1,
            "strategy_slug": slug,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "code_path": str(code_path),
            "hypothesis_path": str(hypothesis_path),
            "prediction": extract_prediction(raw_text),
            "data_sources_consumed": extract_data_sources(raw_text),
            "backtest_metrics": backtest_metrics,
            "provenance": {
                "agent_name": response.agent_name,
                "model": response.model,
                "provider": response.provider,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "usd_cost": response.usd_cost,
                "elapsed_sec": response.elapsed_sec,
            },
            "gates": {
                "syntax": "passed",
                "backtest": "ran" if backtest_metrics else "skipped",
            },
        }
        # Spread the backtest_metrics keys at the top level — but never
        # let them clobber the report's own keys (slug, gates, etc.).
        reserved = set(report.keys())
        for k, v in backtest_metrics.items():
            if k not in reserved:
                report[k] = v
        return report
