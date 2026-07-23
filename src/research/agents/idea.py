"""Idea Generator agent (CL-n0m8) — converts paper extracts into pre-
registered hypothesis briefs that the Implementer (CL-2pww) consumes.

Pipeline owned by this agent:

  1. Read a paper extract from ``data/research/extracts/{hash}.md``
     (produced by the paper ingester, CL-2klj).
  2. Optionally read the existing hypothesis backlog
     (``docs/research/hypotheses/*.md``) so the LLM can flag duplicates.
  3. Run the LLM with the idea-generator system prompt + the extract
     as context.
  4. Parse the position keyword (PROPOSED / DECLINED) out of the
     response; default DECLINED if absent.
  5. If PROPOSED — validate the brief has all required sections, then
     persist to ``docs/research/hypotheses/{slug}.md``.
  6. If DECLINED — return the result without writing; the orchestrator
     records the decline in the run log but doesn't queue work.

Determinism: the agent runs at temperature=0 (configured in
research_agents.yaml). Same extract → same brief. The acceptance
criterion explicitly demands this, and the prompt instructs DECLINE
rather than fabricate when the extract under-specifies a hypothesis.

Designed for both autonomous use (research_loop) and CLI use (a future
operator-facing script). The validation logic + slug helper sit at
module scope so callers can use them independently of the agent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from src.research.agents.base import Agent, AgentResponse

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------- #


class IdeaStatus(StrEnum):
    PROPOSED = "PROPOSED"
    DECLINED = "DECLINED"


@dataclass
class IdeaResult:
    """Output of one ideate() call. Fully auditable."""

    status: IdeaStatus
    strategy_slug: str
    extract_path: Path
    response: AgentResponse
    raw_text: str
    hypothesis_path: Path | None = None
    reason: str = ""  # non-empty when DECLINED or validation failed
    validation_issues: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- #
# Parsing — position + section validation
# --------------------------------------------------------------------- #


_POSITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\*\*FINAL_POSITION\*\*\s*:\s*(PROPOSED|DECLINED)",
        re.IGNORECASE,
    ),
    re.compile(
        r"FINAL[_\s]POSITION\s*:\s*\*?\*?(PROPOSED|DECLINED)\*?\*?",
        re.IGNORECASE,
    ),
    re.compile(r"\*\*(PROPOSED|DECLINED)\*\*", re.IGNORECASE),
)


def parse_position(text: str) -> IdeaStatus:
    """Extract PROPOSED / DECLINED from the brief. Last match wins so an
    earlier exploratory mention doesn't override the final declaration.
    Defaults DECLINED — the safe default; never proposes without an
    explicit declaration."""
    for pattern in _POSITION_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return IdeaStatus(matches[-1].group(1).upper())
    logger.warning(
        "No position keyword found in idea-agent output — defaulting to DECLINED",
    )
    return IdeaStatus.DECLINED


# Required section headings the brief MUST contain. Order doesn't
# matter — the prompt asks for a specific order but the LLM may emit
# slightly different whitespace; we just check presence.
_REQUIRED_SECTIONS: tuple[str, ...] = (
    "Source extract",
    "Change to baseline",
    "Prediction",
    "Abandon condition",
    "Data requirements",
    "References",
)


# Predicted-metrics fields the Prediction section must list.
_REQUIRED_PREDICTION_FIELDS: tuple[str, ...] = (
    "predicted_sharpe_range",
    "predicted_hit_rate_range",
    "expected_n_trades_per_year",
    "regime_dependence",
    "time_to_signal",
)


def validate_brief(text: str, extract_path: Path) -> list[str]:
    """Return a list of validation issues; empty list = brief is valid.

    Checks:
      * each required ``## {section}`` heading is present
      * the Prediction section lists each required predicted_* field
      * the References section cites back to the source extract path
    """
    issues: list[str] = []

    for section in _REQUIRED_SECTIONS:
        if not re.search(rf"##\s+{re.escape(section)}\b", text, re.IGNORECASE):
            issues.append(f"missing required section: ## {section}")

    # Extract the Prediction section content for sub-field checking.
    pred_match = re.search(
        r"##\s+Prediction\s*\n(.+?)(?=\n##\s|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if pred_match:
        pred_block = pred_match.group(1)
        for field_name in _REQUIRED_PREDICTION_FIELDS:
            if field_name not in pred_block:
                issues.append(
                    f"prediction section missing required field: {field_name}",
                )

    # References section must cite the source extract path. We accept
    # either the full path or the bare filename (hash.md).
    ref_match = re.search(
        r"##\s+References\s*\n(.+?)(?=\n##\s|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if ref_match:
        ref_block = ref_match.group(1)
        if str(extract_path) not in ref_block and extract_path.name not in ref_block:
            issues.append(
                f"References section does not cite the source extract path ({extract_path})",
            )

    return issues


# --------------------------------------------------------------------- #
# Slug helper
# --------------------------------------------------------------------- #


_SLUG_TRANSLATE = re.compile(r"[^a-z0-9]+")


def default_slug(extract_path: Path, fallback_prefix: str = "idea") -> str:
    """Derive a kebab-case slug from the extract's title (read from the
    first ``# Title`` line). Falls back to the extract's hash filename
    if the title can't be read."""
    try:
        text = extract_path.read_text()
    except OSError:
        return f"{fallback_prefix}-{extract_path.stem}"

    # First H1 line in the extract is the paper title.
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if not m:
        return f"{fallback_prefix}-{extract_path.stem}"
    raw = m.group(1).strip().lower()
    slug = _SLUG_TRANSLATE.sub("-", raw).strip("-")
    # Cap length so generated paths don't blow up.
    if len(slug) > 64:
        slug = slug[:64].rstrip("-")
    return slug or f"{fallback_prefix}-{extract_path.stem}"


def _extract_decline_reason(text: str) -> str:
    """Pull the rationale paragraph that follows DECLINED. Best-effort."""
    m = re.search(
        r"\*\*FINAL_POSITION\*\*\s*:\s*DECLINED\s*\n+(.+?)(?:\n\n|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return "(no rationale extracted from DECLINED response)"


# --------------------------------------------------------------------- #
# Idea agent
# --------------------------------------------------------------------- #


class IdeaGenerator(Agent):
    """Reads a paper extract, emits a pre-registered hypothesis brief.

    Construct via ``IdeaGenerator.from_config(name="idea_generator",
    research_config=cfg)`` — the YAML wires the system prompt + provider
    (DeepSeek by default per the agent declaration).
    """

    def ideate(
        self,
        extract_path: Path | str,
        strategy_slug: str | None = None,
        backlog_summary: str | None = None,
        hypothesis_dir: Path | str = "docs/research/hypotheses",
    ) -> IdeaResult:
        """Run the agent against one extract.

        ``strategy_slug`` defaults to the auto-generated slug derived
        from the extract title. ``backlog_summary`` is optional context
        the caller can pass so the LLM can flag duplicates against
        existing hypotheses; pass None to skip dedup help.
        """
        extract_path = Path(extract_path)
        if not extract_path.exists():
            msg = f"extract not found at {extract_path}"
            raise FileNotFoundError(msg)
        extract_text = extract_path.read_text()

        slug = strategy_slug or default_slug(extract_path)

        # 1) Run LLM
        context_files: dict[str, str] = {
            f"paper_extract:{extract_path.name}": extract_text,
        }
        if backlog_summary is not None:
            context_files["hypothesis_backlog_summary"] = backlog_summary

        resp = self.run(
            user_prompt=(
                f"Generate a pre-registered hypothesis brief from the "
                f"paper extract below. The strategy slug is {slug!r}. "
                f"Produce the markdown document specified by your system "
                f"prompt — every section heading must be present and "
                f"every Prediction field listed."
            ),
            context_files=context_files,
        )

        # 2) Parse position
        status = parse_position(resp.text)
        if status == IdeaStatus.DECLINED:
            return IdeaResult(
                status=IdeaStatus.DECLINED,
                strategy_slug=slug,
                extract_path=extract_path,
                response=resp,
                raw_text=resp.text,
                reason=_extract_decline_reason(resp.text),
            )

        # 3) Validate the brief shape
        issues = validate_brief(resp.text, extract_path)
        if issues:
            return IdeaResult(
                status=IdeaStatus.DECLINED,
                strategy_slug=slug,
                extract_path=extract_path,
                response=resp,
                raw_text=resp.text,
                reason="brief failed structural validation",
                validation_issues=issues,
            )

        # 4) Persist
        out_dir = Path(hypothesis_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}.md"
        out_path.write_text(resp.text)

        return IdeaResult(
            status=IdeaStatus.PROPOSED,
            strategy_slug=slug,
            extract_path=extract_path,
            response=resp,
            raw_text=resp.text,
            hypothesis_path=out_path,
        )
