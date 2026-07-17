"""Dry-run scaffolding for the research loop.

A ``--dry-run`` invocation of ``scripts/research_loop.py`` walks every
phase of the pipeline (ingest → idea → GATE 1 → implementer → debate →
verdict → GATE 2 → registrar) using stubbed LLM calls, stubbed HTTP,
and stubbed backtest data. No real API calls; no real network; no
real money risk; nothing pushed.

Useful as a preflight before CL-77rq (the real end-to-end smoke that
DOES burn real tokens). If a wiring bug exists — a config typo, a
prompt-fingerprint mismatch, a renamed function — it'll surface here
without spending API budget.

This module exposes three primitives that the CLI assembles:

  * ``DryRunDriver`` — Driver subclass that routes by system-prompt
    fingerprint to canned responses for paper_extractor / idea_generator
    / implementer / bull_reviewer / bear_reviewer.
  * ``stub_http_get(url)`` — returns a single canned arXiv Atom feed
    with one synthetic paper, regardless of URL.
  * ``stub_backtest_runner(code_path)`` — returns metrics that all
    pass the threshold rules so the dry-run produces a PROMOTE
    verdict and exercises GATE 2 + (skip-git) registrar paths.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.research.llm.client import (
    Driver,
    LLMResponse,
    register_driver,
)

# ---------------------------------------------------------------------- #
# Canned LLM responses
# ---------------------------------------------------------------------- #


_PAPER_EXTRACT = (
    "## Methodology\n"
    "Dry-run stub extract.\n\n"
    "## Findings\nStub.\n\n"
    "## FX trading applicability\nDirect.\n\n"
    "## Data sources cited\n- (none specified in abstract)\n\n"
    "## Key citations\n- (none in abstract)\n"
)

_IDEA_BRIEF_TEMPLATE = (
    "# Hypothesis: dry-run regime carry\n\n"
    "## Source extract\n"
    "- **path**: `{extract_path}`\n"
    "- **paper title**: dry-run stub paper\n"
    "- **paper hash**: `dry-run`\n\n"
    "## Change to baseline\n"
    "Replaces existing carry with regime gate.\n\n"
    "## Prediction\n"
    "- `predicted_sharpe_range`: [0.4, 0.8]\n"
    "- `predicted_hit_rate_range`: [0.55, 0.62]\n"
    "- `expected_n_trades_per_year`: 60\n"
    "- `regime_dependence`: regime-agnostic\n"
    "- `time_to_signal`: daily\n\n"
    "## Abandon condition\n"
    "- OOS Sharpe < 0 over rolling 6m\n\n"
    "## Data requirements\n"
    "- FRED IRLTLT01DEM156N\n\n"
    "## References\n"
    "- {extract_path}\n\n"
    "**FINAL_POSITION**: PROPOSED\n"
)

_IMPLEMENTER_OUTPUT = (
    "# IMPLEMENTATION\n\n"
    "## Strategy code\n"
    "```python\n"
    "class DryRunStrategy:\n"
    "    id = 'dry-run-stub'\n"
    "    symbols = ['EURUSD']\n"
    "    def fit(self, train_data):\n"
    "        self._mean = 0.0\n"
    "    def generate_signals(self, data):\n"
    "        return data * 0\n"
    "```\n\n"
    "## Prediction\n- `predicted_sharpe_range`: [0.4, 0.8]\n\n"
    "## Data sources actually consumed\n- FRED IRLTLT01DEM156N\n\n"
    "## Validation notes\nDry-run stub.\n\n"
    "**FINAL_POSITION**: IMPLEMENTED\n"
)

_BULL_REVIEW = (
    "# PROMOTE_CASE\n\nDry-run stub.\n\n**FINAL_POSITION**: PROMOTE\n"
)

_BEAR_REVIEW = (
    "# REJECT_CASE\n\nDry-run stub.\n\n**FINAL_POSITION**: PROMOTE\n"
)


# Maps a fingerprint substring to its canned-response handler. Each
# prompt file begins with ``# {Agent Name} — System Prompt`` on the
# first line, so the H1 prefix is a uniquely-identifying fingerprint
# that isn't shadowed by cross-references in the body (e.g. the idea-
# generator prompt mentions "Implementer" — without the H1 prefix
# match, the implementer's fingerprint would win first).
#
# Some agents (the idea generator) need to reference fields from the
# user-supplied context (the extract path) so the validator passes.
# Handlers receive (sys_text, user_text) and return the canned text.


def _idea_brief_handler(_sys: str, user: str) -> str:
    """Pull the extract path out of the user prompt's ``<context
    label='paper_extract:NAME'>`` block. The validator requires the
    References section to cite the actual extract path or filename;
    splicing it in here makes the dry-run idea-agent output pass."""
    m = re.search(r"paper_extract:([^']+)", user)
    extract_path = (
        f"data/research/extracts/{m.group(1)}" if m else "data/research/extracts/abc.md"
    )
    return _IDEA_BRIEF_TEMPLATE.format(extract_path=extract_path)


def _static(text: str) -> Callable[[str, str], str]:
    def handler(_sys: str, _user: str) -> str:
        return text
    return handler


_RESPONSE_TABLE: list[tuple[str, Callable[[str, str], str]]] = [
    ("# Bear Reviewer", _static(_BEAR_REVIEW)),
    ("# Bull Reviewer", _static(_BULL_REVIEW)),
    ("# Implementer", _static(_IMPLEMENTER_OUTPUT)),
    ("# Idea Generator", _idea_brief_handler),
    ("# Paper Extractor", _static(_PAPER_EXTRACT)),
]


# ---------------------------------------------------------------------- #
# DryRunDriver
# ---------------------------------------------------------------------- #


class DryRunDriver(Driver):
    """Returns canned responses based on system-prompt fingerprints.
    Registered as a provider via ``install_dry_run_driver()`` below;
    once installed, every agent's ``provider`` field can resolve to
    this driver and skip real API calls."""

    name = "dry-run"

    def __init__(self, api_key: str = "fake") -> None:
        super().__init__(api_key)

    def complete(
        self,
        messages: Any,
        model: str,
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResponse:
        sys_text = ""
        user_text = ""
        for m in messages:
            if m.role == "system":
                sys_text = m.content
            elif m.role == "user":
                user_text = m.content
        text = "(dry-run: no fingerprint match)"
        for fp, handler in _RESPONSE_TABLE:
            if fp in sys_text:
                text = handler(sys_text, user_text)
                break
        return LLMResponse(
            text=text, model=model, provider=self.name,
            input_tokens=10, output_tokens=20,
            usd_cost=0.0, elapsed_sec=0.001,
        )


def install_dry_run_driver() -> None:
    """Register the dry-run driver under the canonical provider names
    so loading research_agents.yaml picks it up regardless of which
    provider each agent declared. The original drivers stay registered
    too — overwrite is intentional for the duration of a dry run."""
    register_driver("claude", DryRunDriver)
    register_driver("claude-code", DryRunDriver)
    register_driver("deepseek", DryRunDriver)
    register_driver("grok", DryRunDriver)
    register_driver("dry-run", DryRunDriver)


# ---------------------------------------------------------------------- #
# Stub HTTP for the arXiv fetcher
# ---------------------------------------------------------------------- #


_STUB_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/dry-run.0001v1</id>
    <updated>2026-04-29T12:00:00Z</updated>
    <published>2026-04-29T12:00:00Z</published>
    <title>Dry-run synthetic paper for pipeline preflight</title>
    <summary>This is a stub abstract for the dry-run preflight. It does not exist on arXiv.</summary>
    <author><name>Dry-Run Author</name></author>
  </entry>
</feed>
"""


def stub_http_get(_url: str) -> str:
    """HttpGet stub: returns the same canned arXiv Atom feed for any
    URL. The ingester parses it normally and produces one Paper record;
    the loop then walks the rest of the pipeline against that paper."""
    return _STUB_ATOM


# ---------------------------------------------------------------------- #
# Stub backtest_runner
# ---------------------------------------------------------------------- #


def stub_backtest_runner(_code_path: Path) -> dict[str, Any]:
    """BacktestRunner stub: returns metrics that pass every threshold
    rule, so the dry-run produces a PROMOTE verdict and exercises the
    GATE 2 + registrar paths. The metrics are obviously synthetic
    (round numbers, no fold detail) — the audit trail makes this
    clear."""
    return {
        "oos_metrics": {
            "sharpe": 0.85,
            "n_trades": 50,
            "hit_rate": 0.58,
            "max_drawdown": -0.12,
            "profit_factor": 1.6,
        },
        "sharpe_ci_95": {"low": 0.20, "high": 1.30},
        "is_oos_sharpe_ratio": 1.4,
        "edge_concentration": 0.45,
        "regime_diversified": True,
        "decay_severity": "NONE",
        "_metrics_provenance": {
            "all": "DRY RUN — synthetic stubs from src.research.dry_run, not measured",
        },
    }
