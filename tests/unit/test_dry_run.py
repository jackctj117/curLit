"""Tests for the dry-run scaffolding (src/research/dry_run.py).

Covers: fingerprint routing (each agent gets its own canned), the
idea-brief handler splices in the real extract path so validation
passes, the static handlers are immutable, the stub_http_get always
returns a valid arXiv Atom feed, and stub_backtest_runner returns
metrics that pass every threshold rule.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.dry_run import (
    DryRunDriver,
    install_dry_run_driver,
    stub_backtest_runner,
    stub_http_get,
)
from src.research.ingest import ArxivFetcher, FeedConfig
from src.research.llm.client import Message
from src.research.verdict import compute_verdict, parse_rules


class TestFingerprintRouting:
    def _complete(self, sys_text: str, user_text: str = "") -> str:
        d = DryRunDriver()
        resp = d.complete(
            messages=[
                Message(role="system", content=sys_text),
                Message(role="user", content=user_text),
            ],
            model="m1",
        )
        return resp.text

    def test_paper_extractor_routes_to_extract(self) -> None:
        text = self._complete("# Paper Extractor — System Prompt")
        assert "## Methodology" in text and "Dry-run stub extract" in text

    def test_idea_generator_routes_to_brief(self) -> None:
        text = self._complete(
            "# Idea Generator — System Prompt\nMentions Implementer too",
            user_text="<context label='paper_extract:abc.md'>...</context>",
        )
        # Should be the idea brief, not the implementer output (the
        # idea prompt mentions 'Implementer' in its body — without H1
        # prefix matching, the implementer fingerprint would win)
        assert "**FINAL_POSITION**: PROPOSED" in text
        assert "## Source extract" in text
        assert "abc.md" in text

    def test_implementer_routes_to_impl_output(self) -> None:
        text = self._complete("# Implementer — System Prompt")
        assert "**FINAL_POSITION**: IMPLEMENTED" in text
        assert "class DryRunStrategy" in text

    def test_bull_reviewer_routes_to_promote(self) -> None:
        text = self._complete("# Bull Reviewer — System Prompt")
        assert "**FINAL_POSITION**: PROMOTE" in text

    def test_bear_reviewer_routes_to_promote(self) -> None:
        # The dry run intentionally has Bear vote PROMOTE so PROMOTE
        # verdict fires end-to-end. Verify the bear fingerprint
        # specifically matches (not the bull one).
        text = self._complete("# Bear Reviewer — System Prompt")
        assert "**FINAL_POSITION**: PROMOTE" in text

    def test_unknown_prompt_fingerprint(self) -> None:
        text = self._complete("# Some Unknown Agent")
        assert "no fingerprint match" in text


class TestIdeaBriefHandler:
    def test_extracts_path_from_user_message(self) -> None:
        d = DryRunDriver()
        resp = d.complete(
            messages=[
                Message(
                    role="system",
                    content="# Idea Generator — System Prompt",
                ),
                Message(
                    role="user",
                    content=(
                        "<context label='paper_extract:abc123.md'>extract"
                        " body</context>\n\nGenerate the brief."
                    ),
                ),
            ],
            model="m1",
        )
        # The validator requires the extract path or filename in
        # references — verify it lands there
        assert "abc123.md" in resp.text


class TestInstallDryRunDriver:
    def test_overwrites_canonical_provider_drivers(self) -> None:
        from src.research.llm.client import _DRIVERS

        # Snapshot pre-install
        before = dict(_DRIVERS)
        install_dry_run_driver()
        try:
            assert _DRIVERS["claude"] is DryRunDriver
            assert _DRIVERS["deepseek"] is DryRunDriver
            assert _DRIVERS["grok"] is DryRunDriver
            assert _DRIVERS["dry-run"] is DryRunDriver
        finally:
            # Restore so we don't poison sibling tests in the same
            # process (registries are global module state).
            _DRIVERS.clear()
            _DRIVERS.update(before)


class TestStubHttpGet:
    def test_returns_parseable_atom(self) -> None:
        body = stub_http_get("http://example.com/whatever")
        feed = FeedConfig(
            name="t",
            adapter="arxiv",
            query_url="x",
            source_label="dry-run",
        )
        papers = ArxivFetcher(http_get=lambda _u: body).fetch(feed)
        assert len(papers) == 1
        assert papers[0].title == ("Dry-run synthetic paper for pipeline preflight")

    def test_same_response_for_any_url(self) -> None:
        assert stub_http_get("http://a") == stub_http_get("http://b")


class TestStubBacktestRunner:
    def test_returns_passing_metrics(self, tmp_path: Path) -> None:
        metrics = stub_backtest_runner(tmp_path / "stub.py")
        # Every threshold path the verdict engine reads is present
        for k in (
            "sharpe",
            "n_trades",
            "hit_rate",
            "max_drawdown",
            "profit_factor",
        ):
            assert k in metrics["oos_metrics"]
        assert "low" in metrics["sharpe_ci_95"]
        for top in (
            "is_oos_sharpe_ratio",
            "edge_concentration",
            "regime_diversified",
            "decay_severity",
        ):
            assert top in metrics

    def test_metrics_pass_real_review_rules(
        self,
        tmp_path: Path,
    ) -> None:
        """Sanity: the stub metrics should produce no rule failures
        when fed through the real REVIEW_RULES.md verdict engine.
        Otherwise the dry-run wouldn't end in PROMOTE."""
        from src.research.agents.reviewer import Position

        metrics = stub_backtest_runner(tmp_path / "stub.py")
        rules = parse_rules("docs/research/REVIEW_RULES.md")
        # Build a minimal candidate report shape (top-level keys
        # match how the loop's Implementer flattens them)
        report = dict(metrics)
        verdict = compute_verdict(
            candidate_report=report,
            rules=rules,
            bull_position=Position.PROMOTE,
            bear_position=Position.PROMOTE,
        )
        assert verdict.verdict.value == "PROMOTE", (
            f"dry-run stub metrics didn't PROMOTE: {verdict.reason} | "
            + "; ".join(
                f"{e.rule_id}={'pass' if e.passed else 'FAIL'}"
                for e in verdict.rule_evaluations
                if not e.passed or e.missing
            )
        )


# Suppress unused-import warning for pd — it's pulled by transitive
# imports and tests of fold_metrics shape, kept for forward use.
_ = pd
_ = pytest
