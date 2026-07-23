"""Tests for the LLM A/B harness scoring (CL-gyjz).

The actual provider calls hit live APIs and aren't tested here. We
verify the deterministic scoring rubric and the summary aggregation
on synthesized rows so the harness itself can't drift.
"""

from __future__ import annotations

import pytest
from scripts.compare_llm_providers import (
    ComparisonRow,
    PromptSpec,
    QualityScore,
    _score_density,
    _score_keywords,
    _score_structure,
    score_response,
    summarize,
)


class TestScoreStructure:
    def test_valid_json_full_credit(self) -> None:
        assert _score_structure('{"a": 1}', "json") == 1.0

    def test_fenced_json_half_credit(self) -> None:
        # Wrapped in ```json fences but valid after strip.
        assert _score_structure('```json\n{"a": 1}\n```', "json") == 0.5

    def test_invalid_json_zero(self) -> None:
        assert _score_structure("not json at all", "json") == 0.0

    def test_short_freeform_zero(self) -> None:
        assert _score_structure("yes", "free-form") == 0.0

    def test_freeform_with_paragraphs(self) -> None:
        text = "First paragraph here that goes on a bit.\n\nSecond paragraph."
        assert _score_structure(text, "free-form") == 1.0


class TestScoreDensity:
    def test_empty_zero(self) -> None:
        assert _score_density("") == 0.0

    def test_pure_filler_low_score(self) -> None:
        text = "However, moreover, furthermore, additionally."
        assert _score_density(text) < 0.5

    def test_dense_prose_full(self) -> None:
        text = "Carry trade Sharpe 0.7 over 1983-2010 G10 sample."
        assert _score_density(text) >= 0.95


class TestScoreKeywords:
    def test_no_expected_keywords_returns_one(self) -> None:
        assert _score_keywords("any text", []) == 1.0

    def test_partial_match(self) -> None:
        assert _score_keywords(
            "Carry trade and volatility risk",
            ["carry", "volatility", "momentum"],
        ) == pytest.approx(2 / 3)

    def test_case_insensitive(self) -> None:
        assert _score_keywords("CARRY TRADE", ["carry"]) == 1.0


class TestWeightedScore:
    def test_combines_components(self) -> None:
        prompt = PromptSpec(
            name="t",
            system="",
            user="",
            expected_keywords=["carry"],
            expected_format="json",
        )
        # JSON-valid, dense (no filler), keyword present → all dims 1.0.
        s = score_response('{"side": "long carry"}', prompt)
        assert s.structure == 1.0
        assert s.keyword_presence == 1.0
        # Default weights sum to 1; perfect inputs → weighted = 1.0
        assert s.weighted == pytest.approx(1.0, abs=0.05)


class TestSummarize:
    def test_groups_by_provider(self) -> None:
        rows = [
            ComparisonRow(
                prompt_name="p1",
                provider="claude",
                model="m",
                text="t",
                input_tokens=10,
                output_tokens=20,
                usd_cost=0.001,
                elapsed_sec=2.0,
                quality=QualityScore(1.0, 1.0, 1.0, 1.0),
            ),
            ComparisonRow(
                prompt_name="p1",
                provider="deepseek",
                model="m",
                text="t",
                input_tokens=10,
                output_tokens=20,
                usd_cost=0.0001,
                elapsed_sec=2.5,
                quality=QualityScore(0.8, 0.8, 0.8, 0.8),
            ),
        ]
        out = summarize(rows)
        assert "claude" in out and "deepseek" in out
        assert out["claude"]["avg_quality"] == 1.0
        assert out["deepseek"]["avg_quality"] == 0.8
        # USD per quality point: claude 0.001 / 1.0 = 0.001;
        # deepseek 0.0001 / 0.8 ≈ 0.000125
        assert out["claude"]["usd_per_quality"] > out["deepseek"]["usd_per_quality"]


class TestPromptFile:
    def test_seed_file_loads(self) -> None:
        from pathlib import Path

        from scripts.compare_llm_providers import load_prompts

        prompts = load_prompts(Path("scripts/_seed_prompts/idea_extract.json"))
        assert len(prompts) >= 3
        assert all(isinstance(p, PromptSpec) for p in prompts)
        # Each prompt declares a format the harness understands.
        for p in prompts:
            assert p.expected_format in ("json", "free-form")
