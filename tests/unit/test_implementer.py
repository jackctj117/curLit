"""Unit tests for the Implementer agent (CL-2pww).

Mocks the LLM with the existing _NullDriver pattern so no live API
calls. Validates:
  * extract_code picks the largest python-fenced block
  * parse_position extracts IMPLEMENTED / REJECTED, defaults REJECTED
  * extract_prediction parses the prediction section as a key-value map
  * syntax_check fails REJECTED on broken code
  * implement() writes code + report when IMPLEMENTED
  * implement() returns REJECTED early on REJECTED keyword
  * implement() returns REJECTED when no python block found
  * implement() invokes backtest_runner and persists metrics
  * implement() returns REJECTED when backtest_runner raises
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.research.agents.implementer import (
    Implementer,
    ImplementerStatus,
    extract_code,
    extract_data_sources,
    extract_prediction,
    parse_position,
    syntax_check,
)
from src.research.config import (
    AgentConfig,
    ProviderConfig,
    ResearchConfig,
)
from src.research.llm.client import (
    Driver,
    LLMResponse,
    register_driver,
)

# Mock driver -------------------------------------------------------------


class _CannedDriver(Driver):
    """Returns a configurable canned response per call."""

    name = "canned-impl"

    def __init__(self, api_key: str = "x", canned_text: str = "stub") -> None:
        super().__init__(api_key)
        self.canned_text = canned_text
        self.calls: list[Any] = []

    def complete(
        self,
        messages: Any,  # noqa: ARG002
        model: str,
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({"model": model})
        return LLMResponse(
            text=self.canned_text,
            model=model,
            provider=self.name,
            input_tokens=10,
            output_tokens=20,
            usd_cost=0.001,
            elapsed_sec=0.01,
        )


register_driver("canned-impl", _CannedDriver)


# Fixtures ----------------------------------------------------------------


@pytest.fixture
def implementer_prompt(tmp_path: Path) -> Path:
    f = tmp_path / "implementer_prompt.md"
    f.write_text("You are a stub implementer for tests.")
    return f


@pytest.fixture
def hypothesis_brief(tmp_path: Path) -> Path:
    f = tmp_path / "test_hyp.md"
    f.write_text(
        "# Hypothesis\n\nMean-revert on US-DE 2Y rate diff at z=2.\n"
        "Data sources: FRED IRLTLT01DEM156N, US 2Y from FRED.\n",
    )
    return f


@pytest.fixture
def cfg(
    implementer_prompt: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ResearchConfig:
    monkeypatch.setenv("CANNED_IMPL_KEY", "fake")
    return ResearchConfig(
        providers={
            "canned-impl": ProviderConfig(
                api_key_env="CANNED_IMPL_KEY",
                default_model="m1",
            ),
        },
        agents={
            "implementer": AgentConfig(
                provider="canned-impl",
                role="implementer",
                prompt_path=str(implementer_prompt),
                model=None,
            ),
        },
        debates={},  # type: ignore[arg-type]
    )


def _build_implementer(cfg: ResearchConfig, canned_text: str) -> Implementer:
    """Construct the Implementer with the canned LLM response patched in."""
    impl = Implementer.from_config(name="implementer", research_config=cfg)
    # Swap the registered driver instance with one that returns this text.
    # The client's get_client returns a fresh driver each call; we mutate
    # its driver in-place.
    impl.client.driver.canned_text = canned_text  # type: ignore[attr-defined]
    return impl


# --------------------------------------------------------------------- #
# Pure parsing
# --------------------------------------------------------------------- #


class TestExtractCode:
    def test_returns_largest_python_block(self) -> None:
        text = (
            "preamble\n"
            "```python\n"
            "x = 1\n"
            "```\n"
            "more text\n"
            "```python\n"
            "def long_function():\n    return 42\n"
            "```\n"
        )
        assert extract_code(text) == "def long_function():\n    return 42"

    def test_returns_none_when_no_block(self) -> None:
        assert extract_code("no code here") is None

    def test_ignores_non_python_fences(self) -> None:
        text = "```yaml\nfoo: bar\n```"
        assert extract_code(text) is None


class TestParsePosition:
    def test_explicit_final_position_implemented(self) -> None:
        assert parse_position("**FINAL_POSITION**: IMPLEMENTED") == ImplementerStatus.IMPLEMENTED

    def test_explicit_final_position_rejected(self) -> None:
        assert parse_position("**FINAL_POSITION**: REJECTED") == ImplementerStatus.REJECTED

    def test_no_keyword_defaults_rejected(self) -> None:
        # Caution-default: never IMPLEMENTED without explicit declaration.
        assert parse_position("blah blah") == ImplementerStatus.REJECTED

    def test_last_match_wins(self) -> None:
        text = (
            "Initial draft might be IMPLEMENTED.\n\n"
            "Updated thinking...\n\n"
            "**FINAL_POSITION**: REJECTED"
        )
        assert parse_position(text) == ImplementerStatus.REJECTED


class TestExtractPrediction:
    def test_parses_bulleted_section(self) -> None:
        text = (
            "## Prediction\n"
            "- `predicted_sharpe_range`: [0.5, 0.9]\n"
            "- `predicted_hit_rate_range`: [0.55, 0.62]\n"
            "- `expected_n_trades_per_year`: 80\n"
            "## Data sources actually consumed\n"
            "ignored\n"
        )
        out = extract_prediction(text)
        assert out["predicted_sharpe_range"] == "[0.5, 0.9]"
        assert out["predicted_hit_rate_range"] == "[0.55, 0.62]"
        assert out["expected_n_trades_per_year"] == "80"

    def test_returns_empty_when_no_section(self) -> None:
        assert extract_prediction("no prediction here") == {}


class TestExtractDataSources:
    def test_parses_list(self) -> None:
        text = (
            "## Data sources actually consumed\n- FRED IRLTLT01DEM156N\n- prices.EURUSD\n## next\n"
        )
        sources = extract_data_sources(text)
        assert "FRED IRLTLT01DEM156N" in sources
        assert "prices.EURUSD" in sources


class TestSyntaxCheck:
    def test_passes_valid_python(self) -> None:
        ok, _ = syntax_check("x = 1 + 2", "test")
        assert ok

    def test_fails_syntax_error(self) -> None:
        ok, detail = syntax_check("def broken(:", "test")
        assert not ok
        assert "SyntaxError" in detail


# --------------------------------------------------------------------- #
# implement() integration
# --------------------------------------------------------------------- #


_VALID_RESPONSE = """# IMPLEMENTATION for test-strat

## Hypothesis citation
Mean-revert on US-DE 2Y rate diff (per brief).

## Strategy code
```python
class TestStrategy:
    id = "test-strat"
    symbols = ["EURUSD"]

    def fit(self, train_data):
        self._mean = 0.0

    def generate_signals(self, data):
        return data["close"] - self._mean
```

## Prediction
- `predicted_sharpe_range`: [0.5, 0.9]
- `predicted_hit_rate_range`: [0.55, 0.62]

## Data sources actually consumed
- FRED IRLTLT01DEM156N

## Validation notes
Stub implementation for testing.

**FINAL_POSITION**: IMPLEMENTED
"""


_REJECTED_RESPONSE = """The hypothesis brief asks for tomorrow's NFP
data which is not available — this is a look-ahead at the brief level.

**FINAL_POSITION**: REJECTED

The data source 'tomorrow_nfp' is not in src/data/.
"""


_NO_CODE_RESPONSE = """## Hypothesis citation
A reasonable thesis.

## Validation notes
Forgot to include the code, oops.

**FINAL_POSITION**: IMPLEMENTED
"""


_BROKEN_CODE_RESPONSE = """# IMPLEMENTATION

## Strategy code
```python
def broken(:
    return None
```

## Prediction
- `predicted_sharpe_range`: [0.5, 0.9]

**FINAL_POSITION**: IMPLEMENTED
"""


class TestImplementIntegration:
    def test_implemented_writes_code_and_report(
        self,
        cfg: ResearchConfig,
        hypothesis_brief: Path,
        tmp_path: Path,
    ) -> None:
        impl = _build_implementer(cfg, _VALID_RESPONSE)
        result = impl.implement(
            hypothesis_path=hypothesis_brief,
            strategy_slug="test-strat",
            code_dir=tmp_path / "experimental",
            report_dir=tmp_path / "candidates",
        )
        assert result.status == ImplementerStatus.IMPLEMENTED
        assert result.code_path is not None and result.code_path.exists()
        assert result.report_path is not None and result.report_path.exists()

        # Code on disk should be exactly the extracted block
        on_disk = result.code_path.read_text()
        assert "class TestStrategy" in on_disk
        assert "fit(self, train_data)" in on_disk

        # Report JSON should be valid + contain prediction + provenance
        report = json.loads(result.report_path.read_text())
        assert report["strategy_slug"] == "test-strat"
        assert report["prediction"]["predicted_sharpe_range"] == "[0.5, 0.9]"
        assert report["data_sources_consumed"] == ["FRED IRLTLT01DEM156N"]
        assert report["gates"]["syntax"] == "passed"
        assert report["gates"]["backtest"] == "skipped"
        assert report["provenance"]["model"] == "m1"

    def test_rejected_short_circuits_no_code_written(
        self,
        cfg: ResearchConfig,
        hypothesis_brief: Path,
        tmp_path: Path,
    ) -> None:
        impl = _build_implementer(cfg, _REJECTED_RESPONSE)
        result = impl.implement(
            hypothesis_path=hypothesis_brief,
            strategy_slug="rej-strat",
            code_dir=tmp_path / "experimental",
            report_dir=tmp_path / "candidates",
        )
        assert result.status == ImplementerStatus.REJECTED
        assert result.code_path is None
        assert result.report_path is None
        # _extract_rejection_reason pulls the paragraph AFTER the keyword
        assert "tomorrow_nfp" in result.reason

    def test_no_code_block_rejects_even_when_keyword_says_implemented(
        self,
        cfg: ResearchConfig,
        hypothesis_brief: Path,
        tmp_path: Path,
    ) -> None:
        impl = _build_implementer(cfg, _NO_CODE_RESPONSE)
        result = impl.implement(
            hypothesis_path=hypothesis_brief,
            strategy_slug="no-code",
            code_dir=tmp_path / "experimental",
            report_dir=tmp_path / "candidates",
        )
        assert result.status == ImplementerStatus.REJECTED
        assert "no ```python code block" in result.reason

    def test_syntax_gate_rejects_broken_code(
        self,
        cfg: ResearchConfig,
        hypothesis_brief: Path,
        tmp_path: Path,
    ) -> None:
        impl = _build_implementer(cfg, _BROKEN_CODE_RESPONSE)
        result = impl.implement(
            hypothesis_path=hypothesis_brief,
            strategy_slug="broken",
            code_dir=tmp_path / "experimental",
            report_dir=tmp_path / "candidates",
        )
        assert result.status == ImplementerStatus.REJECTED
        assert "syntax gate failed" in result.reason
        # We DO write the code (for forensic inspection) even on syntax fail
        assert result.code_path is not None and result.code_path.exists()
        # But no report — broken code doesn't get a candidate report
        assert result.report_path is None

    def test_backtest_runner_invoked_and_metrics_in_report(
        self,
        cfg: ResearchConfig,
        hypothesis_brief: Path,
        tmp_path: Path,
    ) -> None:
        impl = _build_implementer(cfg, _VALID_RESPONSE)

        called_with: list[Path] = []

        def fake_backtest(code_path: Path) -> dict[str, Any]:
            called_with.append(code_path)
            return {
                "oos_metrics": {"sharpe": 0.72, "max_dd": -0.12},
                "n_trades": 47,
            }

        result = impl.implement(
            hypothesis_path=hypothesis_brief,
            strategy_slug="bt-strat",
            backtest_runner=fake_backtest,
            code_dir=tmp_path / "experimental",
            report_dir=tmp_path / "candidates",
        )
        assert result.status == ImplementerStatus.IMPLEMENTED
        assert len(called_with) == 1
        assert called_with[0] == result.code_path
        report = json.loads(result.report_path.read_text())  # type: ignore[arg-type]
        assert report["backtest_metrics"]["oos_metrics"]["sharpe"] == 0.72
        assert report["gates"]["backtest"] == "ran"

    def test_backtest_runner_failure_rejects(
        self,
        cfg: ResearchConfig,
        hypothesis_brief: Path,
        tmp_path: Path,
    ) -> None:
        impl = _build_implementer(cfg, _VALID_RESPONSE)

        def failing_backtest(code_path: Path) -> dict[str, Any]:
            raise RuntimeError("walk-forward exploded")

        result = impl.implement(
            hypothesis_path=hypothesis_brief,
            strategy_slug="bt-fail",
            backtest_runner=failing_backtest,
            code_dir=tmp_path / "experimental",
            report_dir=tmp_path / "candidates",
        )
        assert result.status == ImplementerStatus.REJECTED
        assert "backtest failed" in result.reason
        assert "walk-forward exploded" in result.reason

    def test_missing_hypothesis_raises_filenotfound(
        self,
        cfg: ResearchConfig,
        tmp_path: Path,
    ) -> None:
        impl = _build_implementer(cfg, _VALID_RESPONSE)
        with pytest.raises(FileNotFoundError, match="hypothesis brief not found"):
            impl.implement(
                hypothesis_path=tmp_path / "nope.md",
                strategy_slug="ghost",
                code_dir=tmp_path / "experimental",
                report_dir=tmp_path / "candidates",
            )


# --------------------------------------------------------------------- #
# Real prompt file
# --------------------------------------------------------------------- #


class TestRealPromptFile:
    def test_implementer_prompt_loads_and_references_evidence_first(
        self,
    ) -> None:
        path = Path("configs/research_prompts/implementer.md")
        assert path.exists()
        text = path.read_text()
        # Doctrine reference
        assert "EVIDENCE_FIRST.md" in text
        # Strategy protocol mentioned
        assert "fit" in text and "generate_signals" in text
        # Look-ahead constraint named
        assert "No look-ahead" in text or "no look-ahead" in text.lower()
        # Position keyword spelled out
        assert "IMPLEMENTED" in text and "REJECTED" in text
