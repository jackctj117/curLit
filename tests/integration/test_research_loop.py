"""Integration test for the full research loop (CL-rmmu).

Exercises src/research/loop.py end-to-end with mock LLM responses —
no live API calls, no network. Walks the pipeline:

    ingest → idea → GATE 1 (auto-approved here) → implementer → debate
        → verdict → GATE 2 (no side-effect; registrar mocked)

Four cases:

  1. **Known-good** — implementer emits a candidate report whose
     metrics pass every REVIEW_RULES.md gate; both reviewers PROMOTE.
     Expect verdict=PROMOTE.
  2. **Known-bad (low Sharpe)** — implementer's report has
     oos_metrics.sharpe well below the A.1 threshold. Expect
     verdict=REJECT (rule fails, agent positions don't matter).
  3. **Ambiguous (missing metric)** — report omits a required field.
     Verdict engine returns ESCALATE (any_missing path).
  4. **Mixed positions** — all gates pass on metrics, but Bull says
     PROMOTE and Bear says REJECT → ESCALATE per the verdict engine's
     mixed-positions rule.

Smart-question reformulation (case 4 in the bead) is covered in the
unit-level test_question_resolver / test_debate_orchestrator suites;
not duplicated here.

The mock LLM is wired via a custom Driver registered against a
provider name we configure into a tmp_path agent YAML. The driver
routes by sniffing the system prompt for a fingerprint string —
"Paper Extractor" / "Idea Generator" / "Implementer" / "Bull
Reviewer" / "Bear Reviewer" — so each agent gets its case-specific
canned response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.research.agents.idea import IdeaGenerator
from src.research.agents.implementer import Implementer
from src.research.config import load_config
from src.research.ingest import (
    ExtractStore,
    IngestRunner,
    PaperExtractor,
)
from src.research.llm.client import (
    Driver,
    LLMResponse,
    register_driver,
)
from src.research.loop import (
    ResearchLoop,
    load_state,
    save_state,
)
from src.research.orchestrator import DebateOrchestrator
from src.research.promote import PromoteRegistrar
from src.research.verdict import parse_rules

# --------------------------------------------------------------------------- #
# Mock LLM driver — routes by system-prompt fingerprint
# --------------------------------------------------------------------------- #


class _RouteByPromptDriver(Driver):
    """Mock driver. Set ``RouteByPromptDriver.responses`` (class-level)
    before constructing agents; each ``complete`` call inspects the
    system prompt and returns the matching canned text.

    Class-level attribute is fine here: the driver is instantiated per-
    client by the LLM client's ``get_client`` and we want every instance
    to share the routing table for the duration of one test."""

    name = "route-by-prompt"

    # case-name → fingerprint → canned text
    responses: dict[str, str] = {}
    calls: list[dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls.responses = {}
        cls.calls = []

    def __init__(self, api_key: str = "x") -> None:
        super().__init__(api_key)

    def complete(
        self,
        messages: Any,
        model: str,
        max_tokens: int = 4096,  # noqa: ARG002
        temperature: float = 0.0,  # noqa: ARG002
        **kwargs: Any,
    ) -> LLMResponse:
        sys_text = next(
            (m.content for m in messages if m.role == "system"),
            "",
        )
        # First fingerprint match wins.
        canned = self._lookup(sys_text)
        type(self).calls.append({
            "fingerprint_used": canned[0] if canned else "(no match)",
            "model": model,
        })
        text = canned[1] if canned else "(no canned response wired)"
        return LLMResponse(
            text=text, model=model, provider=self.name,
            input_tokens=10, output_tokens=20,
            usd_cost=0.0001, elapsed_sec=0.01,
        )

    @classmethod
    def _lookup(cls, sys_text: str) -> tuple[str, str] | None:
        for fingerprint, canned in cls.responses.items():
            if fingerprint in sys_text:
                return fingerprint, canned
        return None


register_driver("route-by-prompt", _RouteByPromptDriver)


# --------------------------------------------------------------------------- #
# Canned responses (shared across cases; metrics customized per case)
# --------------------------------------------------------------------------- #


_PAPER_EXTRACT_BODY = (
    "## Methodology\n"
    "Stub methodology for tests.\n\n"
    "## Findings\n"
    "Stub finding with effect size.\n\n"
    "## FX trading applicability\n"
    "Direct.\n\n"
    "## Data sources cited\n"
    "- (none specified in abstract)\n\n"
    "## Key citations\n"
    "- (none in abstract)\n"
)


def _idea_brief(extract_name: str) -> str:
    """A valid hypothesis brief that passes validate_brief()."""
    return (
        "# Hypothesis: regime-conditional carry\n\n"
        "## Source extract\n"
        f"- **path**: `{extract_name}`\n"
        "- **paper title**: stub\n"
        "- **paper hash**: `abc`\n\n"
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
        f"- {extract_name}\n\n"
        "**FINAL_POSITION**: PROPOSED\n"
    )


def _implementer_response(metrics_block: dict[str, Any]) -> str:
    """Build an implementer markdown response. The candidate report's
    metrics come from the implementer's separate JSON write — which we
    can't influence directly because the implementer reads its own
    response. So we wire the candidate report indirectly via the
    backtest_runner callback when wiring the loop, OR we write the
    metrics into the strategy code as constants the verdict engine
    won't see anyway and patch the report after-the-fact in the test.

    For this integration we override the implementer's report writing
    by patching json.dumps via fixture; see test setup below.
    """
    # The implementer extracts the largest python-fenced block. We
    # include a no-op class so syntax_check passes.
    return (
        "# IMPLEMENTATION\n\n"
        "## Strategy code\n"
        "```python\n"
        "class StubStrategy:\n"
        "    id = 'stub'\n"
        "    symbols = ['EURUSD']\n"
        "    def fit(self, train_data):\n        self._mean = 0.0\n"
        "    def generate_signals(self, data):\n        return data * 0\n"
        "```\n\n"
        "## Prediction\n"
        "- `predicted_sharpe_range`: [0.4, 0.8]\n\n"
        "## Data sources actually consumed\n"
        "- FRED IRLTLT01DEM156N\n\n"
        "## Validation notes\nStub.\n\n"
        f"## Metrics\n{json.dumps(metrics_block, indent=2)}\n\n"
        "**FINAL_POSITION**: IMPLEMENTED\n"
    )


def _reviewer_response(position: str) -> str:
    return (
        f"# REVIEW\n\nstub review.\n\n**FINAL_POSITION**: {position}\n"
    )


# --------------------------------------------------------------------------- #
# Fixtures — config + mocked components
# --------------------------------------------------------------------------- #


def _passing_metrics() -> dict[str, Any]:
    """Candidate report metrics that pass every threshold rule."""
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
    }


def _failing_metrics() -> dict[str, Any]:
    """Sharpe below A.1 threshold → REJECT."""
    m = _passing_metrics()
    m["oos_metrics"]["sharpe"] = 0.10  # well below 0.50
    return m


def _missing_metric() -> dict[str, Any]:
    """Drop a required field → ESCALATE (any_missing)."""
    m = _passing_metrics()
    del m["sharpe_ci_95"]
    return m


@pytest.fixture(autouse=True)
def reset_driver() -> None:
    _RouteByPromptDriver.reset()


@pytest.fixture
def repo_paths(tmp_path: Path) -> dict[str, Path]:
    """Build a tmp_path mock-repo with config files + dirs the loop
    expects."""
    paths: dict[str, Path] = {
        "tmp": tmp_path,
        "extracts": tmp_path / "data" / "research" / "extracts",
        "hypotheses": tmp_path / "docs" / "research" / "hypotheses",
        "candidates": tmp_path / "reports" / "candidates",
        "experimental": tmp_path / "src" / "strategies" / "_experimental",
        "production": tmp_path / "src" / "strategies",
        "transcripts": tmp_path / "docs" / "research" / "debates",
        "state": tmp_path / "data" / "research" / "state.json",
        "runs": tmp_path / "data" / "research" / "runs",
        "rules": tmp_path / "REVIEW_RULES.md",
        "agent_yaml": tmp_path / "research_agents.yaml",
        "extractor_prompt": tmp_path / "extractor_prompt.md",
        "idea_prompt": tmp_path / "idea_prompt.md",
        "implementer_prompt": tmp_path / "implementer_prompt.md",
        "bull_prompt": tmp_path / "bull_prompt.md",
        "bear_prompt": tmp_path / "bear_prompt.md",
    }
    for d in (
        paths["extracts"], paths["hypotheses"], paths["candidates"],
        paths["experimental"], paths["production"], paths["transcripts"],
    ):
        d.mkdir(parents=True, exist_ok=True)
    paths["rules"].write_text(Path("docs/research/REVIEW_RULES.md").read_text())
    # System prompts contain the routing fingerprints.
    paths["extractor_prompt"].write_text("Paper Extractor — test stub")
    paths["idea_prompt"].write_text("Idea Generator — test stub")
    paths["implementer_prompt"].write_text("Implementer — test stub")
    paths["bull_prompt"].write_text("Bull Reviewer — test stub")
    paths["bear_prompt"].write_text("Bear Reviewer — test stub")

    # Agent YAML wired to the route-by-prompt provider for all roles.
    paths["agent_yaml"].write_text(yaml.safe_dump({
        "providers": {
            "route-by-prompt": {
                "api_key_env": "ROUTE_KEY", "default_model": "m1",
            },
        },
        "agents": {
            "paper_extractor": {
                "provider": "route-by-prompt",
                "role": "paper_extractor",
                "prompt_path": str(paths["extractor_prompt"]),
                "max_tokens": 1024,
            },
            "idea_generator": {
                "provider": "route-by-prompt",
                "role": "idea_generator",
                "prompt_path": str(paths["idea_prompt"]),
                "max_tokens": 4096,
            },
            "implementer": {
                "provider": "route-by-prompt",
                "role": "implementer",
                "prompt_path": str(paths["implementer_prompt"]),
                "max_tokens": 8192,
            },
            "bull_reviewer": {
                "provider": "route-by-prompt",
                "role": "bull",
                "prompt_path": str(paths["bull_prompt"]),
                "max_tokens": 4096,
            },
            "bear_reviewer": {
                "provider": "route-by-prompt",
                "role": "bear",
                "prompt_path": str(paths["bear_prompt"]),
                "max_tokens": 4096,
            },
        },
        "debates": {
            "promotion_review": {
                "participants": ["bull_reviewer", "bear_reviewer"],
                "rules_path": str(paths["rules"]),
                "verdict_engine": "rule_based",
                "rounds": [
                    {"name": "initial_positions", "type": "parallel"},
                    {"name": "final_position", "type": "parallel"},
                ],
            },
        },
    }))
    return paths


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTE_KEY", "fake")


def _silent_notifier(*_args: Any, **_kwargs: Any) -> Any:
    from src.research.notifications import DispatchResult
    return DispatchResult()


def _seed_extract(repo_paths: dict[str, Path]) -> Path:
    """Drop a single extract in the extracts dir to feed the loop."""
    extract = repo_paths["extracts"] / "abc.md"
    extract.write_text(
        "# Stub Paper Title\n\n"
        "- **paper_hash**: `abc`\n\n"
        "---\n\n" + _PAPER_EXTRACT_BODY,
    )
    return extract


def _build_loop(
    repo_paths: dict[str, Path],
    metrics: dict[str, Any],
    bull_position: str = "PROMOTE",
    bear_position: str = "PROMOTE",
) -> ResearchLoop:
    """Wire the loop with mocked LLM driver routing + a no-op
    registrar."""
    extract = _seed_extract(repo_paths)
    _RouteByPromptDriver.responses = {
        "Paper Extractor": _PAPER_EXTRACT_BODY,
        "Idea Generator": _idea_brief(extract.name),
        "Implementer": _implementer_response(metrics),
        "Bull Reviewer": _reviewer_response(bull_position),
        "Bear Reviewer": _reviewer_response(bear_position),
    }

    research_config = load_config(repo_paths["agent_yaml"])
    extractor = PaperExtractor.from_config(
        name="paper_extractor", research_config=research_config,
    )
    idea = IdeaGenerator.from_config(
        name="idea_generator", research_config=research_config,
    )
    impl = Implementer.from_config(
        name="implementer", research_config=research_config,
    )
    extract_store = ExtractStore(root=repo_paths["extracts"])
    ingest_runner = IngestRunner(
        extractor=extractor,  # type: ignore[arg-type]
        store=extract_store,
    )
    orchestrator = DebateOrchestrator(
        research_config=research_config, debate_name="promotion_review",
        transcript_root=repo_paths["transcripts"],
    )

    # The Implementer now spreads its backtest_runner result at top
    # level (CL-p9ix), so the verdict engine reads oos_metrics.sharpe
    # directly. Inject the test's metrics via a backtest_runner
    # callback rather than post-processing the report on disk.
    original_implement = impl.implement  # type: ignore[attr-defined]

    def patched_implement(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("code_dir", repo_paths["experimental"])
        kwargs.setdefault("report_dir", repo_paths["candidates"])
        kwargs["backtest_runner"] = lambda _code_path: metrics
        return original_implement(*args, **kwargs)

    impl.implement = patched_implement  # type: ignore[method-assign]

    # No-op registrar so GATE 2 doesn't try to git/gh.
    registrar = PromoteRegistrar(
        experimental_dir=repo_paths["experimental"],
        production_dir=repo_paths["production"],
        portfolio_yaml=repo_paths["tmp"] / "live_portfolio.yaml",
        skip_git=True, skip_pr=True,
    )
    # Seed minimal portfolio yaml so the registrar's YAML mutator works.
    (repo_paths["tmp"] / "live_portfolio.yaml").write_text(
        yaml.safe_dump({"strategies": [], "initial_weights": {}}),
    )

    return ResearchLoop(
        ingest_runner=ingest_runner,
        idea_agent=idea,  # type: ignore[arg-type]
        implementer=impl,  # type: ignore[arg-type]
        debate_orchestrator=orchestrator,
        rules_loader=lambda: parse_rules(repo_paths["rules"]),
        feed_configs=[],  # ingest will run with no feeds; we seed the
                          # extract directly on disk
        extract_store=extract_store,
        state_path=repo_paths["state"],
        runs_dir=repo_paths["runs"],
        hypothesis_dir=repo_paths["hypotheses"],
        candidate_dir=repo_paths["candidates"],
        notify_fn=_silent_notifier,
        registrar=registrar,
    )


def _approve_all_pending(state_path: Path) -> int:
    state = load_state(state_path)
    flipped = 0
    for entry in state.ideas_processed.values():
        if entry.get("status") == "PENDING_OPERATOR_APPROVAL":
            entry["status"] = "APPROVED"
            flipped += 1
    save_state(state, state_path)
    return flipped


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("env")
class TestResearchLoopIntegration:
    def test_known_good_candidate_promotes(
        self, repo_paths: dict[str, Path],
    ) -> None:
        loop = _build_loop(
            repo_paths, _passing_metrics(),
            bull_position="PROMOTE", bear_position="PROMOTE",
        )
        # Pass 1: ingest (no-op feeds) + idea agent → PENDING.
        loop.run()
        _approve_all_pending(repo_paths["state"])
        # Pass 2: implementer + debate + verdict.
        summary = loop.run()
        assert summary.candidates_implemented == 1
        assert summary.debates_run == 1
        assert summary.verdicts_promote == 1
        assert summary.verdicts_reject == 0
        assert summary.verdicts_escalate == 0

    def test_known_bad_low_sharpe_rejects(
        self, repo_paths: dict[str, Path],
    ) -> None:
        loop = _build_loop(
            repo_paths, _failing_metrics(),
            bull_position="PROMOTE", bear_position="PROMOTE",
        )
        loop.run()
        _approve_all_pending(repo_paths["state"])
        summary = loop.run()
        # A.1 fails on sharpe=0.10 < 0.50 → REJECT regardless of agent
        # positions
        assert summary.verdicts_reject == 1
        assert summary.verdicts_promote == 0

    def test_missing_metric_escalates(
        self, repo_paths: dict[str, Path],
    ) -> None:
        loop = _build_loop(
            repo_paths, _missing_metric(),
            bull_position="PROMOTE", bear_position="PROMOTE",
        )
        loop.run()
        _approve_all_pending(repo_paths["state"])
        summary = loop.run()
        # Missing sharpe_ci_95 → ESCALATE (rule A.2 marked missing)
        assert summary.verdicts_escalate == 1
        assert summary.verdicts_promote == 0
        assert summary.verdicts_reject == 0
        # ESCALATE side-effect should have fired (silent notifier here)
        assert summary.escalate_notifications_sent == 0
        state = load_state(repo_paths["state"])
        # Slug is derived from extract H1 ("Stub Paper Title") via
        # default_slug → "stub-paper-title".
        assert state.debates_completed["stub-paper-title"]["verdict"] == (
            "ESCALATE"
        )

    def test_mixed_positions_escalate(
        self, repo_paths: dict[str, Path],
    ) -> None:
        loop = _build_loop(
            repo_paths, _passing_metrics(),
            bull_position="PROMOTE", bear_position="REJECT",
        )
        loop.run()
        _approve_all_pending(repo_paths["state"])
        summary = loop.run()
        # All thresholds pass on metrics, but agents disagree → ESCALATE
        assert summary.verdicts_escalate == 1
        assert summary.verdicts_promote == 0
