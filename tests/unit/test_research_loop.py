"""Unit tests for the research loop (CL-x561).

The loop is mostly orchestration — its value is wiring the right
sub-pieces in the right order with idempotent state. Tests inject
fake versions of every sub-component so the loop is exercised without
LLM/network/Postgres.

Covers:
  * load_state / save_state roundtrip
  * full happy-path run produces RunSummary + populates state
  * dedup: rerunning skips already-processed extracts/hypotheses/candidates
  * an idea-agent failure on one extract doesn't kill the run
  * implementer REJECTED hypotheses don't reach the debate phase
  * verdict tally (PROMOTE/REJECT/ESCALATE) maps from compute_verdict
  * per-run summary file written under runs_dir
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from src.research.agents.idea import IdeaResult, IdeaStatus
from src.research.agents.implementer import ImplementerResult, ImplementerStatus
from src.research.agents.reviewer import Position
from src.research.ingest import IngestRunSummary
from src.research.loop import (
    LoopState,
    ResearchLoop,
    RunSummary,
    load_state,
    save_state,
)
from src.research.orchestrator import DebateResult

# --------------------------------------------------------------------------- #
# Fake helpers — minimal stand-ins for the real sub-components
# --------------------------------------------------------------------------- #


@dataclass
class _FakeIngestRunner:
    """Pretend ingester. ``new_extracts`` is a list of (hash, body)
    pairs — the runner writes them into the store the first time
    ``run`` is called and reports them as new."""

    extract_store: Any
    new_extracts: list[tuple[str, str]]

    def run(self, _feed_configs: list[Any]) -> IngestRunSummary:
        summary = IngestRunSummary(feeds_total=1)
        for paper_hash, body in self.new_extracts:
            path = self.extract_store.root / f"{paper_hash}.md"
            if path.exists():
                summary.papers_skipped_duplicate += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body)
            summary.papers_extracted += 1
            summary.extract_paths.append(path)
        # Subsequent calls find everything already on disk → 0 new
        return summary


class _FakeIdeaAgent:
    """Programmable Idea agent. ``responses`` is a dict
    extract_hash → IdeaResult-ish dict. Missing entry = raise."""

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[Path] = []

    def ideate(
        self,
        extract_path: Path,
        strategy_slug: str | None = None,  # noqa: ARG002
        backlog_summary: str | None = None,  # noqa: ARG002
        hypothesis_dir: Path | str = "docs/research/hypotheses",
    ) -> IdeaResult:
        self.calls.append(extract_path)
        h = extract_path.stem
        if h not in self.responses:
            raise RuntimeError(f"unmocked idea call for {h}")
        spec = self.responses[h]
        if spec.get("raise"):
            raise RuntimeError(spec["raise"])
        status = IdeaStatus(spec["status"])
        slug = spec["slug"]
        # Write hypothesis file when PROPOSED to mirror real agent
        out_path = None
        if status == IdeaStatus.PROPOSED:
            out_dir = Path(hypothesis_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{slug}.md"
            out_path.write_text(f"# Hypothesis: {slug}\n")
        return IdeaResult(
            status=status,
            strategy_slug=slug,
            extract_path=extract_path,
            response=Mock(),
            raw_text=spec.get("raw_text", ""),
            hypothesis_path=out_path,
            reason=spec.get("reason", ""),
        )


class _FakeImplementer:
    """Programmable Implementer. ``responses`` keyed by slug."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any]],
        candidate_dir: Path,
    ) -> None:
        self.responses = responses
        self.candidate_dir = candidate_dir
        self.calls: list[str] = []

    def implement(
        self,
        hypothesis_path: Path,  # noqa: ARG002
        strategy_slug: str,
        backtest_runner: Any = None,  # noqa: ARG002
        code_dir: Path | str = "src/strategies/_experimental",
        report_dir: Path | str = "reports/candidates",  # noqa: ARG002
    ) -> ImplementerResult:
        self.calls.append(strategy_slug)
        if strategy_slug not in self.responses:
            raise RuntimeError(f"unmocked implementer call for {strategy_slug}")
        spec = self.responses[strategy_slug]
        if spec.get("raise"):
            raise RuntimeError(spec["raise"])
        status = ImplementerStatus(spec["status"])
        code_path = Path(code_dir) / f"{strategy_slug}.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(spec.get("code", "# stub\n"))
        report_path: Path | None = None
        if status == ImplementerStatus.IMPLEMENTED:
            report_path = self.candidate_dir / f"{strategy_slug}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(spec.get("report", {"oos_metrics": {"sharpe": 0.6}})),
            )
        return ImplementerResult(
            status=status,
            strategy_slug=strategy_slug,
            response=Mock(),
            raw_text="",
            code_path=code_path,
            report_path=report_path,
            reason=spec.get("reason", ""),
            candidate_report=spec.get("report", {}),
        )


class _FakeOrchestrator:
    """Returns canned DebateResults keyed by slug."""

    def __init__(self, results: dict[str, DebateResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    def run_debate(
        self,
        strategy_slug: str,
        candidate_report_text: str,  # noqa: ARG002
    ) -> DebateResult:
        self.calls.append(strategy_slug)
        if strategy_slug not in self.results:
            raise RuntimeError(f"unmocked debate for {strategy_slug}")
        return self.results[strategy_slug]


def _make_debate_result(
    slug: str,
    bull: Position,
    bear: Position,
    transcript: Path,
    open_questions: list[str] | None = None,
) -> DebateResult:
    return DebateResult(
        debate_name="promotion_review",
        strategy_slug=slug,
        transcript_path=transcript,
        transcript_entries=[],
        final_positions={"bull": bull, "bear": bear},
        open_questions=open_questions or [],
        resolved_questions=[],
        total_cost_usd=0.0,
        total_elapsed_sec=0.0,
    )


@dataclass
class _FakeExtractStore:
    root: Path


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def loop_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "state": tmp_path / "state.json",
        "runs": tmp_path / "runs",
        "hypotheses": tmp_path / "hypotheses",
        "candidates": tmp_path / "candidates",
        "extracts": tmp_path / "extracts",
        "transcripts": tmp_path / "debates",
    }


# --------------------------------------------------------------------------- #
# State persistence
# --------------------------------------------------------------------------- #


class TestStatePersistence:
    def test_load_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        s = load_state(tmp_path / "ghost.json")
        assert s == LoopState()

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        original = LoopState(
            ideas_processed={"abc": {"status": "PROPOSED", "slug": "x", "reason": ""}},
            candidates_processed={"x": {
                "status": "IMPLEMENTED", "reason": "",
                "code_path": "p", "report_path": "r",
            }},
            debates_completed={"x": {
                "verdict": "PROMOTE", "reason": "all gates pass",
                "transcript_path": "t", "bull": "PROMOTE", "bear": "PROMOTE",
            }},
        )
        save_state(original, path)
        loaded = load_state(path)
        assert loaded == original


# --------------------------------------------------------------------------- #
# Happy-path run
# --------------------------------------------------------------------------- #


class TestFullPipelineRun:
    def test_happy_path_end_to_end(self, loop_paths: dict[str, Path]) -> None:
        # Two extracts. Both ideated → one PROPOSED, one DECLINED.
        # PROPOSED → implementer IMPLEMENTED → debate PROMOTE.
        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store,
            new_extracts=[
                ("hash1", "# Paper A\nbody"),
                ("hash2", "# Paper B\nbody"),
            ],
        )
        idea = _FakeIdeaAgent(responses={
            "hash1": {"status": "PROPOSED", "slug": "alpha"},
            "hash2": {"status": "DECLINED", "slug": "beta",
                      "reason": "extract too thin"},
        })
        impl = _FakeImplementer(
            responses={
                "alpha": {
                    "status": "IMPLEMENTED",
                    "report": {"oos_metrics": {"sharpe": 0.7}},
                },
            },
            candidate_dir=loop_paths["candidates"],
        )
        debate_transcript = loop_paths["transcripts"] / "alpha" / "transcript.md"
        debate_transcript.parent.mkdir(parents=True)
        debate_transcript.write_text("(stub transcript)")
        orch = _FakeOrchestrator(results={
            "alpha": _make_debate_result(
                "alpha", Position.PROMOTE, Position.PROMOTE,
                transcript=debate_transcript,
            ),
        })
        loop = ResearchLoop(
            ingest_runner=ingest,
            idea_agent=idea,  # type: ignore[arg-type]
            implementer=impl,  # type: ignore[arg-type]
            debate_orchestrator=orch,  # type: ignore[arg-type]
            rules_loader=lambda: [],  # no threshold rules → all-pass
            feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
        )
        summary = loop.run()
        assert summary.extracts_new == 2
        assert summary.ideas_proposed == 1
        assert summary.ideas_declined == 1
        assert summary.candidates_implemented == 1
        assert summary.candidates_rejected == 0
        assert summary.debates_run == 1
        assert summary.verdicts_promote == 1
        assert summary.verdicts_reject == 0
        assert summary.verdicts_escalate == 0
        assert summary.errors == []

        # State persisted with all expected entries
        state = load_state(loop_paths["state"])
        assert state.ideas_processed["hash1"]["status"] == "PROPOSED"
        assert state.ideas_processed["hash2"]["status"] == "DECLINED"
        assert state.candidates_processed["alpha"]["status"] == "IMPLEMENTED"
        assert state.debates_completed["alpha"]["verdict"] == "PROMOTE"

        # Run summary file written
        run_files = list(loop_paths["runs"].glob("*.json"))
        assert len(run_files) == 1
        run_data = json.loads(run_files[0].read_text())
        assert run_data["verdicts_promote"] == 1


class TestIdempotency:
    def test_second_run_skips_processed_work(
        self, loop_paths: dict[str, Path],
    ) -> None:
        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store,
            new_extracts=[("hash1", "# Paper A\nbody")],
        )
        idea = _FakeIdeaAgent(responses={
            "hash1": {"status": "PROPOSED", "slug": "alpha"},
        })
        impl = _FakeImplementer(
            responses={"alpha": {"status": "IMPLEMENTED"}},
            candidate_dir=loop_paths["candidates"],
        )
        debate_transcript = loop_paths["transcripts"] / "alpha" / "transcript.md"
        debate_transcript.parent.mkdir(parents=True)
        debate_transcript.write_text("(stub)")
        orch = _FakeOrchestrator(results={
            "alpha": _make_debate_result(
                "alpha", Position.PROMOTE, Position.PROMOTE,
                transcript=debate_transcript,
            ),
        })
        loop = ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=impl,  # type: ignore[arg-type]
            debate_orchestrator=orch,  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
        )

        loop.run()
        first_calls = (len(idea.calls), len(impl.calls), len(orch.calls))
        assert first_calls == (1, 1, 1)

        loop.run()
        # Second run should NOT re-call any sub-component on the same
        # extract/hypothesis/candidate.
        assert (len(idea.calls), len(impl.calls), len(orch.calls)) == first_calls


class TestErrorHandling:
    def test_idea_failure_doesnt_kill_run(
        self, loop_paths: dict[str, Path],
    ) -> None:
        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store,
            new_extracts=[
                ("hash1", "# Paper A\n"),
                ("hash2", "# Paper B\n"),
            ],
        )
        idea = _FakeIdeaAgent(responses={
            "hash1": {"raise": "LLM timed out"},
            "hash2": {"status": "PROPOSED", "slug": "beta"},
        })
        impl = _FakeImplementer(
            responses={"beta": {"status": "IMPLEMENTED"}},
            candidate_dir=loop_paths["candidates"],
        )
        debate_transcript = loop_paths["transcripts"] / "beta" / "transcript.md"
        debate_transcript.parent.mkdir(parents=True)
        debate_transcript.write_text("(stub)")
        orch = _FakeOrchestrator(results={
            "beta": _make_debate_result(
                "beta", Position.PROMOTE, Position.PROMOTE,
                transcript=debate_transcript,
            ),
        })
        loop = ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=impl,  # type: ignore[arg-type]
            debate_orchestrator=orch,  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
        )
        summary = loop.run()
        # hash1 errored, hash2 went through
        assert summary.errors  # one entry for hash1
        assert summary.candidates_implemented == 1
        assert summary.verdicts_promote == 1
        state = load_state(loop_paths["state"])
        assert state.ideas_processed["hash1"]["status"] == "ERROR"

    def test_implementer_rejected_skips_debate(
        self, loop_paths: dict[str, Path],
    ) -> None:
        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store,
            new_extracts=[("hash1", "# A\n")],
        )
        idea = _FakeIdeaAgent(responses={
            "hash1": {"status": "PROPOSED", "slug": "alpha"},
        })
        impl = _FakeImplementer(
            responses={"alpha": {"status": "REJECTED",
                                 "reason": "syntax gate failed"}},
            candidate_dir=loop_paths["candidates"],
        )
        orch = _FakeOrchestrator(results={})
        loop = ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=impl,  # type: ignore[arg-type]
            debate_orchestrator=orch,  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
        )
        summary = loop.run()
        assert summary.candidates_rejected == 1
        assert summary.debates_run == 0
        # Orchestrator never called for a REJECTED candidate
        assert orch.calls == []


class TestVerdictTally:
    def test_escalate_counted(self, loop_paths: dict[str, Path]) -> None:
        # Mixed positions → ESCALATE per the verdict engine's rule
        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store,
            new_extracts=[("hash1", "# A\n")],
        )
        idea = _FakeIdeaAgent(responses={
            "hash1": {"status": "PROPOSED", "slug": "alpha"},
        })
        impl = _FakeImplementer(
            responses={"alpha": {"status": "IMPLEMENTED"}},
            candidate_dir=loop_paths["candidates"],
        )
        debate_transcript = loop_paths["transcripts"] / "alpha" / "transcript.md"
        debate_transcript.parent.mkdir(parents=True)
        debate_transcript.write_text("(stub)")
        orch = _FakeOrchestrator(results={
            "alpha": _make_debate_result(
                "alpha",
                Position.PROMOTE, Position.REJECT,  # mixed
                transcript=debate_transcript,
            ),
        })
        loop = ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=impl,  # type: ignore[arg-type]
            debate_orchestrator=orch,  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
        )
        summary = loop.run()
        assert summary.verdicts_escalate == 1
        assert summary.verdicts_promote == 0
        assert summary.verdicts_reject == 0


class TestRunSummaryDataclass:
    def test_starts_with_zeros(self) -> None:
        s = RunSummary(started_at="2026-01-01T00:00:00")
        assert s.extracts_new == 0
        assert s.errors == []
