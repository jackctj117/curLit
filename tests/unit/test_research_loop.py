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
            out_path.write_text(
                spec.get("brief", f"# Hypothesis: {slug}\n"),
            )
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
    """Programmable Implementer. ``responses`` keyed by slug.

    Takes both code_dir and candidate_dir at construction so the fake
    never falls through to the real ``src/strategies/_experimental/``
    when the loop calls ``implement(...)`` without per-call dir args
    (the loop intentionally doesn't pass them — the real Implementer
    has them as defaults pointing at the production paths)."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any]],
        candidate_dir: Path,
        code_dir: Path | None = None,
    ) -> None:
        self.responses = responses
        self.candidate_dir = candidate_dir
        # Default to the candidate_dir's sibling so tests never write
        # outside tmp_path.
        self.code_dir = (
            code_dir if code_dir is not None
            else candidate_dir.parent / "experimental"
        )
        self.calls: list[str] = []

    def implement(
        self,
        hypothesis_path: Path,  # noqa: ARG002
        strategy_slug: str,
        backtest_runner: Any = None,  # noqa: ARG002
        code_dir: Path | str | None = None,  # noqa: ARG002
        report_dir: Path | str | None = None,  # noqa: ARG002
    ) -> ImplementerResult:
        self.calls.append(strategy_slug)
        if strategy_slug not in self.responses:
            raise RuntimeError(f"unmocked implementer call for {strategy_slug}")
        spec = self.responses[strategy_slug]
        if spec.get("raise"):
            raise RuntimeError(spec["raise"])
        status = ImplementerStatus(spec["status"])
        # Always use the fake's tmp-path-rooted dir, ignoring whatever
        # the loop passes — keeps tests from polluting the production
        # ``src/strategies/_experimental`` path.
        code_path = self.code_dir / f"{strategy_slug}.py"
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


def _approve_all_pending(state_path: Path) -> int:
    """Test helper: flip every PENDING_OPERATOR_APPROVAL entry to
    APPROVED, simulating an operator who clicked GO via the dashboard
    or scripts/research_approve.py. Returns the number flipped."""
    state = load_state(state_path)
    flipped = 0
    for entry in state.ideas_processed.values():
        if entry.get("status") == "PENDING_OPERATOR_APPROVAL":
            entry["status"] = "APPROVED"
            flipped += 1
    save_state(state, state_path)
    return flipped


def _silent_notifier(*_args: Any, **_kwargs: Any) -> Any:
    """No-op notifier so tests don't try to dispatch over the network."""
    from src.research.notifications import DispatchResult
    return DispatchResult()


class _FakeRegistrar:
    """Records register() calls and returns a configurable result."""

    def __init__(self, *, succeeded: bool = True, error: str = "") -> None:
        self._succeeded = succeeded
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def register(
        self,
        strategy_slug: str,
        candidate_report_path: Any,
        debate_transcript_path: Any,
        verdict_reason: str = "",
    ) -> Any:
        from src.research.promote import RegistrationResult
        self.calls.append({
            "slug": strategy_slug,
            "report": str(candidate_report_path),
            "transcript": str(debate_transcript_path),
            "reason": verdict_reason,
        })
        return RegistrationResult(
            succeeded=self._succeeded,
            strategy_slug=strategy_slug,
            error=self._error,
            pr_url="https://example.com/pr/1" if self._succeeded else "",
            branch_name=f"experiment/{strategy_slug}",
            steps_completed=(
                ["move_file", "portfolio_yaml", "git", "pr"]
                if self._succeeded else ["move_file"]
            ),
        )


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
            notify_fn=_silent_notifier,
        )
        # Pass 1: get through ingest + idea + GATE 1 notification.
        summary = loop.run()
        assert summary.extracts_new == 2
        assert summary.ideas_proposed == 1
        assert summary.ideas_declined == 1
        assert summary.gate1_notifications_sent == 0  # silent notifier
        # No implementer / debate yet — held by GATE 1
        assert summary.candidates_implemented == 0
        assert summary.debates_run == 0

        state = load_state(loop_paths["state"])
        assert state.ideas_processed["hash1"]["status"] == (
            "PENDING_OPERATOR_APPROVAL"
        )
        assert state.ideas_processed["hash2"]["status"] == "DECLINED"

        # Operator approves via the helper.
        assert _approve_all_pending(loop_paths["state"]) == 1

        # Pass 2: implementer + debate now run for the approved entry.
        summary2 = loop.run()
        assert summary2.candidates_implemented == 1
        assert summary2.debates_run == 1
        assert summary2.verdicts_promote == 1

        state = load_state(loop_paths["state"])
        assert state.ideas_processed["hash1"]["status"] == "APPROVED"
        assert state.candidates_processed["alpha"]["status"] == "IMPLEMENTED"
        assert state.debates_completed["alpha"]["verdict"] == "PROMOTE"

        # Run summary file written. Both runs share a second-precision
        # filename in this test (real cron runs are minutes apart);
        # the latest write reflects pass 2's verdict tally.
        run_files = list(loop_paths["runs"].glob("*.json"))
        assert run_files
        latest_run = json.loads(
            max(run_files, key=lambda p: p.stat().st_mtime).read_text(),
        )
        assert latest_run["verdicts_promote"] == 1


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
            notify_fn=_silent_notifier,
        )

        # Pass 1: idea runs, GATE 1 holds.
        loop.run()
        assert (len(idea.calls), len(impl.calls), len(orch.calls)) == (1, 0, 0)
        _approve_all_pending(loop_paths["state"])

        # Pass 2: implementer + debate run.
        loop.run()
        first_calls = (len(idea.calls), len(impl.calls), len(orch.calls))
        assert first_calls == (1, 1, 1)

        # Pass 3: nothing new — every phase finds its work already done.
        loop.run()
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
            notify_fn=_silent_notifier,
        )
        # Pass 1: hash1 errored in idea phase, hash2 PROPOSED → PENDING.
        summary = loop.run()
        assert summary.errors  # one entry for hash1
        assert summary.candidates_implemented == 0
        state = load_state(loop_paths["state"])
        assert state.ideas_processed["hash1"]["status"] == "ERROR"
        assert state.ideas_processed["hash2"]["status"] == (
            "PENDING_OPERATOR_APPROVAL"
        )

        # Pass 2: operator approves hash2 → it makes it through.
        _approve_all_pending(loop_paths["state"])
        summary2 = loop.run()
        assert summary2.candidates_implemented == 1
        assert summary2.verdicts_promote == 1

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
            notify_fn=_silent_notifier,
        )
        # Pass 1: idea → PENDING. Pass 2: implementer rejects.
        loop.run()
        _approve_all_pending(loop_paths["state"])
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
            notify_fn=_silent_notifier,
        )
        # Pass 1: idea → PENDING. Pass 2: implementer + debate run.
        loop.run()
        _approve_all_pending(loop_paths["state"])
        summary = loop.run()
        assert summary.verdicts_escalate == 1
        assert summary.verdicts_promote == 0
        assert summary.verdicts_reject == 0


class TestRunSummaryDataclass:
    def test_starts_with_zeros(self) -> None:
        s = RunSummary(started_at="2026-01-01T00:00:00")
        assert s.extracts_new == 0
        assert s.errors == []


# --------------------------------------------------------------------------- #
# GATE 1 — pre-research operator approval (CL-0hr3)
# --------------------------------------------------------------------------- #


class TestGate1:
    def test_propose_holds_at_pending_and_fires_notification(
        self, loop_paths: dict[str, Path],
    ) -> None:
        notifications: list[tuple[str, str, int]] = []

        def recorder(title: str, message: str, priority: int) -> Any:
            from src.research.notifications import DispatchResult
            notifications.append((title, message, priority))
            r = DispatchResult()
            r.telegram_attempted = True
            r.telegram_succeeded = True
            return r

        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store, new_extracts=[("h1", "# A\n")],
        )
        idea = _FakeIdeaAgent(responses={
            "h1": {
                "status": "PROPOSED", "slug": "alpha",
                "brief": (
                    "# Hypothesis: SCI filter beats momentum\n\n"
                    "## Data requirements\n"
                    "- `prices.{symbol}` for EURUSD and GBPUSD\n"
                    "- Macro input: FRED `DGS10`\n"
                ),
            },
        })
        loop = ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=_FakeImplementer({}, loop_paths["candidates"]),  # type: ignore[arg-type]
            debate_orchestrator=_FakeOrchestrator({}),  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
            notify_fn=recorder,
        )
        summary = loop.run()
        assert summary.ideas_proposed == 1
        assert summary.gate1_notifications_sent == 1
        # Implementer NOT called for a PENDING entry
        assert summary.candidates_implemented == 0

        state = load_state(loop_paths["state"])
        assert state.ideas_processed["h1"]["status"] == (
            "PENDING_OPERATOR_APPROVAL"
        )
        assert "pending_since" in state.ideas_processed["h1"]
        assert state.ideas_processed["h1"]["hypothesis_path"]

        # Phone-first HTML body (CL-frn7): bold slug + thesis, which
        # instruments the plan would trade, reply line with short id.
        assert len(notifications) == 1
        title, message, priority = notifications[0]
        assert "GATE 1" in title
        assert "<b>alpha</b>" in message
        assert "SCI filter beats momentum" in message
        assert "<b>Trades:</b> EURUSD, GBPUSD" in message
        assert "<b>Inputs:</b> DGS10" in message
        assert "Reply: approve h1 | reject h1" in message
        assert priority == 0

    def test_skipped_entry_doesnt_reach_implementer(
        self, loop_paths: dict[str, Path],
    ) -> None:
        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store, new_extracts=[("h1", "# A\n")],
        )
        idea = _FakeIdeaAgent(responses={
            "h1": {"status": "PROPOSED", "slug": "alpha"},
        })
        impl = _FakeImplementer(
            responses={"alpha": {"status": "IMPLEMENTED"}},
            candidate_dir=loop_paths["candidates"],
        )
        loop = ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=impl,  # type: ignore[arg-type]
            debate_orchestrator=_FakeOrchestrator({}),  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
            notify_fn=_silent_notifier,
        )
        loop.run()
        # Operator skips
        state = load_state(loop_paths["state"])
        state.ideas_processed["h1"]["status"] = "SKIPPED"
        state.ideas_processed["h1"]["reason"] = "duplicate of existing"
        save_state(state, loop_paths["state"])

        loop.run()
        assert impl.calls == []  # implementer NEVER ran

    def test_auto_skip_after_timeout(
        self, loop_paths: dict[str, Path],
    ) -> None:
        from datetime import UTC, datetime, timedelta

        # First run "happens" at t=0; gate1 timeout is 60s for the test.
        clock_time = [datetime(2026, 1, 1, tzinfo=UTC)]

        def fake_clock() -> datetime:
            return clock_time[0]

        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store, new_extracts=[("h1", "# A\n")],
        )
        idea = _FakeIdeaAgent(responses={
            "h1": {"status": "PROPOSED", "slug": "alpha"},
        })
        impl = _FakeImplementer({}, loop_paths["candidates"])
        loop = ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=impl,  # type: ignore[arg-type]
            debate_orchestrator=_FakeOrchestrator({}),  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
            notify_fn=_silent_notifier,
            gate1_timeout_sec=60.0,
            clock=fake_clock,
        )

        # First run: PROPOSED → PENDING (with pending_since=t0)
        loop.run()
        state = load_state(loop_paths["state"])
        assert state.ideas_processed["h1"]["status"] == (
            "PENDING_OPERATOR_APPROVAL"
        )

        # Advance clock past the timeout. Don't re-ingest (idea already
        # processed) — second run only does the GATE 1 sweep + downstream.
        clock_time[0] = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
            seconds=120,
        )
        # Make ingest a no-op for pass 2 by clearing its new_extracts
        ingest.new_extracts = []

        summary = loop.run()
        assert summary.gate1_auto_skipped_expired == 1
        state = load_state(loop_paths["state"])
        assert state.ideas_processed["h1"]["status"] == "SKIPPED"
        assert "auto-SKIPPED" in state.ideas_processed["h1"]["reason"]
        # Implementer never reached
        assert impl.calls == []

    def test_notifier_failure_doesnt_kill_run(
        self, loop_paths: dict[str, Path],
    ) -> None:
        def boom(*_a: Any, **_kw: Any) -> Any:
            raise ConnectionError("telegram unreachable")

        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store, new_extracts=[("h1", "# A\n")],
        )
        idea = _FakeIdeaAgent(responses={
            "h1": {"status": "PROPOSED", "slug": "alpha"},
        })
        loop = ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=_FakeImplementer({}, loop_paths["candidates"]),  # type: ignore[arg-type]
            debate_orchestrator=_FakeOrchestrator({}),  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
            notify_fn=boom,
        )
        # Should not raise — notifier failure is contained
        summary = loop.run()
        assert summary.ideas_proposed == 1
        # Notification didn't actually send
        assert summary.gate1_notifications_sent == 0
        # State still records the entry as PENDING
        state = load_state(loop_paths["state"])
        assert state.ideas_processed["h1"]["status"] == (
            "PENDING_OPERATOR_APPROVAL"
        )


# --------------------------------------------------------------------------- #
# GATE 2 — pre-deploy operator confirmation (CL-yta6)
# --------------------------------------------------------------------------- #


def _build_promote_loop(
    loop_paths: dict[str, Path],
    *,
    bull: Position = Position.PROMOTE,
    bear: Position = Position.PROMOTE,
    registrar: Any = None,
    notify_fn: Any = None,
    clock: Any = None,
    gate2_timeout_sec: float | None = None,
    impl_spec: dict[str, Any] | None = None,
) -> tuple[ResearchLoop, _FakeRegistrar]:
    """Helper: build a loop wired through to a PROMOTE verdict so the
    test can drive GATE 2 transitions."""
    store = _FakeExtractStore(root=loop_paths["extracts"])
    ingest = _FakeIngestRunner(
        extract_store=store, new_extracts=[("h1", "# A\n")],
    )
    idea = _FakeIdeaAgent(responses={
        "h1": {"status": "PROPOSED", "slug": "alpha"},
    })
    impl = _FakeImplementer(
        responses={"alpha": impl_spec or {"status": "IMPLEMENTED"}},
        candidate_dir=loop_paths["candidates"],
    )
    transcript = loop_paths["transcripts"] / "alpha" / "transcript.md"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("(stub)")
    orch = _FakeOrchestrator(results={
        "alpha": _make_debate_result(
            "alpha", bull, bear, transcript=transcript,
        ),
    })
    used_registrar = registrar if registrar is not None else _FakeRegistrar()
    kwargs: dict[str, Any] = {
        "notify_fn": notify_fn or _silent_notifier,
        "registrar": used_registrar,
    }
    if clock is not None:
        kwargs["clock"] = clock
    if gate2_timeout_sec is not None:
        kwargs["gate2_timeout_sec"] = gate2_timeout_sec
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
        **kwargs,
    )
    return loop, used_registrar


class TestGate2:
    def test_promote_holds_pending_and_fires_priority1_notification(
        self, loop_paths: dict[str, Path],
    ) -> None:
        notifications: list[tuple[str, str, int]] = []

        def recorder(title: str, message: str, priority: int) -> Any:
            from src.research.notifications import DispatchResult
            notifications.append((title, message, priority))
            r = DispatchResult()
            r.telegram_attempted = True
            r.telegram_succeeded = True
            return r

        loop, registrar = _build_promote_loop(
            loop_paths, notify_fn=recorder,
            impl_spec={
                "status": "IMPLEMENTED",
                # Real-shaped candidate code so the notification can
                # extract the instruments (execution symbol first).
                "code": (
                    "class Strategy:\n"
                    "    symbols = ['DXY', 'EURUSD']\n"
                    "    execution_symbol = 'EURUSD'\n"
                    "    def fit(self, data):\n        return self\n"
                    "    def generate_signals(self, data):\n"
                    "        return None\n"
                ),
                "report": {"oos_metrics": {
                    "sharpe": 0.45, "max_drawdown": -0.021, "n_trades": 12,
                }},
            },
        )
        # Pass 1: gate 1 holds.
        loop.run()
        _approve_all_pending(loop_paths["state"])
        # Pass 2: implementer + debate. Debate produces PROMOTE → gate 2
        # holds at PENDING_DEPLOY_CONFIRMATION.
        summary = loop.run()
        assert summary.verdicts_promote == 1
        assert summary.gate2_notifications_sent == 1
        # Registrar NOT called on PENDING entries
        assert registrar.calls == []
        # State reflects pending deploy
        state = load_state(loop_paths["state"])
        entry = state.debates_completed["alpha"]
        assert entry["deploy_status"] == "PENDING_DEPLOY_CONFIRMATION"
        assert "pending_since" in entry
        assert entry["candidate_report_path"]
        # Notification: only the GATE 2 one is priority=1 (deploy
        # decision); GATE 1 was priority=0 on pass 1. Phone-first HTML
        # body (CL-frn7): slug, instruments from the candidate code,
        # key backtest numbers, reply line.
        gate2_notifs = [n for n in notifications if "GATE 2" in n[0]]
        assert len(gate2_notifs) == 1
        title, message, priority = gate2_notifs[0]
        assert priority == 1
        assert "<b>alpha</b>" in message
        assert "<b>Trades:</b> EURUSD, DXY" in message
        assert "Sharpe 0.45" in message
        assert "max DD -2.10%" in message
        assert "12 trades" in message
        assert "allocation=0" in message
        assert "Reply: approve alpha | reject alpha" in message

    def test_approved_runs_registrar_on_next_pass(
        self, loop_paths: dict[str, Path],
    ) -> None:
        loop, registrar = _build_promote_loop(loop_paths)
        loop.run()  # gate 1
        _approve_all_pending(loop_paths["state"])
        loop.run()  # gate 2 holds at PENDING

        # Operator approves deploy
        state = load_state(loop_paths["state"])
        state.debates_completed["alpha"]["deploy_status"] = (
            "DEPLOY_APPROVED"
        )
        save_state(state, loop_paths["state"])

        summary = loop.run()
        assert summary.deployments_succeeded == 1
        assert registrar.calls == [{
            "slug": "alpha",
            "report": str(
                loop_paths["candidates"] / "alpha.json",
            ),
            "transcript": str(
                loop_paths["transcripts"] / "alpha" / "transcript.md",
            ),
            "reason": "All gates pass; both reviewers PROMOTE",
        }]
        state = load_state(loop_paths["state"])
        assert state.debates_completed["alpha"]["deploy_status"] == "DEPLOYED"
        assert state.debates_completed["alpha"]["pr_url"]
        assert state.debates_completed["alpha"]["branch_name"]

    def test_rejected_doesnt_run_registrar(
        self, loop_paths: dict[str, Path],
    ) -> None:
        loop, registrar = _build_promote_loop(loop_paths)
        loop.run()
        _approve_all_pending(loop_paths["state"])
        loop.run()  # gate 2 holds

        # Operator rejects deploy
        state = load_state(loop_paths["state"])
        state.debates_completed["alpha"]["deploy_status"] = (
            "DEPLOY_REJECTED"
        )
        state.debates_completed["alpha"]["deploy_reason"] = (
            "duplicates carry strategy"
        )
        save_state(state, loop_paths["state"])

        summary = loop.run()
        assert summary.deployments_succeeded == 0
        assert summary.deployments_failed == 0
        assert registrar.calls == []

    def test_auto_reject_after_timeout(
        self, loop_paths: dict[str, Path],
    ) -> None:
        from datetime import UTC, datetime, timedelta

        clock_time = [datetime(2026, 1, 1, tzinfo=UTC)]

        def fake_clock() -> datetime:
            return clock_time[0]

        loop, registrar = _build_promote_loop(
            loop_paths, clock=fake_clock, gate2_timeout_sec=60.0,
        )
        loop.run()
        _approve_all_pending(loop_paths["state"])
        loop.run()  # gate 2 holds at t=0

        state = load_state(loop_paths["state"])
        assert state.debates_completed["alpha"]["deploy_status"] == (
            "PENDING_DEPLOY_CONFIRMATION"
        )

        # Advance clock past timeout
        clock_time[0] = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
            seconds=120,
        )
        summary = loop.run()
        assert summary.gate2_auto_rejected_expired == 1
        state = load_state(loop_paths["state"])
        assert state.debates_completed["alpha"]["deploy_status"] == (
            "DEPLOY_REJECTED"
        )
        assert "auto-REJECTED" in state.debates_completed["alpha"][
            "deploy_reason"]
        # Registrar never ran
        assert registrar.calls == []

    def test_registrar_failure_marks_deploy_failed(
        self, loop_paths: dict[str, Path],
    ) -> None:
        bad_registrar = _FakeRegistrar(
            succeeded=False, error="git: branch already exists",
        )
        loop, _ = _build_promote_loop(loop_paths, registrar=bad_registrar)
        loop.run()
        _approve_all_pending(loop_paths["state"])
        loop.run()

        # Operator approves
        state = load_state(loop_paths["state"])
        state.debates_completed["alpha"]["deploy_status"] = (
            "DEPLOY_APPROVED"
        )
        save_state(state, loop_paths["state"])

        summary = loop.run()
        assert summary.deployments_failed == 1
        assert summary.deployments_succeeded == 0
        assert any(
            "branch already exists" in e for e in summary.errors
        )
        state = load_state(loop_paths["state"])
        assert state.debates_completed["alpha"]["deploy_status"] == (
            "DEPLOY_FAILED"
        )

    def test_non_promote_verdict_no_gate2(
        self, loop_paths: dict[str, Path],
    ) -> None:
        # Mixed positions → ESCALATE; no gate 2 entry created
        loop, registrar = _build_promote_loop(
            loop_paths,
            bull=Position.PROMOTE, bear=Position.REJECT,  # ESCALATE
        )
        loop.run()
        _approve_all_pending(loop_paths["state"])
        summary = loop.run()
        assert summary.verdicts_escalate == 1
        assert summary.verdicts_promote == 0
        assert summary.gate2_notifications_sent == 0
        state = load_state(loop_paths["state"])
        # Entry exists but has no deploy_status
        assert "deploy_status" not in state.debates_completed["alpha"]
        # Registrar never ran
        assert registrar.calls == []


# --------------------------------------------------------------------------- #
# ESCALATE side-effect (CL-o2vb)
# --------------------------------------------------------------------------- #


class TestEscalateAlert:
    def test_escalate_fires_priority1_alert_with_context(
        self, loop_paths: dict[str, Path],
    ) -> None:
        notifications: list[tuple[str, str, int]] = []

        def recorder(title: str, message: str, priority: int) -> Any:
            from src.research.notifications import DispatchResult
            notifications.append((title, message, priority))
            r = DispatchResult()
            r.telegram_attempted = True
            r.telegram_succeeded = True
            return r

        # Mixed positions → ESCALATE
        loop, _ = _build_promote_loop(
            loop_paths,
            bull=Position.PROMOTE, bear=Position.REJECT,
            notify_fn=recorder,
        )
        loop.run()
        _approve_all_pending(loop_paths["state"])
        summary = loop.run()
        assert summary.verdicts_escalate == 1
        assert summary.escalate_notifications_sent == 1

        escalate_notifs = [n for n in notifications if "ESCALATE" in n[0]]
        assert len(escalate_notifs) == 1
        title, message, priority = escalate_notifs[0]
        assert priority == 1
        assert "alpha" in title
        # Body has bull/bear positions and links to transcript + report
        assert "Bull: PROMOTE" in message
        assert "Bear: REJECT" in message
        assert "transcript" in message.lower()
        assert "report" in message.lower()

    def test_escalate_notification_deduped_across_runs(
        self, loop_paths: dict[str, Path],
    ) -> None:
        # Outer loop skips already-debated slugs, so an ESCALATE alert
        # naturally fires only once across reruns of the same slug.
        notifications: list[tuple[str, str, int]] = []

        def recorder(title: str, message: str, priority: int) -> Any:
            from src.research.notifications import DispatchResult
            notifications.append((title, message, priority))
            r = DispatchResult()
            r.telegram_attempted = True
            return r

        loop, _ = _build_promote_loop(
            loop_paths,
            bull=Position.PROMOTE, bear=Position.REJECT,
            notify_fn=recorder,
        )
        loop.run()
        _approve_all_pending(loop_paths["state"])
        loop.run()  # ESCALATE fires here
        # Re-run twice; ESCALATE notification should NOT fire again.
        loop.run()
        loop.run()
        escalate_count = sum(
            1 for n in notifications if "ESCALATE" in n[0]
        )
        assert escalate_count == 1

    def test_escalate_notifier_failure_doesnt_kill_run(
        self, loop_paths: dict[str, Path],
    ) -> None:
        def boom(*_a: Any, **_kw: Any) -> Any:
            raise ConnectionError("telegram unreachable")

        loop, _ = _build_promote_loop(
            loop_paths,
            bull=Position.PROMOTE, bear=Position.REJECT,
            notify_fn=boom,
        )
        loop.run()
        _approve_all_pending(loop_paths["state"])
        summary = loop.run()  # ESCALATE alert raises, but loop continues
        assert summary.verdicts_escalate == 1
        assert summary.escalate_notifications_sent == 0
        # State still recorded the verdict
        state = load_state(loop_paths["state"])
        assert state.debates_completed["alpha"]["verdict"] == "ESCALATE"


# =============================================================================
# Transient-error retry (CL-837v)
# =============================================================================

from src.research.loop import (  # noqa: E402
    ERROR_RETRY_ATTEMPT_CAP,
    _error_entry,
    _is_transient_error,
    _should_retry,
)


class _APITimeoutError(Exception):
    """SDK-style name — must classify transient via the name fragment."""


class _FlakyIdeaAgent(_FakeIdeaAgent):
    """Raises ``exc`` for the first ``fail_times`` calls, then delegates."""

    def __init__(self, responses, exc: Exception, fail_times: int = 1) -> None:
        super().__init__(responses)
        self._exc = exc
        self._fail_times = fail_times

    def ideate(self, extract_path, **kwargs):
        if len(self.calls) < self._fail_times:
            self.calls.append(extract_path)
            raise self._exc
        return super().ideate(extract_path, **kwargs)


class TestTransientErrorRetry:
    def test_transient_classification(self) -> None:
        assert _is_transient_error(TimeoutError("t"))
        assert _is_transient_error(ConnectionError("c"))
        assert _is_transient_error(_APITimeoutError("sdk"))  # name fragment
        assert not _is_transient_error(ValueError("bad json"))
        assert not _is_transient_error(KeyError("schema"))

    def test_error_entry_and_retry_cap(self) -> None:
        e1 = _error_entry(TimeoutError("t"), None)
        assert e1["transient"] is True and e1["attempts"] == 1
        assert _should_retry(e1)
        e2 = _error_entry(TimeoutError("t"), e1)
        assert e2["attempts"] == 2
        e_cap = _error_entry(TimeoutError("t"), {"attempts": ERROR_RETRY_ATTEMPT_CAP - 1})
        assert e_cap["attempts"] == ERROR_RETRY_ATTEMPT_CAP
        assert not _should_retry(e_cap)  # cap reached
        assert not _should_retry(_error_entry(ValueError("v"), None))  # content
        assert not _should_retry(None)
        assert not _should_retry({"status": "IMPLEMENTED"})

    def _make_loop(self, loop_paths, idea):
        store = _FakeExtractStore(root=loop_paths["extracts"])
        ingest = _FakeIngestRunner(
            extract_store=store, new_extracts=[("hash1", "# Paper A\nbody")],
        )
        return ResearchLoop(
            ingest_runner=ingest, idea_agent=idea,  # type: ignore[arg-type]
            implementer=_FakeImplementer(
                responses={}, candidate_dir=loop_paths["candidates"]),  # type: ignore[arg-type]
            debate_orchestrator=_FakeOrchestrator(results={}),  # type: ignore[arg-type]
            rules_loader=lambda: [], feed_configs=[],
            extract_store=store,  # type: ignore[arg-type]
            state_path=loop_paths["state"],
            runs_dir=loop_paths["runs"],
            hypothesis_dir=loop_paths["hypotheses"],
            candidate_dir=loop_paths["candidates"],
            notify_fn=_silent_notifier,
        )

    def test_idea_phase_retries_transient_then_succeeds(
        self, loop_paths: dict[str, Path],
    ) -> None:
        idea = _FlakyIdeaAgent(
            responses={"hash1": {"status": "PROPOSED", "slug": "alpha"}},
            exc=TimeoutError("network outage"), fail_times=1,
        )
        loop = self._make_loop(loop_paths, idea)
        loop.run()  # run 1: transient ERROR recorded
        state1 = json.loads(loop_paths["state"].read_text())
        entry = state1["ideas_processed"]["hash1"]
        assert entry["status"] == "ERROR"
        assert entry["transient"] is True and entry["attempts"] == 1
        loop.run()  # run 2: retried -> PROPOSED (previously wedged forever)
        state2 = json.loads(loop_paths["state"].read_text())
        assert state2["ideas_processed"]["hash1"]["status"] == (
            "PENDING_OPERATOR_APPROVAL")
        assert len(idea.calls) == 2

    def test_idea_phase_content_error_stays_terminal(
        self, loop_paths: dict[str, Path],
    ) -> None:
        idea = _FlakyIdeaAgent(
            responses={"hash1": {"status": "PROPOSED", "slug": "alpha"}},
            exc=ValueError("unparseable output"), fail_times=99,
        )
        loop = self._make_loop(loop_paths, idea)
        loop.run()
        loop.run()  # must NOT retry a content failure
        assert len(idea.calls) == 1
        state = json.loads(loop_paths["state"].read_text())
        assert state["ideas_processed"]["hash1"]["transient"] is False
