"""Research loop (CL-x561) — autonomous orchestrator that ties the
research pipeline phases into a single cron-friendly entry.

Each ``ResearchLoop.run()`` call walks the pipeline:

  1. **Ingest** — pull new papers, extract, persist to
     ``data/research/extracts/{hash}.md``. Natural dedup via the
     paper-hash filenames (CL-2klj's ExtractStore).
  2. **Idea**  — for each NEW extract (one we haven't ideated yet), call
     the Idea Generator. PROPOSED briefs land in
     ``docs/research/hypotheses/{slug}.md``; DECLINED briefs are logged.
  3. **Implement** — for each PROPOSED hypothesis (not yet implemented),
     call the Implementer. IMPLEMENTED candidates land in
     ``src/strategies/_experimental/{slug}.py`` plus a candidate report
     at ``reports/candidates/{slug}.json``. REJECTED implementations
     are logged.
  4. **Debate** — for each IMPLEMENTED candidate (not yet debated), run
     the promotion-review debate via the orchestrator. Transcripts are
     persisted under ``docs/research/debates/{slug}/``.
  5. **Verdict** — compute_verdict() over the candidate report +
     debater positions; PROMOTE/REJECT/ESCALATE recorded in state.

Idempotency: each phase reads the on-disk state file at the start and
records its outputs as it proceeds. A second invocation skips any
extract/hypothesis/candidate already processed in a prior run. A run
that crashes mid-pipeline leaves a partially-updated state file, and
the next run picks up exactly where the previous one stopped.

Side-effects (paper-shadow registration, Pushover/Telegram alerting)
are intentionally out of scope here — those are separate beads
(CL-3xn1, CL-o2vb) that read the loop's verdict log and act on it.
The loop's job is to populate that log.

Designed for dependency injection: the constructor takes already-built
ingest_runner / idea / implementer / orchestrator / verdict_rules so
tests run with mocked components and no LLM/network/Postgres.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.agents.idea import IdeaGenerator, IdeaStatus
from src.research.agents.implementer import Implementer, ImplementerStatus
from src.research.agents.reviewer import Position
from src.research.ingest import (
    DEFAULT_EXTRACT_ROOT,
    ExtractStore,
    FeedConfig,
    IngestRunner,
)
from src.research.notifications import DispatchResult, notify_operator
from src.research.orchestrator import DebateOrchestrator
from src.research.promote import PromoteRegistrar, RegistrationResult
from src.research.verdict import (
    ParsedRule,
    Verdict,
    VerdictResult,
    compute_verdict,
)

logger = logging.getLogger(__name__)


# Default state-file location. Kept under data/, which is gitignored.
DEFAULT_STATE_PATH: Path = Path("data/research/state.json")
DEFAULT_RUNS_DIR: Path = Path("data/research/runs")
DEFAULT_HYPOTHESIS_DIR: Path = Path("docs/research/hypotheses")
DEFAULT_CANDIDATE_DIR: Path = Path("reports/candidates")
DEFAULT_EXPERIMENTAL_CODE_DIR: Path = Path("src/strategies/_experimental")

# GATE 1: pre-research operator approval (CL-0hr3).
#
# After the Idea Agent emits PROPOSED, the loop holds the hypothesis at
# PENDING_OPERATOR_APPROVAL until the operator marks it APPROVED or
# SKIPPED (via the dashboard or scripts/research_approve.py). Default
# timeout: 7 days; pending entries older than this auto-SKIP so the
# pipeline doesn't pile up indefinitely. Configurable per-run.
GATE1_PENDING_STATUS: str = "PENDING_OPERATOR_APPROVAL"
GATE1_APPROVED_STATUS: str = "APPROVED"
GATE1_SKIPPED_STATUS: str = "SKIPPED"
DEFAULT_GATE1_TIMEOUT_SEC: float = 7 * 24 * 60 * 60  # 7 days

# GATE 2: pre-deploy operator confirmation (CL-yta6).
#
# After verdict=PROMOTE, the loop holds at deploy_status=
# PENDING_DEPLOY_CONFIRMATION until the operator marks DEPLOY_APPROVED
# (registrar runs on the next pass) or DEPLOY_REJECTED (strategy
# archived). Default timeout: 7 days; expired pending entries auto-
# REJECT (safer-default — a deploy decision should never be made by
# silence). Configurable per-run.
GATE2_PENDING_STATUS: str = "PENDING_DEPLOY_CONFIRMATION"
GATE2_APPROVED_STATUS: str = "DEPLOY_APPROVED"
GATE2_REJECTED_STATUS: str = "DEPLOY_REJECTED"
GATE2_DEPLOYED_STATUS: str = "DEPLOYED"
GATE2_DEPLOY_FAILED_STATUS: str = "DEPLOY_FAILED"
DEFAULT_GATE2_TIMEOUT_SEC: float = 7 * 24 * 60 * 60  # 7 days


# --------------------------------------------------------------------------- #
# State persistence
# --------------------------------------------------------------------------- #


@dataclass
class LoopState:
    """Cross-run state. Each phase records which inputs it has already
    consumed so the next invocation can skip them.

    Shape is intentionally flat JSON: easy to inspect with `jq`, easy
    to reset by deleting the file."""

    schema_version: int = 1
    # extract_hash → {"status": "PROPOSED"|"DECLINED",
    #                 "slug": str|None, "reason": str}
    ideas_processed: dict[str, dict[str, Any]] = field(default_factory=dict)
    # slug → {"status": "IMPLEMENTED"|"REJECTED", "reason": str,
    #         "code_path": str|None, "report_path": str|None}
    candidates_processed: dict[str, dict[str, Any]] = field(default_factory=dict)
    # slug → {"verdict": "PROMOTE"|"REJECT"|"ESCALATE", "reason": str,
    #         "transcript_path": str, "bull": str, "bear": str}
    debates_completed: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_state(path: Path | str = DEFAULT_STATE_PATH) -> LoopState:
    """Read state JSON from disk. Returns an empty LoopState if the
    file doesn't exist (fresh run)."""
    p = Path(path)
    if not p.exists():
        return LoopState()
    raw = json.loads(p.read_text())
    return LoopState(
        schema_version=raw.get("schema_version", 1),
        ideas_processed=raw.get("ideas_processed", {}),
        candidates_processed=raw.get("candidates_processed", {}),
        debates_completed=raw.get("debates_completed", {}),
    )


def save_state(state: LoopState, path: Path | str = DEFAULT_STATE_PATH) -> None:
    """Persist state JSON. Creates parent dirs if missing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2, default=str))


# --------------------------------------------------------------------------- #
# Run summary
# --------------------------------------------------------------------------- #


@dataclass
class RunSummary:
    """Per-run summary written to data/research/runs/{ts}.json."""

    started_at: str
    finished_at: str = ""
    extracts_new: int = 0
    extracts_failed: int = 0
    ideas_proposed: int = 0
    ideas_declined: int = 0
    # GATE 1 (CL-0hr3) bookkeeping
    gate1_notifications_sent: int = 0
    gate1_auto_skipped_expired: int = 0
    candidates_implemented: int = 0
    candidates_rejected: int = 0
    debates_run: int = 0
    verdicts_promote: int = 0
    verdicts_reject: int = 0
    verdicts_escalate: int = 0
    # GATE 2 (CL-yta6) bookkeeping
    gate2_notifications_sent: int = 0
    gate2_auto_rejected_expired: int = 0
    deployments_succeeded: int = 0
    deployments_failed: int = 0
    # ESCALATE side-effect (CL-o2vb)
    escalate_notifications_sent: int = 0
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# ResearchLoop
# --------------------------------------------------------------------------- #


# Type alias for the verdict-rules loader. Default reads + parses the
# configured REVIEW_RULES.md once per run; tests inject a fixed list.
RulesLoader = Callable[[], list[ParsedRule]]

# GATE 1 notifier. Default = production pushover/telegram via
# src.research.notifications.notify_operator; tests inject a recorder.
# Signature: (title, message, priority) → DispatchResult.
NotifyFn = Callable[[str, str, int], DispatchResult]


class ResearchLoop:
    """One full pipeline pass per ``run()`` call.

    All sub-components are required at construction time — no lazy
    wiring inside the loop, so config errors fail loud at boot rather
    than mid-run. The CLI builds these from configs/research_agents.yaml
    + configs/paper_streams.yaml; tests inject mocks.
    """

    def __init__(
        self,
        ingest_runner: IngestRunner,
        idea_agent: IdeaGenerator,
        implementer: Implementer,
        debate_orchestrator: DebateOrchestrator,
        rules_loader: RulesLoader,
        feed_configs: list[FeedConfig],
        extract_store: ExtractStore | None = None,
        state_path: Path | str = DEFAULT_STATE_PATH,
        runs_dir: Path | str = DEFAULT_RUNS_DIR,
        hypothesis_dir: Path | str = DEFAULT_HYPOTHESIS_DIR,
        candidate_dir: Path | str = DEFAULT_CANDIDATE_DIR,
        experimental_code_dir: Path | str = DEFAULT_EXPERIMENTAL_CODE_DIR,
        notify_fn: NotifyFn | None = None,
        gate1_timeout_sec: float = DEFAULT_GATE1_TIMEOUT_SEC,
        gate2_timeout_sec: float = DEFAULT_GATE2_TIMEOUT_SEC,
        registrar: PromoteRegistrar | None = None,
        backtest_runner: Callable[[Path], dict[str, Any]] | None = None,
        auto_approve: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.ingest_runner = ingest_runner
        self.idea_agent = idea_agent
        self.implementer = implementer
        self.debate_orchestrator = debate_orchestrator
        self.rules_loader = rules_loader
        self.feed_configs = feed_configs
        self.extract_store = extract_store or ExtractStore(
            root=DEFAULT_EXTRACT_ROOT,
        )
        self.state_path = Path(state_path)
        self.runs_dir = Path(runs_dir)
        self.hypothesis_dir = Path(hypothesis_dir)
        self.candidate_dir = Path(candidate_dir)
        self.experimental_code_dir = Path(experimental_code_dir)
        # Default notifier hits prod channels (no-op when env-vars unset).
        self.notify_fn: NotifyFn = notify_fn or (
            lambda t, m, p: notify_operator(title=t, message=m, priority=p)
        )
        self.gate1_timeout_sec = gate1_timeout_sec
        self.gate2_timeout_sec = gate2_timeout_sec
        # Default registrar runs git + gh; tests/dev environments can
        # set skip_git=True or pass a mocked registrar.
        self.registrar = registrar or PromoteRegistrar()
        # Real backtest_runner (CL-p9ix). When None, the implementer
        # writes a candidate report with empty backtest_metrics and the
        # verdict engine ESCALATEs everything for missing metrics —
        # which is the correct behavior for a misconfigured loop. The
        # CLI wires up src/research/backtest_runner.py:make_backtest_runner
        # by default; tests inject fakes.
        self.backtest_runner = backtest_runner
        # Operator-bypass switch for --dry-run preflights. Flips
        # PENDING entries to APPROVED between phases so a single
        # invocation walks the entire pipeline including the registrar.
        # Never set in production — the gates exist for a reason.
        self.auto_approve = auto_approve
        # Injected clock so tests can simulate the timeout window.
        self._clock: Callable[[], datetime] = clock or (
            lambda: datetime.now(UTC)
        )

    # ------------------------------------------------------------------ #
    # Top-level entry
    # ------------------------------------------------------------------ #

    def run(self) -> RunSummary:
        """Execute one full pipeline pass. State is saved after each
        phase so a mid-pipeline crash leaves the loop resumable."""
        started = datetime.now(UTC).isoformat(timespec="seconds")
        summary = RunSummary(started_at=started)

        state = load_state(self.state_path)
        try:
            self._phase_ingest(summary)
            save_state(state, self.state_path)

            self._phase_ideas(state, summary)
            save_state(state, self.state_path)

            self._phase_gate1(state, summary)
            if self.auto_approve:
                self._auto_approve_gate1(state)
            save_state(state, self.state_path)

            self._phase_implement(state, summary)
            save_state(state, self.state_path)

            self._phase_debate(state, summary)
            if self.auto_approve:
                self._auto_approve_gate2(state)
            save_state(state, self.state_path)

            self._phase_gate2(state, summary)
            save_state(state, self.state_path)
        except Exception as exc:
            logger.exception("research loop crashed")
            summary.errors.append(f"{type(exc).__name__}: {exc}")
            save_state(state, self.state_path)
            raise
        finally:
            summary.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
            self._persist_run_summary(summary, started)

        return summary

    # ------------------------------------------------------------------ #
    # Phase 1 — ingest
    # ------------------------------------------------------------------ #

    def _phase_ingest(self, summary: RunSummary) -> None:
        """Pull new papers via the IngestRunner. Dedup is automatic via
        ExtractStore's hash-keyed filenames."""
        logger.info("phase 1: ingest — %d feeds", len(self.feed_configs))
        ingest_summary = self.ingest_runner.run(self.feed_configs)
        summary.extracts_new = ingest_summary.papers_extracted
        summary.extracts_failed = ingest_summary.papers_extract_failed

    # ------------------------------------------------------------------ #
    # Phase 2 — idea agent
    # ------------------------------------------------------------------ #

    def _phase_ideas(self, state: LoopState, summary: RunSummary) -> None:
        """For each extract on disk that we haven't ideated yet, run the
        Idea Generator. Records PROPOSED slug or DECLINED reason in
        state.ideas_processed keyed by extract hash."""
        extract_root = self.extract_store.root
        logger.info("phase 2: ideas — scanning %s", extract_root)
        for extract_path in sorted(extract_root.glob("*.md")):
            extract_hash = extract_path.stem
            if extract_hash in state.ideas_processed:
                continue
            try:
                result = self.idea_agent.ideate(
                    extract_path=extract_path,
                    hypothesis_dir=self.hypothesis_dir,
                )
            except Exception as exc:
                logger.exception(
                    "idea agent failed for extract %s", extract_hash,
                )
                state.ideas_processed[extract_hash] = {
                    "status": "ERROR",
                    "slug": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                summary.errors.append(
                    f"idea/{extract_hash}: {type(exc).__name__}: {exc}"
                )
                continue
            if result.status == IdeaStatus.PROPOSED:
                summary.ideas_proposed += 1
                # GATE 1: hold for operator approval rather than feed
                # straight into the implementer.
                pending_since = self._clock().isoformat(timespec="seconds")
                state.ideas_processed[extract_hash] = {
                    "status": GATE1_PENDING_STATUS,
                    "slug": result.strategy_slug,
                    "reason": "",
                    "pending_since": pending_since,
                    "hypothesis_path": (
                        str(result.hypothesis_path)
                        if result.hypothesis_path else None
                    ),
                }
                self._notify_gate1(result, summary, extract_hash)
            else:
                summary.ideas_declined += 1
                state.ideas_processed[extract_hash] = {
                    "status": "DECLINED",
                    "slug": result.strategy_slug,
                    "reason": result.reason,
                }

    # ------------------------------------------------------------------ #
    # GATE 1 — pre-research operator approval (CL-0hr3)
    # ------------------------------------------------------------------ #

    def _phase_gate1(self, state: LoopState, summary: RunSummary) -> None:
        """Auto-skip pending entries older than the configured timeout.

        The actual approval transition (PENDING → APPROVED / SKIPPED) is
        operator-driven via scripts/research_approve.py or the dashboard
        — the loop only enforces the timeout safety so the queue
        doesn't grow unbounded when the operator is on vacation.
        """
        now = self._clock()
        for extract_hash, entry in state.ideas_processed.items():
            if entry.get("status") != GATE1_PENDING_STATUS:
                continue
            pending_since_str = entry.get("pending_since")
            if not pending_since_str:
                continue
            try:
                pending_since = datetime.fromisoformat(pending_since_str)
            except ValueError:
                logger.warning(
                    "ideas_processed[%s].pending_since malformed: %r",
                    extract_hash, pending_since_str,
                )
                continue
            elapsed = (now - pending_since).total_seconds()
            if elapsed > self.gate1_timeout_sec:
                entry["status"] = GATE1_SKIPPED_STATUS
                entry["reason"] = (
                    f"auto-SKIPPED: pending {elapsed/86400:.1f}d > "
                    f"{self.gate1_timeout_sec/86400:.1f}d operator timeout"
                )
                summary.gate1_auto_skipped_expired += 1
                logger.info(
                    "GATE 1 auto-skipped expired pending entry %s",
                    extract_hash,
                )

    def _notify_gate1(
        self,
        result: Any,
        summary: RunSummary,
        extract_hash: str,
    ) -> None:
        """Fire pre-research approval notification. Best-effort —
        notifier failures don't kill the loop, they just get logged."""
        title = f"GATE 1: research approval — {result.strategy_slug}"
        body_preview = (
            result.raw_text[:600] if result.raw_text else "(no preview)"
        )
        message = (
            f"New hypothesis pending operator GO/SKIP.\n\n"
            f"Slug: {result.strategy_slug}\n"
            f"Extract: {extract_hash}\n"
            f"Hypothesis: {result.hypothesis_path}\n\n"
            f"--- preview ---\n{body_preview}\n"
            f"\nApprove via:\n"
            f"  python -m scripts.research_approve --slug {result.strategy_slug} "
            f"--action GO\n"
            f"Or SKIP with --action SKIP --reason 'why'."
        )
        try:
            disp = self.notify_fn(title, message, 0)
        except Exception as exc:
            logger.warning(
                "GATE 1 notification dispatch raised: %s: %s",
                type(exc).__name__, exc,
            )
            return
        if disp.any_attempted:
            summary.gate1_notifications_sent += 1

    # ------------------------------------------------------------------ #
    # Phase 3 — implementer
    # ------------------------------------------------------------------ #

    def _phase_implement(self, state: LoopState, summary: RunSummary) -> None:
        """For each APPROVED hypothesis not yet implemented (passed
        GATE 1), run the Implementer. Records IMPLEMENTED/REJECTED
        status in state.candidates_processed keyed by slug."""
        logger.info("phase 3: implementer")
        for entry in state.ideas_processed.values():
            if entry.get("status") != GATE1_APPROVED_STATUS:
                continue
            slug = entry.get("slug")
            if not slug or slug in state.candidates_processed:
                continue
            hyp_path = self.hypothesis_dir / f"{slug}.md"
            if not hyp_path.exists():
                logger.warning(
                    "hypothesis file %s missing for slug %s — skipping",
                    hyp_path, slug,
                )
                state.candidates_processed[slug] = {
                    "status": "ERROR",
                    "reason": f"hypothesis file not found at {hyp_path}",
                    "code_path": None,
                    "report_path": None,
                }
                continue
            try:
                result = self.implementer.implement(
                    hypothesis_path=hyp_path,
                    strategy_slug=slug,
                    backtest_runner=self.backtest_runner,
                    code_dir=self.experimental_code_dir,
                    report_dir=self.candidate_dir,
                )
            except Exception as exc:
                logger.exception("implementer failed for slug %s", slug)
                state.candidates_processed[slug] = {
                    "status": "ERROR",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "code_path": None,
                    "report_path": None,
                }
                summary.errors.append(
                    f"implement/{slug}: {type(exc).__name__}: {exc}"
                )
                continue
            if result.status == ImplementerStatus.IMPLEMENTED:
                summary.candidates_implemented += 1
                state.candidates_processed[slug] = {
                    "status": "IMPLEMENTED",
                    "reason": "",
                    "code_path": (
                        str(result.code_path) if result.code_path else None
                    ),
                    "report_path": (
                        str(result.report_path) if result.report_path else None
                    ),
                }
            else:
                summary.candidates_rejected += 1
                state.candidates_processed[slug] = {
                    "status": "REJECTED",
                    "reason": result.reason,
                    "code_path": (
                        str(result.code_path) if result.code_path else None
                    ),
                    "report_path": None,
                }

    # ------------------------------------------------------------------ #
    # Phase 4 + 5 — debate + verdict
    # ------------------------------------------------------------------ #

    def _phase_debate(self, state: LoopState, summary: RunSummary) -> None:
        """For each IMPLEMENTED candidate not yet debated, run the
        debate orchestrator + verdict engine. Records the verdict +
        transcript path in state.debates_completed."""
        logger.info("phase 4+5: debate + verdict")
        rules = self.rules_loader()
        for slug, entry in state.candidates_processed.items():
            if entry.get("status") != "IMPLEMENTED":
                continue
            if slug in state.debates_completed:
                continue
            report_path_s = entry.get("report_path")
            if not report_path_s:
                logger.warning(
                    "candidate %s missing report_path — skipping debate", slug,
                )
                continue
            report_path = Path(report_path_s)
            if not report_path.exists():
                logger.warning(
                    "candidate %s report file %s missing — skipping debate",
                    slug, report_path,
                )
                continue
            try:
                candidate_report_dict = json.loads(report_path.read_text())
                debate_result = self.debate_orchestrator.run_debate(
                    strategy_slug=slug,
                    candidate_report_text=json.dumps(
                        candidate_report_dict, indent=2,
                    ),
                )
                bull_pos = debate_result.final_positions.get(
                    "bull_reviewer",
                ) or debate_result.final_positions.get("bull", Position.ABSTAIN)
                bear_pos = debate_result.final_positions.get(
                    "bear_reviewer",
                ) or debate_result.final_positions.get("bear", Position.ABSTAIN)
                verdict = compute_verdict(
                    candidate_report=candidate_report_dict,
                    rules=rules,
                    bull_position=bull_pos,
                    bear_position=bear_pos,
                    open_questions=debate_result.open_questions,
                )
            except Exception as exc:
                logger.exception("debate/verdict failed for slug %s", slug)
                state.debates_completed[slug] = {
                    "verdict": "ERROR",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "transcript_path": "",
                    "bull": "",
                    "bear": "",
                }
                summary.errors.append(
                    f"debate/{slug}: {type(exc).__name__}: {exc}"
                )
                continue
            summary.debates_run += 1
            self._tally_verdict(summary, verdict)
            new_entry: dict[str, Any] = {
                "verdict": verdict.verdict.value,
                "reason": verdict.reason,
                "transcript_path": str(debate_result.transcript_path),
                "bull": str(bull_pos),
                "bear": str(bear_pos),
            }
            # GATE 2: PROMOTE verdicts hold for operator deploy
            # confirmation before the registrar runs.
            if verdict.verdict == Verdict.PROMOTE:
                new_entry["deploy_status"] = GATE2_PENDING_STATUS
                new_entry["pending_since"] = self._clock().isoformat(
                    timespec="seconds",
                )
                new_entry["candidate_report_path"] = str(report_path)
                self._notify_gate2(slug, new_entry, summary)
            # ESCALATE: fire operator alert with debate context. Dedup
            # is automatic — a slug only gets debated once (the outer
            # loop skips slugs already in debates_completed), so this
            # notification fires exactly once per escalated candidate.
            elif verdict.verdict == Verdict.ESCALATE:
                self._notify_escalate(
                    slug=slug, verdict=verdict,
                    candidate_report_path=report_path,
                    transcript_path=debate_result.transcript_path,
                    candidate_report=candidate_report_dict,
                    summary=summary,
                )
            state.debates_completed[slug] = new_entry

    def _auto_approve_gate1(self, state: LoopState) -> None:
        """--auto-approve helper: flip every PENDING_OPERATOR_APPROVAL
        to APPROVED. Only safe for dry-run preflights."""
        for entry in state.ideas_processed.values():
            if entry.get("status") == GATE1_PENDING_STATUS:
                entry["status"] = GATE1_APPROVED_STATUS
                entry["reason"] = "auto-approved (--auto-approve / dry-run)"

    def _auto_approve_gate2(self, state: LoopState) -> None:
        """--auto-approve helper: flip every PENDING_DEPLOY_CONFIRMATION
        to DEPLOY_APPROVED."""
        for entry in state.debates_completed.values():
            if entry.get("deploy_status") == GATE2_PENDING_STATUS:
                entry["deploy_status"] = GATE2_APPROVED_STATUS
                entry["deploy_reason"] = (
                    "auto-approved (--auto-approve / dry-run)"
                )

    @staticmethod
    def _tally_verdict(summary: RunSummary, verdict: VerdictResult) -> None:
        if verdict.verdict == Verdict.PROMOTE:
            summary.verdicts_promote += 1
        elif verdict.verdict == Verdict.REJECT:
            summary.verdicts_reject += 1
        elif verdict.verdict == Verdict.ESCALATE:
            summary.verdicts_escalate += 1

    # ------------------------------------------------------------------ #
    # GATE 2 — pre-deploy operator confirmation (CL-yta6)
    # ------------------------------------------------------------------ #

    def _phase_gate2(self, state: LoopState, summary: RunSummary) -> None:
        """Two responsibilities:

          1. Auto-REJECT pending entries older than the timeout. Note
             this is the OPPOSITE default from GATE 1 — silence on a
             deploy decision means "no", not "yes". A skipped deploy
             can always be re-promoted later; an erroneous deploy is
             harder to undo even at allocation=0.
          2. Run the PROMOTE registrar for entries the operator has
             marked DEPLOY_APPROVED. Records DEPLOYED on success or
             DEPLOY_FAILED with the registrar's error.
        """
        now = self._clock()
        for slug, entry in state.debates_completed.items():
            deploy_status = entry.get("deploy_status")
            if deploy_status == GATE2_PENDING_STATUS:
                pending_since_str = entry.get("pending_since")
                if not pending_since_str:
                    continue
                try:
                    pending_since = datetime.fromisoformat(pending_since_str)
                except ValueError:
                    logger.warning(
                        "debates_completed[%s].pending_since malformed: %r",
                        slug, pending_since_str,
                    )
                    continue
                elapsed = (now - pending_since).total_seconds()
                if elapsed > self.gate2_timeout_sec:
                    entry["deploy_status"] = GATE2_REJECTED_STATUS
                    entry["deploy_reason"] = (
                        f"auto-REJECTED: pending {elapsed/86400:.1f}d > "
                        f"{self.gate2_timeout_sec/86400:.1f}d operator timeout"
                    )
                    summary.gate2_auto_rejected_expired += 1
                    logger.info(
                        "GATE 2 auto-rejected expired pending deploy %s", slug,
                    )
            elif deploy_status == GATE2_APPROVED_STATUS:
                self._run_registrar(slug, entry, summary)

    def _run_registrar(
        self,
        slug: str,
        entry: dict[str, Any],
        summary: RunSummary,
    ) -> None:
        """Execute the PROMOTE side-effect for a DEPLOY_APPROVED entry."""
        try:
            result: RegistrationResult = self.registrar.register(
                strategy_slug=slug,
                candidate_report_path=entry.get("candidate_report_path", ""),
                debate_transcript_path=entry.get("transcript_path", ""),
                verdict_reason=entry.get("reason", ""),
            )
        except Exception as exc:
            logger.exception("registrar raised for slug %s", slug)
            entry["deploy_status"] = GATE2_DEPLOY_FAILED_STATUS
            entry["deploy_reason"] = f"{type(exc).__name__}: {exc}"
            summary.deployments_failed += 1
            summary.errors.append(f"deploy/{slug}: {type(exc).__name__}: {exc}")
            return
        if result.succeeded:
            entry["deploy_status"] = GATE2_DEPLOYED_STATUS
            entry["pr_url"] = result.pr_url
            entry["branch_name"] = result.branch_name
            entry["deploy_reason"] = (
                f"steps: {','.join(result.steps_completed)}"
            )
            summary.deployments_succeeded += 1
        else:
            entry["deploy_status"] = GATE2_DEPLOY_FAILED_STATUS
            entry["deploy_reason"] = result.error or "registrar reported failure"
            summary.deployments_failed += 1
            summary.errors.append(f"deploy/{slug}: {result.error}")

    def _notify_escalate(
        self,
        slug: str,
        verdict: VerdictResult,
        candidate_report_path: Path,
        transcript_path: Path,
        candidate_report: dict[str, Any],
        summary: RunSummary,
    ) -> None:
        """Fire ESCALATE alert (CL-o2vb). priority=1 so the operator
        notices; the dedup invariant holds because the outer phase
        skips slugs already in debates_completed."""
        # Identify which rule(s) triggered ESCALATE: missing metrics
        # are the most common cause; ambiguous-positions case shows up
        # in verdict.reason.
        problem_rules = [
            f"{e.rule_id} ({e.detail})"
            for e in verdict.rule_evaluations
            if e.missing or not e.passed
        ]
        oos = candidate_report.get("backtest_metrics", {}).get(
            "oos_metrics", {},
        ) or candidate_report.get("oos_metrics", {})
        metrics_blurb = ", ".join(
            f"{k}={v}" for k, v in oos.items()
        ) if isinstance(oos, dict) else ""
        title = f"ESCALATE: debate result needs operator review — {slug}"
        message = (
            f"Verdict: ESCALATE\n"
            f"Slug: {slug}\n"
            f"Reason: {verdict.reason}\n"
            f"Bull: {verdict.bull_position}  "
            f"Bear: {verdict.bear_position}\n\n"
            f"Problem rules / metrics:\n"
            + (
                "\n".join(f"  - {r}" for r in problem_rules)
                if problem_rules else "  (none — agent positions diverged)"
            )
            + (f"\n\nOOS metrics: {metrics_blurb}" if metrics_blurb else "")
            + f"\n\nTranscript: {transcript_path}\n"
            + f"Candidate report: {candidate_report_path}\n\n"
            + "Open the transcript to see the full debate, then file an "
            "operator decision (manually retry / archive / data-seed)."
        )
        try:
            disp = self.notify_fn(title, message, 1)
        except Exception as exc:
            logger.warning(
                "ESCALATE notification dispatch raised: %s: %s",
                type(exc).__name__, exc,
            )
            return
        if disp.any_attempted:
            summary.escalate_notifications_sent += 1

    def _notify_gate2(
        self,
        slug: str,
        entry: dict[str, Any],
        summary: RunSummary,
    ) -> None:
        """Fire pre-deploy confirmation notification. priority=1 since
        this is a deploy decision and we want the operator to notice."""
        title = f"GATE 2: deploy confirmation — {slug}"
        message = (
            f"PROMOTE verdict — strategy ready for paper-shadow.\n\n"
            f"Slug: {slug}\n"
            f"Verdict reason: {entry.get('reason', '(none)')}\n"
            f"Bull: {entry.get('bull')}  Bear: {entry.get('bear')}\n"
            f"Transcript: {entry.get('transcript_path')}\n"
            f"Candidate report: {entry.get('candidate_report_path')}\n\n"
            f"NOTE: paper-shadow registration starts at allocation=0 so "
            f"there is no real-money risk. This gate is operator-awareness, "
            f"not financial-loss prevention.\n\n"
            f"Approve via:\n"
            f"  python -m scripts.research_approve --gate=2 --slug {slug} "
            f"--action GO\n"
            f"Reject with --action SKIP --reason 'why'."
        )
        try:
            disp = self.notify_fn(title, message, 1)
        except Exception as exc:
            logger.warning(
                "GATE 2 notification dispatch raised: %s: %s",
                type(exc).__name__, exc,
            )
            return
        if disp.any_attempted:
            summary.gate2_notifications_sent += 1

    # ------------------------------------------------------------------ #
    # Run-summary persistence
    # ------------------------------------------------------------------ #

    def _persist_run_summary(self, summary: RunSummary, started: str) -> None:
        """Write per-run summary to data/research/runs/{ts}.json."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        # Use the start timestamp as the filename to make per-run inspection
        # deterministic. Replace : with - for cross-platform filesystem safety.
        safe_ts = started.replace(":", "-")
        out_path = self.runs_dir / f"{safe_ts}.json"
        out_path.write_text(json.dumps(asdict(summary), indent=2, default=str))
