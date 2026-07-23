"""Approvals-panel helpers for the soak dashboard (CL-7t8d).

Exposes pure functions the dashboard endpoints call:

  * ``list_pending_approvals(state)`` — flatten GATE 1 + GATE 2 pending
    entries into a single list the dashboard can render.
  * ``apply_decision(state, gate, slug, action, reason, decisions_log)``
    — mutate the state file and append to the decisions log.

Kept out of ``scripts/soak_dashboard.py`` so the logic is unit-testable
without spinning up FastAPI. The dashboard module is a thin adapter
that wires HTTP request → these functions → state save.

Decisions log format (one line per decision):

    {iso_ts}  GATE{n}  {action}  {slug}  | {reason}

The log is append-only; never rewritten. Operator history accrues
across runs, useful for the post-mortem when a strategy that was
APPROVED in retrospect should have been SKIPPED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.research.loop import (
    GATE1_APPROVED_STATUS,
    GATE1_PENDING_STATUS,
    GATE1_SKIPPED_STATUS,
    GATE2_APPROVED_STATUS,
    GATE2_PENDING_STATUS,
    GATE2_REJECTED_STATUS,
    LoopState,
    load_state,
    save_state,
)

logger = logging.getLogger(__name__)


# Shape returned by the API; the dashboard JS renders this.
@dataclass
class PendingEntry:
    gate: int  # 1 or 2
    slug: str
    pending_since: str  # ISO timestamp
    extra: dict[str, Any]  # gate-specific extras (hypothesis_path, etc.)

    def to_json(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "slug": self.slug,
            "pending_since": self.pending_since,
            **self.extra,
        }


# --------------------------------------------------------------------- #
# List pending
# --------------------------------------------------------------------- #


def list_pending_approvals(state: LoopState) -> list[PendingEntry]:
    """Flatten GATE 1 + GATE 2 pending entries. GATE 1 entries come
    keyed by extract hash in ``state.ideas_processed``; GATE 2 entries
    come keyed by slug in ``state.debates_completed``."""
    out: list[PendingEntry] = []
    for extract_hash, entry in state.ideas_processed.items():
        if entry.get("status") != GATE1_PENDING_STATUS:
            continue
        out.append(
            PendingEntry(
                gate=1,
                slug=str(entry.get("slug") or extract_hash),
                pending_since=str(entry.get("pending_since") or ""),
                extra={
                    "extract_hash": extract_hash,
                    "hypothesis_path": entry.get("hypothesis_path"),
                    "title": entry.get("slug") or extract_hash,
                },
            )
        )
    for slug, entry in state.debates_completed.items():
        if entry.get("deploy_status") != GATE2_PENDING_STATUS:
            continue
        out.append(
            PendingEntry(
                gate=2,
                slug=slug,
                pending_since=str(entry.get("pending_since") or ""),
                extra={
                    "verdict": entry.get("verdict"),
                    "verdict_reason": entry.get("reason"),
                    "bull": entry.get("bull"),
                    "bear": entry.get("bear"),
                    "transcript_path": entry.get("transcript_path"),
                    "candidate_report_path": entry.get("candidate_report_path"),
                    "title": slug,
                },
            )
        )
    # Stable sort: gate ascending, then pending_since ascending so
    # oldest-pending floats to the top of each gate.
    out.sort(key=lambda e: (e.gate, e.pending_since))
    return out


# --------------------------------------------------------------------- #
# Apply decision
# --------------------------------------------------------------------- #


class DecisionError(ValueError):
    """Raised when an operator decision can't be applied — e.g. the
    target entry isn't in PENDING status, or the gate/slug pair is
    unknown. Mapped to HTTP 400 by the dashboard endpoint."""


def apply_decision(
    state_path: Path | str,
    gate: int,
    slug: str,
    action: str,
    reason: str = "",
    decisions_log: Path | str = "data/research/decisions.log",
) -> dict[str, Any]:
    """Apply an APPROVE/REJECT decision to the loop state file. Returns
    a small dict the dashboard echoes back to the operator UI.

    Validates:
      * gate in {1, 2}
      * action in {"APPROVE", "REJECT"}
      * the target entry is in the right PENDING status
    Raises ``DecisionError`` on any of the above.
    """
    if gate not in (1, 2):
        msg = f"unknown gate: {gate!r}"
        raise DecisionError(msg)
    if action not in ("APPROVE", "REJECT"):
        msg = f"unknown action: {action!r}"
        raise DecisionError(msg)

    state = load_state(state_path)
    if gate == 1:
        new_status = _apply_gate1(state, slug, action, reason)
    else:
        new_status = _apply_gate2(state, slug, action, reason)

    save_state(state, Path(state_path))
    _append_decision_log(decisions_log, gate=gate, action=action, slug=slug, reason=reason)
    return {
        "ok": True,
        "gate": gate,
        "slug": slug,
        "new_status": new_status,
    }


def _apply_gate1(
    state: LoopState,
    slug: str,
    action: str,
    reason: str,
) -> str:
    target_hash: str | None = None
    for h, e in state.ideas_processed.items():
        if e.get("slug") == slug:
            target_hash = h
            break
    if target_hash is None:
        msg = f"no GATE 1 entry found with slug={slug!r}"
        raise DecisionError(msg)

    entry = state.ideas_processed[target_hash]
    if entry.get("status") != GATE1_PENDING_STATUS:
        msg = (
            f"GATE 1 entry for {slug!r} is in status "
            f"{entry.get('status')!r}, not {GATE1_PENDING_STATUS!r}"
        )
        raise DecisionError(msg)

    if action == "APPROVE":
        entry["status"] = GATE1_APPROVED_STATUS
        if reason:
            entry["reason"] = reason
    else:  # REJECT
        entry["status"] = GATE1_SKIPPED_STATUS
        entry["reason"] = reason or "skipped via dashboard"
    return str(entry["status"])


def _apply_gate2(
    state: LoopState,
    slug: str,
    action: str,
    reason: str,
) -> str:
    if slug not in state.debates_completed:
        msg = f"no GATE 2 (debate) entry found for slug={slug!r}"
        raise DecisionError(msg)
    entry = state.debates_completed[slug]
    if entry.get("deploy_status") != GATE2_PENDING_STATUS:
        msg = (
            f"GATE 2 entry for {slug!r} is in deploy_status "
            f"{entry.get('deploy_status')!r}, not {GATE2_PENDING_STATUS!r}"
        )
        raise DecisionError(msg)

    if action == "APPROVE":
        entry["deploy_status"] = GATE2_APPROVED_STATUS
        if reason:
            entry["deploy_reason"] = reason
    else:  # REJECT
        entry["deploy_status"] = GATE2_REJECTED_STATUS
        entry["deploy_reason"] = reason or "rejected via dashboard"
    return str(entry["deploy_status"])


# --------------------------------------------------------------------- #
# Decisions log
# --------------------------------------------------------------------- #


def _append_decision_log(
    log_path: Path | str,
    *,
    gate: int,
    action: str,
    slug: str,
    reason: str,
) -> None:
    """Append a single decision line. Best-effort; log failures don't
    block the state mutation (decision is already persisted)."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    line = f"{ts}  GATE{gate}  {action:<7}  {slug}  | {reason}\n"
    try:
        with p.open("a") as f:
            f.write(line)
    except OSError as exc:
        logger.warning("decisions.log write failed: %s: %s", type(exc).__name__, exc)
