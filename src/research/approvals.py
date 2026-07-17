"""Shared GATE 1 / GATE 2 approval state mutations (CL-b1l6).

Extracted from ``scripts/research_approve.py`` (CL-0hr3, CL-yta6) so the
CLI approver and the Telegram approval bot
(``src/research/telegram_approvals.py``) share one implementation of
the operator gate transitions:

  * GATE 1: ``PENDING_OPERATOR_APPROVAL`` → ``APPROVED`` / ``SKIPPED``
  * GATE 2: ``PENDING_DEPLOY_CONFIRMATION`` → ``DEPLOY_APPROVED`` /
    ``DEPLOY_REJECTED``

All functions here are pure state mutations — no I/O, no printing —
so both frontends (argparse CLI, Telegram command handler) can render
the outcome however suits their channel. Persistence goes through
``save_state_atomic`` which serializes exactly like
``src.research.loop.save_state`` but writes tmp-file + ``os.replace``
so a concurrently-reading research loop never sees a torn file.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
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
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionResult:
    """Outcome of a gate transition attempt.

    ``ok=True`` means the state object was mutated and should be
    persisted; ``message`` is a human-readable summary either way
    (e.g. ``"APPROVED: carry-regime"`` or the refusal reason).
    """

    ok: bool
    message: str


def save_state_atomic(state: LoopState, path: Path | str) -> None:
    """Persist state JSON atomically (tmp file + ``os.replace``).

    Byte-for-byte the same serialization as ``loop.save_state`` — the
    research loop reads this file at the start of every pass, so the
    write must never be observable half-finished.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(state), indent=2, default=str)
    fd, tmp_name = tempfile.mkstemp(
        dir=p.parent, prefix=f".{p.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp_name, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# --------------------------------------------------------------------- #
# Pending listings
# --------------------------------------------------------------------- #


def gate1_pending(state: LoopState) -> list[tuple[str, dict[str, Any]]]:
    """``(extract_hash, entry)`` pairs awaiting GATE 1 approval."""
    return [
        (h, e) for h, e in state.ideas_processed.items()
        if e.get("status") == GATE1_PENDING_STATUS
    ]


def gate2_pending(state: LoopState) -> list[tuple[str, dict[str, Any]]]:
    """``(slug, entry)`` pairs awaiting GATE 2 deploy confirmation."""
    return [
        (slug, e) for slug, e in state.debates_completed.items()
        if e.get("deploy_status") == GATE2_PENDING_STATUS
    ]


def find_gate1_hash_by_slug(state: LoopState, slug: str) -> str | None:
    """Reverse-lookup the ideas_processed key for a strategy slug."""
    for h, e in state.ideas_processed.items():
        if e.get("slug") == slug:
            return h
    return None


# --------------------------------------------------------------------- #
# Gate transitions
# --------------------------------------------------------------------- #


def act_gate1(
    state: LoopState,
    extract_hash: str,
    approve: bool,
    reason: str = "",
) -> ActionResult:
    """GATE 1 transition keyed by extract hash.

    ``approve=True`` → APPROVED (implementer runs next loop pass);
    ``approve=False`` → SKIPPED (loop archives the hypothesis).
    Refuses to act on entries that aren't PENDING — an already-decided
    entry is never silently clobbered.
    """
    entry = state.ideas_processed.get(extract_hash)
    if entry is None:
        return ActionResult(
            ok=False,
            message=f"no GATE 1 entry for extract {extract_hash!r}",
        )
    slug = entry.get("slug")
    if entry.get("status") != GATE1_PENDING_STATUS:
        return ActionResult(
            ok=False,
            message=(
                f"entry for slug={slug!r} is in status "
                f"{entry.get('status')!r}, not {GATE1_PENDING_STATUS!r}; "
                f"refusing to act"
            ),
        )
    if approve:
        entry["status"] = GATE1_APPROVED_STATUS
        if reason:
            entry["reason"] = reason
        return ActionResult(ok=True, message=f"APPROVED: {slug}")
    entry["status"] = GATE1_SKIPPED_STATUS
    entry["reason"] = reason or "skipped by operator"
    return ActionResult(
        ok=True, message=f"SKIPPED: {slug} — {entry['reason']}",
    )


def act_gate2(
    state: LoopState,
    slug: str,
    approve: bool,
    reason: str = "",
) -> ActionResult:
    """GATE 2 transition keyed by strategy slug.

    ``approve=True`` → DEPLOY_APPROVED (registrar runs next loop pass);
    ``approve=False`` → DEPLOY_REJECTED (strategy archived).
    """
    if slug not in state.debates_completed:
        return ActionResult(
            ok=False, message=f"no debate entry for slug={slug!r}",
        )
    entry = state.debates_completed[slug]
    if entry.get("deploy_status") != GATE2_PENDING_STATUS:
        return ActionResult(
            ok=False,
            message=(
                f"entry for slug={slug!r} has deploy_status="
                f"{entry.get('deploy_status')!r}, not "
                f"{GATE2_PENDING_STATUS!r}; refusing to act"
            ),
        )
    if approve:
        entry["deploy_status"] = GATE2_APPROVED_STATUS
        if reason:
            entry["deploy_reason"] = reason
        return ActionResult(ok=True, message=f"DEPLOY_APPROVED: {slug}")
    entry["deploy_status"] = GATE2_REJECTED_STATUS
    entry["deploy_reason"] = reason or "rejected by operator"
    return ActionResult(
        ok=True,
        message=f"DEPLOY_REJECTED: {slug} — {entry['deploy_reason']}",
    )
