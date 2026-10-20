"""Research progress, independent of process liveness (CL-x40b).

Only timestamps and stable failure categories are persisted, never prompts,
provider output, credentials or headlines. This signal does not enable trading.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from src.events._util import atomic_write_json

logger = logging.getLogger(__name__)
# Two hours tolerates a slow batch/cooldown, but catches the documented two-day
# silent assessment outage. An alert threshold, never a trading parameter.
DEFAULT_STALL_AFTER = timedelta(hours=2)


class AssessmentOutcome(Protocol):
    status: str
    failure_reason: str | None


def _path() -> Path:
    return Path(os.environ.get("ASSESSMENT_HEALTH_STATE", "data/assessment_health.json"))


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("invalid assessment health timestamp")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("assessment health timestamp needs timezone")
    return result.astimezone(UTC)


@dataclass(frozen=True)
class AssessmentHealth:
    updated_at: datetime
    last_assessed_at: datetime | None = None
    last_auth_failure_at: datetime | None = None


def read_assessment_health() -> AssessmentHealth | None:
    try:
        raw = json.loads(_path().read_text())
        updated = _timestamp(raw["updated_at"])
        if updated is None:
            raise ValueError("missing assessment health timestamp")
        return AssessmentHealth(
            updated, _timestamp(raw.get("last_assessed_at")),
            _timestamp(raw.get("last_auth_failure_at")),
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        logger.warning("assessment health unavailable; not proof of recovery", exc_info=True)
        return None


def record_assessment_health(
    results: Sequence[AssessmentOutcome], *, now: datetime | None = None,
) -> None:
    """Called only after the batch's database writes succeed, before enrichment."""
    now = now or datetime.now(UTC)
    previous = read_assessment_health()
    success = previous.last_assessed_at if previous else None
    auth = previous.last_auth_failure_at if previous else None
    if any(r.status == "ASSESSED" for r in results):
        success = now
    if any(r.failure_reason == "subscription_auth_unavailable" for r in results):
        auth = now
    logger.info("Recording assessment health: batch=%d auth_failure=%s", len(results), auth == now)
    atomic_write_json(_path(), {
        "updated_at": now.isoformat(),
        "last_assessed_at": success.isoformat() if success else None,
        "last_auth_failure_at": auth.isoformat() if auth else None,
    })


def decide_assessment_alert(
    health: AssessmentHealth | None,
    oldest_pending: datetime | None,
    backlog_known: bool,
    previous: dict[str, str],
    now: datetime,
    *,
    stall_after: timedelta = DEFAULT_STALL_AFTER,
) -> tuple[str | None, dict[str, str]]:
    """Page once on failure; only subsequent assessment proves recovery.

    Empty/triaged-out batches and unreadable status/DB never clear an incident.
    Backlog age bounds the quiet-period clock so old successes alone don't page.
    """
    assert stall_after.total_seconds() > 0
    success = health.last_assessed_at if health else None
    auth = health.last_auth_failure_at if health else None
    # Future timestamps cannot prove health (e.g. a host clock adjustment).
    if success is not None and success > now:
        success = None
    auth_failed = auth is not None and (success is None or auth >= success)
    stalled = (
        backlog_known and oldest_pending is not None
        and now - max(oldest_pending, success or oldest_pending) >= stall_after
    )
    reason = "auth" if auth_failed else "stalled" if stalled else None
    if reason is not None:
        if previous:
            return None, previous
        message = (
            "Claude CLI login expired or unavailable; event assessments remain queued. "
            "Operator: restore the CLI login, then verify assessments resume."
            if reason == "auth" else
            "Event assessment stalled: NEW backlog exists with no successful assessment "
            f"for at least {stall_after.total_seconds() / 3600:g} hours. "
            "Operator: inspect pipeline assessment failures and transport availability."
        )
        return message, {"reason": reason, "since": now.isoformat()}
    if previous and success is not None and backlog_known:
        try:
            since = _timestamp(previous.get("since"))
        except ValueError:
            since = None
        if since is not None and success > since:
            return "Event assessment recovered: successful assessments have resumed.", {}
    return None, previous
