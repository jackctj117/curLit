"""Fleet watchdog logic (CL-fmqp) — pure, testable decisions.

Born from a real incident: x_monitor died silently for 34 minutes after a
restart race and nobody noticed until the operator asked. The watchdog
daemon (scripts/health_watch.py) runs every 5 minutes and pages Telegram
on:

  1. daemon DEATH — an up→down transition in ``daemons.sh status`` output
     (pages once per onset; a recovery notice when it comes back; no
     re-page while it stays down);
  2. X-ingest STALENESS — the newest x-sourced geo_event older than a
     threshold (secondary signal: catches a monitor that is alive but
     silently failing to ingest).

This module holds the parsing/decision logic only — no subprocess, no
Telegram, no DB — so every transition rule is unit-tested. The daemon
script owns I/O.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

_STATUS_RE = re.compile(r"^\s*(✓|✗)\s+(\S+)")

#: Newest x-sourced event older than this many hours → staleness page.
#: 44 accounts incl. 24/7 news wires post far more often than this; a
#: quiet stretch this long means the monitor is broken, not the world.
DEFAULT_X_STALE_HOURS = 3.0


def parse_status(output: str) -> dict[str, bool]:
    """``daemons.sh status`` output → {daemon_name: is_up}."""
    out: dict[str, bool] = {}
    for line in output.splitlines():
        m = _STATUS_RE.match(line)
        if m:
            out[m.group(2)] = m.group(1) == "✓"
    return out


def decide_fleet_alerts(
    current: dict[str, bool],
    prev_down: dict[str, str],
    now: datetime | None = None,
    self_name: str = "health_watch",
) -> tuple[list[str], dict[str, str]]:
    """Transition-only alerting.

    ``prev_down``: {daemon: iso_ts_first_seen_down} carried in the state
    file. Returns (alert message lines, new prev_down). Rules: up→down
    pages once ("DOWN since …"); down→up sends a recovery notice with the
    outage duration; still-down stays silent (the first page stands). The
    watchdog never reports itself — if it runs, it's up.
    """
    now = now or datetime.now(UTC)
    alerts: list[str] = []
    new_down: dict[str, str] = {}
    for name, is_up in current.items():
        if name == self_name:
            continue
        was_down_since = prev_down.get(name)
        if not is_up:
            if was_down_since is None:
                alerts.append(
                    f"🔴 daemon DOWN: {name} (first seen "
                    f"{now.strftime('%H:%M:%S')} UTC) — "
                    f"./scripts/daemons.sh start to recover"
                )
                new_down[name] = now.isoformat()
            else:
                new_down[name] = was_down_since  # still down — stay silent
        elif was_down_since is not None:
            try:
                since = datetime.fromisoformat(was_down_since)
                mins = int((now - since).total_seconds() // 60)
                dur = f" (down ~{mins}min)"
            except ValueError:
                dur = ""
            alerts.append(f"🟢 daemon RECOVERED: {name}{dur}")
    # Daemons that vanished from the status list entirely while down:
    # keep them tracked so a later reappearance still closes the loop.
    for name, down_ts in prev_down.items():
        if name not in current:
            new_down[name] = down_ts
    return alerts, new_down


def decide_x_staleness_alert(
    newest_x_event: datetime | None,
    was_stale: bool,
    now: datetime | None = None,
    stale_hours: float = DEFAULT_X_STALE_HOURS,
) -> tuple[str | None, bool]:
    """(alert_or_recovery_message | None, is_stale_now).

    Pages once per staleness ONSET; a recovery notice when ingest resumes.
    ``newest_x_event`` None means no x-sourced rows at all — treated as
    stale (an empty feed must never look healthy).
    """
    now = now or datetime.now(UTC)
    if newest_x_event is None:
        age_h = float("inf")
    else:
        if newest_x_event.tzinfo is None:
            newest_x_event = newest_x_event.replace(tzinfo=UTC)
        age_h = (now - newest_x_event).total_seconds() / 3600.0
    stale = age_h > stale_hours
    if stale and not was_stale:
        age_txt = "never" if age_h == float("inf") else f"{age_h:.1f}h ago"
        return (
            f"🟡 X-ingest STALE: newest x-sourced event {age_txt} "
            f"(threshold {stale_hours:g}h) — monitor may be up but not "
            f"ingesting (check logs/x_monitor.log, bird cookies)",
            True,
        )
    if not stale and was_stale:
        return ("🟢 X-ingest recovered — fresh x-sourced events flowing", False)
    return (None, stale)


def decide_halt_alert(
    currently_halted: bool | None,
    was_halted: bool,
    reason: str | None = None,
    now: datetime | None = None,
) -> tuple[str | None, bool]:
    """(message | None, is_halted_now).

    Pages ONCE on the not-halted → halted transition (enriched with the
    triggering kill-switch reason when known); a recovery notice on
    halted → not-halted; silent while it stays halted. Today's incident —
    the engine halted itself and nobody knew until the operator asked —
    is exactly what this closes.

    ``currently_halted is None`` means the engine's state is UNKNOWN this
    cycle (API unreachable / OMS not wired). That is deliberately NOT
    treated as "not halted": it holds the last known state and never
    pages — a dead engine is already caught by the daemon-death check, and
    an unreachable API must not fabricate a false "recovered".
    """
    now = now or datetime.now(UTC)
    if currently_halted is None:
        return (None, was_halted)
    if currently_halted and not was_halted:
        detail = f" — {reason}" if reason else ""
        return (
            f"🛑 ENGINE HALTED: OMS new-trades halt is ACTIVE{detail}. "
            f"No new trades will place until the cause clears and you "
            f"call /api/system/resume (or restart).",
            True,
        )
    if not currently_halted and was_halted:
        return ("🟢 Engine halt CLEARED — new trades flowing again.", False)
    return (None, currently_halted)


def summarize_state(state: dict[str, Any]) -> str:
    """One-line state summary for the cycle log."""
    down = sorted((state.get("down") or {}).keys())
    stale = bool(state.get("x_stale"))
    halted = bool(state.get("engine_halted"))
    return f"down={down or 'none'} x_stale={stale} engine_halted={halted}"
