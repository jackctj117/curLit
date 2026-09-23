"""Host suspend/resume detection for fleet readiness evidence (CL-cmg9).

Born from the 2026-09-08 deployment observation: the Mac slept for hours,
monitor samples simply jumped from 23:01 to 02:18 UTC, and the unchanged
pipeline PID "resumed" afterwards. Those hours looked like uninterrupted
healthy uptime; they were not. On 2026-09-19 the same sleep produced a burst of
"Your computer went to sleep mid-response" transport failures.

The detection rests on one measured fact: on macOS ``time.monotonic()`` is
``mach_absolute_time()``, which does NOT advance while the host sleeps, while
the wall clock does. Verified on the operator's Mac (2026-09-22): wall time
since boot 1339.3 h vs monotonic 601.2 h — the 738 h difference is exactly the
accumulated sleep. Linux ``CLOCK_MONOTONIC`` likewise stops during suspend. So
for two samples taken by the SAME process:

* ``wall_elapsed - active_elapsed`` = time the host was suspended;
* a large ``active_elapsed`` with no suspension = the process was awake but
  slow (active-runtime latency — e.g. CL-wv3v's slow process discovery), a
  different problem that must not be reported as sleep;
* samples from DIFFERENT processes are not comparable (monotonic origins
  differ): a long wall gap there is an UNOBSERVED interval of unknown cause.

Readiness after any suspension or unobserved gap is STALE: the fleet was not
observing, so earlier evidence says nothing about the gap. It is restored only
by a later awake cycle that re-establishes fresh evidence. Sleeping hours are
never counted as observed time (:func:`accumulate_observation`).

Pure logic only — no clocks read, no I/O. ``scripts/health_watch.py`` samples
the clocks and owns persistence and paging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

#: Minimum (wall - active) discrepancy treated as host suspension. NTP slews
#: the wall clock by at most ~0.5 ms/s (≈0.15 s over a 300 s cycle), so a
#: 120 s discrepancy cannot come from normal clock discipline — only from a
#: suspend or a manual clock step. Well below the shortest real macOS sleep
#: episodes seen in pmset (minutes), so none are missed.
SUSPEND_THRESHOLD_SEC = 120.0

#: A same-process cycle whose ACTIVE elapsed time exceeds this many loop
#: intervals was awake but slow. 2x: one interval of sleep() plus a whole
#: second interval of cycle work is already pathological for a 300 s loop.
LATENCY_FACTOR = 2.0

#: Wall-clock go-backwards tolerance before a sample pair is treated as a
#: clock step (a small negative jitter from NTP stepping is not an event).
_CLOCK_STEP_TOLERANCE_SEC = 1.0

#: Most recent gap records kept in watchdog state (bounded, audit only).
MAX_GAP_RECORDS = 20

GAP_NONE = "none"
GAP_SUSPENDED = "suspended"
GAP_LATENCY = "latency"
GAP_UNOBSERVED = "unobserved"
GAP_CLOCK_STEP = "clock_step"

#: Gap kinds that invalidate readiness evidence.
STALE_GAP_KINDS = frozenset({GAP_SUSPENDED, GAP_UNOBSERVED, GAP_CLOCK_STEP})

READINESS_OK = "ok"
READINESS_STALE = "stale_after_gap"


@dataclass(frozen=True)
class CycleClock:
    """One cycle's clock sample: wall epoch seconds, monotonic seconds, pid."""

    wall: float
    mono: float
    pid: int

    def to_dict(self) -> dict[str, float | int]:
        return {"wall": self.wall, "mono": self.mono, "pid": self.pid}

    @staticmethod
    def from_dict(raw: Any) -> CycleClock | None:
        """Parse persisted state; anything malformed → None (no comparison)."""
        if not isinstance(raw, dict):
            return None
        try:
            return CycleClock(wall=float(raw["wall"]), mono=float(raw["mono"]), pid=int(raw["pid"]))
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class GapVerdict:
    kind: str
    wall_elapsed_sec: float
    #: None when the samples are not comparable (different process / step).
    active_elapsed_sec: float | None
    suspended_sec: float | None

    @property
    def invalidates_readiness(self) -> bool:
        return self.kind in STALE_GAP_KINDS


def classify_cycle_gap(
    prev: CycleClock | None,
    now: CycleClock,
    interval_sec: float | None,
) -> GapVerdict:
    """Classify the interval between two consecutive watchdog cycles.

    ``interval_sec`` is the loop period (None for a one-shot run, which
    disables the latency and unobserved checks — they need an expectation).
    """
    if prev is None:
        return GapVerdict(GAP_NONE, 0.0, None, None)
    wall = now.wall - prev.wall
    if wall < -_CLOCK_STEP_TOLERANCE_SEC:
        logger.warning("host gap: wall clock went BACKWARDS by %.1fs", -wall)
        return GapVerdict(GAP_CLOCK_STEP, wall, None, None)
    wall = max(wall, 0.0)
    if prev.pid != now.pid:
        # Monotonic origins differ between processes: only the wall gap is
        # meaningful, and anything well beyond one loop was not observed.
        if interval_sec is not None and wall > LATENCY_FACTOR * interval_sec:
            return GapVerdict(GAP_UNOBSERVED, wall, None, None)
        return GapVerdict(GAP_NONE, wall, None, None)
    active = max(now.mono - prev.mono, 0.0)
    suspended = wall - active
    if suspended >= SUSPEND_THRESHOLD_SEC:
        return GapVerdict(GAP_SUSPENDED, wall, active, suspended)
    if interval_sec is not None and active > LATENCY_FACTOR * interval_sec:
        return GapVerdict(GAP_LATENCY, wall, active, max(suspended, 0.0))
    return GapVerdict(GAP_NONE, wall, active, max(suspended, 0.0))


def _minutes(sec: float | None) -> str:
    return "?" if sec is None else f"{sec / 60.0:.0f}m"


def decide_readiness(
    verdict: GapVerdict,
    readiness: dict[str, Any] | None,
    now: datetime,
    *,
    fresh_evidence: bool,
) -> tuple[list[str], dict[str, Any]]:
    """(pages, new readiness state).

    * A suspend / unobserved / clock-step gap → readiness STALE, paged once
      per onset (a further gap while already stale updates the record but
      does not re-page).
    * STALE → OK only on a clean awake cycle (verdict ``none`` or latency —
      the host WAS awake) whose ``fresh_evidence`` is True: every daemon
      up, engine state readable, ingest fresh. One recovery page.
    * Latency is logged by the caller, never paged here and never flips
      readiness: the host was up and observing, just slowly.
    """
    current = dict(readiness or {"status": READINESS_OK})
    pages: list[str] = []
    if verdict.invalidates_readiness:
        was_stale = current.get("status") == READINESS_STALE
        current = {
            "status": READINESS_STALE,
            "since": current.get("since") if was_stale else now.isoformat(),
            "last_gap_kind": verdict.kind,
            "last_gap_wall_sec": round(verdict.wall_elapsed_sec, 1),
            "last_gap_suspended_sec": (
                round(verdict.suspended_sec, 1) if verdict.suspended_sec is not None else None
            ),
        }
        if not was_stale:
            if verdict.kind == GAP_SUSPENDED:
                what = (
                    f"💤 HOST SUSPENDED ~{_minutes(verdict.suspended_sec)} "
                    f"(wall gap {_minutes(verdict.wall_elapsed_sec)}, active "
                    f"{_minutes(verdict.active_elapsed_sec)})"
                )
            elif verdict.kind == GAP_UNOBSERVED:
                what = (
                    f"⏸️ UNOBSERVED GAP {_minutes(verdict.wall_elapsed_sec)} "
                    "(watchdog restarted; cause unknown)"
                )
            else:
                what = "⏱️ WALL CLOCK STEPPED BACKWARDS"
            pages.append(
                f"{what} — the fleet was NOT observing during that window, so it "
                "is not healthy uptime. Readiness is STALE until fresh account/data "
                "state is re-established; verify positions before trusting new exposure."
            )
        return pages, current
    if current.get("status") == READINESS_STALE and fresh_evidence:
        pages.append(
            "🟢 Readiness restored after host gap — all daemons up, engine state "
            "readable, ingest fresh."
        )
        return pages, {"status": READINESS_OK, "restored_at": now.isoformat()}
    return pages, current


def accumulate_observation(obs: dict[str, Any] | None, verdict: GapVerdict) -> dict[str, Any]:
    """Running totals that never count suspended or unobserved time as
    observed. ``active_sec`` accrues only same-process awake time."""
    out = {
        "cycles": 0,
        "wall_sec": 0.0,
        "active_sec": 0.0,
        "suspended_sec": 0.0,
        "unobserved_sec": 0.0,
    }
    out.update({k: v for k, v in (obs or {}).items() if k in out})
    out["cycles"] = int(out["cycles"]) + 1
    wall = max(verdict.wall_elapsed_sec, 0.0)
    out["wall_sec"] = float(out["wall_sec"]) + wall
    if verdict.kind in (GAP_UNOBSERVED, GAP_CLOCK_STEP):
        out["unobserved_sec"] = float(out["unobserved_sec"]) + wall
    elif verdict.active_elapsed_sec is not None:
        # min(): an NTP slew can make active exceed wall by a fraction of a
        # second; observed time can never exceed elapsed wall time.
        out["active_sec"] = float(out["active_sec"]) + min(verdict.active_elapsed_sec, wall)
        out["suspended_sec"] = float(out["suspended_sec"]) + (verdict.suspended_sec or 0.0)
    assert out["active_sec"] <= out["wall_sec"] + 1e-6, ("active exceeds wall", out)
    return out
