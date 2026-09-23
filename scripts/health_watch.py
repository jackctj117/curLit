"""Fleet watchdog daemon (CL-fmqp).

Every 5 minutes: parse ``daemons.sh status``, page Telegram on daemon
up→down transitions (once per onset, recovery notice on return), and on
X-ingest staleness (newest x-sourced geo_event older than
``X_INGEST_STALE_HOURS``, default 3h). Decisions live in
src/monitoring/fleet_watch.py; this script owns the I/O.

Who watches the watchman: nothing — deliberately. It is the simplest
process in the fleet (subprocess + one SQL + one HTTP), and adding a
second watchdog would just move the question. Its own death is visible
in ``daemons.sh status`` and at the next incident post-mortem.

Usage:
    .venv/bin/python scripts/health_watch.py --once
    .venv/bin/python scripts/health_watch.py --loop 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

STATE_PATH = Path(os.environ.get("HEALTH_WATCH_STATE", "data/health_watch_state.json"))
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_state() -> dict:
    try:
        return dict(json.loads(STATE_PATH.read_text()))
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("health watch: unreadable state — starting fresh", exc_info=True)
        return {}


def _fleet_status_output() -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603 — our own script, fixed argv
            ["bash", str(_REPO_ROOT / "scripts" / "daemons.sh"), "status"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_REPO_ROOT),
        )
        return proc.stdout
    except Exception:
        logger.warning("health watch: daemons.sh status failed", exc_info=True)
        return None


def _engine_halt_state() -> bool | None:
    """The engine's OMS halt flag via GET /api/system, or None when the
    state is UNKNOWN (no secret configured, API unreachable, or OMS not
    wired). None is NOT 'not halted' — the decision layer holds last state
    and never pages on unknown."""
    import httpx  # noqa: PLC0415

    secret = os.environ.get("WEB_API_SECRET", "")
    if not secret:
        return None
    host = os.environ.get("CURLIT_API_HOST", "127.0.0.1")
    port = os.environ.get("CURLIT_API_PORT", "8200")
    try:
        resp = httpx.get(
            f"http://{host}:{port}/api/system",
            headers={"X-API-Key": secret},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("oms_wired"):
            return None  # engine up but OMS not wired → unknown, not "clear"
        return bool(data.get("oms_halted"))
    except Exception:
        logger.debug("health watch: engine /api/system unreachable", exc_info=True)
        return None


def _recent_halt_reason() -> str | None:
    """Best-effort enrichment: the most recent kill-switch trigger in the
    engine log (name + action). Returns None if none found / unreadable —
    a manual or non-switch halt still pages, just without a switch name."""
    from collections import deque  # noqa: PLC0415

    log_path = _REPO_ROOT / "logs" / "engine.log"
    try:
        with log_path.open() as f:
            tail = deque(f, maxlen=500)
    except Exception:
        return None
    for line in reversed(tail):
        if "KILL SWITCH:" in line and "triggered" in line:
            m = re.search(
                r"KILL SWITCH: (\S+) triggered — action=(\S+)",
                line,
            )
            return f"kill switch {m.group(1)} (action={m.group(2)})" if m else "kill switch fired"
    return None


def _newest_x_event(engine) -> datetime | None:  # noqa: ANN001
    from sqlalchemy import text  # noqa: PLC0415

    try:
        with engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT MAX(seen_at) FROM geo_events WHERE source LIKE 'x%'",
                )
            ).scalar()
    except Exception:
        logger.warning("health watch: x-freshness query failed", exc_info=True)
        return None


def run_once(
    now: datetime | None = None,
    *,
    clock: Any = None,
    interval_sec: float | None = None,
) -> int:
    """One watchdog cycle. Returns the number of pages sent.

    ``clock`` (a :class:`~src.monitoring.host_gap.CycleClock`) and
    ``interval_sec`` (the loop period) drive host suspend/resume detection
    (CL-cmg9); both are injectable for tests, sampled live otherwise.
    """
    from sqlalchemy import create_engine  # noqa: PLC0415

    from src.data.db_env import build_db_url  # noqa: PLC0415
    from src.events._util import atomic_write_json  # noqa: PLC0415
    from src.monitoring.fleet_watch import (  # noqa: PLC0415
        DEFAULT_X_STALE_HOURS,
        decide_fleet_alerts,
        decide_halt_alert,
        decide_x_staleness_alert,
        parse_status,
        summarize_state,
    )
    from src.monitoring.host_gap import (  # noqa: PLC0415
        GAP_LATENCY,
        GAP_NONE,
        MAX_GAP_RECORDS,
        CycleClock,
        accumulate_observation,
        classify_cycle_gap,
        decide_readiness,
    )
    from src.research.notifications import notify_operator  # noqa: PLC0415

    now = now or datetime.now(UTC)
    # Sample BOTH clocks first thing (CL-cmg9): the wall-vs-monotonic
    # difference since the previous cycle is the time this host was asleep.
    sample = clock or CycleClock(wall=time.time(), mono=time.monotonic(), pid=os.getpid())
    state = _load_state()
    messages: list[str] = []
    gap = classify_cycle_gap(CycleClock.from_dict(state.get("clock")), sample, interval_sec)
    if gap.kind == GAP_LATENCY:
        logger.warning(
            "health watch: ACTIVE-RUNTIME latency — cycle took %.0fs awake "
            "(host was not suspended; loop %.0fs)",
            gap.active_elapsed_sec or 0.0,
            interval_sec or 0.0,
        )
    elif gap.invalidates_readiness:
        logger.warning(
            "health watch: host gap %s — wall %.0fs, active %s, suspended %s",
            gap.kind,
            gap.wall_elapsed_sec,
            f"{gap.active_elapsed_sec:.0f}s" if gap.active_elapsed_sec is not None else "n/a",
            f"{gap.suspended_sec:.0f}s" if gap.suspended_sec is not None else "n/a",
        )

    output = _fleet_status_output()
    if output is not None:
        current = parse_status(output)
        alerts, new_down = decide_fleet_alerts(
            current,
            dict(state.get("down") or {}),
            now,
        )
        messages.extend(alerts)
        state["down"] = new_down
    else:
        logger.warning("health watch: fleet status unavailable this cycle")

    try:
        stale_hours = float(os.environ.get("X_INGEST_STALE_HOURS", DEFAULT_X_STALE_HOURS))
    except ValueError:
        stale_hours = DEFAULT_X_STALE_HOURS
    # DISPOSE the engine every cycle (CL-bdax): run_once builds a fresh
    # Engine per tick, and an undisposed Engine keeps its pooled connection
    # open forever — at --loop 300 that leaked ~1 Postgres connection every
    # 5 minutes until the server hit max_connections and REFUSED everything
    # ("sorry, too many clients already" killed execute_options). The engine
    # is only needed for the x-freshness read, so scope it tightly.
    engine = create_engine(build_db_url())
    try:
        newest_x = _newest_x_event(engine)
    finally:
        engine.dispose()
    msg, is_stale = decide_x_staleness_alert(
        newest_x,
        bool(state.get("x_stale")),
        now,
        stale_hours=stale_hours,
    )
    if msg:
        messages.append(msg)
    state["x_stale"] = is_stale

    # Engine halt / kill-switch page (CL-fmqp extension): the safety layer
    # can halt trading — a bug OR a legit VIX/drawdown/desync trip — and
    # nothing surfaced it before now. Page on the not-halted → halted
    # transition, enriched with the triggering switch.
    halted = _engine_halt_state()
    hmsg, is_halted = decide_halt_alert(
        halted,
        bool(state.get("engine_halted")),
        _recent_halt_reason() if halted else None,
        now,
    )
    if hmsg:
        messages.append(hmsg)
    state["engine_halted"] = is_halted

    # Readiness (CL-cmg9): a suspend/unobserved gap makes it STALE; only a
    # later awake cycle with fresh evidence — every daemon up, engine state
    # readable, ingest fresh — restores it. Sleep never counts as observed.
    fresh_evidence = (
        output is not None and not state.get("down") and halted is not None and not is_stale
    )
    rpages, state["readiness"] = decide_readiness(
        gap, state.get("readiness"), now, fresh_evidence=fresh_evidence
    )
    messages.extend(rpages)
    state["observation"] = accumulate_observation(state.get("observation"), gap)
    if gap.kind != GAP_NONE:
        record = {
            "at": now.isoformat(),
            "kind": gap.kind,
            "wall_sec": round(gap.wall_elapsed_sec, 1),
            "active_sec": (
                round(gap.active_elapsed_sec, 1) if gap.active_elapsed_sec is not None else None
            ),
            "suspended_sec": round(gap.suspended_sec, 1) if gap.suspended_sec is not None else None,
        }
        state["host_gaps"] = [*list(state.get("host_gaps") or []), record][-MAX_GAP_RECORDS:]
    state["clock"] = sample.to_dict()

    sent = 0
    for m in messages:
        result = notify_operator("🩺 Fleet watchdog", m)
        if result.any_succeeded:
            sent += 1
        else:
            logger.warning("health watch: page failed to send: %s", m)
    # State saves even when a page fails — a failed send should not
    # re-page every 5 minutes forever; the log line above is the record.
    atomic_write_json(STATE_PATH, state)
    logger.info("health watch: %s pages=%d", summarize_state(state), sent)
    return sent


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    parser = argparse.ArgumentParser(description="Fleet watchdog.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.loop:
        logger.info("health watch: looping every %ds", args.loop)
        try:
            while True:
                try:
                    run_once(interval_sec=float(args.loop))
                except Exception:
                    logger.exception("health watch: cycle failed — retrying")
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("health watch: stopped")
        return 0

    run_once()
    return 0


if __name__ == "__main__":
    sys.exit(main())
