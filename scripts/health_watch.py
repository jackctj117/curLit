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
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

STATE_PATH = Path(os.environ.get("HEALTH_WATCH_STATE",
                                 "data/health_watch_state.json"))
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_state() -> dict:
    try:
        return dict(json.loads(STATE_PATH.read_text()))
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("health watch: unreadable state — starting fresh",
                       exc_info=True)
        return {}


def _fleet_status_output() -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603 — our own script, fixed argv
            ["bash", str(_REPO_ROOT / "scripts" / "daemons.sh"), "status"],
            capture_output=True, text=True, timeout=30,
            cwd=str(_REPO_ROOT),
        )
        return proc.stdout
    except Exception:
        logger.warning("health watch: daemons.sh status failed",
                       exc_info=True)
        return None


def _newest_x_event(engine) -> datetime | None:  # noqa: ANN001
    from sqlalchemy import text  # noqa: PLC0415
    try:
        with engine.connect() as conn:
            return conn.execute(text(
                "SELECT MAX(seen_at) FROM geo_events WHERE source LIKE 'x%'",
            )).scalar()
    except Exception:
        logger.warning("health watch: x-freshness query failed",
                       exc_info=True)
        return None


def run_once(now: datetime | None = None) -> int:
    """One watchdog cycle. Returns the number of pages sent."""
    from sqlalchemy import create_engine  # noqa: PLC0415

    from src.data.db_env import build_db_url  # noqa: PLC0415
    from src.events._util import atomic_write_json  # noqa: PLC0415
    from src.monitoring.fleet_watch import (  # noqa: PLC0415
        DEFAULT_X_STALE_HOURS,
        decide_fleet_alerts,
        decide_x_staleness_alert,
        parse_status,
        summarize_state,
    )
    from src.research.notifications import notify_operator  # noqa: PLC0415

    now = now or datetime.now(UTC)
    state = _load_state()
    messages: list[str] = []

    output = _fleet_status_output()
    if output is not None:
        current = parse_status(output)
        alerts, new_down = decide_fleet_alerts(
            current, dict(state.get("down") or {}), now,
        )
        messages.extend(alerts)
        state["down"] = new_down
    else:
        logger.warning("health watch: fleet status unavailable this cycle")

    try:
        stale_hours = float(os.environ.get("X_INGEST_STALE_HOURS",
                                           DEFAULT_X_STALE_HOURS))
    except ValueError:
        stale_hours = DEFAULT_X_STALE_HOURS
    engine = create_engine(build_db_url())
    msg, is_stale = decide_x_staleness_alert(
        _newest_x_event(engine), bool(state.get("x_stale")), now,
        stale_hours=stale_hours,
    )
    if msg:
        messages.append(msg)
    state["x_stale"] = is_stale

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
                    run_once()
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
