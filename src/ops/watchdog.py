"""Watchdog — engine health monitor with auto-restart."""

import logging
import os
import subprocess
import time
from pathlib import Path

import httpx

from src.monitoring.logging_setup import setup_logging

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SEC = 30
FAILURE_THRESHOLD = 3
RESTART_COOLDOWN_SEC = 60
HEARTBEAT_MAX_AGE_SEC = 120


def check_engine_heartbeat() -> float | None:
    try:
        r = httpx.get("http://localhost:8000/metrics", timeout=5)
        r.raise_for_status()
        for line in r.text.split("\n"):
            if 'fx_service_last_heartbeat_timestamp{service="live_engine"}' in line:
                return time.time() - float(line.split()[-1])
        return None
    except Exception:
        return None


def restart_engine() -> None:
    logger.critical("Attempting engine restart")
    subprocess.run(["systemctl", "restart", "fx-live-engine"], check=False)


def main() -> None:
    log_dir = Path(os.environ.get("FX_LOG_DIR", "logs"))
    setup_logging("watchdog", log_dir)

    failures = 0
    while True:
        age = check_engine_heartbeat()
        if age is None:
            failures += 1
            logger.warning("Heartbeat check failed (%d/%d)", failures, FAILURE_THRESHOLD)
        elif age > HEARTBEAT_MAX_AGE_SEC:
            failures += 1
            logger.warning("Heartbeat stale: %.0fs (%d/%d)", age, failures, FAILURE_THRESHOLD)
        else:
            failures = 0

        if failures >= FAILURE_THRESHOLD:
            logger.critical("Engine unresponsive — restarting")
            restart_engine()
            failures = 0
            time.sleep(RESTART_COOLDOWN_SEC)

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
