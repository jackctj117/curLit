#!/usr/bin/env python3
"""Canary deployment — parallel practice instance, behavior diff, promotion."""

import logging
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def run_canary(port: int = 8003, duration_days: int = 7) -> None:
    """Start canary engine on separate port, compare with production."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.runtime.run_engine", "--port", str(port), "--canary"],
    )
    logger.info("Canary started on port %d for %d days", port, duration_days)
    time.sleep(duration_days * 86400)
    proc.terminate()
    logger.info("Canary stopped — check diff reports before promoting")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_canary()


if __name__ == "__main__":
    main()
