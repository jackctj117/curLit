#!/usr/bin/env python3
"""Update staging workflow — git branch per update, test suite gate."""

import logging
import subprocess
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def create_update_branch(pkg: str, version: str) -> None:
    """Create a branch and run test suite for an updated package."""
    branch = f"update/{pkg}-{version}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    logger.info("Created branch %s", branch)

    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", pkg], check=True)

    result = subprocess.run(["make", "check"], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("PASS: %s -> %s", pkg, version)
    else:
        logger.error("FAIL: %s -> %s\n%s", pkg, version, result.stderr[:500])


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: apply_updates.py <package> [version]")
        print("  or: apply_updates.py --all (reads check_updates.py output)")
        sys.exit(1)
    pkg = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else "latest"
    create_update_branch(pkg, version)


if __name__ == "__main__":
    main()
