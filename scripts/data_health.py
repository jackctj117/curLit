"""Data-health / series-coverage check (CL-q4n1).

Surfaces data-starved strategies. Exits non-zero when any required series is
starved, so it's usable as a cron/CI gate.

Usage:
    .venv/bin/python scripts/data_health.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.monitoring.data_health import (  # noqa: PLC0415
        check_series,
        format_report,
        starved,
    )

    engine = create_engine(build_db_url())
    results = check_series(engine)
    print(format_report(results))
    return 1 if starved(results) else 0


if __name__ == "__main__":
    sys.exit(main())
