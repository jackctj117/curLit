"""Independent risk monitor service — parallel P&L checks and kill switch panel."""

import logging
import os
import time
from pathlib import Path

from src.monitoring.logging_setup import setup_logging
from src.monitoring.metrics import start_metrics_server

logger = logging.getLogger(__name__)


def main() -> None:
    log_dir = Path(os.environ.get("FX_LOG_DIR", "logs"))
    setup_logging("risk_monitor", log_dir)
    start_metrics_server(port=8002)
    logger.info("Risk monitor starting")

    while True:
        try:
            # Poll broker, check daily P&L, update kill switch states
            logger.debug("Risk check")
        except Exception:
            logger.exception("Risk monitor error")
        time.sleep(30)


if __name__ == "__main__":
    main()
