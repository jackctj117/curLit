"""Live trading engine entrypoint."""

import asyncio
import logging
import os
import signal
from pathlib import Path

from src.monitoring.logging_setup import setup_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    log_dir = Path(os.environ.get("FX_LOG_DIR", "logs"))
    setup_logging("live_engine", log_dir)

    practice = os.environ.get("OANDA_PRACTICE", "true").lower() == "true"
    logger.info("Starting curLit live engine (practice=%s)", practice)

    # Placeholder — full engine wiring in CL-9l7
    loop = asyncio.get_event_loop()

    def shutdown(sig: signal.Signals) -> None:
        logger.info("Received %s — shutting down", sig.name)
        asyncio.create_task(_graceful_shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    logger.info("Engine stopped")


async def _graceful_shutdown() -> None:
    logger.info("Graceful shutdown initiated")
    await asyncio.sleep(1)
    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
