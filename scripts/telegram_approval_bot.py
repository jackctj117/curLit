"""Telegram approval bot entrypoint (CL-b1l6).

Runs the long-polling approval bot from
``src.research.telegram_approvals`` so the operator can approve /
reject / skip GATE 1 and GATE 2 entries by replying in the same
Telegram chat that receives the gate notifications.

Usage:

    # Run forever (systemd / tmux). SIGTERM/SIGINT exit gracefully
    # after the in-flight long poll completes.
    .venv/bin/python -m scripts.telegram_approval_bot

    # Drain one getUpdates batch and exit (testing / cron).
    .venv/bin/python -m scripts.telegram_approval_bot --once

Requires ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID`` (auto-loaded
from ``.env`` via the project dotenv bootstrap; explicit env wins).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from types import FrameType

from src.research.loop import DEFAULT_STATE_PATH
from src.research.telegram_approvals import (
    DEFAULT_OFFSET_PATH,
    DEFAULT_POLL_TIMEOUT_SEC,
    TelegramApprovalBot,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Telegram approval bot for the research-loop gates",
    )
    p.add_argument(
        "--state", default=str(DEFAULT_STATE_PATH),
        help="Path to the research-loop state file",
    )
    p.add_argument(
        "--offset-file", default=str(DEFAULT_OFFSET_PATH),
        help="Path to the getUpdates offset persistence file",
    )
    p.add_argument(
        "--poll-timeout", type=int, default=DEFAULT_POLL_TIMEOUT_SEC,
        help="getUpdates long-poll timeout in seconds",
    )
    p.add_argument(
        "--once", action="store_true",
        help=(
            "Drain one getUpdates batch (timeout=0) and exit — for "
            "testing and cron-driven operation"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Auto-load .env so credentials are available without sourcing the
    # file first. Explicit env vars still win (override=False).
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_parser().parse_args(argv)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print(
            "ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set "
            "(env or .env)",
            file=sys.stderr,
        )
        return 2

    bot = TelegramApprovalBot(
        token=token,
        chat_id=chat_id,
        state_path=args.state,
        offset_path=args.offset_file,
        poll_timeout_sec=args.poll_timeout,
    )

    if args.once:
        n = bot.poll_once(timeout_sec=0)
        logger.info("--once drain complete: %d update(s) processed", n)
        return 0

    stop = threading.Event()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.info(
            "received %s; stopping after current poll",
            signal.Signals(signum).name,
        )
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    bot.run_forever(stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
