"""X watchlist monitor daemon (CL-3j86).

Relays every new post from the curated watchlist in
``configs/x_watchlist.yaml`` to the operator's Telegram chat. Ships
dark: without ``TWITTER_BEARER_TOKEN`` (X API v2 basic tier, paid) the
daemon logs one clear idle line and sleeps — it never crashes and it
NEVER scrapes (Nitter is dead; scraping x.com risks account bans).

Usage:

    # One poll cycle and exit (testing / cron):
    .venv/bin/python -m scripts.x_monitor --once

    # Run forever, one cycle every 300s (systemd / tmux). SIGTERM /
    # SIGINT exit gracefully after the in-flight cycle:
    .venv/bin/python -m scripts.x_monitor --loop 300

    # Print notifications to stdout instead of sending to Telegram:
    .venv/bin/python -m scripts.x_monitor --once --dry-run

Backend selection is via ``X_MONITOR_BACKEND`` (``api`` default;
``cli`` runs an operator-supplied external command — see the
CliTransport docstring in ``src/data/x_monitor/transport.py`` for the
ToS / ban-risk honesty notes before even thinking about it). The monthly
API read budget is ``X_MONITOR_MONTHLY_CAP`` (default 9000).

Credentials auto-load from ``.env`` via the project dotenv bootstrap;
explicit env vars win.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any

# Make `src` importable when launched as a file (scripts/x_monitor.py),
# matching scripts/event_pipeline.py. Harmless under `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.x_monitor import (  # noqa: E402
    DEFAULT_STATE_PATH,
    DEFAULT_USER_IDS_PATH,
    DEFAULT_WATCHLIST_PATH,
    CycleSummary,
    XMonitorConfig,
    XWatchlistMonitor,
    build_transport,
    load_watchlist,
)
from src.research.notifications import notify_operator  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_LOOP_INTERVAL_SEC = 300


def _print_notify(
    title: str,
    message: str,
    priority: int = 0,  # noqa: ARG001 — notify_operator-compatible
    *,
    html: bool = False,
) -> None:
    """--dry-run sink: print what would have gone to Telegram."""
    mode = "HTML" if html else "plain"
    print(f"--- would notify [{mode}] {title} ---")
    print(message)
    print("---")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="X account watchlist monitor → Telegram (CL-3j86)",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Run one poll cycle and exit",
    )
    p.add_argument(
        "--loop", type=int, default=DEFAULT_LOOP_INTERVAL_SEC, metavar="SEC",
        help=f"Seconds between cycles (default {DEFAULT_LOOP_INTERVAL_SEC})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print notifications to stdout instead of sending to Telegram",
    )
    p.add_argument(
        "--watchlist", default=str(DEFAULT_WATCHLIST_PATH),
        help="Path to the watchlist YAML",
    )
    p.add_argument(
        "--state-file", default=str(DEFAULT_STATE_PATH),
        help="Path to the since_id / budget state file",
    )
    p.add_argument(
        "--user-ids-file", default=str(DEFAULT_USER_IDS_PATH),
        help="Path to the handle → user-id cache file",
    )
    return p


def _log_summary(summary: CycleSummary) -> None:
    if summary.idle:
        return  # the monitor already logged the one clear idle line
    if summary.paced:
        logger.info("cycle paced (budget spread across the month); no polls")
        return
    if summary.rate_limited:
        logger.info("cycle skipped: rate-limit backoff active")
        return
    if summary.budget_exhausted:
        logger.info("cycle stopped: monthly read budget exhausted")
        return
    logger.info(
        "cycle complete: polled=%d new_posts=%d ingested=%d "
        "notifications=%d reads_used=%d", summary.polled, summary.new_posts,
        summary.ingested, summary.notifications, summary.reads_used,
    )


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_parser().parse_args(argv)

    config = XMonitorConfig(
        watchlist_path=Path(args.watchlist),
        state_path=Path(args.state_file),
        user_ids_path=Path(args.user_ids_file),
    )
    try:
        accounts = load_watchlist(config.watchlist_path)
        transport = build_transport(config)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    notify: Any = _print_notify if args.dry_run else notify_operator
    try:
        monitor = XWatchlistMonitor(
            config=config,
            accounts=accounts,
            transport=transport,
            notify=notify,
        )
    except RuntimeError as exc:  # corrupt state file — fail loud
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    logger.info(
        "x_monitor starting: %d accounts, backend=%s, cap=%d reads/mo",
        len(accounts), type(transport).__name__, config.monthly_cap,
    )

    if args.once:
        _log_summary(monitor.poll_cycle())
        return 0

    stop = threading.Event()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.info(
            "received %s; stopping after current cycle",
            signal.Signals(signum).name,
        )
        stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not stop.is_set():
        try:
            _log_summary(monitor.poll_cycle())
        except Exception:
            # Belt & braces: a poll cycle must never kill the daemon.
            logger.exception("poll cycle raised unexpectedly; continuing")
        stop.wait(max(1, args.loop))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
