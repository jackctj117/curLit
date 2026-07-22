"""Alpaca paper options executor daemon (CL-ldd2).

Auto-executes the desk's red-team-survived niche options ideas on Alpaca paper
(policy defaults: niche + red-team-survived, confidence >= 0.55, 1 contract,
premium <= $500, <= 5/day). Market-hours-aware — a no-op when the options
market is closed. The Telegram advisory feed is separate and unaffected.

Usage:
    .venv/bin/python scripts/execute_options.py --once
    .venv/bin/python scripts/execute_options.py --loop 300
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def _build_db_url() -> str:
    explicit = os.environ.get("DATABASE_URL")
    if explicit:
        return explicit
    user = os.environ.get("POSTGRES_USER", "fx")
    password = os.environ.get("POSTGRES_PASSWORD", "changeme")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "fx")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _config_from_env():  # noqa: ANN202
    from src.execution.alpaca_options_executor import OptionsExecConfig  # noqa: PLC0415

    return OptionsExecConfig(
        min_confidence=_f("ALPACA_OPT_MIN_CONFIDENCE", 0.55),
        max_premium_usd=_f("ALPACA_OPT_MAX_PREMIUM", 500.0),
        qty=_i("ALPACA_OPT_QTY", 1),
        max_per_day=_i("ALPACA_OPT_MAX_PER_DAY", 5),
        # Paper-phase knobs: default True (strict); set 0 to widen the pool so
        # the paper track record actually accumulates a sample.
        require_niche=_b("ALPACA_OPT_REQUIRE_NICHE", default=True),
        require_red_team=_b("ALPACA_OPT_REQUIRE_RED_TEAM", default=True),
        # Technical-alignment gate (CL-3xoj): skip ideas whose computed price
        # structure scores below this against the thesis. -1.01 disables.
        min_alignment=_f("ALPACA_OPT_MIN_ALIGNMENT", -0.4),
    )


def _exit_config_from_env():  # noqa: ANN202
    from src.execution.alpaca_options_exit import OptionsExitConfig  # noqa: PLC0415

    return OptionsExitConfig(
        stop_loss_pct=_f("ALPACA_OPT_STOP_LOSS_PCT", 0.40),
        profit_target_pct=_f("ALPACA_OPT_PROFIT_TARGET_PCT", 0.80),
        expiry_protect_days=_i("ALPACA_OPT_EXPIRY_PROTECT_DAYS", 4),
        default_time_stop_days=_i("ALPACA_OPT_DEFAULT_TIME_STOP_DAYS", 10),
    )


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()

    parser = argparse.ArgumentParser(description="Alpaca paper options executor.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or not secret:
        logger.error("ALPACA_API_KEY / ALPACA_API_SECRET not set — cannot run")
        return 2
    if os.environ.get("ALPACA_OPTIONS_ENABLED", "").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        logger.error("ALPACA_OPTIONS_ENABLED is not set — refusing to trade")
        return 3

    from src.execution.alpaca_options import AlpacaOptionsClient  # noqa: PLC0415
    from src.execution.alpaca_options_executor import (  # noqa: PLC0415
        execute_pending_options,
    )
    from src.execution.alpaca_options_exit import (  # noqa: PLC0415
        manage_option_exits,
    )

    paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    engine = create_engine(_build_db_url())
    client = AlpacaOptionsClient(key, secret, paper=paper)
    cfg = _config_from_env()
    exit_enabled = _b("ALPACA_OPT_EXIT_ENABLED", default=True)
    exit_cfg = _exit_config_from_env()
    logger.info("alpaca options executor: paper=%s policy=%s exit=%s %s",
                paper, cfg, exit_enabled, exit_cfg)

    def _run() -> None:
        # Exits BEFORE entries (CL-3rho): manage what we hold, then buy.
        if exit_enabled:
            exits = manage_option_exits(engine, client, cfg=exit_cfg)
            print(f"alpaca option exits: {exits}")
        counts = execute_pending_options(engine, client, cfg=cfg)
        print(f"alpaca options: {counts}")

    if args.loop:
        logger.info("looping every %ds (Ctrl-C to stop)", args.loop)
        try:
            while True:
                _run()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("alpaca options executor: stopped")
        return 0

    _run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
