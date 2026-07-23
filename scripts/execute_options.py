"""Alpaca paper options executor daemon (CL-ldd2).

Auto-executes the desk's red-team-survived niche options ideas on Alpaca paper
(policy defaults: niche + red-team-survived, confidence >= 0.55, 1 contract,
premium <= $500, <= 5/day). Market-hours-aware — a no-op when the options
market is closed. The Telegram advisory feed is separate and unaffected.

Live-trading gate (CL-8lv6): paper mode is the default and needs nothing.
Setting ``ALPACA_PAPER=false`` alone is NOT enough to trade real money —
mirroring OANDA's ``--confirm-live``, live mode requires BOTH the env var
``ALPACA_LIVE_UNLOCK=1`` AND the ``--confirm-live`` CLI flag; anything
less refuses to start with a loud error.

Usage:
    .venv/bin/python scripts/execute_options.py --once
    .venv/bin/python scripts/execute_options.py --loop 300
    # LIVE (real money — dual gate required):
    ALPACA_PAPER=false ALPACA_LIVE_UNLOCK=1 \\
        .venv/bin/python scripts/execute_options.py --confirm-live --once
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

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)


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
        max_per_day=_i("ALPACA_OPT_MAX_PER_DAY", 10),
        # Intraday pacing so buys spread across the day (CL-h02l).
        max_per_hour=_i("ALPACA_OPT_MAX_PER_HOUR", 2),
        # Paper-phase knobs: default True (strict); set 0 to widen the pool so
        # the paper track record actually accumulates a sample.
        require_niche=_b("ALPACA_OPT_REQUIRE_NICHE", default=True),
        require_red_team=_b("ALPACA_OPT_REQUIRE_RED_TEAM", default=True),
        # Technical-alignment gate (CL-3xoj): skip ideas whose computed price
        # structure scores below this against the thesis. -1.01 disables.
        min_alignment=_f("ALPACA_OPT_MIN_ALIGNMENT", -0.4),
        # Open-spread protection: no entries in the first N minutes of the
        # session (0 disables); conf >= override enters immediately.
        entry_delay_min=_i("ALPACA_OPT_ENTRY_DELAY_MIN", 15),
        entry_delay_override_conf=_f("ALPACA_OPT_ENTRY_DELAY_OVERRIDE_CONF", 0.80),
    )


def _exit_config_from_env():  # noqa: ANN202
    from src.execution.alpaca_options_exit import OptionsExitConfig  # noqa: PLC0415

    return OptionsExitConfig(
        stop_loss_pct=_f("ALPACA_OPT_STOP_LOSS_PCT", 0.40),
        entry_day_extreme_stop_pct=_f("ALPACA_OPT_ENTRY_DAY_EXTREME_STOP", 0.60),
        profit_target_pct=_f("ALPACA_OPT_PROFIT_TARGET_PCT", 0.80),
        expiry_protect_days=_i("ALPACA_OPT_EXPIRY_PROTECT_DAYS", 4),
        expiry_protect_min_profit=_f("ALPACA_OPT_EXPIRY_MIN_PROFIT", 0.25),
        default_time_stop_days=_i("ALPACA_OPT_DEFAULT_TIME_STOP_DAYS", 10),
        # Settle window: suppress premium stops for the first N min after
        # entry so opening spread on cheap contracts can't trip them (CL-h02l).
        entry_settle_min=_i("ALPACA_OPT_ENTRY_SETTLE_MIN", 15),
    )


def _live_gate_error(*, paper: bool, confirm_live: bool) -> str | None:
    """Dual live-trading gate (CL-8lv6), mirroring OANDA's --confirm-live.

    Returns an error message when live (non-paper) trading is requested
    without BOTH ``ALPACA_LIVE_UNLOCK=1`` (env) and ``--confirm-live``
    (CLI); ``None`` means clear to start. Paper mode always passes — the
    gate exists so an ``ALPACA_PAPER=false`` left in a .env can never
    silently flip a restarting daemon to real money.
    """
    if paper:
        return None
    missing: list[str] = []
    if not _b("ALPACA_LIVE_UNLOCK", default=False):
        missing.append("env ALPACA_LIVE_UNLOCK=1")
    if not confirm_live:
        missing.append("the --confirm-live CLI flag")
    if missing:
        return (
            "ALPACA_PAPER=false requests LIVE trading with REAL MONEY, but "
            f"the live gate is not satisfied: missing {' and '.join(missing)}. "
            "Refusing to start. Either unset ALPACA_PAPER (paper is the "
            "default) or supply BOTH gates deliberately."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    parser = argparse.ArgumentParser(description="Alpaca paper options executor.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None)
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Second half of the LIVE-trading dual gate (with "
        "ALPACA_LIVE_UNLOCK=1). Required when ALPACA_PAPER=false; "
        "a no-op in paper mode.",
    )
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
        "1",
        "true",
        "yes",
        "on",
    ):
        logger.error("ALPACA_OPTIONS_ENABLED is not set — refusing to trade")
        return 3

    paper = _b("ALPACA_PAPER", default=True)
    gate_error = _live_gate_error(paper=paper, confirm_live=args.confirm_live)
    if gate_error is not None:
        logger.error(gate_error)
        return 4
    if not paper:
        logger.warning(
            "LIVE MODE ARMED: ALPACA_PAPER=false with ALPACA_LIVE_UNLOCK=1 "
            "and --confirm-live — orders will use REAL MONEY",
        )

    from src.execution.alpaca_options import AlpacaOptionsClient  # noqa: PLC0415
    from src.execution.alpaca_options_executor import (  # noqa: PLC0415
        execute_pending_options,
    )
    from src.execution.alpaca_options_exit import (  # noqa: PLC0415
        manage_option_exits,
    )

    engine = create_engine(build_db_url())
    client = AlpacaOptionsClient(key, secret, paper=paper)
    cfg = _config_from_env()
    exit_enabled = _b("ALPACA_OPT_EXIT_ENABLED", default=True)
    exit_cfg = _exit_config_from_env()
    logger.info(
        "alpaca options executor: paper=%s policy=%s exit=%s %s", paper, cfg, exit_enabled, exit_cfg
    )

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
