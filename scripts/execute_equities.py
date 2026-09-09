"""Alpaca paper EQUITY (shares) executor daemon (CL-ncbq).

Runs the share book: exits first, then entries, every cycle. Expresses the
SAME advisory ideas the options daemon trades as contracts (CL-ldd2), so the
two books are a live A/B on one signal set.

WHY (CL-4c7o): the desk's equity ideas were right on DIRECTION 80% of the
time (41/51) while the short-dated OTM options expressing them won 7% —
spread and theta ate the 1-2% moves the theses produced. Shares carry the
same call with a penny spread and no decay.

Master switch: ``ALPACA_EQUITY_ENABLED``. Default OFF — an unset switch logs
"disabled" and idles, so deploying this file (or adding it to the fleet
roster) can never start trading a second book by accident.

Live-trading gate (CL-8lv6, shared with the options daemon): paper is the
default. ``ALPACA_PAPER=false`` alone is NOT enough — live mode needs BOTH
``ALPACA_LIVE_UNLOCK=1`` and ``--confirm-live``.

Usage:
    .venv/bin/python scripts/execute_equities.py --once
    .venv/bin/python scripts/execute_equities.py --loop 300
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
    from src.execution.alpaca_equity_executor import EquityExecConfig  # noqa: PLC0415

    return EquityExecConfig(
        min_confidence=_f("ALPACA_EQ_MIN_CONFIDENCE", 0.55),
        # Fixed dollar sleeve per idea — qty = floor(notional / price), so
        # outcomes are comparable across very different share prices.
        notional_usd=_f("ALPACA_EQ_NOTIONAL_USD", 1000.0),
        max_per_day=_i("ALPACA_EQ_MAX_PER_DAY", 10),
        max_per_hour=_i("ALPACA_EQ_MAX_PER_HOUR", 2),
        # Paper-phase knobs: default True (strict); set 0 to widen the pool so
        # the paper track record accumulates a sample faster.
        require_niche=_b("ALPACA_EQ_REQUIRE_NICHE", default=True),
        require_red_team=_b("ALPACA_EQ_REQUIRE_RED_TEAM", default=True),
        min_alignment=_f("ALPACA_EQ_MIN_ALIGNMENT", -0.4),
        entry_delay_min=_i("ALPACA_EQ_ENTRY_DELAY_MIN", 15),
        entry_delay_override_conf=_f("ALPACA_EQ_ENTRY_DELAY_OVERRIDE_CONF", 0.80),
        min_idea_life_days=_f("ALPACA_EQ_MIN_IDEA_LIFE_DAYS", 3.0),
        # Separately killable: shorting carries borrow/locate risk and
        # unbounded loss that the long side does not.
        allow_short=_b("ALPACA_EQ_ALLOW_SHORT", default=True),
    )


def _exit_config_from_env():  # noqa: ANN202
    from src.execution.alpaca_equity_exit import EquityExitConfig  # noqa: PLC0415

    return EquityExitConfig(
        # SHARE moves, not premium moves — an order of magnitude tighter than
        # the options book's 40/80% for exactly that reason.
        stop_loss_pct=_f("ALPACA_EQ_STOP_LOSS_PCT", 0.05),
        profit_target_pct=_f("ALPACA_EQ_PROFIT_TARGET_PCT", 0.10),
        default_time_stop_days=_i("ALPACA_EQ_TIME_STOP_DAYS", 10),
    )


def _live_gate_error(*, paper: bool, confirm_live: bool) -> str | None:
    """Dual live-trading gate (CL-8lv6), identical to the options daemon's.

    Returns an error message when live (non-paper) trading is requested
    without BOTH ``ALPACA_LIVE_UNLOCK=1`` and ``--confirm-live``; ``None``
    means clear to start. The gate exists so an ``ALPACA_PAPER=false`` left in
    a .env can never silently flip a restarting daemon to real money.
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

    parser = argparse.ArgumentParser(description="Alpaca paper equity (shares) executor.")
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

    # Master switch FIRST and default-off: the daemon is in the fleet roster,
    # so an unset switch must be a quiet idle, not a crash loop the watchdog
    # keeps paging about.
    if not _b("ALPACA_EQUITY_ENABLED", default=False):
        logger.info("ALPACA_EQUITY_ENABLED is not set — equity book disabled, idling")
        if args.loop:
            try:
                while True:
                    time.sleep(args.loop)
            except KeyboardInterrupt:
                logger.info("alpaca equity executor: stopped")
        return 0

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or not secret:
        logger.error("ALPACA_API_KEY / ALPACA_API_SECRET not set — cannot run")
        return 2

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

    from src.execution.alpaca_equity import AlpacaEquityClient  # noqa: PLC0415
    from src.execution.alpaca_equity_executor import execute_pending_equities  # noqa: PLC0415
    from src.execution.alpaca_equity_exit import manage_equity_exits  # noqa: PLC0415

    engine = create_engine(build_db_url())
    client = AlpacaEquityClient(key, secret, paper=paper)
    cfg = _config_from_env()
    exit_enabled = _b("ALPACA_EQ_EXIT_ENABLED", default=True)
    exit_cfg = _exit_config_from_env()
    logger.info(
        "alpaca equity executor: paper=%s policy=%s exit=%s %s", paper, cfg, exit_enabled, exit_cfg
    )

    def _run() -> None:
        if _b("ALPACA_LEDGER_CLOSE_ONLY", default=False):
            if not paper:
                raise RuntimeError("Recovery ledger mode is paper-only")
            from src.execution.alpaca_ledger_exits import close_only_cycle  # noqa: PLC0415
            from src.execution.alpaca_recovery import PaperEvidenceClient  # noqa: PLC0415

            evidence = PaperEvidenceClient(key, secret)
            try:
                if exit_enabled:
                    logger.info(
                        "Ledger close-only: %s",
                        close_only_cycle(engine, evidence, client, book="equities", cfg=exit_cfg),
                    )
            except Exception as exc:
                logger.error(
                    "Ledger recovery blocked: %s; entries remain paused", type(exc).__name__
                )
            finally:
                evidence.close()
            return  # NEVER fall through to legacy writers or new entries.
        # Exits BEFORE entries: manage what we hold, then buy.
        if exit_enabled:
            exits = manage_equity_exits(engine, client, cfg=exit_cfg)
            logger.info("alpaca equity exits: %s", exits)
        counts = execute_pending_equities(engine, client, cfg=cfg)
        logger.info("alpaca equity entries: %s", counts)

    if args.loop:
        logger.info("looping every %ds (Ctrl-C to stop)", args.loop)
        try:
            while True:
                _run()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("alpaca equity executor: stopped")
        return 0

    _run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
