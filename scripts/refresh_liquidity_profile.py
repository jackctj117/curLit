"""Liquidity-window profile refresh (CL-y412 / CL-4wi5).

Builds the hour-of-week × pair median-spread heatmap that the strategies'
entry gate (``PositionSizer.adjust_for_liquidity``) consults, and persists it
to ``data/liquidity_profile.json``. The engine loads that file at boot; a
missing file means the gate is inert (every entry passes at full size).

Signal source is ``intraday_quotes`` (real OANDA bid/ask for every polled
instrument). That table only retains ~24h, so a single run sees at most one
day's hours — the refresh MERGES onto the existing profile so successive
daily runs accumulate the full 168-hour week while preferring the freshest
sample for any bucket re-measured today.

Usage:
    .venv/bin/python scripts/refresh_liquidity_profile.py --once
    .venv/bin/python scripts/refresh_liquidity_profile.py --loop 86400   # daily
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402
from src.risk.liquidity_window import (  # noqa: E402
    LiquidityProfile,
    build_profile_from_spreads,
    load_profile,
    merge_profiles,
    save_profile,
)

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "data/liquidity_profile.json"
# 30 days requested; the table only retains ~24h, so this is an upper bound —
# the merge accumulates coverage across runs regardless.
_DEFAULT_LOOKBACK_HOURS = 720


def spreads_from_rows(
    rows: list[tuple[datetime, str, float, float]],
) -> list[tuple[datetime, str, float]]:
    """Map ``(ts, symbol, bid, ask)`` DB rows → ``(ts, symbol, spread_bps)``.

    Drops any row without a sane two-sided quote (crossed, non-positive mid).
    Kept separate from the DB call so it is unit-testable without Postgres.
    """
    out: list[tuple[datetime, str, float]] = []
    for ts, symbol, bid, ask in rows:
        try:
            b = float(bid)
            a = float(ask)
        except (TypeError, ValueError):
            continue
        if b <= 0 or a <= 0 or a < b:
            continue
        mid = (a + b) / 2.0
        spread_bps = (a - b) / mid * 10_000.0
        if spread_bps > 0:
            out.append((ts, symbol, spread_bps))
    return out


def build_from_engine(
    engine: Any,
    lookback_hours: int = _DEFAULT_LOOKBACK_HOURS,
) -> LiquidityProfile:
    """Query ``intraday_quotes`` over the lookback window → a fresh profile."""
    sql = text(
        "SELECT ts, symbol, bid, ask FROM intraday_quotes "
        "WHERE ts >= now() - make_interval(hours => :h) "
        "AND bid > 0 AND ask > 0",
    )
    with engine.connect() as conn:
        rows = [
            (r[0], r[1], r[2], r[3]) for r in conn.execute(sql, {"h": lookback_hours}).fetchall()
        ]
    triples = spreads_from_rows(rows)
    logger.info(
        "liquidity refresh: %d quotes -> %d usable spreads over %dh",
        len(rows),
        len(triples),
        lookback_hours,
    )
    return build_profile_from_spreads(triples)


def refresh(
    engine: Any,
    path: str = _DEFAULT_PATH,
    lookback_hours: int = _DEFAULT_LOOKBACK_HOURS,
) -> dict[str, int]:
    """Build from the DB, merge onto any existing profile, persist atomically.

    Returns ``{"buckets", "pairs", "new_buckets"}`` for logging.
    """
    fresh = build_from_engine(engine, lookback_hours)
    existing = load_profile(path)
    merged = merge_profiles(existing, fresh)
    save_profile(merged, path)
    return {
        "buckets": len(merged.median_spread_bps),
        "pairs": len(merged.pair_median_bps),
        "new_buckets": len(fresh.median_spread_bps),
    }


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    from sqlalchemy import create_engine  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Refresh the liquidity-window spread profile.",
    )
    parser.add_argument("--once", action="store_true", help="Refresh once, exit.")
    parser.add_argument(
        "--loop",
        type=int,
        metavar="SECONDS",
        default=None,
        help="Refresh every SECONDS (daemon mode).",
    )
    parser.add_argument(
        "--path", default=_DEFAULT_PATH, help=f"Profile output path (default {_DEFAULT_PATH})."
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=_DEFAULT_LOOKBACK_HOURS,
        help="Quote history window to read.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine = create_engine(build_db_url())

    def _run() -> None:
        stats = refresh(engine, args.path, args.lookback_hours)
        print(
            f"liquidity profile -> {args.path}: buckets={stats['buckets']} "
            f"pairs={stats['pairs']} refreshed_today={stats['new_buckets']}",
        )

    if args.loop:
        logger.info("liquidity refresh: looping every %ds (Ctrl-C to stop)", args.loop)
        try:
            while True:
                _run()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("liquidity refresh: stopped")
            return 0
    else:
        _run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
