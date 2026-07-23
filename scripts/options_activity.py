"""Daily options-activity snapshot daemon (CL-mtum).

Scans the option chains of the ACTIVE-idea equity universe — tickers
carrying a still-live trade idea (``pending`` advisory OR ``taken``/open
position) — and snapshots volume/OI/P-C/IV into options_activity, the
free confirmation signal (see src/events/options_activity.py for honest
scope).

CL-7vn9: the snapshot set is deliberately the ACTIVE ideas only, not a
static full watch universe — dead ideas (``expired``/``cancelled``/
``closed``) carry no live thesis to confirm, so their chains are never
fetched. This keeps the per-cycle yfinance call count proportional to
open exposure, not to the whole history of ever-considered tickers.

Usage:
    .venv/bin/python scripts/options_activity.py --once
    .venv/bin/python scripts/options_activity.py --loop 86400
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,5}$")
MAX_TICKERS = 60

#: CL-7vn9 — "active" = still-live idea. ``pending`` (advisory, awaiting
#: a decision) and ``taken`` (acted, position open) both carry a thesis
#: worth confirming with fresh chain data; ``expired``/``cancelled``/
#: ``closed`` do not, so they are excluded and never fetched from Yahoo.
_ACTIVE_STATUSES = ("pending", "taken")


def _candidate_tickers(engine) -> list[str]:  # noqa: ANN001
    """Equity tickers with an ACTIVE idea (pending or taken/open), most
    recent first, capped. Dead ideas are excluded to keep the yfinance
    chain-fetch set proportional to live exposure (CL-7vn9)."""
    placeholders = ", ".join(f":s{i}" for i in range(len(_ACTIVE_STATUSES)))
    params = {f"s{i}": s for i, s in enumerate(_ACTIVE_STATUSES)}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT ticker, MAX(created_at) AS latest FROM trade_ideas "
                f"WHERE status IN ({placeholders}) GROUP BY ticker "
                "ORDER BY latest DESC",
            ),
            params,
        ).all()
    out: list[str] = []
    for ticker, _latest in rows:
        t = str(ticker or "").strip().upper()
        if _TICKER_RE.match(t):
            out.append(t)
        if len(out) >= MAX_TICKERS:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    parser = argparse.ArgumentParser(description="Options-activity snapshots.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", type=int, metavar="SECONDS", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.events.options_activity import snapshot_tickers  # noqa: PLC0415

    engine = create_engine(build_db_url())

    def _run() -> None:
        tickers = _candidate_tickers(engine)
        logger.info("options activity: scanning %d candidate tickers", len(tickers))
        counts = snapshot_tickers(engine, tickers)
        print(
            f"options-activity: written={counts['written']} "
            f"skipped={counts['skipped']} of {len(tickers)}"
        )

    if args.loop:
        logger.info("looping every %ds (Ctrl-C to stop)", args.loop)
        try:
            while True:
                _run()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            logger.info("options activity: stopped")
        return 0

    _run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
