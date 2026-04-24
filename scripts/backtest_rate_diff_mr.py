#!/usr/bin/env python3
"""Rate diff mean reversion backtest — 10+ year walk-forward on EUR/USD."""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Rate diff backtest — stub (wire to WalkForwardRunner + RateDiffMRStrategy)")
    logger.info("Expected: Sharpe 0.5-1.0, max DD 10-18%, hit rate 55-65%")


if __name__ == "__main__":
    main()
