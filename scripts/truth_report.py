"""Truth Social event-study report (CL-s9as) — RESEARCH ONLY.

Batch analysis over the measured reactions. Answers exactly the research
questions the operator scoped: which topics move markets, how much, how
fast the reaction decays, and how often the initial move reverses.

Usage:
    .venv/bin/python scripts/truth_report.py            # SPY focus
    .venv/bin/python scripts/truth_report.py --instrument QQQ
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
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


def load_frame(engine, instrument: str) -> pd.DataFrame:  # noqa: ANN001
    return pd.read_sql(
        """
        SELECT r.post_id, r.window_minutes, r.return_pct, r.volume_ratio,
               r.max_favorable_pct, r.max_adverse_pct,
               c.primary_topic, c.tone, c.explicit_market_language,
               p.posted_at
        FROM truth_market_reactions r
        JOIN truth_classifications c ON c.post_id = r.post_id
        JOIN truth_posts p ON p.post_id = r.post_id
        WHERE r.instrument = %(inst)s AND r.return_pct IS NOT NULL
        """,
        engine, params={"inst": instrument},
    )


def reversal_rate(df: pd.DataFrame) -> float | None:
    """Fraction of posts whose 5-min move is mostly gone by 60 min
    (sign flip, or |60m| < half of |5m|)."""
    wide = df.pivot_table(index="post_id", columns="window_minutes",
                          values="return_pct")
    if 5 not in wide.columns or 60 not in wide.columns:
        return None
    both = wide[[5, 60]].dropna()
    both = both[both[5].abs() > 0.05]  # ignore sub-noise 5m moves
    if both.empty:
        return None
    reversed_mask = (
        ((both[5] > 0) & (both[60] < 0))
        | ((both[5] < 0) & (both[60] > 0))
        | (both[60].abs() < both[5].abs() * 0.5)
    )
    return float(reversed_mask.mean())


def print_report(df: pd.DataFrame, instrument: str) -> None:
    if df.empty:
        print(f"No measured reactions for {instrument} yet — let the "
              "daemon accumulate data.")
        return
    n_posts = df["post_id"].nunique()
    print(f"=== Truth Social event study — {instrument} "
          f"({n_posts} posts) ===\n")

    print("Mean / median return_pct by topic × window (percent):")
    pivot = df.pivot_table(index="primary_topic", columns="window_minutes",
                           values="return_pct", aggfunc=["mean", "median",
                                                         "count"])
    print(pivot.round(3).to_string())

    rv = reversal_rate(df)
    if rv is not None:
        print(f"\nReversal rate (5m move mostly gone by 60m): {rv:.0%}")

    xm = df[df["window_minutes"] == 30]
    if not xm.empty:
        by_buy = xm.groupby("explicit_market_language")["return_pct"]
        print("\n30-min return by explicit market language "
              "(True = 'buy'-style phrasing):")
        print(by_buy.agg(["count", "mean", "median"]).round(3).to_string())
        by_tone = xm.groupby("tone")["return_pct"]
        print("\n30-min return by tone:")
        print(by_tone.agg(["count", "mean", "median"]).round(3).to_string())

    df = df.copy()
    df["hour_et"] = (pd.to_datetime(df["posted_at"], utc=True)
                     .dt.tz_convert("America/New_York").dt.hour)
    tod = df[df["window_minutes"] == 30].groupby("hour_et")["return_pct"]
    print("\n30-min |return| by ET hour of post:")
    print(tod.apply(lambda s: s.abs().mean()).round(3).to_string())


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="SPY")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    engine = create_engine(_build_db_url())
    print_report(load_frame(engine, args.instrument), args.instrument)
    return 0


if __name__ == "__main__":
    sys.exit(main())
