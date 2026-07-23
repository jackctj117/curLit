"""Truth Social event-study report (CL-s9as) — RESEARCH ONLY.

Batch analysis over the measured reactions. Answers exactly the research
questions the operator scoped: which topics move markets, how much, how
fast the reaction decays, and how often the initial move reverses.

Usage:
    .venv/bin/python scripts/truth_report.py            # SPY focus
    .venv/bin/python scripts/truth_report.py --instrument QQQ
    .venv/bin/python scripts/truth_report.py --sample 10   # label spot-check
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.db_env import build_db_url  # noqa: E402

logger = logging.getLogger(__name__)


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
        engine,
        params={"inst": instrument},
    )


def reversal_rate(df: pd.DataFrame) -> float | None:
    """Fraction of posts whose 5-min move is mostly gone by 60 min
    (sign flip, or |60m| < half of |5m|)."""
    wide = df.pivot_table(index="post_id", columns="window_minutes", values="return_pct")
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


def print_label_sample(engine, n: int) -> None:  # noqa: ANN001
    """Random labeled posts for classifier-drift spot-checks: read the
    text, judge whether the labels still look right. Version-stamped rows
    make a re-label diff possible if drift is confirmed."""
    df = pd.read_sql(
        """
        SELECT c.post_id, c.is_market_relevant, c.primary_topic, c.tone,
               c.explicit_market_language, c.confidence,
               c.classifier_version, p.text
        FROM truth_classifications c JOIN truth_posts p USING (post_id)
        WHERE p.text <> ''
        ORDER BY RANDOM() LIMIT %(n)s
        """,
        engine,
        params={"n": n},
    )
    if df.empty:
        print("No labeled posts yet.")
        return
    print(f"=== Classifier spot-check sample (n={len(df)}) — do the labels match the text? ===\n")
    for _, r in df.iterrows():
        conf = f"{r.confidence:.2f}" if pd.notna(r.confidence) else "n/a"
        print(
            f"[{r.classifier_version}] relevant={bool(r.is_market_relevant)} "
            f"topic={r.primary_topic} tone={r.tone} "
            f"buy_lang={bool(r.explicit_market_language)} conf={conf}"
        )
        print(f"  {str(r.text)[:220]}\n")


def distribution_table(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Per-topic distribution at one window: count, hit rate (% positive),
    quantiles, std. Averages alone mislead — a +0.1% mean can hide a
    bimodal spike-and-reverse shape; the quantiles show it."""
    w = df[df["window_minutes"] == window]
    if w.empty:
        return pd.DataFrame()
    g = w.groupby("primary_topic")["return_pct"]
    out = pd.DataFrame(
        {
            "n": g.count(),
            "hit_rate": g.apply(lambda s: float((s > 0).mean())),
            "p10": g.quantile(0.10),
            "p25": g.quantile(0.25),
            "p50": g.quantile(0.50),
            "p75": g.quantile(0.75),
            "p90": g.quantile(0.90),
            "std": g.std(),
        }
    )
    return out.round(3)


def print_report(df: pd.DataFrame, instrument: str) -> None:
    if df.empty:
        print(f"No measured reactions for {instrument} yet — let the daemon accumulate data.")
        return
    n_posts = df["post_id"].nunique()
    print(f"=== Truth Social event study — {instrument} ({n_posts} posts) ===\n")

    print("Mean / median return_pct by topic × window (percent):")
    pivot = df.pivot_table(
        index="primary_topic",
        columns="window_minutes",
        values="return_pct",
        aggfunc=["mean", "median", "count"],
    )
    print(pivot.round(3).to_string())

    dist = distribution_table(df, window=30)
    if not dist.empty:
        print(
            "\n30-min return DISTRIBUTION by topic (hit_rate = share "
            "positive; read p10/p90 before trusting any mean):"
        )
        print(dist.to_string())

    rv = reversal_rate(df)
    if rv is not None:
        print(f"\nReversal rate (5m move mostly gone by 60m): {rv:.0%}")

    xm = df[df["window_minutes"] == 30]
    if not xm.empty:
        by_buy = xm.groupby("explicit_market_language")["return_pct"]
        print("\n30-min return by explicit market language (True = 'buy'-style phrasing):")
        print(by_buy.agg(["count", "mean", "median"]).round(3).to_string())
        by_tone = xm.groupby("tone")["return_pct"]
        print("\n30-min return by tone:")
        print(by_tone.agg(["count", "mean", "median"]).round(3).to_string())

    df = df.copy()
    df["hour_et"] = (
        pd.to_datetime(df["posted_at"], utc=True).dt.tz_convert("America/New_York").dt.hour
    )
    tod = df[df["window_minutes"] == 30].groupby("hour_et")["return_pct"]
    print("\n30-min |return| by ET hour of post:")
    print(tod.apply(lambda s: s.abs().mean()).round(3).to_string())


def main(argv: list[str] | None = None) -> int:
    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="SPY")
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        default=None,
        help="print N random labeled posts for a classifier-drift spot-check, then exit",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    engine = create_engine(build_db_url())
    if args.sample:
        print_label_sample(engine, args.sample)
        return 0
    print_report(load_frame(engine, args.instrument), args.instrument)
    return 0


if __name__ == "__main__":
    sys.exit(main())
