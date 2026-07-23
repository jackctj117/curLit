#!/usr/bin/env python3
"""Weekly performance scorecard — real engine/DB/log metrics only (CL-4nnr).

Streamlit app summarizing the trading week from the system's actual
artifacts (no hardcoded numbers, no synthetic data):

    logs/live_engine.jsonl*          — Health equity samples + KILL SWITCH lines
    portfolio_orders (Postgres)      — order count + per-strategy contributions
    data/research/runs/*.json        — research-loop verdict counts + errors
    docs/research/debates/*/transcript.jsonl — LLM spend ledger (usd_cost)
    prices / macro_data (Postgres)   — data freshness (max row age)

Each metric renders value + target threshold + PASS/FAIL chip; the final
week verdict is GREEN (all pass), AMBER (one non-critical miss or any
metric unavailable) or RED (any critical miss, or two-plus misses).

Run:
    .venv/bin/python -m streamlit run scripts/scorecard.py --server.headless true

Headless-friendly: page config set up-front, no blocking inputs. The
pure parsing/threshold helpers below are unit-tested without DB/network
(tests/unit/test_scorecard.py).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Make src.* imports work when streamlit runs this file from repo root.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

LOG_DIR = ROOT / "logs"
RESEARCH_RUNS_DIR = ROOT / "data" / "research" / "runs"
DEBATES_DIR = ROOT / "docs" / "research" / "debates"

_HEALTH_RE = re.compile(r"^Health: equity=([0-9]+(?:\.[0-9]+)?)")
_KILL_SWITCH_MARKER = "KILL SWITCH"

# --------------------------------------------------------------------------- #
# Targets — value + direction each metric is checked against.
#   mode "max": pass when value <= threshold
#   mode "min": pass when value >= threshold
# critical metrics turn the whole week RED on a miss.
# --------------------------------------------------------------------------- #

TARGETS: dict[str, dict[str, Any]] = {
    # Losing more than 1% of equity in a practice week means sizing or
    # signal quality is off — critical.
    "weekly_pnl_pct": {"mode": "min", "threshold": -1.0, "critical": True, "label": ">= -1.0%"},
    # Any kill-switch trigger is an incident — critical.
    "kill_switch_triggers": {"mode": "max", "threshold": 0, "critical": True, "label": "0"},
    # Over-trading guard; zero trades passes (practice run may be idle).
    "trade_count": {"mode": "max", "threshold": 50, "critical": False, "label": "<= 50 / week"},
    # LLM research budget for the week.
    "research_spend_usd": {
        "mode": "max",
        "threshold": 25.0,
        "critical": False,
        "label": "<= $25 / week",
    },
    # Daily price bars: 5 days tolerates weekend + one stalled ingest.
    "prices_age_days": {"mode": "max", "threshold": 5.0, "critical": False, "label": "<= 5 days"},
    # macro_data has daily series (DFF, DGS10) so a week of silence
    # means FRED ingest is stuck.
    "macro_age_days": {"mode": "max", "threshold": 7.0, "critical": False, "label": "<= 7 days"},
}


@dataclass
class Metric:
    """One scorecard row: real value vs target with pass/fail state.

    passed is None when the underlying source was unavailable (DB down,
    log missing) — surfaced as UNKNOWN and treated as an AMBER
    contributor rather than silently passing.
    """

    key: str
    name: str
    value: float | int | None
    display: str
    target: str
    passed: bool | None
    critical: bool = False
    note: str = ""


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested — no DB / network / streamlit)
# --------------------------------------------------------------------------- #


def check_threshold(value: float, mode: str, threshold: float) -> bool:
    """mode 'max': value <= threshold. mode 'min': value >= threshold."""
    if mode == "max":
        return value <= threshold
    if mode == "min":
        return value >= threshold
    raise ValueError(f"unknown threshold mode {mode!r}")


def _parse_ts(raw: str) -> datetime | None:
    try:
        ts = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def parse_health_equity(lines: Iterable[str]) -> list[tuple[datetime, float]]:
    """Extract (ts, equity) points from engine JSONL Health lines.

    Matches ``"msg": "Health: equity=100000.00"`` records emitted by
    live_engine._health_check_task; malformed lines are skipped.
    """
    points: list[tuple[datetime, float]] = []
    for line in lines:
        line = line.strip()
        if "Health" not in line:
            continue  # cheap pre-filter; JSON decides the rest
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        m = _HEALTH_RE.match(str(rec.get("msg", "")))
        if not m:
            continue
        ts = _parse_ts(str(rec.get("ts", "")))
        if ts is None:
            continue
        points.append((ts, float(m.group(1))))
    points.sort(key=lambda p: p[0])
    return points


def count_kill_switch_triggers(
    lines: Iterable[str],
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    """Count engine JSONL records whose msg contains 'KILL SWITCH'
    (src/risk/kill_switches.py logs 'KILL SWITCH: %s triggered ...')."""
    n = 0
    for line in lines:
        if _KILL_SWITCH_MARKER not in line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if _KILL_SWITCH_MARKER not in str(rec.get("msg", "")):
            continue
        ts = _parse_ts(str(rec.get("ts", "")))
        if ts is None:
            continue
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        n += 1
    return n


def equity_stats(
    points: list[tuple[datetime, float]],
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    """Window the equity samples and derive weekly P&L + daily closes.

    daily_close maps UTC date -> last equity sample of that day;
    daily_pnl is the day-over-day diff. weekly_pnl_pct is relative to
    the first in-window sample. Empty dict when no samples in window.
    """
    window = [(ts, eq) for ts, eq in points if start <= ts <= end]
    if not window:
        return {}
    daily_close: dict[str, float] = {}
    for ts, eq in window:  # points sorted ascending — last write wins
        daily_close[ts.date().isoformat()] = eq
    days = sorted(daily_close)
    daily_pnl = {
        days[i]: daily_close[days[i]] - daily_close[days[i - 1]] for i in range(1, len(days))
    }
    first_eq = window[0][1]
    last_eq = window[-1][1]
    weekly_pnl = last_eq - first_eq
    weekly_pnl_pct = (weekly_pnl / first_eq * 100.0) if first_eq else 0.0
    return {
        "first_equity": first_eq,
        "last_equity": last_eq,
        "weekly_pnl": weekly_pnl,
        "weekly_pnl_pct": weekly_pnl_pct,
        "daily_close": daily_close,
        "daily_pnl": daily_pnl,
        "n_samples": len(window),
    }


def summarize_research_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum verdict/debate counters across research run summaries
    (data/research/runs/*.json, schema src/research/loop.py::RunSummary)."""
    keys = (
        "debates_run",
        "verdicts_promote",
        "verdicts_reject",
        "verdicts_escalate",
        "ideas_proposed",
        "ideas_declined",
        "candidates_implemented",
        "candidates_rejected",
    )
    out: dict[str, Any] = dict.fromkeys(keys, 0)
    errors = 0
    for run in runs:
        for k in keys:
            try:
                out[k] += int(run.get(k, 0) or 0)
            except (TypeError, ValueError):
                continue
        errs = run.get("errors")
        if isinstance(errs, list):
            errors += len(errs)
    out["n_runs"] = len(runs)
    out["errors"] = errors
    return out


def sum_debate_spend(
    lines: Iterable[str],
    start: datetime | None = None,
    end: datetime | None = None,
) -> float:
    """Sum usd_cost across debate-ledger JSONL entries within window
    (docs/research/debates/*/transcript.jsonl, field ``usd_cost``)."""
    total = 0.0
    for line in lines:
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        ts = _parse_ts(str(rec.get("timestamp", "")))
        if start is not None and (ts is None or ts < start):
            continue
        if end is not None and (ts is None or ts > end):
            continue
        try:
            total += float(rec.get("usd_cost", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def build_metric(
    key: str,
    name: str,
    value: float | int | None,
    display: str | None = None,
    note: str = "",
) -> Metric:
    """Assemble a Metric row, evaluating the TARGETS entry for ``key``.
    value=None → UNKNOWN (source unavailable)."""
    spec = TARGETS[key]
    passed = (
        None
        if value is None
        else check_threshold(float(value), spec["mode"], float(spec["threshold"]))
    )
    return Metric(
        key=key,
        name=name,
        value=value,
        display=display if display is not None else ("n/a" if value is None else str(value)),
        target=spec["label"],
        passed=passed,
        critical=bool(spec["critical"]),
        note=note,
    )


def week_verdict(metrics: list[Metric]) -> str:
    """GREEN: every metric passed. RED: any critical miss, or >= 2
    misses. AMBER: one non-critical miss and/or any UNKNOWN source."""
    fails = [m for m in metrics if m.passed is False]
    unknowns = [m for m in metrics if m.passed is None]
    if any(m.critical for m in fails) or len(fails) >= 2:
        return "RED"
    if fails or unknowns:
        return "AMBER"
    return "GREEN"


def engine_log_files(log_dir: Path, start: datetime) -> list[Path]:
    """live_engine.jsonl plus rotated siblings whose date-suffix falls
    on/after the window start (suffix format live_engine.jsonl.YYYY-MM-DD)."""
    files: list[Path] = []
    for p in sorted(log_dir.glob("live_engine.jsonl*")):
        suffix = p.name.removeprefix("live_engine.jsonl").lstrip(".")
        if not suffix:
            files.append(p)
            continue
        try:
            file_date = datetime.strptime(suffix, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            continue
        # Rotated file <date> holds records written up to that date —
        # include it when it could contain in-window records.
        if file_date >= start - timedelta(days=1):
            files.append(p)
    return files


# --------------------------------------------------------------------------- #
# Data collection (real sources — logs, DB, research artifacts)
# --------------------------------------------------------------------------- #


@dataclass
class WeekData:
    """Everything the renderer needs, gathered once per page run."""

    start: datetime
    end: datetime
    equity: dict[str, Any] = field(default_factory=dict)
    equity_points: list[tuple[datetime, float]] = field(default_factory=list)
    kill_switches: int | None = None
    orders_count: int | None = None
    per_strategy: dict[str, int] = field(default_factory=dict)
    orders_note: str = ""
    research: dict[str, Any] = field(default_factory=dict)
    spend_usd: float | None = None
    prices_age_days: float | None = None
    macro_age_days: float | None = None
    freshness_note: str = ""
    log_note: str = ""


def _read_log_lines(files: list[Path]) -> list[str]:
    lines: list[str] = []
    for p in files:
        try:
            lines.extend(p.read_text(errors="replace").splitlines())
        except OSError:
            logger.warning("Could not read %s", p)
    return lines


def _collect_db(data: WeekData) -> None:
    """portfolio_orders + freshness from Postgres; failures leave the
    corresponding metrics UNKNOWN with a note instead of raising."""
    try:
        from sqlalchemy import text

        from src.runtime.run_engine import _build_db_engine

        engine = _build_db_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT ts, symbol, contributions FROM portfolio_orders "
                    "WHERE ts >= :start AND ts <= :end"
                ),
                {"start": data.start, "end": data.end},
            ).fetchall()
            data.orders_count = len(rows)
            per_strategy: dict[str, int] = {}
            for _ts, _sym, contribs in rows:
                if isinstance(contribs, str):
                    try:
                        contribs = json.loads(contribs)
                    except json.JSONDecodeError:
                        contribs = {}
                for sid in contribs or {}:
                    per_strategy[sid] = per_strategy.get(sid, 0) + 1
            data.per_strategy = per_strategy
            if not rows:
                data.orders_note = (
                    "portfolio_orders is empty for this window — the "
                    "practice run recorded no fills."
                )

            now = datetime.now(UTC)
            max_price_ts = conn.execute(text("SELECT max(ts) FROM prices")).scalar()
            if max_price_ts is not None:
                if max_price_ts.tzinfo is None:
                    max_price_ts = max_price_ts.replace(tzinfo=UTC)
                data.prices_age_days = (now - max_price_ts).total_seconds() / 86400.0
            max_macro = conn.execute(text("SELECT max(observation_date) FROM macro_data")).scalar()
            if max_macro is not None:
                macro_ts = datetime(
                    max_macro.year,
                    max_macro.month,
                    max_macro.day,
                    tzinfo=UTC,
                )
                data.macro_age_days = (now - macro_ts).total_seconds() / 86400.0
    except Exception as exc:  # DB down / table missing — report, don't crash
        logger.warning("DB collection failed: %s", exc)
        data.freshness_note = f"DB unavailable: {type(exc).__name__}"
        data.orders_note = data.freshness_note


def collect_week_data(days: int = 7) -> WeekData:
    """Gather all real inputs for the scorecard window (last ``days``)."""
    from src.dotenv_bootstrap import load_project_env

    load_project_env()

    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    data = WeekData(start=start, end=end)

    # --- engine logs: equity health samples + kill switches ---------------- #
    files = engine_log_files(LOG_DIR, start)
    if files:
        lines = _read_log_lines(files)
        data.equity_points = parse_health_equity(lines)
        data.equity = equity_stats(data.equity_points, start, end)
        data.kill_switches = count_kill_switch_triggers(lines, start, end)
    else:
        data.log_note = f"No live_engine.jsonl files found under {LOG_DIR}"

    # --- DB: orders + freshness ------------------------------------------- #
    _collect_db(data)

    # --- research runs + spend --------------------------------------------- #
    runs: list[dict[str, Any]] = []
    if RESEARCH_RUNS_DIR.is_dir():
        for p in sorted(RESEARCH_RUNS_DIR.glob("*.json")):
            try:
                run = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            started = _parse_ts(str(run.get("started_at", "")))
            if started is not None and start <= started <= end:
                runs.append(run)
    data.research = summarize_research_runs(runs)

    if DEBATES_DIR.is_dir():
        ledger_lines: list[str] = []
        for p in DEBATES_DIR.glob("*/transcript.jsonl"):
            try:
                ledger_lines.extend(p.read_text(errors="replace").splitlines())
            except OSError:
                continue
        data.spend_usd = sum_debate_spend(ledger_lines, start, end)
    else:
        data.spend_usd = 0.0

    return data


def build_metrics(data: WeekData) -> list[Metric]:
    """Turn collected week data into the scorecard's Metric rows."""
    eq = data.equity
    pnl_pct = eq.get("weekly_pnl_pct") if eq else None
    pnl_display = f"{eq['weekly_pnl']:+,.2f} ({eq['weekly_pnl_pct']:+.2f}%)" if eq else "n/a"
    metrics = [
        build_metric(
            "weekly_pnl_pct",
            "Weekly P&L",
            pnl_pct,
            pnl_display,
            note=data.log_note or ("" if eq else "no Health equity samples in window"),
        ),
        build_metric(
            "kill_switch_triggers",
            "Kill-switch triggers",
            data.kill_switches,
            note=data.log_note,
        ),
        build_metric(
            "trade_count",
            "Portfolio orders",
            data.orders_count,
            note=data.orders_note,
        ),
        build_metric(
            "research_spend_usd",
            "Research LLM spend",
            data.spend_usd,
            display=("n/a" if data.spend_usd is None else f"${data.spend_usd:,.2f}"),
        ),
        build_metric(
            "prices_age_days",
            "Prices freshness",
            data.prices_age_days,
            display=(
                "n/a" if data.prices_age_days is None else f"{data.prices_age_days:.1f} days old"
            ),
            note=data.freshness_note,
        ),
        build_metric(
            "macro_age_days",
            "Macro data freshness",
            data.macro_age_days,
            display=(
                "n/a" if data.macro_age_days is None else f"{data.macro_age_days:.1f} days old"
            ),
            note=data.freshness_note,
        ),
    ]
    return metrics


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #


def _chip(passed: bool | None) -> str:
    if passed is True:
        return ":green[PASS]"
    if passed is False:
        return ":red[FAIL]"
    return ":orange[UNKNOWN]"


def render() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="curLit Weekly Scorecard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("curLit Weekly Scorecard")

    days = st.sidebar.slider("Window (days)", min_value=1, max_value=30, value=7)

    @st.cache_data(ttl=300)
    def _cached(days_: int) -> WeekData:
        return collect_week_data(days_)

    data = _cached(days)
    st.caption(
        f"Window: {data.start:%Y-%m-%d %H:%M} → {data.end:%Y-%m-%d %H:%M} UTC "
        f"(real engine logs, Postgres, research artifacts — no synthetic numbers)"
    )

    metrics = build_metrics(data)
    verdict = week_verdict(metrics)
    banner = {
        "GREEN": st.success,
        "AMBER": st.warning,
        "RED": st.error,
    }[verdict]
    banner(f"Week verdict: {verdict}")

    st.subheader("Metrics")
    for m in metrics:
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        c1.markdown(f"**{m.name}**" + (" *(critical)*" if m.critical else ""))
        c2.markdown(m.display)
        c3.markdown(f"target {m.target}")
        c4.markdown(_chip(m.passed))
        if m.note:
            st.caption(m.note)

    st.subheader("Equity")
    if data.equity:
        eq_df = pd.DataFrame(
            data.equity_points,
            columns=["ts", "equity"],
        ).set_index("ts")
        eq_df = eq_df[(eq_df.index >= data.start) & (eq_df.index <= data.end)]
        st.line_chart(eq_df["equity"])
        pnl = data.equity.get("daily_pnl", {})
        if pnl:
            st.bar_chart(pd.Series(pnl, name="daily P&L"))
        else:
            st.caption("Fewer than two daily closes in window — no daily P&L bars.")
    else:
        st.caption(data.log_note or "No Health equity samples found in this window — engine idle?")

    st.subheader("Fills by strategy")
    if data.per_strategy:
        st.dataframe(
            pd.DataFrame(
                sorted(data.per_strategy.items()),
                columns=["strategy_id", "orders"],
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption(data.orders_note or "No per-strategy fills recorded this window.")

    st.subheader("Research pipeline")
    r = data.research
    if r.get("n_runs"):
        st.markdown(
            f"Runs: **{r['n_runs']}** · debates: **{r['debates_run']}** · "
            f"promote/reject/escalate: **{r['verdicts_promote']}/"
            f"{r['verdicts_reject']}/{r['verdicts_escalate']}** · "
            f"errors: **{r['errors']}**"
        )
    else:
        st.caption("No research runs recorded in this window.")


if __name__ == "__main__":
    render()
