"""Data-health / series-coverage preflight (CL-q4n1).

Turns silent data starvation into a loud, scannable report. A strategy that
gates itself off because its input series is empty or stale looks identical to
"markets are quiet" — until you dig. This checks each series a strategy depends
on for EXISTENCE, minimum COVERAGE, and FRESHNESS, and flags which strategies
are effectively data-starved.

Run at engine startup (logs a WARN summary; never blocks boot — fail loud, keep
running) and standalone via ``scripts/data_health.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Statuses, worst-first for readiness rollups.
EMPTY, SPARSE, STALE, OK = "EMPTY", "SPARSE", "STALE", "OK"
_SEVERITY = {EMPTY: 3, SPARSE: 2, STALE: 1, OK: 0}
_ICON = {EMPTY: "🔴", SPARSE: "🟠", STALE: "🟡", OK: "✅"}


@dataclass(frozen=True)
class SeriesRequirement:
    series_id: str
    kind: str  # "macro" | "price"
    needed_by: str
    min_rows: int = 100
    max_staleness_days: int = 7


@dataclass
class SeriesHealth:
    series_id: str
    kind: str
    needed_by: str
    rows: int
    last: date | None
    status: str
    detail: str


#: The series the live strategies actually gate on (from live_portfolio.yaml).
#: When one of these is EMPTY/SPARSE, the consuming strategy cannot trade.
DEFAULT_REQUIREMENTS: tuple[SeriesRequirement, ...] = (
    SeriesRequirement("US2Y_MINUS_DE2Y", "macro", "rate_diff (signal)", 250, 5),
    SeriesRequirement("EURUSD", "price", "rate_diff (price)", 250, 5),
    SeriesRequirement("CVIX", "macro", "rate_diff/carry_vol (regime)", 120, 5),
    SeriesRequirement("US_3M_OIS", "macro", "rate_diff/carry_vol (carry)", 60, 7),
    SeriesRequirement("EUR_3M_ESTR_OIS", "macro", "rate_diff/carry_vol (carry)", 60, 7),
    SeriesRequirement("DGS2", "macro", "rate context", 250, 5),
    SeriesRequirement("USDJPY", "price", "carry_vol", 250, 5),
    SeriesRequirement("GBPUSD", "price", "carry_vol", 250, 5),
)


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _check_one(
    engine: Any, req: SeriesRequirement, today: date,
) -> SeriesHealth:
    if req.kind == "price":
        sql = "SELECT COUNT(*), MAX(ts) FROM prices WHERE symbol = :s"
    else:
        sql = ("SELECT COUNT(*), MAX(observation_date) FROM macro_data "
               "WHERE series_id = :s")
    try:
        with engine.connect() as conn:
            rows, last_raw = conn.execute(text(sql), {"s": req.series_id}).one()
    except Exception:
        logger.debug("data health: query failed for %s", req.series_id,
                     exc_info=True)
        rows, last_raw = 0, None
    rows = int(rows or 0)
    last = _as_date(last_raw)

    if rows == 0:
        status, detail = EMPTY, "0 rows — series never populated"
    elif rows < req.min_rows:
        status, detail = SPARSE, f"only {rows} rows (< {req.min_rows} needed)"
    elif last is not None and (today - last).days > req.max_staleness_days:
        status = STALE
        detail = f"last {last} ({(today - last).days}d old > {req.max_staleness_days}d)"
    else:
        status, detail = OK, f"{rows} rows, last {last}"
    return SeriesHealth(req.series_id, req.kind, req.needed_by, rows, last,
                        status, detail)


def check_series(
    engine: Any,
    requirements: Sequence[SeriesRequirement] | None = None,
    now: datetime | None = None,
) -> list[SeriesHealth]:
    """Check every required series → per-series health, worst-status-first."""
    reqs = requirements if requirements is not None else DEFAULT_REQUIREMENTS
    today = (now or datetime.now(UTC)).date()
    results = [_check_one(engine, r, today) for r in reqs]
    results.sort(key=lambda h: (-_SEVERITY[h.status], h.series_id))
    return results


def strategy_readiness(results: Sequence[SeriesHealth]) -> dict[str, str]:
    """Worst series status per consuming strategy label."""
    out: dict[str, str] = {}
    for h in results:
        cur = out.get(h.needed_by)
        if cur is None or _SEVERITY[h.status] > _SEVERITY[cur]:
            out[h.needed_by] = h.status
    return out


def starved(results: Sequence[SeriesHealth]) -> list[SeriesHealth]:
    """Series that BLOCK trading (empty or too sparse to fit a model)."""
    return [h for h in results if h.status in (EMPTY, SPARSE)]


def format_report(results: Sequence[SeriesHealth]) -> str:
    """Scannable multi-line report."""
    lines = [f"DATA HEALTH ({len(results)} series checked):"]
    for h in results:
        lines.append(
            f"  {_ICON[h.status]} {h.status:6} {h.series_id:18} "
            f"{h.kind:5} — {h.detail}  [{h.needed_by}]",
        )
    blocked = starved(results)
    if blocked:
        names = ", ".join(f"{h.series_id}" for h in blocked)
        lines.append(f"⚠ STARVED (blocks trading): {names}")
    else:
        lines.append("✅ no data-starved strategies")
    return "\n".join(lines)


def log_startup_health(engine: Any, now: datetime | None = None) -> list[SeriesHealth]:
    """Preflight for engine startup: check + log the report (WARN when any
    series is starved, INFO otherwise). NEVER raises — a health-check failure
    must not stop the engine booting."""
    try:
        results = check_series(engine, now=now)
    except Exception:
        logger.warning("data-health preflight failed to run", exc_info=True)
        return []
    report = format_report(results)
    if starved(results):
        logger.warning("DATA-HEALTH PREFLIGHT — starvation detected:\n%s", report)
    else:
        logger.info("DATA-HEALTH PREFLIGHT ok:\n%s", report)
    return results
