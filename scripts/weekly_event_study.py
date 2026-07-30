"""Weekly event-study runner (CL-s1gb) — recurring edge-decay monitoring.

Once every ``INTERVAL_DAYS`` (7) it runs the CL-z95p event study over the
accumulated data (``scripts/event_study.py --days 30 --options``), writes the
full report to ``data/research/event_study_YYYYMMDD.md``, and Telegrams a
compact summary: sample size, the headline-vs-confirmation decay topline, and
the placebo verdicts — the numbers that decide whether the event edge is
holding as the sample grows. The full report stays on disk; the message is a
pointer plus the deciding rows.

The study itself is read-only research (no trading behavior); this runner
adds only scheduling + summary dispatch. State (last successful run) persists
atomically in ``data/event_study_schedule.json`` so restarts never double-run.

Usage:
    .venv/bin/python scripts/weekly_event_study.py --once     # run if due
    .venv/bin/python scripts/weekly_event_study.py --force    # run now
    .venv/bin/python scripts/weekly_event_study.py --loop 21600
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/event_study_schedule.json")
REPORT_DIR = Path("data/research")
INTERVAL_DAYS = 7.0
#: The study can grind a large window; never let a hung child wedge the loop.
STUDY_TIMEOUT_SEC = 1800


def _load_last_run() -> datetime | None:
    import json

    try:
        raw = json.loads(STATE_PATH.read_text())
        value = datetime.fromisoformat(str(raw.get("last_run", "")))
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    except Exception:
        return None


def is_due(last_run: datetime | None, now: datetime, *, force: bool = False) -> bool:
    """Due when forced, never run, or the interval has elapsed."""
    if force or last_run is None:
        return True
    return (now - last_run) >= timedelta(days=INTERVAL_DAYS)


def extract_summary(report: str, max_lines: int = 18) -> str:
    """Compact Telegram body from the full report (CL-s1gb).

    Pulls the coverage line, the primary-question table, and the placebo
    section's verdict rows. DEFENSIVE: if the report shape ever changes and
    nothing matches, fall back to a pointer-only message rather than failing
    the run — the full report is already on disk either way."""
    lines: list[str] = []
    grabbing = None
    for ln in report.splitlines():
        if ln.startswith("## 5. THE PRIMARY QUESTION"):
            grabbing = "primary"
            lines.append("HEADLINE vs CONFIRMATION (mean bps / hit):")
            continue
        if ln.startswith("## 5b. PLACEBO"):
            grabbing = "placebo"
            lines.append("PLACEBO VERDICTS:")
            continue
        if ln.startswith("## ") and grabbing:
            grabbing = None
            continue
        if grabbing == "primary" and re.match(r"^\| \d+m ", ln):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            # horizon | n | mean | hit | n | mean | hit | n | mean | hit | t | flag
            if len(cells) >= 10:
                lines.append(
                    f"{cells[0]}: HL {cells[2]}bps/{cells[3]} vs CONF {cells[8]}bps/{cells[9]}"
                )
        if grabbing == "placebo":
            if ln.startswith("### "):
                lines.append(ln.removeprefix("### "))
            elif re.match(r"^\| \d+m ", ln):
                cells = [c.strip() for c in ln.strip("|").split("|")]
                if len(cells) >= 6:
                    lines.append(f"  {cells[0]}: {cells[1]}bps, {cells[4]} pctile — {cells[5]}")
    return "\n".join(lines[:max_lines]) if lines else "(summary extraction found no tables)"


def run_once(now: datetime | None = None, *, force: bool = False) -> bool:
    """Run the study if due. Returns True when a run happened."""
    from src.events._util import atomic_write_json  # noqa: PLC0415
    from src.research.notifications import notify_operator  # noqa: PLC0415

    now = now or datetime.now(UTC)
    if not is_due(_load_last_run(), now, force=force):
        return False

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"event_study_{now:%Y%m%d}.md"
    proc = subprocess.run(  # noqa: S603 — our own CLI, fixed argv
        [sys.executable, "scripts/event_study.py", "--days", "30", "--options"],
        capture_output=True,
        text=True,
        timeout=STUDY_TIMEOUT_SEC,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        # Failed runs do NOT stamp state — the next cycle retries.
        logger.error(
            "weekly event study failed (rc=%d): %s",
            proc.returncode,
            (proc.stderr or "")[-400:],
        )
        return False
    out_path.write_text(proc.stdout)
    summary = extract_summary(proc.stdout)
    notify_operator(
        "📐 Weekly event study",
        f"{summary}\n\nFull report: {out_path}",
    )
    atomic_write_json(STATE_PATH, {"last_run": now.isoformat()})
    logger.info("weekly event study: ran, report=%s (%d bytes)", out_path, len(proc.stdout))
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run if due, then exit")
    mode.add_argument("--loop", type=int, metavar="SECONDS", default=None)
    parser.add_argument("--force", action="store_true", help="Run now regardless of schedule")
    args = parser.parse_args(argv)

    from src.dotenv_bootstrap import load_project_env  # noqa: PLC0415

    load_project_env()

    if args.loop:
        logger.info(
            "weekly event study: looping every %ds (interval %.0fd)", args.loop, INTERVAL_DAYS
        )
        while True:
            try:
                run_once(force=args.force)
                args.force = False  # force applies to the first iteration only
            except Exception:
                logger.exception("weekly event study cycle failed — retrying next cycle")
            time.sleep(args.loop)
    else:
        run_once(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
