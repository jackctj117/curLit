#!/usr/bin/env python3
"""Soak test monitor — capture engine state every 10 minutes for stability analysis."""

import json
import time
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
SOAK_LOG = LOG_DIR / "soak_test.jsonl"
CHECK_INTERVAL = 600  # 10 minutes
DURATION_HOURS = 24


def capture_state() -> dict:
    """Capture relevant system state for soak test analysis."""
    state = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "engine_pid": _find_engine_pid(),
    }
    try:
        import httpx
        r = httpx.get("http://localhost:8099/metrics", timeout=5)
        for line in r.text.split("\n"):
            if "fx_account_equity_usd" in line and not line.startswith("#"):
                state["equity"] = float(line.split()[-1])
            if "fx_signals_generated_total" in line and not line.startswith("#"):
                state["signals"] = float(line.split()[-1])
            if "fx_errors_total" in line and not line.startswith("#"):
                state["errors"] = float(line.split()[-1])
    except Exception:
        state["metrics"] = "unreachable"

    try:
        import psutil
        proc = psutil.Process(state["engine_pid"]) if state["engine_pid"] else None
        if proc:
            mem = proc.memory_info()
            state["memory_mb"] = round(mem.rss / 1024 / 1024, 1)
            state["cpu_pct"] = round(proc.cpu_percent(interval=0.1), 1)
    except Exception:
        pass

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOAK_LOG, "a") as f:
        f.write(json.dumps(state) + "\n")
    return state


def _find_engine_pid() -> int | None:
    """Find the actual python engine PID, not the shell wrapper that launched it.

    A run launched via `python -m src.runtime.run_engine` from a shell ends up
    with two pgrep matches: the zsh wrapper holding the eval string AND the
    python process. We want the python process — filter by checking each
    candidate's `comm` (basename of the executable) for "python".
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "src\\.runtime\\.run_engine"],
            capture_output=True, text=True,
        )
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip().isdigit()]
        for pid in pids:
            comm = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm="],
                capture_output=True, text=True,
            )
            basename = comm.stdout.strip().lower()
            if "python" in basename:
                return pid
        return pids[0] if pids else None
    except Exception:
        return None


def main() -> None:
    print(f"Soak test starting — {DURATION_HOURS}h duration, {CHECK_INTERVAL}s intervals")
    start = datetime.now(timezone.utc)
    checks = 0

    while (datetime.now(timezone.utc) - start).total_seconds() < DURATION_HOURS * 3600:
        state = capture_state()
        checks += 1
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() / 3600
        mem = state.get("memory_mb", "?")
        sig = state.get("signals", "?")
        err = state.get("errors", "?")
        print(f"[{elapsed:.1f}h] check={checks} mem={mem}MB signals={sig} errors={err}")
        time.sleep(CHECK_INTERVAL)

    print("Soak test complete")


if __name__ == "__main__":
    main()
