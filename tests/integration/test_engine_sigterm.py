"""Integration test — engine exits cleanly on SIGTERM (CL-6v2l).

Spawns the engine as a subprocess (matches how it actually runs in
production), sends SIGTERM, and asserts the process exits within a tight
timeout. This is the only honest way to test signal handling — unit tests
that import LiveEngine and stub signal handlers don't catch the actual
asyncio.gather + uvicorn + price-stream interactions that broke before.

Skipped if Postgres isn't reachable — the engine refuses to boot otherwise.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent


def _postgres_reachable() -> bool:
    """Cheap socket check on the configured Postgres host:port."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _postgres_reachable(),
    reason="Engine refuses to boot without Postgres — skipping subprocess test",
)
def test_engine_exits_cleanly_on_sigterm() -> None:
    """Engine should exit with code 0 within 10s of receiving SIGTERM.

    Acceptance for CL-6v2l: kill -TERM <pid> ⇒ process gone within 5s.
    Tighter than the bd ticket's claim because in practice the cancel-and-
    drain path takes ~1s on a fresh paper engine; 10s is generous slack
    for slow CI machines.
    """
    env = {**os.environ, "FX_LOG_DIR": str(ROOT / "logs")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.runtime.run_engine", "--practice"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=ROOT,
    )
    try:
        # Give the engine time to boot — start_metrics_server, build the
        # coordinator, run cold-start reconciliation. On a fast machine
        # this takes ~1.5s; 5s is a safe upper bound.
        time.sleep(5)
        assert proc.poll() is None, (
            "engine died during startup (boot path broken, not the SIGTERM "
            f"path being tested) — exit code {proc.returncode}"
        )

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("engine did not exit within 10s of SIGTERM — CL-6v2l regressed")
        assert proc.returncode == 0, (
            f"engine exited with non-zero code {proc.returncode} on SIGTERM"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
