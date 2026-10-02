"""Bounded, identity-verified daemon process control (CL-wv3v).

Every test drives processes it spawns itself, each tagged with a unique
argv token, so no pattern can match a real fleet daemon. Oracles are
independent of the helper: ``pgrep -f`` (cross-validation of discovery),
``Popen.poll()`` / ``psutil`` for liveness, wall-clock time for deadlines,
and ``fleet_watch.parse_status`` for the status line contract.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from src.monitoring.fleet_watch import parse_status
from src.runtime import proc_discovery as pd
from src.runtime.proc_discovery import (
    ControlConfig,
    ProcIdentity,
    ScanResult,
    bounded_scan,
    identity_alive,
    identity_changed,
    launch_nohup,
    select_signal_targets,
    signal_verified,
)

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "src" / "runtime" / "proc_discovery.py"
DAEMONS_SH = REPO / "scripts" / "daemons.sh"

FAST = ControlConfig(
    scan_timeout_s=5.0,
    stop_grace_s=2.0,
    kill_wait_s=2.0,
    start_settle_s=0.3,
    start_verify_s=3.0,
    start_reserve_s=3.0,
    poll_s=0.05,
)
#: Slack for interpreter/worker startup on a loaded CI box.
SLACK_S = 2.5
SLEEPER = "import time; time.sleep(60)"
STALLED_WORKER = [sys.executable, "-c", "import time; time.sleep(600)"]


def real_scan(pattern: str, timeout_s: float) -> ScanResult:
    return bounded_scan(pattern, timeout_s)


def stalled_scan(pattern: str, timeout_s: float) -> ScanResult:
    return bounded_scan(pattern, timeout_s, worker_argv=STALLED_WORKER)


def scan_then_stall(real_calls: int) -> Callable[[str, float], ScanResult]:
    """Real scans for the first ``real_calls`` calls, then a stalled scanner."""
    calls = {"n": 0}

    def scanner(pattern: str, timeout_s: float) -> ScanResult:
        calls["n"] += 1
        if calls["n"] <= real_calls:
            return real_scan(pattern, timeout_s)
        return stalled_scan(pattern, timeout_s)

    return scanner


def pgrep(tag: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True, check=False)
    return sorted(int(p) for p in out.stdout.split())


def wait_dead(proc: subprocess.Popen[bytes], timeout_s: float = 5.0) -> bool:
    try:
        proc.wait(timeout=timeout_s)
        return True
    except subprocess.TimeoutExpired:
        return False


def pid_dead(pid: int, timeout_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def tag() -> Iterator[str]:
    """Unique argv token; every process carrying it is killed at teardown."""
    t = f"clwv3v_{uuid.uuid4().hex}"
    yield t
    for proc in psutil.process_iter(["cmdline"]):
        if t in " ".join(proc.info.get("cmdline") or []):
            with contextlib.suppress(psutil.Error):
                proc.kill()


@pytest.fixture
def spawn(tag: str) -> Iterator[Callable[..., subprocess.Popen[bytes]]]:
    procs: list[subprocess.Popen[bytes]] = []

    def _spawn(code: str = SLEEPER) -> subprocess.Popen[bytes]:
        p = subprocess.Popen([sys.executable, "-c", code, tag])
        procs.append(p)
        return p

    yield _spawn
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait(timeout=5)


class RecordingLauncher:
    """Real nohup launch (same as production) that records each call."""

    def __init__(self, extra: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.extra = extra  # additional racing writers to start alongside

    def __call__(self, argv: Sequence[str], log_path: str) -> tuple[int, float | None]:
        self.calls.append(list(argv))
        launched = launch_nohup(argv, log_path)
        for _ in range(self.extra):
            launch_nohup(argv, log_path)
        return launched


def launch_argv(tag: str) -> list[str]:
    return [sys.executable, "-c", SLEEPER, tag]


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def test_scan_finds_identity_and_agrees_with_pgrep(spawn, tag):
    a, b = spawn(), spawn()
    res = bounded_scan(tag, 10.0)
    assert res.ok, res.detail
    assert [m.pid for m in res.matches] == pgrep(tag) == sorted([a.pid, b.pid])
    for m in res.matches:
        assert m.create_time == psutil.Process(m.pid).create_time()
        assert tag in m.cmdline


def test_scan_does_not_match_its_own_helper_or_unrelated(tag):
    res = bounded_scan(tag, 10.0)
    assert res.ok and res.matches == ()


def test_stalled_scan_is_abandoned_at_timeout():
    t0 = time.monotonic()
    res = bounded_scan("anything", 0.5, worker_argv=STALLED_WORKER)
    elapsed = time.monotonic() - t0
    assert not res.ok
    assert "stalled" in res.detail
    assert elapsed < 0.5 + SLACK_S


# --------------------------------------------------------------------------
# Identity / PID reuse
# --------------------------------------------------------------------------


def test_identity_alive_rejects_reused_pid(spawn, tag):
    p = spawn()
    real = bounded_scan(tag, 10.0).matches[0]
    assert identity_alive(real)
    reused = ProcIdentity(real.pid, real.create_time - 100.0, real.cmdline)
    assert not identity_alive(reused)  # same PID, different start → not ours
    p.kill()
    assert wait_dead(p)
    assert not identity_alive(real)


def test_signal_refused_for_reused_pid_and_process_survives(spawn, tag):
    p = spawn()
    real = bounded_scan(tag, 10.0).matches[0]
    reused = ProcIdentity(real.pid, real.create_time + 7.0, real.cmdline)
    assert signal_verified(reused, signal.SIGKILL) is False
    time.sleep(0.3)
    assert p.poll() is None  # the live process was NOT signalled


def test_select_targets_rejects_reuse_and_changed_command():
    old = ProcIdentity(100, 1000.0, "python scripts/x.py --loop 300")
    assert select_signal_targets([old], [old]) == [old]
    reused = ProcIdentity(100, 2000.0, old.cmdline)
    assert select_signal_targets([old], [reused]) == []
    exec_d = ProcIdentity(100, 1000.0, "/bin/sleep 30")
    assert select_signal_targets([old], [exec_d]) == []
    assert select_signal_targets([old], []) == []


def test_identity_changed_counts_reused_pid_as_new_but_not_same_process():
    old = ProcIdentity(100, 1000.0, "c")
    assert identity_changed(old, ProcIdentity(100, 1001.0, "c"))  # PID reused by new
    assert identity_changed(old, ProcIdentity(101, 1001.0, "c"))
    assert not identity_changed(old, ProcIdentity(100, 1000.0, "c"))
    assert identity_changed(None, old)


def test_stop_never_signals_a_process_that_exec_d_into_another_command(spawn, tag):
    code = "import os,time; time.sleep(0.5); os.execv('/bin/sleep', ['/bin/sleep', '30'])"
    p = spawn(code)
    old = bounded_scan(tag, 10.0).matches[0]
    end = time.monotonic() + 5
    while psutil.Process(p.pid).cmdline()[:1] != ["/bin/sleep"] and time.monotonic() < end:
        time.sleep(0.05)
    assert psutil.Process(p.pid).create_time() == old.create_time  # same pid+start
    out = pd._stop_identities("x", tag, [old], real_scan, FAST, pd._Clock(10.0), 1.0)
    assert out.ok and "nothing signalled" in out.lines[0]
    time.sleep(0.3)
    assert p.poll() is None


# --------------------------------------------------------------------------
# Status line contract (fleet_watch.parse_status)
# --------------------------------------------------------------------------


def test_status_lines_are_byte_compatible(spawn, tag):
    none = pd.status("svc", tag, real_scan, 10.0)
    assert none.lines == ("  ✗ svc NOT RUNNING",)
    assert parse_status(none.lines[0]) == {"svc": False}
    p = spawn()
    one = pd.status("svc", tag, real_scan, 10.0)
    assert one.lines == (f"  ✓ svc (pid {p.pid})",)
    assert parse_status(one.lines[0]) == {"svc": True}


def test_status_reports_duplicates_and_stays_parseable(spawn, tag):
    a, b = spawn(), spawn()
    out = pd.status("svc", tag, real_scan, 10.0)
    low = min(a.pid, b.pid)
    assert out.lines[0].startswith(f"  ✓ svc (pid {low}) — DUPLICATE: 2 matching processes")
    assert parse_status(out.lines[0]) == {"svc": True}


def test_stalled_status_is_unknown_never_a_false_up_or_down():
    t0 = time.monotonic()
    out = pd.status("svc", "whatever", stalled_scan, 0.5)
    assert time.monotonic() - t0 < 0.5 + SLACK_S
    assert out.lines[0].startswith("  ? svc UNKNOWN")
    assert parse_status(out.lines[0]) == {}


# --------------------------------------------------------------------------
# start / stop
# --------------------------------------------------------------------------


def test_start_launches_once_and_verifies_launched_pid(tag, tmp_path):
    launcher = RecordingLauncher()
    out = pd.start(
        "svc", tag, launch_argv(tag), str(tmp_path / "svc.log"), real_scan, launcher, FAST
    )
    assert out.ok, out.lines
    assert len(launcher.calls) == 1
    assert out.running is not None
    assert pgrep(tag) == [out.running.pid]
    assert out.lines == (f"  ▶ svc started (pid {out.running.pid})",)
    again = pd.start(
        "svc", tag, launch_argv(tag), str(tmp_path / "svc.log"), real_scan, launcher, FAST
    )
    assert again.ok and "already running" in again.lines[0]
    assert len(launcher.calls) == 1  # idempotent: no second writer


def test_start_refuses_when_duplicates_already_running(spawn, tag, tmp_path):
    spawn(), spawn()
    launcher = RecordingLauncher()
    out = pd.start("svc", tag, launch_argv(tag), str(tmp_path / "l"), real_scan, launcher, FAST)
    assert not out.ok and "DUPLICATE" in out.lines[0]
    assert launcher.calls == []
    assert len(pgrep(tag)) == 2  # never a third


def test_start_refuses_to_launch_when_scan_stalls(tag, tmp_path):
    launcher = RecordingLauncher()
    cfg = ControlConfig(scan_timeout_s=0.5, poll_s=0.05)
    t0 = time.monotonic()
    out = pd.start("svc", tag, launch_argv(tag), str(tmp_path / "l"), stalled_scan, launcher, cfg)
    assert time.monotonic() - t0 < 0.5 + SLACK_S
    assert not out.ok and "UNKNOWN" in out.lines[0] and "[phase=start]" in out.lines[0]
    assert launcher.calls == []


def test_start_detects_racing_duplicate_writer_after_launch(tag, tmp_path):
    launcher = RecordingLauncher(extra=1)  # a second writer appears concurrently
    out = pd.start("svc", tag, launch_argv(tag), str(tmp_path / "l"), real_scan, launcher, FAST)
    assert not out.ok
    assert "[phase=verify]" in out.lines[0] and "DUPLICATE" in out.lines[0]
    assert len(launcher.calls) == 1  # reported, not "fixed" by launching again
    assert len(pgrep(tag)) == 2


def test_stop_terms_then_kills_a_writer_ignoring_sigterm(spawn, tag):
    p = spawn("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)")
    time.sleep(0.3)  # let it install the handler
    cfg = ControlConfig(scan_timeout_s=5.0, stop_grace_s=0.5, kill_wait_s=2.0, poll_s=0.05)
    out = pd.stop("svc", tag, real_scan, cfg)
    assert out.ok and "KILLED" in out.lines[0]
    assert wait_dead(p)


def test_stop_graceful_and_not_running(spawn, tag):
    assert pd.stop("svc", tag, real_scan, FAST).lines == ("  - svc not running",)
    p = spawn()
    out = pd.stop("svc", tag, real_scan, FAST)
    assert out.ok and out.lines[0].startswith(f"  ■ svc stopped (pid {p.pid}, ")
    assert wait_dead(p)


# --------------------------------------------------------------------------
# restart: deadline, phases, duplicates
# --------------------------------------------------------------------------


def test_restart_happy_path_changes_identity(tag, tmp_path):
    log = str(tmp_path / "svc.log")
    first = pd.start("svc", tag, launch_argv(tag), log, real_scan, RecordingLauncher(), FAST)
    assert first.ok and first.running
    old = first.running
    out = pd.restart("svc", tag, launch_argv(tag), log, real_scan, 15.0, RecordingLauncher(), FAST)
    assert out.ok, out.lines
    assert out.running and out.running.pid != old.pid
    assert out.lines[-1] == f"  ↻ svc restart verified (pid {old.pid} → {out.running.pid})"
    assert pid_dead(old.pid)
    assert pgrep(tag) == [out.running.pid]


def test_restart_refuses_duplicate_writers_and_signals_nothing(spawn, tag, tmp_path):
    a, b = spawn(), spawn()
    launcher = RecordingLauncher()
    out = pd.restart(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), real_scan, 10.0, launcher, FAST
    )
    assert not out.ok and out.phase == "discover"
    assert "DUPLICATE" in out.lines[-1] and f"{min(a.pid, b.pid)}" in out.lines[-1]
    assert launcher.calls == []
    time.sleep(0.3)
    assert a.poll() is None and b.poll() is None


def test_restart_with_permanently_stalled_scan_is_bounded(tag, tmp_path):
    launcher = RecordingLauncher()
    deadline = 3.0
    t0 = time.monotonic()
    out = pd.restart(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), stalled_scan, deadline, launcher, FAST
    )
    assert time.monotonic() - t0 < deadline + SLACK_S
    assert not out.ok and out.phase == "discover"
    assert "[phase=discover]" in out.lines[-1] and "writer state: UNKNOWN" in out.lines[-1]
    assert launcher.calls == []


def test_restart_scan_stalls_after_stop_reports_start_phase(spawn, tag, tmp_path):
    old = spawn()
    launcher = RecordingLauncher()
    t0 = time.monotonic()
    out = pd.restart(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), scan_then_stall(2), 8.0, launcher, FAST
    )
    assert time.monotonic() - t0 < 8.0 + SLACK_S
    assert not out.ok and out.phase == "start"
    assert f"old pid {old.pid} exited" in out.lines[-1]
    assert "replacement NOT started" in out.lines[-1]
    assert launcher.calls == []
    assert wait_dead(old)


def test_restart_scan_stalls_during_verify_reports_launched_state(spawn, tag, tmp_path):
    spawn()
    launcher = RecordingLauncher()
    t0 = time.monotonic()
    out = pd.restart(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), scan_then_stall(3), 8.0, launcher, FAST
    )
    assert time.monotonic() - t0 < 8.0 + SLACK_S
    assert not out.ok and out.phase == "verify"
    assert len(launcher.calls) == 1
    assert any("launched pid" in line and "ALIVE" in line for line in out.lines)


def test_restart_with_unkillable_old_writer_fails_stop_wait_in_time(
    spawn, tag, tmp_path, monkeypatch
):
    old = spawn()
    # Simulate a writer that survives every signal (e.g. wedged in the kernel).
    monkeypatch.setattr(pd, "signal_verified", lambda ident, sig: True)
    launcher = RecordingLauncher()
    deadline = 6.0
    t0 = time.monotonic()
    out = pd.restart(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), real_scan, deadline, launcher, FAST
    )
    assert time.monotonic() - t0 < deadline + SLACK_S
    assert not out.ok and out.phase == "stop-wait"
    assert f"old pid {old.pid} STILL RUNNING" in out.lines[-1]
    assert launcher.calls == []  # never a second writer next to the old one
    assert old.poll() is None


def test_restart_verify_detects_duplicate_and_never_launches_third(spawn, tag, tmp_path):
    spawn()
    launcher = RecordingLauncher(extra=1)
    out = pd.restart(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), real_scan, 15.0, launcher, FAST
    )
    assert not out.ok and out.phase == "verify"
    assert any("DUPLICATE" in line for line in out.lines)
    assert len(launcher.calls) == 1
    assert len(pgrep(tag)) == 2


# --------------------------------------------------------------------------
# CLI / daemons.sh contract
# --------------------------------------------------------------------------


def _cli(op: str, tag: str, log: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env[pd.PATTERN_ENV] = tag
    # Whitespace-split like the shell's unquoted $launch, so no spaces in -c.
    env[pd.LAUNCH_ENV] = " ".join([sys.executable, "-c", "__import__('time').sleep(60)", tag])
    return subprocess.run(
        [sys.executable, str(HELPER), op, *extra, "--name", "svc", "--log", str(log)],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
        check=False,
    )


def test_cli_start_status_stop_round_trip(tag, tmp_path):
    log = tmp_path / "svc.log"
    started = _cli("start", tag, log)
    assert started.returncode == 0, started.stdout + started.stderr
    [pid] = pgrep(tag)
    assert started.stdout == f"  ▶ svc started (pid {pid})\n"
    status = _cli("status", tag, log)
    assert status.stdout == f"  ✓ svc (pid {pid})\n"
    stopped = _cli("stop", tag, log)
    assert stopped.returncode == 0 and stopped.stdout.startswith(f"  ■ svc stopped (pid {pid}, ")
    assert pid_dead(pid)
    assert _cli("status", tag, log).stdout == "  ✗ svc NOT RUNNING\n"


def test_cli_restart_exit_codes(tag, tmp_path):
    log = tmp_path / "svc.log"
    assert _cli("start", tag, log).returncode == 0
    ok = _cli("restart", tag, log, "--deadline", "20")
    assert ok.returncode == 0 and "restart verified" in ok.stdout
    bad = _cli("restart", tag, log, "--deadline", "0")
    assert bad.returncode == 2


def test_daemons_sh_has_no_unbounded_pgrep_and_is_valid_bash():
    src = DAEMONS_SH.read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "pgrep" not in code and "pkill" not in code
    assert "src/runtime/proc_discovery.py" in code
    assert subprocess.run(["bash", "-n", str(DAEMONS_SH)], check=False).returncode == 0
    if shutil.which("shellcheck"):
        res = subprocess.run(
            ["shellcheck", str(DAEMONS_SH)], capture_output=True, text=True, check=False
        )
        assert res.returncode == 0, res.stdout


# --------------------------------------------------------------------------
# Review round 1 (Codex) regressions
# --------------------------------------------------------------------------


class _FakeProc:
    """psutil.Process stand-in; ``argv=None`` means the argv read is denied."""

    def __init__(self, pid: int, euid: int, *, argv: list[str] | None = None) -> None:
        self.pid = pid
        self._euid = euid
        self._argv = argv

    def cmdline(self) -> list[str]:
        if self._argv is None:
            raise psutil.AccessDenied(self.pid)
        return self._argv

    def create_time(self) -> float:
        return 1000.0

    def uids(self) -> Any:
        return SimpleNamespace(real=self._euid, effective=self._euid)

    def status(self) -> str:
        return psutil.STATUS_SLEEPING


def test_unreadable_same_user_process_makes_scan_inconclusive(monkeypatch):
    me, other = os.geteuid(), os.geteuid() + 1
    procs = [
        _FakeProc(11, other),  # e.g. setuid login: hidden argv, not ours
        _FakeProc(12, me, argv=["python", "x.py", "--loop"]),
    ]
    monkeypatch.setattr(pd.psutil, "process_iter", lambda *a, **k: iter(procs))
    matches, unreadable = pd.scan_inline("x.py --loop", frozenset())
    assert [m.pid for m in matches] == [12] and unreadable == []
    procs.append(_FakeProc(13, me))  # same-user process we cannot read
    procs.append(_FakeProc(14, me, argv=[]))  # same-user, live, empty argv
    _, unreadable = pd.scan_inline("x.py --loop", frozenset())
    assert unreadable == [13, 14]


def test_inconclusive_scan_refuses_start_and_restart(tag, tmp_path):
    worker = [sys.executable, "-c", 'print(\'{"matches": [], "unreadable": [4242]}\')']

    def scanner(pattern: str, timeout_s: float) -> ScanResult:
        return bounded_scan(pattern, timeout_s, worker_argv=worker)

    res = scanner(tag, 5.0)
    assert not res.ok and "4242" in res.detail
    launcher = RecordingLauncher()
    out = pd.start("svc", tag, launch_argv(tag), str(tmp_path / "l"), scanner, launcher, FAST)
    assert not out.ok and "START REFUSED" in out.lines[0]
    out = pd.restart(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), scanner, 5.0, launcher, FAST
    )
    assert not out.ok and out.phase == "discover"
    assert launcher.calls == []
    st = pd.status("svc", tag, scanner, 5.0)
    assert st.lines[0].startswith("  ? svc UNKNOWN") and parse_status(st.lines[0]) == {}


def test_sigkill_not_sent_to_survivor_that_reexecd_after_sigterm(spawn, tag):
    code = (
        "import os,signal,time\n"
        "signal.signal(signal.SIGTERM, lambda *a: os.execv('/bin/sleep', ['/bin/sleep', '30']))\n"
        "time.sleep(60)"
    )
    p = spawn(code)
    time.sleep(0.3)
    cfg = ControlConfig(scan_timeout_s=5.0, stop_grace_s=0.5, kill_wait_s=2.0, poll_s=0.05)
    out = pd.stop("svc", tag, real_scan, cfg)
    assert out.ok and "no longer matches after SIGTERM" in out.lines[0], out.lines
    time.sleep(0.3)
    assert p.poll() is None  # the unrelated command was NOT killed
    assert psutil.Process(p.pid).cmdline() == ["/bin/sleep", "30"]


def _empty_then(scanner: Callable[[str, float], ScanResult]) -> Callable[[str, float], ScanResult]:
    """First call (pre-launch check) sees nothing; later calls use ``scanner``."""
    calls = {"n": 0}

    def wrapped(pattern: str, timeout_s: float) -> ScanResult:
        calls["n"] += 1
        if calls["n"] == 1:
            return ScanResult(True, (), "", 0.0)
        return scanner(pattern, timeout_s)

    return wrapped


def test_launch_verify_rejects_recycled_launched_pid(tag, tmp_path):
    launched: dict[str, int] = {}

    def launcher(argv: Sequence[str], log_path: str) -> tuple[int, float | None]:
        pid, ct = launch_nohup(argv, log_path)
        launched["pid"] = pid
        return pid, ct

    def scanner(pattern: str, timeout_s: float) -> ScanResult:
        # Same pid, different start time: a recycled pid, not our launch.
        pid = launched["pid"]
        ident = ProcIdentity(pid, psutil.Process(pid).create_time() + 5.0, tag)
        return ScanResult(True, (ident,), "", 0.0)

    cfg = ControlConfig(scan_timeout_s=1.0, start_settle_s=0.1, start_verify_s=1.0, poll_s=0.05)
    out = pd.start(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), _empty_then(scanner), launcher, cfg
    )
    assert not out.ok and out.phase == "verify", out.lines
    assert "started" not in out.lines[0]


def test_launch_nohup_pins_start_time_before_handle_is_dropped(tag, tmp_path):
    pid, ct = launch_nohup(launch_argv(tag), str(tmp_path / "l"))
    assert ct is not None and ct == psutil.Process(pid).create_time()


def test_verify_trusts_launcher_pinned_identity_not_a_later_pid_lookup(spawn, tag, tmp_path):
    # The pid now holds a matching process, but with a different start time
    # than the one pinned at launch: models the child being reaped and its
    # pid recycled BEFORE any later lookup. Must not be reported as started.
    other = spawn()
    real_ct = psutil.Process(other.pid).create_time()

    def launcher(argv: Sequence[str], log_path: str) -> tuple[int, float | None]:
        return other.pid, real_ct - 50.0

    cfg = ControlConfig(scan_timeout_s=5.0, start_settle_s=0.1, start_verify_s=1.0, poll_s=0.05)
    out = pd.start(
        "svc", tag, launch_argv(tag), str(tmp_path / "l"), _empty_then(real_scan), launcher, cfg
    )
    assert not out.ok and out.phase == "verify", out.lines
    assert "started" not in out.lines[0]
    # A launch whose child was already gone at pin time is reported as exited.
    gone = pd.start(
        "svc",
        tag + "_none",
        launch_argv(tag),
        str(tmp_path / "l"),
        _empty_then(real_scan),
        lambda a, lp: (other.pid, None),
        cfg,
    )
    assert not gone.ok and "exited" in gone.lines[0]


def test_denied_signal_is_reported_not_false_stopped(spawn, tag, monkeypatch):
    p = spawn()
    monkeypatch.setattr(pd, "signal_verified", lambda ident, sig: False)  # AccessDenied
    cfg = ControlConfig(scan_timeout_s=5.0, stop_grace_s=0.5, kill_wait_s=0.5, poll_s=0.05)
    out = pd.stop("svc", tag, real_scan, cfg)
    assert not out.ok and out.phase == "stop-wait"
    assert f"pid {p.pid} still alive" in out.lines[0]
    assert p.poll() is None


def test_launched_daemon_ignores_sigint_and_gets_no_helper_env(tag, tmp_path, monkeypatch):
    monkeypatch.setenv(pd.PATTERN_ENV, "leak")
    monkeypatch.setenv(pd.LAUNCH_ENV, "leak")
    marker = tmp_path / "env.txt"
    code = (
        "import os,time\n"
        f"keys = sorted(k for k in os.environ if k.startswith('CURLIT_PROC'))\n"
        f"open({str(marker)!r}, 'w').write(str(keys))\n"
        "time.sleep(60)"
    )
    pid, _ = launch_nohup([sys.executable, "-c", code, tag], str(tmp_path / "l"))
    end = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < end:
        time.sleep(0.05)
    assert marker.read_text() == "[]"
    os.kill(pid, signal.SIGINT)  # bash `cmd &` children ignore SIGINT
    time.sleep(0.5)
    assert psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
