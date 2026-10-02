"""Bounded, identity-verified daemon process control for daemons.sh (CL-wv3v).

Why this exists: during the authorized 2026-09-08 options restart, each
``pgrep -f`` call in ``scripts/daemons.sh`` took tens of seconds on macOS.
``pgrep -f`` must read EVERY process's argv (``sysctl KERN_PROCARGS2``), and
the launcher called it in an unbounded 1-second poll loop, so a slow or
stalled argv read turned a nominal 30 s restart into minutes with no
deadline and no report of which phase was stuck.

This helper replaces those calls with:

* a **bounded scan** — the full argv scan runs in a throwaway worker
  subprocess; the caller stops waiting at a hard timeout and reports the
  scan as stalled (writer state UNKNOWN) instead of hanging;
* **verified identities** — a match is ``(pid, create_time, cmdline)``.
  After discovery, liveness is polled per PID via ``create_time`` only (no
  argv read), so a reused PID (same number, different start time) is never
  mistaken for the old writer, and a signal is sent only to an identity a
  fresh scan still shows with the same start time AND matching command;
* a **total wall-clock deadline** on restart, with failures naming the
  phase (``discover`` / ``stop-wait`` / ``start`` / ``verify``) and the last
  known writer state;
* **duplicate-writer refusal** — two matching processes are reported and
  nothing is launched; an unconfirmed (stalled) scan never launches either.

Launch semantics are unchanged from daemons.sh: ``nohup <argv>`` appending
stdout+stderr to ``logs/<name>.log``.

Shell contract: the pgrep-style regex arrives in ``$CURLIT_PROC_PATTERN`` and
the launch command in ``$CURLIT_PROC_LAUNCH`` (split on whitespace, exactly
like the shell's unquoted ``$launch``) — never in argv, so a helper can not
match its own pattern or another concurrent helper's. Status lines keep the
exact ``  ✓ name (pid N)`` / ``  ✗ name NOT RUNNING`` format parsed by
``src/monitoring/fleet_watch.parse_status``; a stalled scan prints a
``  ? name UNKNOWN`` line that the parser deliberately ignores (neither a
false DOWN page nor a false UP).

Self-contained (stdlib + psutil) so daemons.sh can run it by file path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import psutil  # type: ignore[import-untyped]  # OS process-inspection boundary.

logger = logging.getLogger("curlit.proc_discovery")

PATTERN_ENV = "CURLIT_PROC_PATTERN"
LAUNCH_ENV = "CURLIT_PROC_LAUNCH"
_WORKER_CMD = "_scan-worker"

#: One full argv scan normally takes ~0.03-0.06 s on the operator Mac
#: (955 processes, measured 2026-10-02); 10 s is >100x headroom yet still
#: bounds the 2026-09-08 tens-of-seconds stall.
SCAN_TIMEOUT_S = 10.0
#: Status is polled by health_watch with a 30 s budget for the whole
#: roster, so a single stalled status scan gives up sooner.
STATUS_SCAN_TIMEOUT_S = 5.0
#: Daemons finish their in-flight cycle on SIGTERM; 30 s is the historical
#: daemons.sh graceful window before SIGKILL (2026-07-22 restart race).
STOP_GRACE_S = 30.0
#: After SIGKILL the kernel normally reaps within milliseconds; 5 s covers
#: a busy host tearing down a large (torch-loaded) address space.
KILL_WAIT_S = 5.0
#: The historical ``sleep 1`` before checking a fresh launch.
START_SETTLE_S = 1.0
#: Budget for a standalone ``start`` to observe its launched process.
START_VERIFY_S = 10.0
#: Total wall-clock deadline for restart: grace 30 + kill 5 + start/verify.
RESTART_DEADLINE_S = 60.0
#: Time kept back from the stop phase so start+verify still get a chance.
START_RESERVE_S = 15.0
#: Identity/scan poll cadence: per-PID create_time checks are microseconds.
POLL_S = 0.2


@dataclass(frozen=True)
class ProcIdentity:
    """A process as discovered: PID alone is not an identity (PIDs recycle)."""

    pid: int
    create_time: float
    cmdline: str


@dataclass(frozen=True)
class ScanResult:
    """Outcome of one bounded scan. ``ok=False`` means writer state UNKNOWN."""

    ok: bool
    matches: tuple[ProcIdentity, ...]
    detail: str
    elapsed_s: float


Scanner = Callable[[str, float], ScanResult]
#: Returns ``(pid, start_time | None)``, start time pinned before reaping.
Launcher = Callable[[Sequence[str], str], tuple[int, float | None]]


@dataclass(frozen=True)
class ControlConfig:
    """Timing knobs (seconds); tests shrink them, production uses defaults."""

    scan_timeout_s: float = SCAN_TIMEOUT_S
    stop_grace_s: float = STOP_GRACE_S
    kill_wait_s: float = KILL_WAIT_S
    start_settle_s: float = START_SETTLE_S
    start_verify_s: float = START_VERIFY_S
    start_reserve_s: float = START_RESERVE_S
    poll_s: float = POLL_S


@dataclass(frozen=True)
class Outcome:
    """User-facing lines plus success flag (and the identity left running)."""

    ok: bool
    lines: tuple[str, ...]
    phase: str = ""
    running: ProcIdentity | None = None


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


def _may_be_ours(proc: psutil.Process, euid: int) -> bool:
    """Could this process be one of our writers? (runs with our effective uid)

    Other users' processes — and setuid ones such as ``login``, whose argv
    macOS hides even from their real-uid owner — cannot be a writer this
    launcher started. Unknown ownership is treated as possibly ours.
    """
    try:
        return bool(proc.uids().effective == euid)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def scan_inline(pattern: str, exclude: frozenset[int]) -> tuple[list[ProcIdentity], list[int]]:
    """Unbounded in-process argv scan — ONLY run inside the worker process.

    Matches like ``pgrep -f``: regex search over the space-joined argv.
    Returns ``(matches, unreadable)``: ``unreadable`` lists live processes
    running as our effective uid whose argv/start time could not be read —
    any of them might be a writer, so the caller must treat the scan as
    inconclusive rather than as "none running". Exited processes and
    zombies are skipped (they cannot write).
    """
    rx = re.compile(pattern)
    euid = os.geteuid()
    found: list[ProcIdentity] = []
    unreadable: list[int] = []
    for proc in psutil.process_iter():
        pid = int(proc.pid)
        if pid in exclude:
            continue
        try:
            argv = proc.cmdline()
            ctime = float(proc.create_time())
        except psutil.NoSuchProcess:  # includes ZombieProcess
            continue
        except psutil.AccessDenied:
            if _may_be_ours(proc, euid):
                unreadable.append(pid)
            continue
        if not argv:
            with contextlib.suppress(psutil.NoSuchProcess):
                if proc.status() != psutil.STATUS_ZOMBIE and _may_be_ours(proc, euid):
                    unreadable.append(pid)
            continue
        line = " ".join(argv)
        if rx.search(line):
            found.append(ProcIdentity(pid, ctime, line))
    found.sort(key=lambda i: i.pid)  # pgrep order: lowest pid first
    return found, sorted(unreadable)


def _worker_main() -> int:
    pattern = os.environ.get(PATTERN_ENV, "")
    if not pattern:
        print(f"{PATTERN_ENV} is empty", file=sys.stderr)
        return 2
    matches, unreadable = scan_inline(pattern, frozenset({os.getpid(), os.getppid()}))
    payload = {
        "matches": [[m.pid, m.create_time, m.cmdline] for m in matches],
        "unreadable": unreadable,
    }
    json.dump(payload, sys.stdout)
    return 0


def default_worker_argv() -> list[str]:
    return [sys.executable, os.path.abspath(__file__), _WORKER_CMD]


def bounded_scan(
    pattern: str, timeout_s: float, worker_argv: Sequence[str] | None = None
) -> ScanResult:
    """Run the argv scan in a worker and abandon it at ``timeout_s``.

    A stalled kernel argv read can only wedge the worker; this process
    kills it and returns ``ok=False`` without waiting on it indefinitely.
    """
    assert pattern, "empty pattern would match every process"
    start = time.monotonic()
    if timeout_s <= 0:
        return ScanResult(False, (), "no time left for a process scan", 0.0)
    argv = list(worker_argv) if worker_argv is not None else default_worker_argv()
    env = dict(os.environ)
    env[PATTERN_ENV] = pattern
    logger.debug("process scan starting (timeout %.1fs)", timeout_s)
    try:
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, our own worker
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            close_fds=True,
        )
    except OSError as exc:
        return ScanResult(False, (), f"process scan could not start: {exc}", 0.0)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        logger.warning("process scan stalled after %.1fs — abandoning worker", elapsed)
        proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=1.0)  # reap if it dies promptly; never block on a wedged worker
        return ScanResult(False, (), f"process scan stalled (>{timeout_s:.1f}s)", elapsed)
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        msg = err.decode(errors="replace").strip()[-200:]
        return ScanResult(
            False, (), f"process scan failed (exit {proc.returncode}): {msg}", elapsed
        )
    try:
        payload = json.loads(out)
        rows = payload["matches"]
        unreadable = [int(p) for p in payload["unreadable"]]
        matches = tuple(ProcIdentity(int(p), float(c), str(cmd)) for p, c, cmd in rows)
    except (ValueError, TypeError, KeyError) as exc:
        return ScanResult(False, (), f"process scan output unreadable: {exc}", elapsed)
    if unreadable:
        shown = " ".join(str(p) for p in unreadable[:10])
        logger.warning("process scan inconclusive: unreadable same-user pid(s) %s", shown)
        return ScanResult(
            False,
            matches,
            f"argv unreadable for same-user pid(s) {shown} — cannot confirm writer state",
            elapsed,
        )
    logger.debug("process scan done in %.3fs: %d match(es)", elapsed, len(matches))
    return ScanResult(True, matches, "", elapsed)


# --------------------------------------------------------------------------
# Identity checks (no argv read — cannot hit the slow path)
# --------------------------------------------------------------------------


def identity_alive(ident: ProcIdentity) -> bool:
    """True while THIS process (pid + start time) exists and is not a zombie.

    A recycled PID has a different create_time → False (old writer gone).
    """
    try:
        proc = psutil.Process(ident.pid)
        if proc.create_time() != ident.create_time:
            return False
        return bool(proc.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # cannot prove it is gone — conservative


def select_signal_targets(
    recorded: Sequence[ProcIdentity], fresh: Sequence[ProcIdentity]
) -> list[ProcIdentity]:
    """Recorded identities that a fresh scan still shows unchanged.

    Same PID with a different start time (reuse) or a different command
    (exec'd into something else) is NOT our writer and is never signalled.
    """
    current = set(fresh)
    return [r for r in recorded if r in current]


def identity_changed(old: ProcIdentity | None, new: ProcIdentity) -> bool:
    """CL-obgy verify: a restart happened iff the surviving identity differs.

    A replacement that happens to reuse the old PID number still counts as
    new (different start time); same pid + same start time is NOT a restart.
    """
    return old is None or (old.pid, old.create_time) != (new.pid, new.create_time)


def signal_verified(ident: ProcIdentity, sig: int) -> bool:
    """Send ``sig`` only if ``ident``'s PID still has the same start time."""
    try:
        proc = psutil.Process(ident.pid)
        if proc.create_time() != ident.create_time:
            logger.warning("pid %d reused (start time changed) — NOT signalling", ident.pid)
            return False
        logger.info("sending signal %d to pid %d", sig, ident.pid)
        proc.send_signal(sig)  # psutil re-checks identity before kill()
        return True
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        logger.warning("access denied signalling pid %d", ident.pid)
        return False


def _ignore_int_quit() -> None:
    """Child pre-exec: bash starts ``cmd &`` (no job control) with SIGINT and
    SIGQUIT ignored; keep that so a Ctrl-C in the launching terminal can not
    kill freshly started daemons."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)


def launch_nohup(argv: Sequence[str], log_path: str) -> tuple[int, float | None]:
    """Same launch as daemons.sh: ``nohup argv >> log 2>&1 &``.

    Returns ``(pid, start_time)``; start_time is None if the child is already
    gone. The start time is read while this function still holds the Popen
    handle: an unreaped child (even a zombie) keeps its pid, so the identity
    pinned here is the launch's own. Once the handle is dropped,
    ``Popen.__del__`` may reap it and the pid can be recycled (CL-wv3v review).
    """
    assert argv, "empty launch command"
    env = {k: v for k, v in os.environ.items() if k not in (PATTERN_ENV, LAUNCH_ENV)}
    logger.info("launching %s (log %s)", " ".join(argv), log_path)
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(  # noqa: S603 — fixed daemons.sh roster argv
            ["nohup", *argv],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            env=env,
            preexec_fn=_ignore_int_quit,  # noqa: PLW1509 — single-threaded helper
        )
    started = _create_time(proc.pid)  # `proc` still referenced: pid not reaped
    logger.info("launched pid %d (start time %s)", proc.pid, started)
    return proc.pid, started


def _pids(idents: Sequence[ProcIdentity]) -> str:
    return " ".join(str(i.pid) for i in idents) or "none"


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


class _Clock:
    def __init__(self, total_s: float) -> None:
        self.start = time.monotonic()
        self.deadline = self.start + total_s

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def elapsed(self) -> int:
        return int(time.monotonic() - self.start)


def status(name: str, pattern: str, scanner: Scanner, timeout_s: float) -> Outcome:
    scan = scanner(pattern, timeout_s)
    if not scan.ok and not scan.matches:
        return Outcome(True, (f"  ? {name} UNKNOWN — {scan.detail}",))
    if not scan.matches:
        return Outcome(True, (f"  ✗ {name} NOT RUNNING",))
    # A seen match is a fact even when other processes were unreadable.
    line = f"  ✓ {name} (pid {scan.matches[0].pid})"
    if len(scan.matches) > 1:
        line += f" — DUPLICATE: {len(scan.matches)} matching processes (pids {_pids(scan.matches)})"
    if not scan.ok:
        line += f" — scan incomplete: {scan.detail}"
    return Outcome(True, (line,), running=scan.matches[0])


def _wait_gone(idents: Sequence[ProcIdentity], until: float, poll_s: float) -> list[ProcIdentity]:
    while True:
        alive = [i for i in idents if identity_alive(i)]
        now = time.monotonic()
        if not alive or now >= until:
            return alive
        time.sleep(min(poll_s, until - now))


def _stop_identities(
    name: str,
    pattern: str,
    recorded: Sequence[ProcIdentity],
    scanner: Scanner,
    cfg: ControlConfig,
    clock: _Clock,
    grace_s: float,
) -> Outcome:
    """SIGTERM → wait → SIGKILL → wait, signalling verified identities only."""
    fresh = scanner(pattern, min(cfg.scan_timeout_s, clock.remaining()))
    if not fresh.ok:
        alive = [i for i in recorded if identity_alive(i)]
        return Outcome(
            False,
            (
                f"  ✗ {name} STOP FAILED [phase=stop-wait] — {fresh.detail}; nothing "
                f"signalled; writer state: pid {_pids(alive)} still alive",
            ),
            phase="stop-wait",
        )
    targets = select_signal_targets(recorded, fresh.matches)
    for skipped in set(recorded) - set(targets):
        logger.warning("pid %d no longer the recorded writer — not signalled", skipped.pid)
    if not targets:
        return Outcome(
            True,
            (
                f"  - {name} pid {_pids(recorded)} no longer matches (exited, reused or re-exec'd) — nothing signalled",
            ),
        )
    for t in targets:
        signal_verified(t, signal.SIGTERM)
    # Wait on EVERY target, not just those signalled: a denied signal must
    # surface as a survivor, never as a false "stopped".
    grace_until = time.monotonic() + max(0.0, grace_s)
    survivors = _wait_gone(targets, grace_until, cfg.poll_s)
    if not survivors:
        return Outcome(True, (f"  ■ {name} stopped (pid {_pids(targets)}, {clock.elapsed()}s)",))
    waited = clock.elapsed()
    # Re-verify command identity before escalating: a survivor that exec'd
    # into another command keeps pid + start time but is no longer ours.
    recheck = scanner(pattern, min(cfg.scan_timeout_s, clock.remaining()))
    if not recheck.ok:
        return Outcome(
            False,
            (
                f"  ✗ {name} STOP FAILED [phase=stop-wait] — cannot re-verify before "
                f"SIGKILL: {recheck.detail}; writer state: pid {_pids(survivors)} still "
                f"alive after SIGTERM ({waited}s)",
            ),
            phase="stop-wait",
        )
    kill_targets = select_signal_targets(survivors, recheck.matches)
    changed = [s for s in survivors if s not in kill_targets]
    for c in changed:
        logger.warning("pid %d changed command after SIGTERM — not killed", c.pid)
    if not kill_targets:
        return Outcome(
            True,
            (
                f"  ■ {name} stopped — pid {_pids(changed)} no longer matches after "
                f"SIGTERM (exited or re-exec'd; not killed) ({waited}s)",
            ),
        )
    for s in kill_targets:
        signal_verified(s, signal.SIGKILL)
    kill_until = min(time.monotonic() + cfg.kill_wait_s, clock.deadline)
    survivors = _wait_gone(kill_targets, kill_until, cfg.poll_s)
    if survivors:
        return Outcome(
            False,
            (
                f"  ✗ {name} STOP FAILED [phase=stop-wait] — pid {_pids(survivors)} "
                f"still alive after SIGTERM+SIGKILL ({clock.elapsed()}s); "
                f"writer state: OLD WRITER RUNNING",
            ),
            phase="stop-wait",
        )
    return Outcome(True, (f"  ■ {name} KILLED after {waited}s (graceful stop timed out)",))


def stop(
    name: str,
    pattern: str,
    scanner: Scanner,
    cfg: ControlConfig = ControlConfig(),  # noqa: B008 — frozen dataclass
) -> Outcome:
    # discover + pre-signal + pre-SIGKILL scans, grace, kill wait.
    clock = _Clock(cfg.scan_timeout_s * 3 + cfg.stop_grace_s + cfg.kill_wait_s)
    scan = scanner(pattern, cfg.scan_timeout_s)
    if not scan.ok:
        return Outcome(
            False,
            (
                f"  ✗ {name} STOP FAILED [phase=discover] — {scan.detail}; writer state UNKNOWN, nothing signalled",
            ),
            phase="discover",
        )
    if not scan.matches:
        return Outcome(True, (f"  - {name} not running",))
    # Stop reduces writers, so every matching identity is stopped (dups too).
    return _stop_identities(name, pattern, scan.matches, scanner, cfg, clock, cfg.stop_grace_s)


def _launch_and_verify(
    name: str,
    pattern: str,
    launch_argv: Sequence[str],
    log_path: str,
    scanner: Scanner,
    launcher: Launcher,
    cfg: ControlConfig,
    clock: _Clock,
) -> Outcome:
    """Precondition: a conclusive scan just showed NO matching process."""
    try:
        # The launcher pins the start time before the child can be reaped;
        # every later check compares pid + start time, so a recycled pid is
        # never mistaken for our launch.
        pid, launched_ct = launcher(launch_argv, log_path)
    except OSError as exc:
        return Outcome(
            False,
            (f"  ✗ {name} FAILED [phase=start] — launch error: {exc}; writer state: NONE running",),
            phase="start",
        )

    def launched_alive() -> bool:
        return launched_ct is not None and identity_alive(ProcIdentity(pid, launched_ct, ""))

    time.sleep(max(0.0, min(cfg.start_settle_s, clock.remaining())))
    last = ScanResult(False, (), "deadline reached before verification", 0.0)
    while True:
        last = scanner(pattern, min(cfg.scan_timeout_s, clock.remaining()))
        if len(last.matches) > 1:  # duplicates are a fact even if scan incomplete
            return Outcome(
                False,
                (
                    f"  ✗ {name} FAILED [phase=verify] — DUPLICATE writers: "
                    f"{len(last.matches)} matching processes (pids {_pids(last.matches)}, "
                    f"launched {pid}); not launching again, investigate — "
                    f"check {log_path}",
                ),
                phase="verify",
            )
        if last.ok:
            mine = [m for m in last.matches if m.pid == pid and m.create_time == launched_ct]
            if mine:
                return Outcome(True, (f"  ▶ {name} started (pid {pid})",), running=mine[0])
            alive_now = launched_alive()
            if not alive_now and not last.matches:
                return Outcome(
                    False,
                    (
                        f"  ✗ {name} FAILED [phase=verify] — launched pid {pid} exited; "
                        f"writer state: NONE running — check {log_path}",
                    ),
                    phase="verify",
                )
            if not alive_now:
                return Outcome(
                    False,
                    (
                        f"  ✗ {name} FAILED [phase=verify] — launched pid {pid} exited but "
                        f"pid {_pids(last.matches)} matches; writer state: FOREIGN "
                        f"process running — investigate",
                    ),
                    phase="verify",
                )
        if clock.remaining() <= cfg.poll_s:
            break
        time.sleep(cfg.poll_s)
    alive = launched_alive()
    detail = (
        last.detail
        if not last.ok
        else f"pid {_pids(last.matches)} matched, launched pid not among them"
    )
    return Outcome(
        False,
        (
            f"  ✗ {name} FAILED [phase=verify] — {detail}; writer state: launched "
            f"pid {pid} {'ALIVE' if alive else 'exited'}, other writers UNKNOWN — "
            f"check {log_path}",
        ),
        phase="verify",
    )


def _create_time(pid: int) -> float | None:
    """Start time of ``pid``, or None if it no longer exists."""
    try:
        return float(psutil.Process(pid).create_time())
    except psutil.NoSuchProcess:
        return None


def start(
    name: str,
    pattern: str,
    launch_argv: Sequence[str],
    log_path: str,
    scanner: Scanner,
    launcher: Launcher = launch_nohup,
    cfg: ControlConfig = ControlConfig(),  # noqa: B008 — frozen dataclass
) -> Outcome:
    clock = _Clock(cfg.scan_timeout_s + cfg.start_settle_s + cfg.start_verify_s)
    scan = scanner(pattern, cfg.scan_timeout_s)
    if not scan.ok:
        return Outcome(
            False,
            (
                f"  ✗ {name} START REFUSED [phase=start] — {scan.detail}; writer state "
                f"UNKNOWN — not launching (could create a duplicate writer)",
            ),
            phase="start",
        )
    if len(scan.matches) > 1:
        return Outcome(
            False,
            (
                f"  ✗ {name} DUPLICATE writers running (pids {_pids(scan.matches)}) — "
                f"not launching another; investigate",
            ),
            phase="start",
        )
    if scan.matches:
        return Outcome(
            True,
            (f"  ✓ {name} already running (pid {scan.matches[0].pid})",),
            running=scan.matches[0],
        )
    return _launch_and_verify(name, pattern, launch_argv, log_path, scanner, launcher, cfg, clock)


def restart(
    name: str,
    pattern: str,
    launch_argv: Sequence[str],
    log_path: str,
    scanner: Scanner,
    deadline_s: float = RESTART_DEADLINE_S,
    launcher: Launcher = launch_nohup,
    cfg: ControlConfig = ControlConfig(),  # noqa: B008 — frozen dataclass
) -> Outcome:
    """Stop + start + verify the identity changed, all within ``deadline_s``."""
    assert deadline_s > 0
    clock = _Clock(deadline_s)
    lines: list[str] = []

    def fail(phase: str, detail: str, state: str) -> Outcome:
        lines.append(
            f"  ✗ {name} RESTART FAILED [phase={phase}] after {clock.elapsed()}s — "
            f"{detail}; writer state: {state} — check {log_path}"
        )
        logger.warning("restart %s failed in phase %s: %s", name, phase, detail)
        return Outcome(False, tuple(lines), phase=phase)

    # discover
    scan = scanner(pattern, min(cfg.scan_timeout_s, clock.remaining()))
    if not scan.ok:
        return fail("discover", scan.detail, "UNKNOWN (nothing signalled or started)")
    if len(scan.matches) > 1:
        return fail(
            "discover",
            f"REFUSED: DUPLICATE writers ({len(scan.matches)} matching processes)",
            f"pids {_pids(scan.matches)} RUNNING (nothing signalled or started)",
        )
    old = scan.matches[0] if scan.matches else None

    # stop-wait
    if old is None:
        lines.append(f"  - {name} not running")
    else:
        grace = min(cfg.stop_grace_s, clock.remaining() - cfg.start_reserve_s)
        stopped = _stop_identities(name, pattern, [old], scanner, cfg, clock, grace)
        if not stopped.ok:
            lines.extend(stopped.lines)
            state = f"old pid {old.pid} {'STILL RUNNING' if identity_alive(old) else 'exited'}"
            return fail("stop-wait", "old writer did not stop in time", state)
        lines.extend(stopped.lines)

    if old is None:
        old_desc = "no previous writer"
    elif identity_alive(old):  # same pid+start time but no longer matches
        old_desc = f"old pid {old.pid} alive but no longer matches the pattern"
    else:
        old_desc = f"old pid {old.pid} exited"

    # start — re-check nothing matches (a stalled or racing scan never launches)
    pre = scanner(pattern, min(cfg.scan_timeout_s, clock.remaining()))
    if not pre.ok:
        return fail("start", pre.detail, f"UNKNOWN ({old_desc}); replacement NOT started")
    if pre.matches:
        return fail(
            "start",
            "a matching process appeared after stop",
            f"pid {_pids(pre.matches)} RUNNING ({old_desc}); replacement NOT started",
        )
    started = _launch_and_verify(
        name, pattern, launch_argv, log_path, scanner, launcher, cfg, clock
    )
    lines.extend(started.lines)
    if not started.ok or started.running is None:
        return fail(
            started.phase or "verify", "replacement not verified", f"see above ({old_desc})"
        )

    # verify identity changed (CL-obgy)
    new = started.running
    if not identity_changed(old, new):
        return fail("verify", "surviving process is the old one", f"old pid {new.pid} RUNNING")
    lines.append(f"  ↻ {name} restart verified (pid {old.pid if old else 'none'} → {new.pid})")
    return Outcome(True, tuple(lines), running=new)


# --------------------------------------------------------------------------
# CLI (called by scripts/daemons.sh)
# --------------------------------------------------------------------------


def _scanner(pattern: str, timeout_s: float) -> ScanResult:
    return bounded_scan(pattern, timeout_s)


def main(argv: Sequence[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in[:1] == [_WORKER_CMD]:
        return _worker_main()
    logging.basicConfig(level=logging.WARNING, format="proc_discovery %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("op", choices=["status", "start", "stop", "restart"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--deadline", type=float, default=RESTART_DEADLINE_S)
    args = ap.parse_args(args_in)
    pattern = os.environ.get(PATTERN_ENV, "")
    if not pattern:
        print(f"  ✗ {args.name} — {PATTERN_ENV} not set", file=sys.stderr)
        return 2
    launch = os.environ.get(LAUNCH_ENV, "").split()
    log_path = args.log or f"logs/{args.name}.log"
    if args.op in ("start", "restart") and not launch:
        print(f"  ✗ {args.name} — {LAUNCH_ENV} not set", file=sys.stderr)
        return 2
    if args.op == "status":
        out = status(args.name, pattern, _scanner, STATUS_SCAN_TIMEOUT_S)
    elif args.op == "stop":
        out = stop(args.name, pattern, _scanner)
    elif args.op == "start":
        out = start(args.name, pattern, launch, log_path, _scanner)
    else:
        if args.deadline <= 0:
            print(f"  ✗ {args.name} — --deadline must be > 0", file=sys.stderr)
            return 2
        out = restart(args.name, pattern, launch, log_path, _scanner, deadline_s=args.deadline)
    for line in out.lines:
        print(line, flush=True)
    return 0 if out.ok else 1


if __name__ == "__main__":
    sys.exit(main())
