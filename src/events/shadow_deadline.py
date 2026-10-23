"""Owned-process wall deadline for no-order shadow experiments (CL-ek1d).

POSIX only. The parent never discovers or signals operational processes: its
only group is the forked worker's new session, acknowledged over a private pipe.
The research deadline stops work; a separately bounded cleanup grace follows.
"""

from __future__ import annotations

import json
import logging
import math
import os
import select
import signal
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.events._util import ThreadLocalClient

logger = logging.getLogger(__name__)
# Allow subprocess.run's exception cleanup to reap its child before escalation.
CLEANUP_GRACE_SECONDS = 2.0
# Reserve half the cleanup allowance for observing/reaping after SIGKILL.
TERM_GRACE_SECONDS = CLEANUP_GRACE_SECONDS / 2
# Bound process-exit polling without busy waiting or delaying cancellation.
POLL_SECONDS = 0.02


class ShadowDeadlineError(RuntimeError):
    reason = "shadow_wall_deadline"


class _Canceled(BaseException):
    pass


class _Interrupted(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum


class Deadline:
    def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("wall timeout must be finite and positive")
        self.clock = clock
        self.started = clock()
        self.expires = self.started + seconds

    def remaining(self) -> float:
        return max(0.0, self.expires - self.clock())

    def check(self) -> None:
        if self.remaining() <= 0:
            raise ShadowDeadlineError("shadow_wall_deadline")


class Diagnostics:
    """Fsync complete records; parent flags any trailing write interrupted by kill."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._sequence = 0

    def append(self, kind: str, **fields: Any) -> int:
        with self._lock:
            self._sequence += 1
            record = {"sequence": self._sequence, "kind": kind, **fields}
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return self._sequence


class DeadlineClient:
    """Check the one experiment deadline immediately before each role dispatch."""

    def __init__(
        self, client: Any, deadline: Deadline, diagnostics: Diagnostics, role: str
    ) -> None:
        self.client, self.deadline, self.diagnostics, self.role = (
            client,
            deadline,
            diagnostics,
            role,
        )

    def complete(self, **kwargs: Any) -> Any:
        self.deadline.check()
        client = self.client.get() if isinstance(self.client, ThreadLocalClient) else self.client
        call_id = self.diagnostics.append("call_started", role=self.role, model=kwargs.get("model"))
        try:
            # Resolution and durable logging consume the SAME deadline too.
            self.deadline.check()
            response = client.complete(**kwargs)
            self.diagnostics.append(
                "call_returned",
                call_id=call_id,
                role=self.role,
                response_text=getattr(response, "text", None),
                actual_model=getattr(response, "model", None),
                input_tokens=getattr(response, "input_tokens", None),
                output_tokens=getattr(response, "output_tokens", None),
                usd_cost=None,
            )
            self.deadline.check()
            return response
        except Exception as exc:
            self.diagnostics.append(
                "call_failed",
                call_id=call_id,
                role=self.role,
                error=type(exc).__name__,
                usd_cost=None,
            )
            raise


def _write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    """Publish a complete durable JSON file; never clobber an existing first report."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, allow_nan=False, default=str, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _cleanup(pid: int, ready: bool) -> dict[str, Any]:
    """Only the directly owned PID, or its acknowledged group, can be signaled."""
    assert pid > 0 and pid != os.getpgrp()
    signal_errors: list[str] = []

    def send(sig: signal.Signals) -> None:
        try:
            if ready:
                os.killpg(pid, sig)
            else:
                os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can reject signalling a group containing only zombies.
            # Record it, still reap, and require verified absence afterwards;
            # a live or inaccessible group cannot become successful cleanup.
            signal_errors.append(f"{sig.name}:PermissionError")
            logger.warning("Owned shadow signal rejected pid=%d signal=%s", pid, sig.name)

    logger.info("Stopping owned shadow worker pid=%d group_verified=%s", pid, ready)
    send(signal.SIGTERM)
    started = time.monotonic()
    until = started + CLEANUP_GRACE_SECONDS
    reaped = False
    status = None
    # Do NOT call waitpid before the final group signal. Even an exited leader
    # reserves its PID while unreaped, preventing PGID reuse from redirecting
    # escalation to an unrelated process group.
    while time.monotonic() < started + TERM_GRACE_SECONDS:
        time.sleep(POLL_SECONDS)
    send(signal.SIGKILL)
    while time.monotonic() < until:
        if not reaped:
            child, status = os.waitpid(pid, os.WNOHANG)
            reaped = child == pid
        if reaped and (not ready or not _group_exists(pid)):
            break
        time.sleep(POLL_SECONDS)
    return {
        "worker_reaped": reaped,
        "worker_wait_status": status,
        "group_verified": ready,
        "termination_escalated": True,
        "signal_errors": signal_errors,
        # Signal zero is only an existence probe, never a terminating signal.
        # PID reuse here can conservatively produce cleanup_unverified, but
        # cannot terminate an unrelated process. All TERM/KILL sends precede reap.
        "group_still_present": ready and _group_exists(pid),
    }


def diagnostic_integrity(path: Path) -> dict[str, Any]:
    """Identify the usable JSONL prefix without discarding a killed partial write."""
    records, prefix_bytes = 0, 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                return {
                    "complete_records": records,
                    "complete_prefix_bytes": prefix_bytes,
                    "truncated_tail": True,
                    "invalid_record": False,
                }
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
            except ValueError:
                return {
                    "complete_records": records,
                    "complete_prefix_bytes": prefix_bytes,
                    "truncated_tail": False,
                    "invalid_record": True,
                }
            records += 1
            prefix_bytes += len(line)
    return {
        "complete_records": records,
        "complete_prefix_bytes": prefix_bytes,
        "truncated_tail": False,
        "invalid_record": False,
    }


def supervise(
    work: Callable[[Deadline, Diagnostics], dict[str, Any]],
    *,
    output: Path,
    seconds: float,
) -> int:
    """Return 0 for finished work, 124 for deadline, nonzero for interruption/failure.

    Fork is intentionally restricted to a single-threaded POSIX entrypoint;
    clients are constructed in the worker, never forked from a live CLI driver.
    """
    deadline = Deadline(seconds)
    if os.name != "posix" or threading.active_count() != 1:
        raise RuntimeError("shadow supervision requires a single-threaded POSIX entrypoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output.with_name(output.name + ".diagnostics.jsonl")
    initial: dict[str, Any] = {
        "mode": "shadow_only",
        "status": "incomplete",
        "reason": "worker_not_finished",
        "quality_acceptance": "not_evaluated",
        "usd_cost": None,
        "diagnostics_path": str(diagnostics_path),
        "wall_timeout_seconds": seconds,
        "cleanup_grace_seconds": CLEANUP_GRACE_SECONDS,
    }
    _write_json(output, initial, exclusive=True)
    descriptor = os.open(diagnostics_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    _sync_directory(output.parent)
    diagnostics = Diagnostics(diagnostics_path)
    diagnostics.append("supervisor_started", wall_timeout_seconds=seconds)
    read_fd, write_fd = os.pipe()
    ack_read, ack_write = os.pipe()
    old_handlers: dict[signal.Signals, Any] = {}
    pid = 0
    ready = False
    reason, code = "worker_failed", 1
    cleanup: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix=".shadow-worker-", dir=output.parent) as scratch:
        result_path = Path(scratch) / "result.json"

        def interrupted(signum: int, frame: Any) -> None:
            raise _Interrupted(signum)

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, interrupted)
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                os.close(ack_write)
                try:
                    os.setsid()
                    os.write(write_fd, b"R")  # Group ownership established BEFORE any work.
                    # No descendants before the parent knows our group exists.
                    if os.read(ack_read, 1) != b"A":
                        os._exit(1)
                    os.close(ack_read)

                    def cancel(signum: int, frame: Any) -> None:
                        raise _Canceled()

                    signal.signal(signal.SIGTERM, cancel)
                    signal.signal(signal.SIGINT, cancel)
                    try:
                        deadline.check()
                        result = work(deadline, diagnostics)
                        deadline.check()
                        _write_json(result_path, result)
                        os.write(write_fd, b"D")
                    except BaseException as exc:
                        diagnostics.append("worker_failed", error=type(exc).__name__, usd_cost=None)
                        os.write(write_fd, b"F")
                    # Keep the group leader owned until parent cleanup, including
                    # successful work that accidentally left a descendant alive.
                    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    signal.signal(signal.SIGINT, signal.SIG_DFL)
                    while True:
                        signal.pause()
                finally:
                    os._exit(1)
            os.close(write_fd)
            write_fd = -1
            os.close(ack_read)
            ack_read = -1
            while True:
                remaining = deadline.remaining()
                if remaining <= 0:
                    reason, code = "shadow_wall_deadline", 124
                    break
                readable, _, _ = select.select([read_fd], [], [], remaining)
                if not readable:
                    reason, code = "shadow_wall_deadline", 124
                    break
                marker = os.read(read_fd, 1)
                if marker == b"R":
                    ready = os.getpgid(pid) == pid and pid != os.getpgrp()
                    if not ready:
                        raise RuntimeError("shadow worker process-group handshake failed")
                    os.write(ack_write, b"A")
                elif marker == b"D" and ready:
                    reason, code = (
                        ("completed", 0)
                        if deadline.remaining() > 0
                        else ("shadow_wall_deadline", 124)
                    )
                    break
                else:
                    break
        except _Interrupted as exc:
            reason, code = "operator_interrupted", 128 + exc.signum
        finally:
            # Further terminal signals cannot abandon this owned cleanup.
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            if pid > 0:
                cleanup = _cleanup(pid, ready)
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
            if ack_read >= 0:
                os.close(ack_read)
            os.close(ack_write)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        if not cleanup.get("worker_reaped") or cleanup.get("group_still_present"):
            reason, code = "cleanup_unverified", 1
        integrity = diagnostic_integrity(diagnostics_path)
        if code == 0 and (integrity["truncated_tail"] or integrity["invalid_record"]):
            reason, code = "diagnostics_incomplete", 1
        report = json.loads(result_path.read_text()) if code == 0 else {**initial, "reason": reason}
        report["supervision"] = {
            "status": reason,
            "elapsed_seconds": time.monotonic() - deadline.started,
            "wall_timeout_seconds": seconds,
            "cleanup_grace_seconds": CLEANUP_GRACE_SECONDS,
            "diagnostics_path": str(diagnostics_path),
            "worker_pid": pid,
            "diagnostics_integrity": integrity,
            **cleanup,
        }
        _write_json(output, report)
        return code
