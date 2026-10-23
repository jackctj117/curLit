"""CL-ek1d: fake clocks and owned POSIX children only, never provider calls."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.events import shadow_deadline
from src.events.shadow_deadline import (
    CLEANUP_GRACE_SECONDS,
    Deadline,
    DeadlineClient,
    Diagnostics,
    ShadowDeadlineError,
    supervise,
)


def test_cleanup_never_sends_terminating_signal_after_reaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    reaped = [False]
    killed = [False]
    ordering: list[str] = []

    def group_signal(pid: int, sig: int) -> None:
        if sig:
            assert not reaped[0], "PID identity released before a terminating signal"
            ordering.append(signal.Signals(sig).name)
            if sig == signal.SIGKILL:
                killed[0] = True
        elif killed[0]:
            raise ProcessLookupError

    def reap(pid: int, options: int) -> tuple[int, int]:
        reaped[0] = True
        ordering.append("reaped")
        return pid, 0

    monkeypatch.setattr(shadow_deadline.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        shadow_deadline.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    monkeypatch.setattr(shadow_deadline.os, "getpgrp", lambda: 1)
    monkeypatch.setattr(shadow_deadline.os, "killpg", group_signal)
    monkeypatch.setattr(shadow_deadline.os, "waitpid", reap)
    outcome = shadow_deadline._cleanup(123, True)
    assert outcome["worker_reaped"]
    assert ordering == ["SIGTERM", "SIGKILL", "reaped"]


def test_fake_clock_blocks_later_role_without_dispatch(tmp_path: Path) -> None:
    now = [0.0]
    calls: list[str] = []
    deadline = Deadline(2, clock=lambda: now[0])
    diagnostics = Diagnostics(tmp_path / "calls.jsonl")

    class Client:
        def complete(self, **kwargs: Any) -> Any:
            calls.append("transport")
            now[0] = 3
            return SimpleNamespace(text="partial answer")

    for role in ("traversal", "entailment", "critic", "baseline"):
        with pytest.raises(ShadowDeadlineError):
            DeadlineClient(Client(), deadline, diagnostics, role).complete(model="fixture")
    assert calls == ["transport"]
    records = [json.loads(line) for line in diagnostics.path.read_text().splitlines()]
    assert [r["kind"] for r in records] == ["call_started", "call_returned", "call_failed"]
    assert records[1]["response_text"] == "partial answer"
    assert records[1]["usd_cost"] is None


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_deadline(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        Deadline(seconds)


@given(st.floats(min_value=0.001, max_value=1e6), st.floats(min_value=0, max_value=1e6))
def test_deadline_remaining_is_bounded_and_decreases(seconds: float, elapsed: float) -> None:
    now = [0.0]
    deadline = Deadline(seconds, clock=lambda: now[0])
    now[0] = elapsed
    remaining = deadline.remaining()
    assert 0 <= remaining <= seconds
    assert remaining == max(0.0, seconds - elapsed)
    now[0] += 1
    assert deadline.remaining() <= remaining


def _assert_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_timeout_cleans_owned_child_and_preserves_partial_diagnostics(tmp_path: Path) -> None:
    def work(deadline: Deadline, diagnostics: Diagnostics) -> dict[str, Any]:
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) as child:
            diagnostics.append("owned_fixture_child", pid=child.pid)
            child.wait()
        return {"unexpected_success": True}

    output = tmp_path / "report.json"
    started = time.monotonic()
    code = supervise(work, output=output, seconds=0.3)
    elapsed = time.monotonic() - started
    assert code == 124
    # One second scheduling tolerance; cleanup has its own disclosed bound.
    assert elapsed < 0.3 + CLEANUP_GRACE_SECONDS + 1
    report = json.loads(output.read_text())
    assert report["status"] == "incomplete" and report["reason"] == "shadow_wall_deadline"
    assert report["usd_cost"] is None and "unexpected_success" not in report
    assert report["supervision"]["worker_reaped"]
    assert not report["supervision"]["group_still_present"]
    records = [
        json.loads(line) for line in Path(report["diagnostics_path"]).read_text().splitlines()
    ]
    [child] = [r for r in records if r["kind"] == "owned_fixture_child"]
    _assert_gone(child["pid"])


def test_success_also_cleans_leftover_owned_descendant(tmp_path: Path) -> None:
    def work(deadline: Deadline, diagnostics: Diagnostics) -> dict[str, Any]:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        diagnostics.append("owned_fixture_child", pid=child.pid)
        return {"mode": "shadow_only", "fixture_complete": True}

    output = tmp_path / "report.json"
    assert supervise(work, output=output, seconds=3) == 0
    report = json.loads(output.read_text())
    assert report["fixture_complete"] and report["supervision"]["worker_reaped"]
    records = [
        json.loads(line)
        for line in Path(report["supervision"]["diagnostics_path"]).read_text().splitlines()
    ]
    _assert_gone(next(r["pid"] for r in records if r["kind"] == "owned_fixture_child"))


def test_stubborn_owned_child_is_killed_and_reaped(tmp_path: Path) -> None:
    def work(deadline: Deadline, diagnostics: Diagnostics) -> dict[str, Any]:
        child_code = (
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)"
        )
        with subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE) as child:
            assert child.stdout is not None and child.stdout.readline() == b"ready\n"
            diagnostics.append("owned_fixture_child", pid=child.pid)
            child.wait()
        return {}

    output = tmp_path / "report.json"
    assert supervise(work, output=output, seconds=0.5) == 124
    report = json.loads(output.read_text())
    assert report["supervision"]["termination_escalated"]
    assert report["supervision"]["worker_reaped"]
    records = [
        json.loads(line) for line in Path(report["diagnostics_path"]).read_text().splitlines()
    ]
    _assert_gone(next(r["pid"] for r in records if r["kind"] == "owned_fixture_child"))


def test_failed_group_handshake_never_signals_a_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_session() -> None:
        raise OSError("synthetic setsid failure")

    def forbidden_group_signal(pid: int, sig: int) -> None:
        raise AssertionError("must not signal an unacknowledged process group")

    monkeypatch.setattr(os, "setsid", fail_session)
    monkeypatch.setattr(os, "killpg", forbidden_group_signal)
    output = tmp_path / "report.json"
    assert supervise(lambda deadline, diagnostics: {}, output=output, seconds=1) == 1
    report = json.loads(output.read_text())
    assert report["supervision"]["worker_reaped"]
    assert report["supervision"]["group_verified"] is False


def test_parent_interrupt_cleans_worker_and_reports_incomplete(tmp_path: Path) -> None:
    def work(deadline: Deadline, diagnostics: Diagnostics) -> dict[str, Any]:
        os.kill(os.getppid(), signal.SIGTERM)
        time.sleep(60)
        return {}

    output = tmp_path / "report.json"
    previous = signal.getsignal(signal.SIGTERM)
    assert supervise(work, output=output, seconds=3) == 128 + signal.SIGTERM
    report = json.loads(output.read_text())
    assert report["reason"] == "operator_interrupted"
    assert report["supervision"]["worker_reaped"]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("original evidence")
    with pytest.raises(FileExistsError):
        supervise(lambda deadline, diagnostics: {}, output=output, seconds=1)
    assert output.read_text() == "original evidence"


def test_partial_diagnostic_tail_is_explicit_and_original_bytes_preserved(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.jsonl"
    raw = b'{"kind":"complete"}\n{"kind":"partial'
    path.write_bytes(raw)
    result = shadow_deadline.diagnostic_integrity(path)
    assert result["complete_records"] == 1
    assert result["complete_prefix_bytes"] == len(b'{"kind":"complete"}\n')
    assert result["truncated_tail"] is True
    assert path.read_bytes() == raw
