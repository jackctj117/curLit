"""Offline source-snapshot/container boundary oracles (CL-i6ke)."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from scripts import test_isolated
from scripts.test_isolated import allowed_source, docker_command, write_snapshot


@pytest.mark.parametrize("name", [
    ".env", ".env.backup", "data/orders.json", ".beads/dolt/db",
    ".claude/worktrees/task/.env", "src/.env", "tests/secrets/key",
    "src/../../.env", "/etc/passwd", "src/__pycache__/module.pyc",
])
def test_private_and_runtime_paths_are_not_source(name: str) -> None:
    assert not allowed_source(name)


def test_new_source_included_but_new_data_exports_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        b"src/tracked.py\0reports/fixture.json\0.env\0",
        b"src/new.py\0tests/new.py\0docs/new.md\0scripts/export.json\0",
    ]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=responses.pop(0))

    monkeypatch.setattr(test_isolated.subprocess, "run", fake_run)
    assert test_isolated.source_names(tmp_path) == [
        "docs/new.md", "reports/fixture.json", "src/new.py", "src/tracked.py", "tests/new.py",
    ]
    assert not responses


def test_snapshot_contains_current_bytes_not_host_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src/code.py").write_text("CURRENT_EDIT = True\n")
    (repo / ".env").write_text("SYNTHETIC_SECRET=not-for-snapshot\n")
    archive = tmp_path / "snapshot.tar"
    write_snapshot(repo, archive, ["src/code.py", "src/deleted.py"])
    with tarfile.open(archive) as captured:
        assert captured.getnames() == ["src/code.py"]
        source = captured.extractfile("src/code.py")
        assert source is not None
        assert source.read() == b"CURRENT_EDIT = True\n"
    assert b"SYNTHETIC_SECRET" not in archive.read_bytes()


def test_snapshot_refuses_symlink_to_private_file(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "private").write_text("synthetic secret")
    (tmp_path / "src/code.py").symlink_to(tmp_path / "private")
    with pytest.raises(ValueError, match="symlink"):
        write_snapshot(tmp_path, tmp_path / "snapshot.tar", ["src/code.py"])


def test_snapshot_refuses_symlinked_parent(tmp_path: Path) -> None:
    (tmp_path / "private").mkdir()
    (tmp_path / "private/code.py").write_text("synthetic secret")
    (tmp_path / "src").symlink_to(tmp_path / "private", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        write_snapshot(tmp_path, tmp_path / "snapshot.tar", ["src/code.py"])


def test_snapshot_rejects_forbidden_explicit_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not an allowed"):
        write_snapshot(tmp_path, tmp_path / "snapshot.tar", [".env"])


def test_snapshot_refuses_fifo_without_opening_it(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    os.mkfifo(tmp_path / "src/channel.py")
    with pytest.raises(ValueError, match="not a regular"):
        write_snapshot(tmp_path, tmp_path / "snapshot.tar", ["src/channel.py"])


def test_container_keeps_host_network_and_state_inaccessible() -> None:
    command = docker_command(["docker", "--context", "fixture"], "sha256:fixture", "unit")
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--user=10001:10001" in command
    assert "--pull=never" in command
    assert "--init" in command
    assert "--no-healthcheck" in command
    assert "--entrypoint=/bin/sh" in command
    assert "--tmpfs=/tmp:rw,nosuid,noexec,size=2g" in command
    assert not any(arg.startswith(("--mount", "--volume", "--env", "--publish"))
                   or arg in {"-v", "-e", "-p", "--privileged"} for arg in command)
    shell = command[-1]
    assert "exec env -i " in shell
    assert "PYTENSOR_FLAGS=base_compiledir=/tmp/pytensor,cxx= " in shell
    assert shell.endswith("python -m pytest tests/unit")
    assert "--ignore" not in shell and " -k " not in shell


def test_archive_does_not_preserve_host_owner_or_absolute_names(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/code.py").write_text("pass\n")
    archive = tmp_path / "snapshot.tar"
    write_snapshot(tmp_path, archive, ["src/code.py"])
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as captured:
        member = captured.getmembers()[0]
        assert member.name == "src/code.py"
        assert member.uid == member.gid == 0
        assert member.uname == member.gname == ""


def test_failed_check_is_not_reported_successful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent nonzero exit oracle, without invoking Docker from a test."""
    image_id = "sha256:" + "a" * 64
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src/code.py").write_text("pass\n")
    monkeypatch.setattr(test_isolated, "__file__", str(tmp_path / "scripts/test_isolated.py"))
    monkeypatch.setattr(test_isolated, "source_names", lambda root: ["src/code.py"])
    monkeypatch.setattr(sys, "argv", [
        "test_isolated.py", "--image", "fixture-image", "--context", "fixture-context",
    ])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "inspect" in command:
            return subprocess.CompletedProcess(command, 0, stdout=image_id + "\n")
        assert image_id in command
        assert "fixture-image" not in command  # Run immutable inspected image.
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(test_isolated.subprocess, "run", fake_run)
    assert test_isolated.main() == 7
    assert len(calls) == 2
