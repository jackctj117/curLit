#!/usr/bin/env python3
"""Credential-free, offline container checks from a source snapshot (CL-i6ke).

Requires an already prepared, locally available test-dependency image. Never
builds/pulls an image or mounts the checkout, Docker socket, home or runtime data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shlex
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

# Explicit source/fixture roots used by the unit suite. Operational state and
# .claude/worktrees are deliberately absent; never archive the checkout wholesale.
SOURCE_ROOTS = (
    "src", "tests", "scripts", "configs", "migrations", "docs", "research",
    "airflow", "grafana", "knowledge/obsidian/templates", "deploy", "alertmanager",
    "labeling", "loki", "models", "prometheus", "promtail", "training", "web-ui",
    "reports", ".github",
)
SOURCE_FILES = (
    "AGENTS.md", "CLAUDE.md", "README.md", "CHANGELOG.md", "curLit-idea.md",
    "pyproject.toml", "pytest.ini", "Makefile", ".gitignore", ".dockerignore",
    ".gitleaks.toml", ".env.example", "Dockerfile", "Dockerfile.airflow",
    "docker-compose.yml", "docker-compose.app.yml", "requirements.lock",
    ".beads/PRIME.md", ".beads/config.yaml", ".claude/settings.json",
    ".claude/settings.local.example.json", ".codex/config.toml",
)
CHECKS = {
    "unit": ["python", "-m", "pytest", "tests/unit"],
    "vault": [
        "python", "-m", "pytest", "tests/unit/test_vault_peer_auth.py",
        "tests/unit/test_vault_wire.py",
    ],
    "lint": ["python", "-m", "ruff", "check", "src/", "tests/", "scripts/test_isolated.py"],
    "typecheck": ["python", "-m", "mypy", "src/"],
}


def allowed_source(name: str) -> bool:
    """Only source/fixtures; reject traversal, private env files and cache state."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    if name in SOURCE_FILES:
        return True
    if any(
        part in {".git", ".venv", "__pycache__", "secrets", ".pytest_cache", ".mypy_cache"}
        or part.startswith(".env")
        for part in path.parts
    ):
        return False
    return any(name.startswith(root + "/") for root in SOURCE_ROOTS)


def source_names(root: Path) -> list[str]:
    """Include tracked edits and new Python/docs, not untracked report exports."""
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "-z", "--", *SOURCE_ROOTS, *SOURCE_FILES],
        cwd=root, capture_output=True, check=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z",
         "--", "src", "tests", "scripts", "docs"],
        cwd=root, capture_output=True, check=True,
    )
    names = {os.fsdecode(name) for name in tracked.stdout.split(b"\0") if name}
    for raw in untracked.stdout.split(b"\0"):
        name = os.fsdecode(raw)
        if name.endswith(".py") or (name.startswith("docs/") and name.endswith(".md")):
            names.add(name)
    return sorted(name for name in names if allowed_source(name))


def write_snapshot(root: Path, archive: Path, names: list[str]) -> str:
    """Reject symlinks/special files rather than follow them into host state."""
    root = root.resolve()
    with tarfile.open(archive, "w") as output:
        for name in names:
            if not allowed_source(name):
                raise ValueError(f"not an allowed source path: {name}")
            path = root / name
            if any(p.is_symlink() for p in (path, *path.parents) if p != root):
                raise ValueError(f"symlink in source snapshot: {name}")
            if not path.exists():  # Tracked deletion, not content from HEAD.
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"not a regular source file: {name}")
            # Do not follow a last-component symlink or block on a FIFO swapped
            # in after lstat. fstat below rechecks the opened descriptor.
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as source:
                file_stat = os.fstat(source.fileno())
                if not stat.S_ISREG(file_stat.st_mode):
                    raise ValueError(f"not a regular source file: {name}")
                content = source.read()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o755 if file_stat.st_mode & stat.S_IXUSR else 0o644
            output.addfile(info, io.BytesIO(content))
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def docker_command(docker: list[str], image_id: str, check: str) -> list[str]:
    """Fixed container boundaries; no inherited image command or caller env."""
    command = shlex.join(CHECKS[check])
    # /tmp is container-private. Unix sockets work there without external network
    # or access to operational sockets. Resource caps bound a runaway test.
    entry = (
        "set -eu; mkdir -p /tmp/curlit; tar -xf - -C /tmp/curlit; "
        "cd /tmp/curlit; exec env -i PATH=/usr/local/bin:/usr/bin:/bin "
        "PYTHONPATH=/tmp/curlit PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
        "TMPDIR=/tmp XDG_CACHE_HOME=/tmp/cache NUMBA_CACHE_DIR=/tmp/numba "
        # Supported Python implementation, not a mock or disabled numerical
        # test. Native compilation cannot load .so files from noexec /tmp.
        "PYTENSOR_FLAGS=base_compiledir=/tmp/pytensor,cxx= "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        + command
    )
    return [
        *docker, "run", "--rm", "--init", "--interactive", "--pull=never", "--no-healthcheck",
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--user=10001:10001",
        "--pids-limit=256", "--memory=4g", "--cpus=2",
        "--tmpfs=/tmp:rw,nosuid,noexec,size=2g", "--workdir=/tmp",
        "--entrypoint=/bin/sh", image_id, "-c", entry,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Existing credential-free test image")
    parser.add_argument("--context", required=True, help="Explicit development Docker context")
    parser.add_argument("--check", choices=sorted(CHECKS), default="unit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    docker = ["docker", "--context", args.context]
    # Resolve once and execute that immutable local image, never a later tag.
    inspected = subprocess.run(
        [*docker, "image", "inspect", args.image, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True,
    )
    image_id = inspected.stdout.strip()
    if not image_id.startswith("sha256:") or len(image_id) != len("sha256:") + 64:
        raise ValueError("Docker did not return one immutable image ID")
    with tempfile.TemporaryDirectory(prefix="curlit-test-source-") as directory:
        archive = Path(directory) / "source.tar"
        names = source_names(root)
        digest = write_snapshot(root, archive, names)
        print(f"check={args.check} image={image_id} source_sha256={digest}", flush=True)
        print(f"source_candidates={len(names)}; no host mounts; network=none", flush=True)
        print("PyTensor backend: Python (cxx=); container /tmp stays noexec", flush=True)
        with archive.open("rb") as source:
            result = subprocess.run(
                docker_command(docker, image_id, args.check), stdin=source, check=False,
            )
        print(f"check={args.check} exit_code={result.returncode}", flush=True)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
