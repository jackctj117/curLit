"""Fresh-process release/configuration provenance, not stale checkout identity (CL-0deu.14)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SECRET_KEY = re.compile(
    r"secret|password|credential|api.?key|authorization|access.?token|database_url|dsn", re.I
)


def redact(value: Any) -> Any:
    """Serialized runtime config boundary; never include known credential fields."""
    if isinstance(value, dict):
        return {
            str(k): "[REDACTED]" if SECRET_KEY.search(str(k)) else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def write_release_manifest(
    root: Path, *, role: str, broker_mode: str, config: dict[str, Any]
) -> Path:
    """Exclusive private startup artifact covering all source and deployed configs."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    files = sorted([*root.joinpath("src").rglob("*.py"), *root.joinpath("configs").glob("*.yaml")])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    lock = root / "requirements.lock"
    payload = {
        "schema_version": 1,
        "role": role,
        "pid": os.getpid(),
        "captured_at": datetime.now(UTC).isoformat(),
        "commit": commit,
        "executable": sys.executable,
        "python": platform.python_version(),
        "working_directory": str(root),
        "broker_mode": broker_mode,
        "effective_configuration": redact(config),
        "source_hashes": hashes,
        "source_manifest_hash": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "dependency_lock_hash": hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.exists()
        else None,
        "migration_version": None,
        "limitations": [
            "migration schema must be independently verified",
            "missing lock hash is unknown",
            "hashes identify source present at startup, not historical process imports",
        ],
    }
    directory = root / "data" / "runtime_manifests"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{role}_{os.getpid()}.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(payload, output, sort_keys=True, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    return path
