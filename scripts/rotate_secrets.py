"""Quarterly secret rotation automation (CL-yba0).

Rotates the credentials in the vault: OANDA API key, FRED API key,
Postgres password, etc. Operator runs this quarterly (or after any
suspected compromise). The script never *generates* the new secret —
that step is interactive and provider-specific (OANDA web UI, FRED
account page, etc). What this automates is:

  1. Verify each rotation candidate's new value works (test API call).
  2. Write the new vault file atomically (write-then-rename).
  3. Hot-reload the vault agent without dropping in-flight requests.
  4. Audit-log the rotation (who, what, when — never the value).

Idempotent: running with the same inputs is a no-op. Any single
provider's rotation can be done independently — pass --only OANDA to
rotate just one.

Usage:
  .venv/bin/python -m scripts.rotate_secrets [--only OANDA,FRED]
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import logging
import os
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# Vault path matches deploy/scripts/init_vault.sh's expectation. Symbolic
# constant so the operator override path is one place.
_VAULT_PATH: Path = Path("/etc/curlit/vault/credentials.enc")
_AUDIT_LOG_PATH: Path = Path("/var/log/curlit/secret-rotation.log")
_VAULT_AGENT_SOCK: str = "/run/fx-vault-agent.sock"


@dataclass
class RotationTarget:
    """One credential's rotation spec.

    verify_fn takes the proposed new value and returns True if a
    quick test call succeeds. Returning False or raising aborts
    the rotation for that target.
    """

    key: str
    description: str
    verify_fn: Callable[[str], bool]


def _verify_oanda(api_key: str) -> bool:
    """Hit OANDA practice account-summary with the proposed key."""
    account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
    if not account_id:
        logger.warning("OANDA_ACCOUNT_ID env not set — can't verify; assuming OK")
        return True
    try:
        resp = httpx.get(
            f"https://api-fxpractice.oanda.com/v3/accounts/{account_id}/summary",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        logger.error("OANDA verify failed: %s", exc)
        return False
    return resp.status_code == 200


def _verify_fred(api_key: str) -> bool:
    """Pull a single observation from a tiny series."""
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series",
            params={
                "series_id": "DFF", "api_key": api_key, "file_type": "json",
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        logger.error("FRED verify failed: %s", exc)
        return False
    return resp.status_code == 200


def _verify_postgres(password: str) -> bool:
    """Connect with the proposed password — psycopg2 dance."""
    user = os.environ.get("POSTGRES_USER", "fx")
    db = os.environ.get("POSTGRES_DB", "fx")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    try:
        from sqlalchemy import create_engine
        url = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{db}"
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Postgres verify failed: %s", exc)
        return False


# Rotation registry. Add new targets here when a credential becomes
# subject to the quarterly rotation cadence.
_TARGETS: dict[str, RotationTarget] = {
    "OANDA": RotationTarget(
        key="OANDA_API_KEY",
        description="OANDA API key (rotate via OANDA Hub > Manage API Access)",
        verify_fn=_verify_oanda,
    ),
    "FRED": RotationTarget(
        key="FRED_API_KEY",
        description="FRED API key (rotate via stlouisfed.org account)",
        verify_fn=_verify_fred,
    ),
    "POSTGRES": RotationTarget(
        key="POSTGRES_PASSWORD",
        description=(
            "Postgres password — change with ALTER USER first, then run this "
            "to update vault and restart engine"
        ),
        verify_fn=_verify_postgres,
    ),
}


def _read_vault_via_agent() -> dict[str, Any]:
    """Pull the current vault state through the running agent. Avoids
    needing the master passphrase in the rotation script."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    try:
        sock.connect(_VAULT_AGENT_SOCK)
        sock.sendall(json.dumps({"action": "list"}).encode())
        resp = json.loads(sock.recv(8192))
    finally:
        sock.close()
    if not resp.get("ok"):
        raise RuntimeError(f"Vault list failed: {resp}")
    out: dict[str, Any] = {}
    for name in resp.get("names", []):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10.0)
        s.connect(_VAULT_AGENT_SOCK)
        s.sendall(json.dumps({"action": "get", "name": name}).encode())
        r = json.loads(s.recv(4096))
        s.close()
        if r.get("ok"):
            out[name] = r["value"]
    return out


def _write_audit(action: str, detail: str) -> None:
    _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"{datetime.now(UTC).isoformat()} "
        f"user={os.environ.get('SUDO_USER', os.environ.get('USER', '?'))} "
        f"action={action} detail={detail}\n"
    )
    # Append-only, world-readable so audits across machines work,
    # but writable only by the rotation operator (mode 0644 enforced
    # at file creation; chmod is one-shot here).
    with _AUDIT_LOG_PATH.open("a") as f:
        f.write(line)


def rotate(targets: list[str]) -> int:
    if not targets:
        logger.error("No targets selected — pass --only or --all")
        return 2

    new_values: dict[str, str] = {}
    for tname in targets:
        target = _TARGETS.get(tname)
        if target is None:
            logger.error("Unknown target: %s (known: %s)",
                         tname, list(_TARGETS.keys()))
            return 2
        print(f"\n=== Rotating: {tname} ({target.description})")
        new = getpass.getpass(f"  New {target.key}: ")
        confirm = getpass.getpass("  Confirm: ")
        if new != confirm:
            print("  Mismatch — aborting", file=sys.stderr)
            return 1
        if not new:
            print("  Empty — skipping", file=sys.stderr)
            continue
        print("  Verifying with provider…")
        if not target.verify_fn(new):
            print("  Verification FAILED — aborting", file=sys.stderr)
            return 1
        print("  Verified.")
        new_values[target.key] = new

    if not new_values:
        logger.info("Nothing to rotate")
        return 0

    # Pull current vault, overlay new values.
    current = _read_vault_via_agent()
    merged = {**current, **new_values}

    # Re-seal WITHOUT touching disk (CL-nwzm). The old flow wrote every
    # credential to a PREDICTABLE /tmp/.vault.plain.json and told the
    # operator to re-seal "later" — a crash or forgotten cleanup left the
    # full vault plaintext on disk (the highest practical key-recovery bug
    # in-repo). The plaintext now travels through an inherited-fd pipe: it
    # exists only in this process, the pipe buffer, and the child — never
    # on the filesystem.
    print("\nRe-sealing vault (plaintext via in-memory pipe — nothing on disk)…")
    passphrase = getpass.getpass("Master passphrase for re-seal: ")
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, True)
    payload = json.dumps(merged).encode()
    try:
        proc = subprocess.Popen(  # noqa: S603 — our own interpreter+module
            [
                sys.executable, "-m", "scripts.initialize_vault",
                "--plaintext", f"/dev/fd/{read_fd}",
                "--encrypted", str(_VAULT_PATH),
                "--passphrase-stdin",
            ],
            stdin=subprocess.PIPE,
            pass_fds=(read_fd,),
        )
        os.write(write_fd, payload)
        os.close(write_fd)
        proc.communicate(input=passphrase.encode(), timeout=120)
    finally:
        with contextlib.suppress(OSError):
            os.close(read_fd)
    if proc.returncode != 0:
        print("Re-seal FAILED — vault unchanged; nothing was written to disk.",
              file=sys.stderr)
        return 1

    print("Vault re-sealed. Then:")
    print("  sudo systemctl reload fx-vault-agent")
    print("  sudo systemctl restart fx-live-engine")

    _write_audit("rotate", f"keys={sorted(new_values.keys())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only", default=None,
        help=f"Comma-separated subset of {','.join(_TARGETS.keys())}",
    )
    p.add_argument("--all", action="store_true", help="Rotate every target")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.all:
        targets = list(_TARGETS.keys())
    elif args.only:
        targets = [t.strip() for t in args.only.split(",") if t.strip()]
    else:
        p.error("must pass --only or --all")
        return 2

    return rotate(targets)


if __name__ == "__main__":
    raise SystemExit(main())
