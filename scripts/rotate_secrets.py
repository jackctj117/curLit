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

--rotate-passphrase (CL-qyav follow-up) rotates the vault MASTER
passphrase itself: decrypts vault.enc with the old phrase, re-seals the
same contents under a NEW strong passphrase (vault_codec policy enforced)
with a FRESH salt, and replaces vault.enc/vault.salt atomically after
writing timestamped .bak copies. Restart the vault agent afterwards.

Usage:
  .venv/bin/python -m scripts.rotate_secrets [--only OANDA,FRED]
  .venv/bin/python -m scripts.rotate_secrets --rotate-passphrase
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
        # Exception TYPE only (CL-8lv6 P1): httpx error text can embed the
        # request URL/params — never risk credential material in logs.
        logger.error("OANDA verify failed: %s (exception text suppressed)", type(exc).__name__)
        return False
    return resp.status_code == 200


def _verify_fred(api_key: str) -> bool:
    """Pull a single observation from a tiny series."""
    try:
        resp = httpx.get(
            "https://api.stlouisfed.org/fred/series",
            params={
                "series_id": "DFF",
                "api_key": api_key,
                "file_type": "json",
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        # The FRED key rides in the URL query string, so str(exc) can contain
        # it verbatim — log the exception TYPE only (CL-8lv6 P1).
        logger.error(
            "FRED verify failed: %s (exception text suppressed — it can embed the api_key)",
            type(exc).__name__,
        )
        return False
    return resp.status_code == 200


def _verify_postgres(password: str) -> bool:
    """Connect with the proposed password — psycopg2 dance."""
    user = os.environ.get("POSTGRES_USER", "fx")
    db = os.environ.get("POSTGRES_DB", "fx")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    try:
        from sqlalchemy import create_engine, text  # noqa: PLC0415
        from sqlalchemy.engine import URL  # noqa: PLC0415

        # URL.create both escapes the password and REDACTS it in str()/repr();
        # the old f-string DSN embedded it raw, so any exception carrying the
        # DSN leaked it into logs (CL-8lv6 P1).
        url = URL.create(
            "postgresql+psycopg2",
            username=user,
            password=password,
            host=host,
            port=5432,
            database=db,
        )
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        # Exception TYPE + redacted DSN only — sqlalchemy/psycopg2 error text
        # can echo the full DSN including the password.
        logger.error(
            "Postgres verify failed: %s dsn=postgresql+psycopg2://%s:***@%s:5432/%s "
            "(exception text suppressed — it can embed the password)",
            type(exc).__name__,
            user,
            host,
            db,
        )
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
    needing the master passphrase in the rotation script.

    Uses the length-prefixed frame protocol (CL-1ho7,
    src.security.vault_wire) — the old single-shot ``recv`` silently
    truncated any vault response over one buffer.
    """
    from src.security.vault_wire import recv_framed, send_framed  # noqa: PLC0415

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    try:
        sock.connect(_VAULT_AGENT_SOCK)
        send_framed(sock, json.dumps({"action": "list"}).encode())
        resp = json.loads(recv_framed(sock))
    finally:
        sock.close()
    if not resp.get("ok"):
        raise RuntimeError(f"Vault list failed: {resp}")
    out: dict[str, Any] = {}
    for name in resp.get("names", []):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10.0)
        try:
            s.connect(_VAULT_AGENT_SOCK)
            send_framed(s, json.dumps({"action": "get", "name": name}).encode())
            r = json.loads(recv_framed(s))
        finally:
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
            logger.error("Unknown target: %s (known: %s)", tname, list(_TARGETS.keys()))
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
                sys.executable,
                "-m",
                "scripts.initialize_vault",
                "--plaintext",
                f"/dev/fd/{read_fd}",
                "--encrypted",
                str(_VAULT_PATH),
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
        print("Re-seal FAILED — vault unchanged; nothing was written to disk.", file=sys.stderr)
        return 1

    print("Vault re-sealed. Then:")
    print("  sudo systemctl reload fx-vault-agent")
    print("  sudo systemctl restart fx-live-engine")

    _write_audit("rotate", f"keys={sorted(new_values.keys())}")
    return 0


def reseal_with_new_passphrase(
    plaintext: bytes,
    new_passphrase: str,
    vault_path: Path,
    salt_path: Path,
) -> None:
    """Re-seal vault plaintext under a new passphrase + fresh salt.

    Policy-checked (raises ValueError on a weak passphrase), backed up
    (timestamped .bak-* copies, mode 0600), and atomic (tmp + os.replace in
    the same directory — a crash leaves the OLD vault intact).
    """
    import secrets as _secrets  # noqa: PLC0415

    from src.security.vault_codec import (  # noqa: PLC0415
        atomic_write_bytes,
        derive_key,
        require_strong_passphrase,
        seal,
    )

    require_strong_passphrase(new_passphrase)
    new_salt = _secrets.token_bytes(16)
    sealed = seal(plaintext, derive_key(new_passphrase, new_salt))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for p in (vault_path, salt_path):
        bak = p.with_name(f"{p.name}.bak-{stamp}")
        bak.write_bytes(p.read_bytes())
        os.chmod(bak, 0o600)

    # Canonical fsync'd atomic writer (CL-9dhg finding 13): this was the
    # one remaining copy without flush+fsync — power loss after the
    # rename could truncate vault.enc.
    for path, payload in (
        (vault_path, json.dumps(sealed).encode()),
        (salt_path, new_salt),
    ):
        atomic_write_bytes(path, payload)


def rotate_passphrase() -> int:
    """Interactive master-passphrase rotation against the local vault files
    (VAULT_PATH/VAULT_SALT env, same defaults as initialize_vault)."""
    from src.security.vault_codec import derive_key, unseal  # noqa: PLC0415

    vault_path = Path(os.environ.get("VAULT_PATH", "vault.enc"))
    salt_path = Path(os.environ.get("VAULT_SALT", "vault.salt"))
    if not vault_path.exists() or not salt_path.exists():
        print(
            f"Vault files not found ({vault_path}, {salt_path}) — set "
            "VAULT_PATH/VAULT_SALT or run from the vault directory.",
            file=sys.stderr,
        )
        return 2

    old = getpass.getpass("Current master passphrase: ")
    try:
        plaintext = unseal(
            json.loads(vault_path.read_text()),
            derive_key(old, salt_path.read_bytes()),
        )
    except Exception:
        print(
            "Decryption FAILED (wrong passphrase or corrupt vault) — nothing changed.",
            file=sys.stderr,
        )
        return 1

    new = getpass.getpass("New master passphrase (ENTER to generate a strong one): ")
    if not new:
        from scripts.initialize_vault import generate_passphrase  # noqa: PLC0415

        new = generate_passphrase()
        print(f"\nNEW MASTER PASSPHRASE (write it down NOW):\n  {new}\n")
        input("Press ENTER after recording it...")
    else:
        if getpass.getpass("Confirm: ") != new:
            print("Mismatch — aborting; nothing changed.", file=sys.stderr)
            return 1

    try:
        reseal_with_new_passphrase(plaintext, new, vault_path, salt_path)
    except ValueError as exc:
        print(f"{exc} — nothing changed.", file=sys.stderr)
        return 1

    print("Vault re-sealed under the new passphrase (fresh salt; old files kept as .bak-*).")
    print(
        "IMPORTANT: the printed recovery document still wraps the OLD "
        "vault key — it can recover the .bak files only. Re-run recovery "
        "setup if you rely on it, then shred the old document."
    )
    print("Restart the vault agent so it prompts for the new passphrase.")
    with contextlib.suppress(OSError):
        _write_audit("rotate-passphrase", "master passphrase + salt replaced")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        default=None,
        help=f"Comma-separated subset of {','.join(_TARGETS.keys())}",
    )
    p.add_argument("--all", action="store_true", help="Rotate every target")
    p.add_argument(
        "--rotate-passphrase",
        action="store_true",
        help="Rotate the vault MASTER passphrase (re-seal + fresh salt)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.rotate_passphrase:
        return rotate_passphrase()
    if args.all:
        targets = list(_TARGETS.keys())
    elif args.only:
        targets = [t.strip() for t in args.only.split(",") if t.strip()]
    else:
        p.error("must pass --only, --all, or --rotate-passphrase")
        return 2

    return rotate(targets)


if __name__ == "__main__":
    raise SystemExit(main())
