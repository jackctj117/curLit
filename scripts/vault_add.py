#!/usr/bin/env python3
"""Vault credential management — add, remove, list.

Seal/unseal and key derivation go through src.security.vault_codec ONLY
(CL-8lv6 P1: this script used to reimplement both, which is exactly the
schema-drift class of bug CL-ujm6 fixed elsewhere), and every vault write
is atomic (tmp + os.replace, mode 0600) so a crash mid-write can never
leave a torn vault.enc. CLI behavior is unchanged.
"""

import contextlib
import getpass
import json
import os
import sys
from pathlib import Path

# Keep the documented `python scripts/vault_add.py <cmd>` invocation working:
# running by file path puts scripts/ (not the repo root) on sys.path, and the
# vault_codec import below needs the root.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.security.vault_codec import derive_key, seal, unseal  # noqa: E402


def _write_vault_atomic(vault_path: Path, vault: dict, key: bytes) -> None:
    """Seal via vault_codec and replace the vault file atomically (0600)."""
    payload = json.dumps(seal(json.dumps(vault).encode(), key)).encode()
    tmp = vault_path.with_name(f"{vault_path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, vault_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def open_vault() -> tuple[dict, bytes, Path]:
    vault_path = Path(os.environ.get("VAULT_PATH", "vault.enc"))
    salt_path = Path(os.environ.get("VAULT_SALT", "vault.salt"))
    passphrase = getpass.getpass("Vault passphrase: ")
    salt = salt_path.read_bytes()
    key = derive_key(passphrase, salt)
    vault = json.loads(unseal(json.loads(vault_path.read_text()), key))
    return vault, key, vault_path


def cmd_add() -> None:
    vault, key, vault_path = open_vault()
    name = input("Credential name: ").strip()
    if not name or not name.replace("_", "").isalnum():
        print("Invalid name. Use alphanumeric + underscore only.")
        sys.exit(1)
    value = getpass.getpass("Credential value: ")
    vault[name] = value
    _write_vault_atomic(vault_path, vault, key)
    print(f"Added: {name}")


def cmd_remove() -> None:
    vault, key, vault_path = open_vault()
    name = input("Credential name to remove: ").strip()
    if name not in vault:
        print(f"Not found: {name}")
        sys.exit(1)
    del vault[name]
    _write_vault_atomic(vault_path, vault, key)
    print(f"Removed: {name}")


def cmd_list() -> None:
    vault, _, _ = open_vault()
    if not vault:
        print("No credentials stored.")
    else:
        for name in sorted(vault):
            print(f"  {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("add", "remove", "list"):
        print("Usage: python scripts/vault_add.py <add|remove|list>")
    elif sys.argv[1] == "add":
        cmd_add()
    elif sys.argv[1] == "remove":
        cmd_remove()
    else:
        cmd_list()
