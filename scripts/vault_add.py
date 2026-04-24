#!/usr/bin/env python3
"""Vault credential management — add, remove, list."""

import getpass
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path


def derive_key(passphrase: str, salt: bytes, iterations: int = 600_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, iterations, 32)


def decrypt_vault(path: Path, key: bytes) -> dict:
    data = json.loads(path.read_text())
    nonce = bytes.fromhex(data["nonce"])
    ct = bytes.fromhex(data["ciphertext"])
    tag = bytes.fromhex(data.get("tag", ""))
    try:
        from wolfcrypt.ciphers import Aes, MODE_GCM
        aes = Aes(key, MODE_GCM, nonce)
        return json.loads(aes.decrypt(ct, tag))
    except ImportError:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        return json.loads(aesgcm.decrypt(nonce, ct, None))


def encrypt_vault(data: bytes, key: bytes) -> dict:
    nonce = secrets.token_bytes(12)
    try:
        from wolfcrypt.ciphers import Aes, MODE_GCM
        aes = Aes(key, MODE_GCM, nonce)
        ct, tag = aes.encrypt(data)
    except ImportError:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, data, None)
        tag = b""
    return {"version": 1, "nonce": nonce.hex(), "ciphertext": ct.hex(), "tag": tag.hex()}


def open_vault() -> tuple[dict, bytes, Path]:
    vault_path = Path(os.environ.get("VAULT_PATH", "vault.enc"))
    salt_path = Path(os.environ.get("VAULT_SALT", "vault.salt"))
    passphrase = getpass.getpass("Vault passphrase: ")
    salt = salt_path.read_bytes()
    key = derive_key(passphrase, salt)
    vault = decrypt_vault(vault_path, key)
    return vault, key, vault_path


def cmd_add() -> None:
    vault, key, vault_path = open_vault()
    name = input("Credential name: ").strip()
    if not name or not name.replace("_", "").isalnum():
        print("Invalid name. Use alphanumeric + underscore only.")
        sys.exit(1)
    value = getpass.getpass("Credential value: ")
    vault[name] = value
    encrypted = encrypt_vault(json.dumps(vault).encode(), key)
    vault_path.write_text(json.dumps(encrypted))
    print(f"Added: {name}")


def cmd_remove() -> None:
    vault, key, vault_path = open_vault()
    name = input("Credential name to remove: ").strip()
    if name not in vault:
        print(f"Not found: {name}")
        sys.exit(1)
    del vault[name]
    encrypted = encrypt_vault(json.dumps(vault).encode(), key)
    vault_path.write_text(json.dumps(encrypted))
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
