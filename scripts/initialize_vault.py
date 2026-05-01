#!/usr/bin/env python3
"""Vault initialization — diceware passphrase + BIP39 recovery seed, one-time setup."""

import hashlib
import json
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path


def generate_passphrase() -> str:
    words = [
        "abacus", "balance", "cactus", "dagger", "eagle", "fabric", "galaxy",
        "habitat", "iceberg", "jungle", "kayak", "lantern", "magnet", "nebula",
        "octopus", "paddle", "quantum", "raccoon", "saddle", "tackle", "umbrella",
        "vapor", "walnut", "xenon", "yacht", "zebra", "anchor", "blizzard",
        "captain", "diamond", "emerald", "falcon", "garden", "horizon", "island",
    ]
    return " ".join(secrets.choice(words) for _ in range(6))


def generate_recovery_seed() -> tuple[bytes, list[str]]:
    from hashlib import sha256
    entropy = secrets.token_bytes(32)
    checksum = sha256(entropy).digest()[0]
    full = entropy + bytes([checksum])
    bit_string = "".join(f"{b:08b}" for b in full)[:264]
    # BIP39 wordlist — load from file or use baked-in subset
    bip39 = (Path(__file__).parent / "bip39_english.txt").read_text().strip().split("\n")[:2048]
    indices = [int(bit_string[i:i+11], 2) for i in range(0, 264, 11)]
    words = [bip39[i] for i in indices[:24]]
    return entropy, words


def derive_key(passphrase: str, salt: bytes, iterations: int = 600000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, iterations, 32)


def encrypt(data: bytes, key: bytes) -> dict:
    nonce = secrets.token_bytes(12)
    try:
        from wolfcrypt.ciphers import MODE_GCM, Aes
        aes = Aes(key, MODE_GCM, nonce)
        ct, tag = aes.encrypt(data)
        return {"v": 1, "nonce": nonce.hex(), "ct": ct.hex(), "tag": tag.hex()}
    except ImportError:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        ct = AESGCM(key).encrypt(nonce, data, None)
        return {"v": 1, "nonce": nonce.hex(), "ct": ct.hex(), "tag": ""}


def main() -> None:
    vault_path = Path(os.environ.get("VAULT_PATH", "vault.enc"))
    salt_path = Path(os.environ.get("VAULT_SALT", "vault.salt"))
    recovery_path = Path(os.environ.get("RECOVERY_PATH", "recovery.enc"))

    if vault_path.exists():
        print("Vault already exists. Delete vault.enc to regenerate.")
        sys.exit(1)

    print("=== curLit Vault Initialization ===\n")
    input("Press ENTER when ready...")

    passphrase = generate_passphrase()
    entropy, seed_words = generate_recovery_seed()
    salt = secrets.token_bytes(16)
    vault_key = derive_key(passphrase, salt)
    recovery_key = derive_key(entropy.hex(), b"fx-recovery-v1")

    vault_data = encrypt(b"{}", vault_key)
    recovery_data = encrypt(vault_key, recovery_key)

    vault_path.write_text(json.dumps(vault_data))
    salt_path.write_bytes(salt)
    recovery_path.write_text(json.dumps(recovery_data))
    for p in [vault_path, salt_path, recovery_path]:
        os.chmod(p, 0o600)

    checksum = hashlib.sha256(entropy).hexdigest()[:8]
    print("\n" + "=" * 56)
    print("FX SYSTEM -- RECOVERY DOCUMENT -- DO NOT LOSE")
    print(f"Generated: {datetime.now(UTC).isoformat()[:10]}")
    print("=" * 56)
    print(f"\nMASTER PASSPHRASE:\n  {passphrase}")
    print("\nRECOVERY SEED (24 words):")
    for i, w in enumerate(seed_words, 1):
        if i % 4 == 1:
            print("\n  ", end="")
        print(f"{i:2d}. {w:<12}", end="")
    print(f"\n\nVERIFICATION CHECKSUM: {checksum}")
    print("\n" + "=" * 56)
    print("PRINT THIS NOW. Store in fireproof safe.")
    print("Do not photograph. Do not store digitally.")
    print("=" * 56)
    input("\nPress ENTER after printing...")
    os.system("clear" if os.name == "posix" else "cls")
    print(f"Vault initialized: {vault_path}")
    print("Add credentials: python scripts/vault_add.py add")


if __name__ == "__main__":
    main()
