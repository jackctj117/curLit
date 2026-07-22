#!/usr/bin/env python3
"""Vault initialization — diceware passphrase + BIP39 recovery seed, one-time setup."""

import hashlib
import json
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path


def load_bip39_words() -> list[str]:
    """The canonical 2048-word BIP39 English list vendored at
    scripts/bip39_english.txt (sha256 2f5eed53…dbda)."""
    words = (
        Path(__file__).parent / "bip39_english.txt"
    ).read_text().strip().split("\n")[:2048]
    if len(words) != 2048:
        raise RuntimeError(
            f"bip39_english.txt has {len(words)} words, expected 2048 — "
            "refusing to generate weak secrets from a truncated list")
    return words


def generate_passphrase() -> str:
    """Diceware over the full BIP39 list: 6 × log2(2048) ≈ 66 bits.

    (The old inline 35-word list gave ~31 bits — the live vault's known
    weakness, CL-qyav; vault_codec now enforces ≥60 bits on new seals.)"""
    words = load_bip39_words()
    return " ".join(secrets.choice(words) for _ in range(6))


def generate_recovery_seed() -> tuple[bytes, list[str]]:
    from hashlib import sha256
    entropy = secrets.token_bytes(32)
    checksum = sha256(entropy).digest()[0]
    full = entropy + bytes([checksum])
    bit_string = "".join(f"{b:08b}" for b in full)[:264]
    bip39 = load_bip39_words()
    indices = [int(bit_string[i:i+11], 2) for i in range(0, 264, 11)]
    words = [bip39[i] for i in indices[:24]]
    return entropy, words


# Format + key derivation live in ONE module (CL-ujm6) — this script used to
# write {"ct": ...} while vault_agent read data["ciphertext"], so vaults it
# created could never be opened. Both sides now share vault_codec.
from src.security.vault_codec import derive_key, require_strong_passphrase  # noqa: E402
from src.security.vault_codec import seal as encrypt  # noqa: E402


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
    require_strong_passphrase(passphrase)  # belt-and-suspenders (CL-qyav)
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
