#!/usr/bin/env python3
"""Vault initialization — diceware passphrase + BIP39 recovery seed, one-time
setup — plus a non-interactive RE-SEAL mode (CL-8lv6) used by
deploy/scripts/init_vault.sh and scripts/rotate_secrets.py::

    python -m scripts.initialize_vault \\
        --plaintext /dev/fd/N --encrypted vault.enc [--salt vault.salt] \\
        --passphrase-stdin

The plaintext is read with plain open()+read() so pipe paths (/dev/fd/N)
work — both callers feed the credentials JSON through an in-memory pipe so
it never touches the filesystem. The encrypted file is written atomically
(tmp + os.replace, mode 0600). Creating a NEW vault enforces the
strong-passphrase policy; overwriting an EXISTING one (rotation re-seal)
only warns on a weak passphrase, so the legacy weak-passphrase vault can
still rotate its own secrets.
"""

import argparse
import contextlib
import getpass
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
from src.security.vault_codec import (  # noqa: E402
    derive_key,
    passphrase_weakness,
    require_strong_passphrase,
)
from src.security.vault_codec import seal as encrypt  # noqa: E402


def _atomic_write(path: Path, payload: bytes) -> None:
    """tmp + os.replace in the same directory, mode 0600 before the rename —
    a crash leaves either the old file or nothing, never a torn write."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def reseal(args: argparse.Namespace) -> int:
    """Non-interactive re-seal (CL-8lv6): plaintext JSON in, sealed vault out."""
    encrypted_path = Path(args.encrypted)
    salt_path = (
        Path(args.salt) if args.salt else encrypted_path.with_name("vault.salt")
    )

    # open()+read(), no stat/seek games — the path is often a pipe like
    # /dev/fd/N (rotate_secrets and init_vault.sh both feed the plaintext
    # that way so it never exists on the filesystem).
    with open(args.plaintext, "rb") as f:
        plaintext = f.read()
    try:
        parsed = json.loads(plaintext)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"--plaintext is not valid JSON ({exc}) — refusing to seal it; "
              "the vault agent could never open the result", file=sys.stderr)
        return 2
    if not isinstance(parsed, dict):
        print("--plaintext must be a JSON object (the credentials mapping)",
              file=sys.stderr)
        return 2

    if args.passphrase_stdin:
        passphrase = sys.stdin.buffer.read().decode("utf-8")
        # Strip exactly ONE trailing newline: shells almost always append one
        # (`printf '%s\n' | ...`), but a newline INSIDE a passphrase is legal.
        passphrase = passphrase.removesuffix("\n")
    else:
        passphrase = getpass.getpass("Vault passphrase: ")
    if not passphrase:
        print("Empty passphrase — aborting", file=sys.stderr)
        return 2

    if not encrypted_path.exists():
        # NEW vault: strong-passphrase policy enforced (CL-qyav), and it runs
        # BEFORE anything touches disk so a rejection writes nothing.
        try:
            require_strong_passphrase(passphrase)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        # Re-seal of an EXISTING vault (rotation): warn-not-fail, otherwise a
        # vault sealed under the known-weak legacy passphrase could never
        # rotate its own secrets (that flow re-seals under the SAME phrase).
        weakness = passphrase_weakness(passphrase)
        if weakness is not None:
            print(f"WARNING: vault passphrase is weak ({weakness}) — rotate "
                  "it: scripts/rotate_secrets.py --rotate-passphrase",
                  file=sys.stderr)

    if salt_path.exists():
        salt = salt_path.read_bytes()
        if len(salt) < 8:
            print(f"Corrupt salt file {salt_path} ({len(salt)} bytes) — "
                  "refusing to derive a key from it", file=sys.stderr)
            return 2
    else:
        salt = secrets.token_bytes(16)
        _atomic_write(salt_path, salt)

    sealed = encrypt(plaintext, derive_key(passphrase, salt))
    _atomic_write(encrypted_path, json.dumps(sealed).encode())
    print(f"Sealed {len(plaintext)} plaintext bytes -> {encrypted_path}")
    return 0


def interactive_init() -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="initialize_vault",
        description="Vault setup. No flags: interactive one-time init "
                    "(generates passphrase + recovery seed). With --plaintext/"
                    "--encrypted: non-interactive re-seal (CL-8lv6).",
    )
    parser.add_argument(
        "--plaintext", metavar="PATH",
        help="Read plaintext credentials JSON from PATH (pipe paths like "
             "/dev/fd/N supported — preferred, so secrets never hit disk)")
    parser.add_argument(
        "--encrypted", metavar="PATH",
        help="Write the sealed vault to PATH (atomic tmp+rename, mode 0600)")
    parser.add_argument(
        "--salt", metavar="PATH", default=None,
        help="Salt file (default: vault.salt alongside --encrypted). "
             "Reused if it exists, else 16 random bytes are written 0600")
    parser.add_argument(
        "--passphrase-stdin", action="store_true",
        help="Read the passphrase from stdin (one trailing newline stripped)")
    args = parser.parse_args(argv)

    if args.plaintext or args.encrypted:
        if not (args.plaintext and args.encrypted):
            parser.error("re-seal mode needs BOTH --plaintext and --encrypted")
        return reseal(args)
    if args.salt or args.passphrase_stdin:
        parser.error("--salt/--passphrase-stdin require --plaintext/--encrypted")

    interactive_init()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
