"""SSH key backup and restore — deterministic derivation from BIP39 or hex+QR export."""

import hashlib
import hmac
import os
import sys
from pathlib import Path


def derive_ssh_key_from_mnemonic(mnemonic: str, purpose: str = "ssh-fx-server") -> tuple[bytes, bytes]:
    """Derive Ed25519 SSH key deterministically from a BIP39 mnemonic."""
    prefix = "ssh-key-v1:"
    seed = hashlib.pbkdf2_hmac("sha512", f"{prefix}{mnemonic}".encode(), b"mnemonic", 2048, 64)
    key_bytes = hmac.new(seed, purpose.encode(), hashlib.sha256).digest()[:32]
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
        priv = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        return priv, pub
    except ImportError:
        print("cryptography package required: pip install cryptography")
        sys.exit(1)


def export_key_hex(key_path: Path) -> str:
    """Read an Ed25519 private key and return hex representation."""
    with open(key_path, "rb") as f:
        key_data = f.read()
    return key_data.hex()


def restore_key_hex(hex_data: str, output_path: Path) -> None:
    """Restore an SSH private key from hex."""
    output_path.write_bytes(bytes.fromhex(hex_data))
    os.chmod(output_path, 0o600)


if __name__ == "__main__":
    print("Usage:")
    print("  python scripts/backup_ssh_key.py --derive '<mnemonic>'    # derive from mnemonic")
    print("  python scripts/backup_ssh_key.py --export ~/.ssh/id_ed25519  # hex export")
    print("  python scripts/restore_ssh_key.py <hex> <output_path>   # restore from hex")
