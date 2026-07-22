"""Vault sealing codec (CL-ujm6) — the ONE module that owns the format.

Fixes two review findings at their shared root:

* **Schema split (P0)**: ``initialize_vault`` wrote ``{"ct": ...}`` while
  ``vault_agent`` read ``data["ciphertext"]`` — every unseal of an
  init-created vault KeyError'd, so the deploy seal path was non-functional
  and operators fell back to plaintext env secrets.
* **Dual-backend tag layout (P2)**: wolfCrypt returns ``(ct, tag)`` separately
  while ``cryptography``'s AESGCM appends the tag to the ciphertext — a vault
  written by one backend could not be opened by the other.

Canonical format (v2), backend-independent::

    {"v": 2, "nonce": <hex 12B>, "ciphertext": <hex>, "tag": <hex 16B>}

``seal`` always writes v2 with the tag SPLIT OUT. ``unseal`` accepts v2, the
legacy v1 ``"ct"`` schema, and both tag layouts (separate tag, or tag appended
to the ciphertext with an empty ``tag`` field) with either backend — so every
historical vault on disk opens, and everything written from now on is
portable. Key derivation is pinned here too (PBKDF2-HMAC-SHA256, 600k
iterations, 32 bytes) so the two sides can never drift again.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

PBKDF2_ITERATIONS = 600_000
_TAG_LEN = 16


def derive_key(passphrase: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, iterations, 32)


def seal(plaintext: bytes, key: bytes) -> dict[str, Any]:
    """Encrypt → canonical v2 dict. Prefers wolfCrypt (repo standard),
    falls back to ``cryptography``; either way the tag is stored SEPARATELY
    so any backend can unseal."""
    nonce = secrets.token_bytes(12)
    try:
        from wolfcrypt.ciphers import MODE_GCM, Aes  # noqa: PLC0415
        aes = Aes(key, MODE_GCM, nonce)
        ct, tag = aes.encrypt(plaintext)
    except ImportError:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415
        blob = AESGCM(key).encrypt(nonce, plaintext, None)
        ct, tag = blob[:-_TAG_LEN], blob[-_TAG_LEN:]
    return {
        "v": 2,
        "nonce": nonce.hex(),
        "ciphertext": ct.hex(),
        "tag": tag.hex(),
    }


def unseal(data: dict[str, Any], key: bytes) -> bytes:
    """Decrypt any historical vault dict → plaintext bytes.

    Accepts: v2 canonical; legacy v1 with ``"ct"``; tag either split out or
    appended to the ciphertext (empty ``tag``). Raises ``ValueError`` on a
    dict that carries neither ciphertext key, and whatever the backend raises
    on authentication failure (fail loud — a tampered vault must not
    half-open).
    """
    raw_ct = data.get("ciphertext", data.get("ct"))
    if raw_ct is None:
        msg = "vault dict has neither 'ciphertext' nor legacy 'ct'"
        raise ValueError(msg)
    nonce = bytes.fromhex(data["nonce"])
    ct = bytes.fromhex(raw_ct)
    tag = bytes.fromhex(data.get("tag") or "")
    if not tag and len(ct) > _TAG_LEN:
        # cryptography-written legacy blob: tag is the last 16 bytes.
        ct, tag = ct[:-_TAG_LEN], ct[-_TAG_LEN:]

    try:
        from wolfcrypt.ciphers import MODE_GCM, Aes  # noqa: PLC0415
        aes = Aes(key, MODE_GCM, nonce)
        return bytes(aes.decrypt(ct, tag))
    except ImportError:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415
        return AESGCM(key).decrypt(nonce, ct + tag, None)
