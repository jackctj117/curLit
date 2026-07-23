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

import contextlib
import hashlib
import math
import os
import secrets
from pathlib import Path
from typing import Any

# 600k PBKDF2-HMAC-SHA256 iterations meets the current (OWASP) floor for
# this KDF — reviewed under CL-qyav, no bump needed. If this is ever raised,
# raise it for NEW seals only; unseal of existing vaults must keep working
# (KDF params are pinned per-vault by the derive_key call site).
PBKDF2_ITERATIONS = 600_000
_TAG_LEN = 16

# --- Passphrase strength policy (CL-qyav P2) -------------------------------
#
# Enforced for NEW vault creation / passphrase changes via
# ``require_strong_passphrase``. Unseal of an EXISTING vault must never fail
# on a weak passphrase — callers on that path use ``passphrase_weakness`` and
# log a rotation warning instead.

MIN_PASSPHRASE_CHARS = 12
MIN_PASSPHRASE_BITS = 60.0

# Mirror of the legacy 35-word generator list in
# scripts/initialize_vault.generate_passphrase (duplicated here because that
# script imports this module — importing back would be circular). A phrase
# built solely from these words has only log2(35) ~= 5.13 bits/word against
# an attacker who has read the repo, regardless of its character length.
_LEGACY_WORDLIST = frozenset({
    "abacus", "balance", "cactus", "dagger", "eagle", "fabric", "galaxy",
    "habitat", "iceberg", "jungle", "kayak", "lantern", "magnet", "nebula",
    "octopus", "paddle", "quantum", "raccoon", "saddle", "tackle", "umbrella",
    "vapor", "walnut", "xenon", "yacht", "zebra", "anchor", "blizzard",
    "captain", "diamond", "emerald", "falcon", "garden", "horizon", "island",
})


def estimate_passphrase_bits(passphrase: str) -> float:
    """Crude entropy estimate: charset-size ** length, in bits.

    Deliberately simple (no new deps). One refinement: a phrase made only of
    words from the known legacy 35-word generator list is scored per-word
    (n * log2(35)) — the attacker knows that list, so character length is
    irrelevant. This is what makes the ~31-bit legacy passphrase (6 words)
    actually register as weak instead of looking like a 200-bit string.
    """
    if not passphrase:
        return 0.0
    words = passphrase.split()
    if len(words) > 1 and all(w in _LEGACY_WORDLIST for w in words):
        return len(words) * math.log2(len(_LEGACY_WORDLIST))
    charset = 0
    if any(c.islower() for c in passphrase):
        charset += 26
    if any(c.isupper() for c in passphrase):
        charset += 26
    if any(c.isdigit() for c in passphrase):
        charset += 10
    if any(not c.isalnum() for c in passphrase):
        charset += 33  # printable specials incl. space
    return len(passphrase) * math.log2(charset) if charset else 0.0


def passphrase_weakness(passphrase: str) -> str | None:
    """Human-readable reason the passphrase fails policy, or None if it passes."""
    if len(passphrase) < MIN_PASSPHRASE_CHARS:
        return (
            f"only {len(passphrase)} characters "
            f"(minimum {MIN_PASSPHRASE_CHARS})"
        )
    bits = estimate_passphrase_bits(passphrase)
    if bits < MIN_PASSPHRASE_BITS:
        return (
            f"estimated entropy ~{bits:.0f} bits "
            f"(minimum {MIN_PASSPHRASE_BITS:.0f})"
        )
    return None


def require_strong_passphrase(passphrase: str) -> None:
    """Raise ValueError if the passphrase fails policy.

    Call this on every NEW vault creation or passphrase change. Do NOT call
    it on the unseal path — existing vaults must keep opening.
    """
    reason = passphrase_weakness(passphrase)
    if reason is not None:
        msg = (
            f"vault passphrase too weak: {reason}. "
            "Use >=12 characters mixing character classes, or >=6 words from "
            "a large (2048+) wordlist such as scripts/bip39_english.txt."
        )
        raise ValueError(msg)


def derive_key(passphrase: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> bytes:
    # Salt-agnostic by design: the caller owns the salt, so a per-vault RANDOM
    # salt is fully supported here with no change (CL-8s2a recovery-KDF note).
    #
    # Two derivation call sites exist in the repo:
    #   * the primary vault key — already uses a per-vault random 16-byte salt
    #     persisted as vault.salt (scripts/initialize_vault.py, vault_add.py,
    #     rotate_secrets.py). No fixed salt.
    #   * the RECOVERY key — scripts/initialize_vault.py wraps the vault key
    #     under derive_key(entropy.hex(), b"fx-recovery-v1"), a FIXED public
    #     salt. entropy is 256 fresh random bits, so a rainbow table is already
    #     infeasible and the fixed salt is only weak defense-in-depth.
    #
    # A per-vault random recovery salt is safe to add for NEW vaults (write a
    # recovery.salt companion the manual recovery procedure reads), but MUST
    # NOT be retrofitted to an EXISTING vault: its printed recovery document +
    # recovery.enc are pinned to b"fx-recovery-v1", and re-deriving under a new
    # salt would make that already-printed document undecryptable — breaking
    # disaster recovery for the live vault (backward compat is mandatory). That
    # call-site change lives in scripts/initialize_vault.py (out of this lane);
    # this KDF needs nothing changed to support it.
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, iterations, 32)


def atomic_write_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    """The ONE atomic durable-write helper for vault files (CL-9dhg).

    Same-directory tmp file, ``flush`` + ``os.fsync`` so the data is on disk
    BEFORE the rename, chmod (default 0600) before the rename so there is no
    umask-readable window, then ``os.replace``. A crash or power loss at any
    point leaves either the old file or nothing — never a torn or truncated
    write. Scripts must use this instead of growing private copies: the last
    drifted copy omitted the fsync, so power loss after the rename could
    surface a truncated vault.enc.
    """
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


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
