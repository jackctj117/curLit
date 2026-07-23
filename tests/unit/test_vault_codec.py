"""Vault codec round-trip regressions (CL-ujm6)."""

from __future__ import annotations

import pytest

from src.security.vault_codec import derive_key, seal, unseal

KEY = derive_key("test-passphrase", b"salt-16-bytes---", iterations=1000)


def test_seal_unseal_roundtrip():
    payload = b'{"OANDA_API_KEY": "abc123"}'
    sealed = seal(payload, KEY)
    assert sealed["v"] == 2
    assert "ciphertext" in sealed and sealed["tag"]  # canonical schema
    assert unseal(sealed, KEY) == payload


def test_unseal_accepts_legacy_ct_schema():
    sealed = seal(b"legacy", KEY)
    legacy = {
        "v": 1,
        "nonce": sealed["nonce"],
        "ct": sealed["ciphertext"],
        "tag": sealed["tag"],
    }  # the initialize_vault v1 shape
    assert unseal(legacy, KEY) == b"legacy"


def test_unseal_accepts_appended_tag_layout():
    sealed = seal(b"appended", KEY)
    merged = {
        "v": 1,
        "nonce": sealed["nonce"],
        "ciphertext": sealed["ciphertext"] + sealed["tag"],
        "tag": "",
    }
    assert unseal(merged, KEY) == b"appended"


def test_unseal_rejects_schema_garbage():
    with pytest.raises(ValueError, match="neither"):
        unseal({"v": 9, "nonce": "00" * 12}, KEY)


def test_unseal_fails_loud_on_tamper():
    sealed = seal(b"secret", KEY)
    tampered = dict(sealed)
    ct = bytearray(bytes.fromhex(tampered["ciphertext"]))
    if ct:
        ct[0] ^= 0xFF
    tampered["ciphertext"] = bytes(ct).hex()
    with pytest.raises(Exception):  # noqa: B017 — backend-specific auth error
        unseal(tampered, KEY)


def test_wrong_key_fails():
    sealed = seal(b"secret", KEY)
    other = derive_key("other", b"salt-16-bytes---", iterations=1000)
    with pytest.raises(Exception):  # noqa: B017
        unseal(sealed, other)


# --- Recovery-KDF salt (CL-8s2a) -------------------------------------------
#
# The recovery path (scripts/initialize_vault.py, out of the vault_codec lane)
# wraps the vault key under a FIXED public salt b"fx-recovery-v1". These pin
# the two facts that govern whether a per-vault random recovery salt can be
# added: derive_key is salt-agnostic (so it CAN), and the fixed-salt
# derivation is byte-stable (so existing recovery.enc/documents keep opening —
# backward compat is mandatory).


def test_derive_key_is_salt_agnostic_supports_random_recovery_salt():
    import secrets

    entropy = secrets.token_bytes(32)
    # Same "passphrase" (entropy), two DIFFERENT random salts → two different
    # recovery keys, both valid: a per-vault random recovery salt needs no
    # change to this KDF.
    k_fixed = derive_key(entropy.hex(), b"fx-recovery-v1", iterations=1000)
    k_rand = derive_key(entropy.hex(), secrets.token_bytes(16), iterations=1000)
    assert len(k_fixed) == 32 and len(k_rand) == 32
    assert k_fixed != k_rand
    # A recovery.enc sealed under either key round-trips.
    for k in (k_fixed, k_rand):
        assert unseal(seal(b"vault-key-material", k), k) == b"vault-key-material"


def test_fixed_recovery_salt_derivation_is_byte_stable():
    """Backward-compat proof: the fixed-salt recovery derivation must stay
    byte-for-byte identical, or existing printed recovery documents +
    recovery.enc for the LIVE vault would no longer decrypt."""
    entropy_hex = "00" * 32  # a fixed known entropy for a stable vector
    k1 = derive_key(entropy_hex, b"fx-recovery-v1", iterations=1000)
    k2 = derive_key(entropy_hex, b"fx-recovery-v1", iterations=1000)
    assert k1 == k2  # deterministic
    # Pin the exact bytes so a future KDF tweak can't silently break recovery.
    import hashlib

    assert k1 == hashlib.pbkdf2_hmac(
        "sha256",
        entropy_hex.encode(),
        b"fx-recovery-v1",
        1000,
        32,
    )


def test_initialize_and_agent_roundtrip():
    """The actual incident: a vault written by initialize_vault's encrypt must
    open via vault_agent's decrypt path."""
    import json

    from scripts.initialize_vault import encrypt

    sealed = encrypt(json.dumps({"K": "v"}).encode(), KEY)
    from src.security.vault_codec import unseal as agent_unseal

    assert json.loads(agent_unseal(sealed, KEY)) == {"K": "v"}


# --- Passphrase strength policy (CL-qyav P2) -------------------------------

from src.security.vault_codec import (  # noqa: E402
    MIN_PASSPHRASE_BITS,
    estimate_passphrase_bits,
    passphrase_weakness,
    require_strong_passphrase,
)

LEGACY_PHRASE = "abacus balance cactus dagger eagle fabric"  # 6 x log2(35) ~ 31 bits


def test_legacy_wordlist_phrase_scored_per_word_not_per_char():
    bits = estimate_passphrase_bits(LEGACY_PHRASE)
    assert 30 < bits < 32  # ~31 bits despite being 41 characters long
    assert passphrase_weakness(LEGACY_PHRASE) is not None


def test_short_passphrase_rejected():
    with pytest.raises(ValueError, match="characters"):
        require_strong_passphrase("Ab3!x9")


def test_low_entropy_passphrase_rejected():
    # 12 lowercase chars: 12 * log2(26) ~ 56 bits < 60
    with pytest.raises(ValueError, match="entropy"):
        require_strong_passphrase("abcdefghijkl")


def test_legacy_generator_phrase_rejected_for_new_seals():
    with pytest.raises(ValueError, match="too weak"):
        require_strong_passphrase(LEGACY_PHRASE)


def test_strong_passphrases_accepted():
    require_strong_passphrase("correct-Horse7battery!staple")
    # 13 mixed-case+digit chars: 13 * log2(62) ~ 77 bits
    require_strong_passphrase("aB3defghijkl9")
    # long phrase from a large wordlist (not the legacy 35-word list)
    require_strong_passphrase("abandon ability able about above absent")
    assert estimate_passphrase_bits("") == 0.0
    assert MIN_PASSPHRASE_BITS == 60.0


def test_existing_weak_passphrase_vault_still_unseals(tmp_path):
    """Backward compat: a vault sealed under the weak legacy passphrase must
    keep opening — weakness is warn-only on the unseal path, never fatal."""
    import json

    weak_key = derive_key(LEGACY_PHRASE, b"salt-16-bytes---", iterations=1000)
    vault_file = tmp_path / "vault.enc"
    vault_file.write_text(json.dumps(seal(b'{"K": "v"}', weak_key)))

    from src.security.vault_agent import decrypt_vault

    assert decrypt_vault(vault_file, weak_key) == {"K": "v"}
    # ...and the policy check that main_async logs a WARNING from still flags it
    assert passphrase_weakness(LEGACY_PHRASE) is not None


# --- Canonical atomic write helper (CL-9dhg) -------------------------------

import stat  # noqa: E402

from src.security.vault_codec import atomic_write_bytes  # noqa: E402


def test_atomic_write_bytes_writes_0600_and_replaces(tmp_path):
    target = tmp_path / "vault.enc"
    target.write_text("old contents")

    atomic_write_bytes(target, b"new contents")

    assert target.read_bytes() == b"new contents"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp*"))  # staging file replaced away


def test_atomic_write_bytes_fsyncs_before_replace(tmp_path, monkeypatch):
    """The durability contract: data must be fsynced to disk BEFORE the
    rename — the drifted vault_add-era copy skipped this, so power loss
    after the rename could surface a truncated vault.enc (CL-9dhg)."""
    import os

    calls: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1]
    )

    atomic_write_bytes(tmp_path / "vault.enc", b"payload")

    assert calls == ["fsync", "replace"]


def test_atomic_write_bytes_failure_leaves_old_file_and_no_litter(tmp_path, monkeypatch):
    import os

    target = tmp_path / "vault.enc"
    target.write_text("old contents")
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("simulated crash")))

    with pytest.raises(OSError, match="simulated crash"):
        atomic_write_bytes(target, b"new contents")

    assert target.read_text() == "old contents"  # old file intact
    assert not list(tmp_path.glob("*.tmp*"))  # staging file cleaned up
