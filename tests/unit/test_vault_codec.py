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
    legacy = {"v": 1, "nonce": sealed["nonce"], "ct": sealed["ciphertext"],
              "tag": sealed["tag"]}  # the initialize_vault v1 shape
    assert unseal(legacy, KEY) == b"legacy"


def test_unseal_accepts_appended_tag_layout():
    sealed = seal(b"appended", KEY)
    merged = {"v": 1, "nonce": sealed["nonce"],
              "ciphertext": sealed["ciphertext"] + sealed["tag"], "tag": ""}
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
