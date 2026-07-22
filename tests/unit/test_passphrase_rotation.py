"""Tests for strong passphrase generation + master-passphrase rotation
(CL-qyav follow-ups: scripts/initialize_vault.py, scripts/rotate_secrets.py).
"""

from __future__ import annotations

import json

import pytest
from scripts.initialize_vault import generate_passphrase, load_bip39_words
from scripts.rotate_secrets import reseal_with_new_passphrase

from src.security.vault_codec import (
    derive_key,
    estimate_passphrase_bits,
    require_strong_passphrase,
    seal,
    unseal,
)

STRONG = "abandon ability able about above absent"  # 6 bip39 words


def test_bip39_list_is_full_2048():
    words = load_bip39_words()
    assert len(words) == 2048
    assert words[0] == "abandon" and words[-1] == "zoo"


def test_generated_passphrase_passes_policy():
    for _ in range(5):
        phrase = generate_passphrase()
        assert len(phrase.split()) == 6
        require_strong_passphrase(phrase)  # must not raise
        assert estimate_passphrase_bits(phrase) >= 60


def test_reseal_round_trip(tmp_path):
    vault_path = tmp_path / "vault.enc"
    salt_path = tmp_path / "vault.salt"
    old_salt = b"\x01" * 16
    plaintext = json.dumps({"OANDA_API_KEY": "k"}).encode()
    vault_path.write_text(json.dumps(seal(plaintext, derive_key("old", old_salt))))
    salt_path.write_bytes(old_salt)

    reseal_with_new_passphrase(plaintext, STRONG, vault_path, salt_path)

    new_salt = salt_path.read_bytes()
    assert new_salt != old_salt  # fresh salt
    out = unseal(json.loads(vault_path.read_text()), derive_key(STRONG, new_salt))
    assert out == plaintext
    # old key no longer opens the new file
    with pytest.raises(Exception):  # noqa: B017 — backend-specific auth error
        unseal(json.loads(vault_path.read_text()), derive_key("old", new_salt))
    # timestamped backups of BOTH files exist
    assert list(tmp_path.glob("vault.enc.bak-*"))
    assert list(tmp_path.glob("vault.salt.bak-*"))


def test_reseal_rejects_weak_passphrase(tmp_path):
    vault_path = tmp_path / "vault.enc"
    salt_path = tmp_path / "vault.salt"
    vault_path.write_text("{}")
    salt_path.write_bytes(b"\x01" * 16)
    before = vault_path.read_text()
    with pytest.raises(ValueError, match="too weak"):
        reseal_with_new_passphrase(b"{}", "cactus dagger", vault_path, salt_path)
    assert vault_path.read_text() == before  # nothing changed
    assert not list(tmp_path.glob("*.bak-*"))
