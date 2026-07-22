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
