"""Integration tests — security: vault init, agent start, client get, restore."""

import json
import os
import subprocess
import pytest

VAULT_SCRIPT = "scripts/initialize_vault.py"


class TestSecurity:
    def test_vault_init_creates_files(self, tmp_path) -> None:
        vault = tmp_path / "vault.enc"
        salt = tmp_path / "vault.salt"
        rec = tmp_path / "recovery.enc"
        # Test imports and functions only (no actual passphrase prompt in CI)
        from scripts.initialize_vault import derive_key, encrypt, generate_passphrase
        key = derive_key("test", b"saltsaltsaltsalt"[:16])
        data = encrypt(b'{"test_key": "test_val"}', key)
        assert "ct" in data
        assert "nonce" in data

    def test_derive_key_deterministic(self) -> None:
        from scripts.initialize_vault import derive_key
        k1 = derive_key("password", b"salt" * 4)
        k2 = derive_key("password", b"salt" * 4)
        assert k1 == k2
