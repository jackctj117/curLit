"""scripts/vault_add.py after CL-8lv6 P1: sealing delegated to vault_codec
(no more parallel schema implementation) and atomic 0600 vault writes.

Hermetic: vault files under tmp_path via VAULT_PATH/VAULT_SALT env; prompts
monkeypatched — the real vault/.env are never touched.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Iterator

import pytest
from scripts import vault_add

from src.security.vault_codec import derive_key, seal, unseal

PASS = "correct-Horse7battery!staple"
SALT = b"salt-16-bytes---"


@pytest.fixture()
def vault_env(tmp_path, monkeypatch):
    key = derive_key(PASS, SALT)
    vault_path = tmp_path / "vault.enc"
    salt_path = tmp_path / "vault.salt"
    vault_path.write_text(json.dumps(seal(json.dumps({"EXISTING": "x"}).encode(), key)))
    salt_path.write_bytes(SALT)
    monkeypatch.setenv("VAULT_PATH", str(vault_path))
    monkeypatch.setenv("VAULT_SALT", str(salt_path))
    return vault_path, key


def _feed_getpass(monkeypatch, answers: list[str]) -> None:
    it: Iterator[str] = iter(answers)
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: next(it))


def test_cmd_add_roundtrips_through_codec(vault_env, monkeypatch, capsys):
    vault_path, key = vault_env
    _feed_getpass(monkeypatch, [PASS, "sekret-value"])
    monkeypatch.setattr("builtins.input", lambda *a: "MY_KEY")

    vault_add.cmd_add()

    assert "Added: MY_KEY" in capsys.readouterr().out
    on_disk = json.loads(unseal(json.loads(vault_path.read_text()), key))
    assert on_disk == {"EXISTING": "x", "MY_KEY": "sekret-value"}
    # canonical vault_codec v2 schema — not the old parallel implementation
    assert json.loads(vault_path.read_text())["v"] == 2
    # atomic write left no staging litter, and the vault is 0600
    assert not list(vault_path.parent.glob("*.tmp*"))
    assert stat.S_IMODE(vault_path.stat().st_mode) == 0o600


def test_cmd_remove(vault_env, monkeypatch, capsys):
    vault_path, key = vault_env
    _feed_getpass(monkeypatch, [PASS])
    monkeypatch.setattr("builtins.input", lambda *a: "EXISTING")

    vault_add.cmd_remove()

    assert "Removed: EXISTING" in capsys.readouterr().out
    assert json.loads(unseal(json.loads(vault_path.read_text()), key)) == {}


def test_cmd_remove_missing_name_exits(vault_env, monkeypatch):
    _feed_getpass(monkeypatch, [PASS])
    monkeypatch.setattr("builtins.input", lambda *a: "NOPE")
    with pytest.raises(SystemExit):
        vault_add.cmd_remove()


def test_cmd_list(vault_env, monkeypatch, capsys):
    _feed_getpass(monkeypatch, [PASS])
    vault_add.cmd_list()
    assert "EXISTING" in capsys.readouterr().out


def test_open_vault_reads_legacy_v1_schema(vault_env, monkeypatch):
    """unseal() keeps accepting the legacy {"ct": ...} shape — vault_add must
    open historical vaults exactly like the agent does."""
    vault_path, key = vault_env
    sealed = seal(json.dumps({"OLD": "y"}).encode(), key)
    legacy = {"v": 1, "nonce": sealed["nonce"], "ct": sealed["ciphertext"], "tag": sealed["tag"]}
    vault_path.write_text(json.dumps(legacy))
    _feed_getpass(monkeypatch, [PASS])

    vault, _, _ = vault_add.open_vault()

    assert vault == {"OLD": "y"}
