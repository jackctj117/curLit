"""Non-interactive re-seal CLI of scripts/initialize_vault.py (CL-8lv6 P0).

This is the interface deploy/scripts/init_vault.sh and
scripts/rotate_secrets.rotate() drive: plaintext JSON through a /dev/fd
pipe path, passphrase on stdin, sealed vault written atomically.

Hermetic: everything lives under tmp_path; the real vault/.env are never
touched.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.initialize_vault import main

from src.security.vault_codec import derive_key, unseal

REPO_ROOT = Path(__file__).resolve().parents[2]
STRONG = "abandon ability able about above absent"  # 6 bip39 words, passes policy
WEAK = "weak pass"  # 9 chars — fails the >=12-char policy floor


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    plaintext: bytes,
    enc: Path,
    passphrase: bytes,
    salt: Path | None = None,
) -> int:
    """Invoke main() in-process with plaintext through an os.pipe /dev/fd path
    and the passphrase on (monkeypatched) stdin."""
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, plaintext)
        os.close(write_fd)
        monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(passphrase), encoding="utf-8"))
        argv = [
            "--plaintext",
            f"/dev/fd/{read_fd}",
            "--encrypted",
            str(enc),
            "--passphrase-stdin",
        ]
        if salt is not None:
            argv += ["--salt", str(salt)]
        return main(argv)
    finally:
        with contextlib.suppress(OSError):
            os.close(read_fd)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_seal_via_pipe_then_unseal_with_codec(tmp_path, monkeypatch):
    enc = tmp_path / "credentials.enc"
    payload = json.dumps({"OANDA_API_KEY": "k1", "FRED_API_KEY": "k2"}).encode()

    rc = _run_cli(monkeypatch, payload, enc, STRONG.encode() + b"\n")

    assert rc == 0
    # default salt path: vault.salt ALONGSIDE the encrypted file
    salt_file = tmp_path / "vault.salt"
    assert salt_file.exists() and len(salt_file.read_bytes()) == 16
    assert _mode(enc) == 0o600 and _mode(salt_file) == 0o600
    key = derive_key(STRONG, salt_file.read_bytes())
    assert unseal(json.loads(enc.read_text()), key) == payload
    # atomicity: the tmp staging files are gone after os.replace
    assert not list(tmp_path.glob("*.tmp*"))


def test_subprocess_matches_rotate_secrets_interface(tmp_path):
    """Exactly what rotate_secrets.rotate() does: Popen with pass_fds and a
    /dev/fd/N plaintext path, passphrase piped to stdin WITHOUT a newline."""
    enc = tmp_path / "credentials.enc"
    payload = json.dumps({"POSTGRES_PASSWORD": "pw"}).encode()
    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, True)
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.initialize_vault",
                "--plaintext",
                f"/dev/fd/{read_fd}",
                "--encrypted",
                str(enc),
                "--passphrase-stdin",
            ],
            stdin=subprocess.PIPE,
            pass_fds=(read_fd,),
            cwd=REPO_ROOT,
        )
        os.write(write_fd, payload)
        os.close(write_fd)
        proc.communicate(input=STRONG.encode(), timeout=120)
    finally:
        with contextlib.suppress(OSError):
            os.close(read_fd)
    assert proc.returncode == 0
    salt = (tmp_path / "vault.salt").read_bytes()
    assert unseal(json.loads(enc.read_text()), derive_key(STRONG, salt)) == payload


def test_existing_salt_is_reused(tmp_path, monkeypatch):
    enc = tmp_path / "vault.enc"
    salt_file = tmp_path / "vault.salt"
    pinned = b"\x02" * 16
    salt_file.write_bytes(pinned)

    rc = _run_cli(monkeypatch, b'{"K": "v"}', enc, STRONG.encode() + b"\n", salt=salt_file)

    assert rc == 0
    assert salt_file.read_bytes() == pinned  # NOT regenerated
    assert unseal(json.loads(enc.read_text()), derive_key(STRONG, pinned)) == b'{"K": "v"}'


def test_explicit_salt_path_created_when_missing(tmp_path, monkeypatch):
    enc = tmp_path / "vault.enc"
    salt_file = tmp_path / "elsewhere.salt"

    rc = _run_cli(monkeypatch, b'{"K": "v"}', enc, STRONG.encode() + b"\n", salt=salt_file)

    assert rc == 0
    assert len(salt_file.read_bytes()) == 16
    assert _mode(salt_file) == 0o600
    assert not (tmp_path / "vault.salt").exists()  # default path untouched


def test_weak_passphrase_fails_on_new_vault_and_writes_nothing(tmp_path, monkeypatch, capsys):
    enc = tmp_path / "vault.enc"

    rc = _run_cli(monkeypatch, b'{"K": "v"}', enc, WEAK.encode() + b"\n")

    assert rc == 1
    assert "too weak" in capsys.readouterr().err
    assert not enc.exists()
    assert not list(tmp_path.iterdir())  # not even a salt file or tmp litter


def test_weak_passphrase_warns_but_reseals_existing_vault(tmp_path, monkeypatch, capsys):
    """The rotation path: the live vault's passphrase is known-weak — re-seal
    under the SAME phrase must succeed (warn-only), or rotation is bricked."""
    enc = tmp_path / "vault.enc"
    enc.write_text("old-sealed-bytes")  # any existing file ⇒ overwrite mode
    payload = json.dumps({"OANDA_API_KEY": "rotated"}).encode()

    rc = _run_cli(monkeypatch, payload, enc, WEAK.encode() + b"\n")

    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "weak" in err
    salt = (tmp_path / "vault.salt").read_bytes()
    assert unseal(json.loads(enc.read_text()), derive_key(WEAK, salt)) == payload
    assert not list(tmp_path.glob("*.tmp*"))


def test_non_json_plaintext_rejected_before_touching_disk(tmp_path, monkeypatch, capsys):
    enc = tmp_path / "vault.enc"

    rc = _run_cli(monkeypatch, b"not json at all", enc, STRONG.encode() + b"\n")

    assert rc == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_interactive_init_writes_atomically_with_0600(tmp_path, monkeypatch, capsys):
    """Interactive one-time init (CL-8cw1): vault/salt/recovery must land
    via the same atomic tmp+os.replace path as reseal — mode 0600, no tmp
    litter. The old write_text-then-chmod left a umask-readable window."""
    vault = tmp_path / "vault.enc"
    salt = tmp_path / "vault.salt"
    recovery = tmp_path / "recovery.enc"
    monkeypatch.setenv("VAULT_PATH", str(vault))
    monkeypatch.setenv("VAULT_SALT", str(salt))
    monkeypatch.setenv("RECOVERY_PATH", str(recovery))
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    monkeypatch.setattr(os, "system", lambda *_a: 0)  # skip screen clear

    rc = main([])

    assert rc == 0
    for p in (vault, salt, recovery):
        assert p.exists() and _mode(p) == 0o600
    assert len(salt.read_bytes()) == 16
    assert not list(tmp_path.glob("*.tmp*"))  # staging files replaced away
    # The printed master passphrase must actually open the sealed vault.
    out = capsys.readouterr().out
    passphrase = out.split("MASTER PASSPHRASE:")[1].strip().splitlines()[0].strip()
    key = derive_key(passphrase, salt.read_bytes())
    assert unseal(json.loads(vault.read_text()), key) == b"{}"


def test_interactive_init_crash_mid_write_leaves_no_sentinel(tmp_path, monkeypatch):
    """CL-9dhg finding 12: vault.enc existing is the refuse-to-reinitialize
    sentinel, so it must be written LAST. Simulate a crash between writes
    (second atomic write raises): the sentinel must be absent afterwards —
    a half-initialized vault would otherwise be permanently undecryptable
    (no salt/recovery) AND refuse re-initialization."""
    import scripts.initialize_vault as iv

    vault = tmp_path / "vault.enc"
    monkeypatch.setenv("VAULT_PATH", str(vault))
    monkeypatch.setenv("VAULT_SALT", str(tmp_path / "vault.salt"))
    monkeypatch.setenv("RECOVERY_PATH", str(tmp_path / "recovery.enc"))
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    monkeypatch.setattr(os, "system", lambda *_a: 0)

    real_write = iv.atomic_write_bytes
    calls: list[Path] = []

    def crash_on_second(path: Path, payload: bytes, mode: int = 0o600) -> None:
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated crash mid-initialization")
        real_write(path, payload, mode)

    monkeypatch.setattr(iv, "atomic_write_bytes", crash_on_second)

    with pytest.raises(OSError, match="simulated crash"):
        main([])

    # The sentinel did NOT land — re-initialization remains possible...
    assert not vault.exists()
    # ...and vault.enc was the LAST scheduled write, after salt + recovery.
    assert calls == [tmp_path / "vault.salt", tmp_path / "recovery.enc"]

    # Prove it: re-running init on the same paths now succeeds cleanly
    # (crash_on_second only raises on call #2; calls 3-5 pass through).
    assert main([]) == 0
    for name in ("vault.enc", "vault.salt", "recovery.enc"):
        assert (tmp_path / name).exists()
    assert not list(tmp_path.glob("*.tmp*"))


def test_passphrase_strips_exactly_one_trailing_newline(tmp_path, monkeypatch):
    """`printf '%s\\n' "$PASS" | ...` appends one newline — it must not become
    part of the passphrase, but an embedded newline must survive."""
    enc = tmp_path / "vault.enc"
    rc = _run_cli(monkeypatch, b'{"K": "v"}', enc, STRONG.encode() + b"\n")
    assert rc == 0
    salt = (tmp_path / "vault.salt").read_bytes()
    # passphrase WITHOUT the newline opens it; WITH the newline does not
    assert unseal(json.loads(enc.read_text()), derive_key(STRONG, salt))
    with pytest.raises(Exception):  # noqa: B017 — backend-specific auth error
        unseal(json.loads(enc.read_text()), derive_key(STRONG + "\n", salt))
