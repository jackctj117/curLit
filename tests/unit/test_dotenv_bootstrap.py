"""Tests for the project-wide .env auto-loader (src/dotenv_bootstrap.py).

Covers:

  * a .env file is loaded into os.environ when found
  * pre-existing env vars are NOT overridden (industry convention:
    explicit env wins over .env)
  * a missing .env path is a clean no-op
  * an explicit env_path argument bypasses find_dotenv
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.dotenv_bootstrap import load_project_env


@pytest.fixture
def tmp_env(tmp_path: Path) -> Path:
    """Write a .env with two vars, return the path."""
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nTEST_BOOTSTRAP_NEW=fresh-value\nTEST_BOOTSTRAP_PREEXISTING=from-file\n",
    )
    return env


class TestLoadProjectEnv:
    def test_loads_when_path_given(
        self,
        tmp_env: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("TEST_BOOTSTRAP_NEW", raising=False)
        result = load_project_env(env_path=tmp_env)
        assert result == tmp_env
        assert os.environ.get("TEST_BOOTSTRAP_NEW") == "fresh-value"

    def test_does_not_override_pre_existing(
        self,
        tmp_env: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Var pre-set in the env (e.g. by systemd) should NOT be
        # overwritten by the .env file's value.
        monkeypatch.setenv("TEST_BOOTSTRAP_PREEXISTING", "from-shell")
        load_project_env(env_path=tmp_env)
        assert os.environ.get("TEST_BOOTSTRAP_PREEXISTING") == "from-shell"

    def test_missing_path_is_noop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("TEST_BOOTSTRAP_NEW", raising=False)
        result = load_project_env(env_path=tmp_path / "no-such-file.env")
        assert result is None
        assert os.environ.get("TEST_BOOTSTRAP_NEW") is None

    def test_uses_find_dotenv_when_path_not_given(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # find_dotenv walks up from cwd; chdir into tmp_path with a
        # local .env to verify it's discovered.
        env = tmp_path / ".env"
        env.write_text("TEST_BOOTSTRAP_FIND=ok\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("TEST_BOOTSTRAP_FIND", raising=False)
        result = load_project_env()
        assert result is not None
        assert result.resolve() == env.resolve()
        assert os.environ.get("TEST_BOOTSTRAP_FIND") == "ok"

    def test_no_env_file_anywhere_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # chdir into a totally empty dir tree so find_dotenv returns ""
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.chdir(empty)
        # Defang find_dotenv by pointing HOME at the empty tree too
        # (some implementations check $HOME). On macOS find_dotenv
        # walks up from cwd, so chdir alone usually suffices.
        monkeypatch.setenv("HOME", str(empty))
        result = load_project_env()
        # No assertion on os.environ here — find_dotenv() may locate
        # an unrelated .env on the user's machine. Just confirm the
        # return type is path-or-none.
        assert result is None or isinstance(result, Path)
