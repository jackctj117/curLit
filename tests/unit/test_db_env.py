"""Tests for the shared Postgres URL builder (CL-8lv6, CL-8cw1).

CL-8cw1: ``build_db_url`` now constructs a ``sqlalchemy.engine.URL`` via
``URL.create`` — special characters in the password survive the round
trip, the repr redacts the password, the DATABASE_URL override passes
through untouched, and the changeme warning still fires once per
process. Adoption: the five modules that carried silent local copies of
the builder (no changeme warning) now delegate to db_env — same idiom
as tests/unit/test_x_monitor.py TestIngestEngineDbUrl.
"""

from __future__ import annotations

import importlib
import logging

import pytest
from sqlalchemy import create_engine, make_url
from sqlalchemy.engine import URL

from src.data import db_env

_ENV_VARS = (
    "DATABASE_URL",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _password_warnings(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    return [r for r in records if "default password" in r.getMessage()]


class TestBuildDbUrl:
    def test_special_char_password_round_trips(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss:word/")
        url = db_env.build_db_url()
        assert isinstance(url, URL)
        # create_engine consumes the URL object directly with the raw
        # password intact (the old f-string builder produced an unparseable
        # or wrong-credential DSN for these characters).
        assert create_engine(url).url.password == "p@ss:word/"
        # The explicit string rendering percent-encodes, so re-parsing it
        # recovers the same password.
        rendered = url.render_as_string(hide_password=False)
        assert make_url(rendered).password == "p@ss:word/"

    def test_repr_redacts_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret-hunter2")
        url = db_env.build_db_url()
        # Accidental logging of the URL (repr or str) must not leak it.
        assert "s3cret-hunter2" not in repr(url)
        assert "s3cret-hunter2" not in str(url)

    def test_default_url_shape_unchanged(self) -> None:
        url = db_env.build_db_url()
        assert isinstance(url, URL)
        assert url.render_as_string(hide_password=False) == (
            "postgresql+psycopg2://fx:changeme@localhost:5432/fx"
        )

    def test_database_url_override_wins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Passed through as-is — not re-parsed, not re-rendered.
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        assert db_env.build_db_url() == "sqlite://"

    def test_changeme_warns_once_per_process(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(db_env, "_warned", False)
        with caplog.at_level(logging.WARNING, logger="src.data.db_env"):
            db_env.build_db_url()
            db_env.build_db_url()
        assert len(_password_warnings(caplog.records)) == 1

    def test_no_warning_when_password_rotated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(db_env, "_warned", False)
        monkeypatch.setenv("POSTGRES_PASSWORD", "rotated")
        with caplog.at_level(logging.WARNING, logger="src.data.db_env"):
            db_env.build_db_url()
        assert not _password_warnings(caplog.records)


class TestAdoption:
    """The former local `_build_db_url` copies silently defaulted to the
    changeme password with no warning; each module must now bind the
    shared helper at import level and the local copy must be gone."""

    @pytest.mark.parametrize(
        "modname",
        [
            "scripts.execute_options",
            "scripts.morning_digest",
            "scripts.truth_monitor",
            "scripts.truth_report",
            "migrations.run",
        ],
    )
    def test_binds_shared_helper_and_drops_local_copy(self, modname: str) -> None:
        mod = importlib.import_module(modname)
        assert mod.build_db_url is db_env.build_db_url
        assert not hasattr(mod, "_build_db_url")

    def test_migrations_get_engine_delegates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from migrations import run as run_mod

        monkeypatch.setattr(run_mod, "build_db_url", lambda: "sqlite://")
        assert str(run_mod.get_engine().url) == "sqlite://"

    def test_migrations_get_engine_database_url_override_wins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # New with CL-8cw1: migrations honour DATABASE_URL like every
        # other consumer of the shared convention.
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        from migrations import run as run_mod

        assert str(run_mod.get_engine().url) == "sqlite://"

    def test_migrations_get_engine_default_shape_unchanged(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from migrations import run as run_mod

        # Engine construction succeeds without a live DB; URL shape is the
        # historical one.
        engine = run_mod.get_engine()
        assert engine.url.render_as_string(hide_password=False) == (
            "postgresql+psycopg2://fx:changeme@localhost:5432/fx"
        )
