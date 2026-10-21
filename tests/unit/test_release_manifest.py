"""Independent secret sentinels and resolved-config startup evidence."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.runtime import release_manifest


@pytest.mark.parametrize("key", [
    "token", "refresh_token", "account_id", "OANDA_ACCOUNT_ID", "private_key",
    "cookie", "session_id", "connection_string", "dsn", "api_key",
])
def test_sensitive_fields_are_never_serialized(key: str) -> None:
    sentinel = "SYNTHETIC_PRIVATE_VALUE"
    assert sentinel not in json.dumps(release_manifest.redact({"nested": [{key: sentinel}]}))


def test_unrecognized_object_is_not_stringified() -> None:
    class CredentialBearingObject:
        def __str__(self) -> str:
            raise AssertionError("must not inspect arbitrary configuration object")

    assert release_manifest.redact({"client": CredentialBearingObject()}) == {
        "client": "[UNSUPPORTED TYPE]"
    }


def test_manifest_records_resolved_values_and_hashes_without_claiming_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/fixture.py").write_text("pass\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/entrypoint.py").write_text("pass\n")
    monkeypatch.setattr(
        release_manifest.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="a" * 40),
    )
    output = release_manifest.write_release_manifest(
        tmp_path, role="alpaca_equities", broker_mode="paper",
        config={"entry": {"notional_usd": 275.0}, "ledger_close_only": True,
                "account_id": "SYNTHETIC_PRIVATE_VALUE"},
    )
    payload = json.loads(output.read_text())
    assert payload["effective_configuration"]["entry"]["notional_usd"] == 275.0
    assert payload["effective_configuration"]["ledger_close_only"] is True
    assert "SYNTHETIC_PRIVATE_VALUE" not in output.read_text()
    assert payload["migration_version"] is None
    assert payload["dependency_lock_hash"] is None
    assert payload["source_hashes"]["src/fixture.py"] == (
        # Independently calculated with printf 'pass\n' | shasum -a 256.
        "9f56e761d79bfdb34304a012586cb04d16b435ef6130091a97702e559260a2f2"
    )
    assert payload["source_hashes"]["scripts/entrypoint.py"] == (
        "9f56e761d79bfdb34304a012586cb04d16b435ef6130091a97702e559260a2f2"
    )
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        release_manifest.write_release_manifest(
            tmp_path, role="alpaca_equities", broker_mode="paper", config={},
        )


@pytest.mark.parametrize("book,env_name,field,value", [
    ("options", "ALPACA_OPT_MIN_CONFIDENCE", "min_confidence", "0.72"),
    ("equities", "ALPACA_EQ_NOTIONAL_USD", "notional_usd", "275.0"),
])
def test_alpaca_startup_captures_resolved_config_before_database_or_broker(
    book: str, env_name: str, field: str, value: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import dotenv_bootstrap

    module = importlib.import_module(f"scripts.execute_{book}")
    monkeypatch.setattr(dotenv_bootstrap, "load_project_env", lambda: None)
    for key, setting in {
        "ALPACA_API_KEY": "SYNTHETIC_KEY", "ALPACA_API_SECRET": "SYNTHETIC_SECRET",
        "ALPACA_OPTIONS_ENABLED": "1", "ALPACA_EQUITY_ENABLED": "1",
        "ALPACA_PAPER": "true", "ALPACA_LEDGER_CLOSE_ONLY": "1", env_name: value,
    }.items():
        monkeypatch.setenv(key, setting)

    class CapturedBeforeIOError(Exception):
        pass

    def capture(root: Path, *, role: str, broker_mode: str, config: dict) -> Path:
        assert role == f"alpaca_{book}"
        assert broker_mode == "alpaca-paper"
        assert config["entry"][field] == float(value)
        assert config["ledger_close_only"] is True
        assert config["loop_seconds"] == 300
        assert "SYNTHETIC_" not in json.dumps(config)
        raise CapturedBeforeIOError

    monkeypatch.setattr(release_manifest, "write_release_manifest", capture)
    monkeypatch.setattr(module, "build_db_url", lambda: pytest.fail("database reached first"))
    with pytest.raises(CapturedBeforeIOError):
        module.main(["--loop", "300"])
