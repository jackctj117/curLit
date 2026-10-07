"""Operational-security regressions (CL-esh6).

Three boundaries, each asserted against the requirement written in the
bead rather than against the implementation's own constants:

  * Telegram approvals — approve/reject/skip only from a sender in
    TELEGRAM_APPROVER_IDS; unset = deny all; refusals logged with the
    sender id and never with the bot token.
  * Prometheus /metrics — loopback (127.0.0.1) unless METRICS_BIND_ADDR
    deliberately says otherwise.
  * claude CLI subprocess — an explicit allowlisted environment: no
    broker / DB / messaging / web-API secrets and no API credentials.

No network, no real CLI, no real Telegram: all boundaries are faked.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring import metrics as metrics_mod
from src.research.approvals import save_state_atomic
from src.research.llm.claude_code import ClaudeCodeDriver, build_cli_env
from src.research.llm.client import Message
from src.research.loop import GATE1_APPROVED_STATUS, GATE1_PENDING_STATUS, LoopState, load_state
from src.research.telegram_approvals import (
    TelegramApprovalBot,
    handle_text,
    parse_approver_ids,
)

pytestmark = pytest.mark.unit

CHAT_ID = "123456789"
TOKEN = "2222222222:BBBBfake-token-must-never-leak"
APPROVER_ID = 5550001
STRANGER_ID = 5550002
HASH_A = "a1b2c3d4e5f6a7b8"


# --------------------------------------------------------------------- #
# Telegram sender authorization
# --------------------------------------------------------------------- #


def _pending_state() -> LoopState:
    return LoopState(
        ideas_processed={
            HASH_A: {
                "status": GATE1_PENDING_STATUS,
                "slug": "alpha",
                "reason": "",
                "pending_since": "2026-07-10T00:00:00+00:00",
                "hypothesis_path": "docs/research/hypotheses/alpha.md",
            }
        },
        debates_completed={},
    )


class _FakeApi:
    def __init__(self, batch: list[dict[str, Any]]) -> None:
        self.batch = batch
        self.sent: list[str] = []

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "getUpdates":
            batch, self.batch = self.batch, []
            return {"ok": True, "result": batch}
        self.sent.append(str(params["text"]))
        return {"ok": True, "result": {}}


def _msg(update_id: int, text: str, **message: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"chat": {"id": int(CHAT_ID)}, "text": text}
    body.update(message)
    return {"update_id": update_id, "message": body}


def _from(uid: Any, is_bot: bool = False) -> dict[str, Any]:
    return {"id": uid, "is_bot": is_bot}


def _run_bot(
    tmp_path: Path,
    update: dict[str, Any],
    approver_ids: frozenset[int] | None,
) -> tuple[_FakeApi, Path]:
    state_path = tmp_path / "state.json"
    save_state_atomic(_pending_state(), state_path)
    api = _FakeApi([update])
    kwargs: dict[str, Any] = {}
    if approver_ids is not None:
        kwargs["approver_ids"] = approver_ids
    bot = TelegramApprovalBot(
        token=TOKEN,
        chat_id=CHAT_ID,
        state_path=state_path,
        offset_path=tmp_path / "offset.json",
        api_call=api,
        **kwargs,
    )
    bot.poll_once(timeout_sec=0)
    return api, state_path


def _status(state_path: Path) -> str:
    return str(load_state(state_path).ideas_processed[HASH_A]["status"])


class TestTelegramSenderAuthorization:
    def test_authorized_sender_approves(self, tmp_path: Path) -> None:
        api, state_path = _run_bot(
            tmp_path,
            _msg(1, "approve a1b2c3", **{"from": _from(APPROVER_ID)}),
            frozenset({APPROVER_ID}),
        )
        assert _status(state_path) == GATE1_APPROVED_STATUS
        assert any("Approved" in m for m in api.sent)

    @pytest.mark.parametrize("verb", ["approve", "reject", "skip"])
    def test_unauthorized_sender_rejected_and_logged(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        verb: str,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="src.research.telegram_approvals"):
            api, state_path = _run_bot(
                tmp_path,
                _msg(1, f"{verb} a1b2c3 sneaky", **{"from": _from(STRANGER_ID)}),
                frozenset({APPROVER_ID}),
            )
        # Ignored: the gate entry is untouched.
        assert _status(state_path) == GATE1_PENDING_STATUS
        assert api.sent and "Not authorized" in api.sent[0]
        refusals = [
            r.getMessage() for r in caplog.records if "unauthorized sender" in r.getMessage()
        ]
        assert len(refusals) == 1
        assert str(STRANGER_ID) in refusals[0]
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert TOKEN not in joined
        assert "sneaky" not in joined  # free-text reason is not logged

    def test_unset_allowlist_denies_everyone_and_logs_once_at_startup(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="src.research.telegram_approvals"):
            api, state_path = _run_bot(
                tmp_path,
                _msg(1, "approve a1b2c3", **{"from": _from(APPROVER_ID)}),
                approver_ids=None,  # constructor default
            )
        assert _status(state_path) == GATE1_PENDING_STATUS
        assert "Not authorized" in api.sent[0]
        disabled = [
            r
            for r in caplog.records
            if "DISABLED" in r.getMessage() and "TELEGRAM_APPROVER_IDS" in r.getMessage()
        ]
        assert len(disabled) == 1
        assert TOKEN not in "\n".join(r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "message",
        [
            {},  # no `from` at all
            {"from": _from(APPROVER_ID, is_bot=True)},  # a bot account
            # posted on behalf of a chat (anonymous admin / channel)
            {"from": _from(APPROVER_ID), "sender_chat": {"id": -100123}},
            {"from": _from("not-an-int")},
            {"from": _from(True)},  # bool is not a user id
        ],
    )
    def test_unattributable_sender_never_authorizes(
        self,
        tmp_path: Path,
        message: dict[str, Any],
    ) -> None:
        _, state_path = _run_bot(
            tmp_path,
            _msg(1, "approve a1b2c3", **message),
            frozenset({APPROVER_ID}),
        )
        assert _status(state_path) == GATE1_PENDING_STATUS

    def test_wrong_chat_still_dropped_even_for_approver(self, tmp_path: Path) -> None:
        update = _msg(1, "approve a1b2c3", **{"from": _from(APPROVER_ID)})
        update["message"]["chat"] = {"id": 987654321}
        api, state_path = _run_bot(tmp_path, update, frozenset({APPROVER_ID}))
        assert _status(state_path) == GATE1_PENDING_STATUS
        assert api.sent == []

    @pytest.mark.parametrize("text", ["pending", "help"])
    def test_read_only_commands_open_to_chat_members(self, tmp_path: Path, text: str) -> None:
        api, state_path = _run_bot(
            tmp_path,
            _msg(1, text, **{"from": _from(STRANGER_ID)}),
            frozenset({APPROVER_ID}),
        )
        assert _status(state_path) == GATE1_PENDING_STATUS
        assert api.sent and "Not authorized" not in api.sent[0]

    def test_handle_text_defaults_to_unauthorized(self) -> None:
        state = _pending_state()
        result = handle_text(state, "approve a1b2c3")
        assert not result.state_changed
        assert state.ideas_processed[HASH_A]["status"] == GATE1_PENDING_STATUS


class TestParseApproverIds:
    @pytest.mark.parametrize("raw", [None, "", "   ", ",", " , "])
    def test_blank_is_empty_default_deny(self, raw: str | None) -> None:
        assert parse_approver_ids(raw) == frozenset()

    def test_comma_separated_with_whitespace(self) -> None:
        assert parse_approver_ids(" 111, 222 ,333,") == frozenset({111, 222, 333})

    @pytest.mark.parametrize("raw", ["111,abc", "1.5", "@operator", "-100123", "0"])
    def test_malformed_or_non_user_ids_fail_loud(self, raw: str) -> None:
        with pytest.raises(ValueError, match="TELEGRAM_APPROVER_IDS"):
            parse_approver_ids(raw)


class TestBotEntrypoint:
    def _env(self, monkeypatch: pytest.MonkeyPatch, approvers: str | None) -> None:
        import src.dotenv_bootstrap as bootstrap

        # Never read a real .env from a test.
        monkeypatch.setattr(bootstrap, "load_project_env", lambda *a, **k: None)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT_ID)
        if approvers is None:
            monkeypatch.delenv("TELEGRAM_APPROVER_IDS", raising=False)
        else:
            monkeypatch.setenv("TELEGRAM_APPROVER_IDS", approvers)

    def test_malformed_allowlist_exits_2(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import scripts.telegram_approval_bot as entry

        self._env(monkeypatch, "12345,oops")
        built = MagicMock()
        monkeypatch.setattr(entry, "TelegramApprovalBot", built)
        assert entry.main(["--once", "--state", str(tmp_path / "s.json")]) == 2
        built.assert_not_called()

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("12345, 67890", frozenset({12345, 67890})), (None, frozenset())],
    )
    def test_allowlist_reaches_bot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw: str | None,
        expected: frozenset[int],
    ) -> None:
        import scripts.telegram_approval_bot as entry

        self._env(monkeypatch, raw)
        built = MagicMock()
        built.return_value.poll_once.return_value = 0
        monkeypatch.setattr(entry, "TelegramApprovalBot", built)
        assert entry.main(["--once", "--state", str(tmp_path / "s.json")]) == 0
        assert built.call_args.kwargs["approver_ids"] == expected


# --------------------------------------------------------------------- #
# Metrics bind address
# --------------------------------------------------------------------- #


class TestMetricsBindAddress:
    def test_default_bind_is_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("METRICS_BIND_ADDR", raising=False)
        server = MagicMock()
        monkeypatch.setattr(metrics_mod, "start_http_server", server)
        metrics_mod.start_metrics_server(port=8099)
        server.assert_called_once_with(8099, addr="127.0.0.1")

    def test_blank_env_is_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("METRICS_BIND_ADDR", "  ")
        server = MagicMock()
        monkeypatch.setattr(metrics_mod, "start_http_server", server)
        metrics_mod.start_metrics_server(port=8099)
        server.assert_called_once_with(8099, addr="127.0.0.1")

    def test_env_override_honoured_and_warned(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("METRICS_BIND_ADDR", "0.0.0.0")
        server = MagicMock()
        monkeypatch.setattr(metrics_mod, "start_http_server", server)
        with caplog.at_level(logging.WARNING, logger="src.monitoring.metrics"):
            metrics_mod.start_metrics_server(port=9100)
        server.assert_called_once_with(9100, addr="0.0.0.0")
        assert any("NON-LOOPBACK" in r.getMessage() for r in caplog.records)

    def test_explicit_addr_argument_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("METRICS_BIND_ADDR", "0.0.0.0")
        server = MagicMock()
        monkeypatch.setattr(metrics_mod, "start_http_server", server)
        metrics_mod.start_metrics_server(port=9100, addr="127.0.0.1")
        server.assert_called_once_with(9100, addr="127.0.0.1")

    def test_live_engine_caller_gets_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The engine's call site (port only) must inherit the loopback default."""
        import inspect

        import src.runtime.live_engine as live_engine

        body = inspect.getsource(live_engine.LiveEngine.run)
        assert "start_metrics_server(" in body
        assert "addr=" not in body.split("start_metrics_server(", 1)[1].split(")", 1)[0]
        monkeypatch.delenv("METRICS_BIND_ADDR", raising=False)
        server = MagicMock()
        monkeypatch.setattr(metrics_mod, "start_http_server", server)
        live_engine.start_metrics_server(port=8099)
        assert server.call_args.kwargs["addr"] == "127.0.0.1"


# --------------------------------------------------------------------- #
# claude CLI subprocess environment
# --------------------------------------------------------------------- #

#: Secrets dotenv_bootstrap can load from .env — none may reach the CLI.
_SECRET_ENV = {
    "OANDA_API_KEY": "oanda-secret",
    "OANDA_ACCOUNT_ID": "101-001-1",
    "ALPACA_API_KEY": "alpaca-key",
    "ALPACA_SECRET_KEY": "alpaca-secret",
    "POSTGRES_PASSWORD": "pg-secret",
    "POSTGRES_USER": "fx",
    "MOONSHOT_API_KEY": "moonshot-secret",
    "TELEGRAM_BOT_TOKEN": TOKEN,
    "TELEGRAM_CHAT_ID": CHAT_ID,
    "WEB_API_SECRET": "web-secret",
    "ANTHROPIC_API_KEY": "sk-ant-live",
    "ANTHROPIC_AUTH_TOKEN": "auth-tok",
    "ANTHROPIC_BASE_URL": "https://proxy.example",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "AWS_SECRET_ACCESS_KEY": "aws-secret",
    "XAI_API_KEY": "xai-secret",
    "CURLIT_VAULT_PASSPHRASE": "vault-secret",
}

#: What the CLI legitimately needs (process basics + subscription login).
_NEEDED_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/Users/op",
    "USER": "op",
    "LANG": "en_US.UTF-8",
    "LC_ALL": "en_US.UTF-8",
    "LC_CTYPE": "UTF-8",
    "TMPDIR": "/tmp/",
    "TERM": "xterm-256color",
    "CLAUDE_CONFIG_DIR": "/Users/op/.claude-alt",
    "CLAUDE_CODE_OAUTH_TOKEN": "subscription-oauth",
}

_FORBIDDEN_PREFIXES = ("OANDA_", "ALPACA_", "POSTGRES_", "MOONSHOT_", "TELEGRAM_", "WEB_API_")


def _assert_clean(env: dict[str, str]) -> None:
    bad = sorted(k for k in env if k.startswith(_FORBIDDEN_PREFIXES))
    assert bad == [], f"secret-bearing vars reached the CLI: {bad}"
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    for secret in _SECRET_ENV.values():
        if secret not in _NEEDED_ENV.values():
            assert secret not in env.values()


class TestClaudeCliEnvironment:
    def test_only_allowlisted_keys_reach_cli(self) -> None:
        parent = {**_SECRET_ENV, **_NEEDED_ENV}
        env = build_cli_env(parent, enforce_cap=False, max_tokens=4096)
        assert env == _NEEDED_ENV
        _assert_clean(env)

    def test_cap_set_when_enforced(self) -> None:
        parent = {**_SECRET_ENV, **_NEEDED_ENV, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"}
        env = build_cli_env(parent, enforce_cap=True, max_tokens=900)
        assert env == {**_NEEDED_ENV, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "900"}
        _assert_clean(env)

    def test_driver_subprocess_env_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Through the real driver.complete() call — subprocess.run mocked."""
        monkeypatch.delenv("CURLIT_CLAUDE_ENFORCE_OUTPUT_CAP", raising=False)
        payload = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "ok",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "modelUsage": {"claude-fable-5": {"outputTokens": 1}},
        }
        proc = MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch("shutil.which", return_value="/fake/claude"):
            drv = ClaudeCodeDriver()
        with (
            patch.dict(os.environ, {**_SECRET_ENV, **_NEEDED_ENV}, clear=True),
            patch("subprocess.run", return_value=proc) as run,
        ):
            drv.complete([Message("user", "hi")], model="claude-fable-5")
        env = run.call_args.kwargs["env"]
        assert env == _NEEDED_ENV
        _assert_clean(env)
