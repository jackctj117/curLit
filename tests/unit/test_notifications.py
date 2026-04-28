"""Unit tests for notification dispatch (CL-0hr3 / CL-yta6).

Tests cover env-var gating + dispatch routing without making any real
network calls. We monkeypatch httpx.post on the notifications module
to record what would have been sent.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.research.notifications import (
    DispatchResult,
    notify_operator,
)

# --------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=Any,  # type: ignore[arg-type]
                response=Any,  # type: ignore[arg-type]
            )


@pytest.fixture
def http_recorder(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace httpx.post with a recorder. Returns the list of
    recorded calls so tests can assert on them."""
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append({"url": url, **kwargs})
        return _FakeResponse(status_code=200)

    monkeypatch.setattr(
        "src.research.notifications.httpx.post", fake_post,
    )
    return calls


# --------------------------------------------------------------------- #
# Env-var gating
# --------------------------------------------------------------------- #


class TestGating:
    def test_no_env_vars_no_dispatch(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        for v in (
            "PUSHOVER_API_TOKEN", "PUSHOVER_USER_KEY",
            "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        ):
            monkeypatch.delenv(v, raising=False)
        result = notify_operator(title="t", message="m")
        assert not result.any_attempted
        assert http_recorder == []

    def test_pushover_only(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        monkeypatch.setenv("PUSHOVER_API_TOKEN", "tok")
        monkeypatch.setenv("PUSHOVER_USER_KEY", "user")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        result = notify_operator(title="GATE 1", message="hi", priority=1)
        assert result.pushover_attempted and result.pushover_succeeded
        assert not result.telegram_attempted
        assert len(http_recorder) == 1
        call = http_recorder[0]
        assert call["url"] == "https://api.pushover.net/1/messages.json"
        assert call["data"]["token"] == "tok"
        assert call["data"]["user"] == "user"
        assert call["data"]["title"] == "GATE 1"
        assert call["data"]["priority"] == 1

    def test_telegram_only(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        monkeypatch.delenv("PUSHOVER_API_TOKEN", raising=False)
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "btok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
        result = notify_operator(title="GATE 1", message="hi")
        assert result.telegram_attempted and result.telegram_succeeded
        assert not result.pushover_attempted
        assert len(http_recorder) == 1
        call = http_recorder[0]
        assert call["url"] == "https://api.telegram.org/botbtok/sendMessage"
        assert call["data"]["chat_id"] == "12345"
        # Title bold-prefixed in body
        assert "*GATE 1*" in call["data"]["text"]

    def test_both_channels(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        monkeypatch.setenv("PUSHOVER_API_TOKEN", "tok")
        monkeypatch.setenv("PUSHOVER_USER_KEY", "user")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "btok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
        result = notify_operator(title="t", message="m")
        assert result.pushover_attempted and result.telegram_attempted
        assert len(http_recorder) == 2


class TestErrorContainment:
    def test_pushover_failure_doesnt_block_telegram(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PUSHOVER_API_TOKEN", "tok")
        monkeypatch.setenv("PUSHOVER_USER_KEY", "user")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "btok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

        call_count = [0]

        def split(url: str, **kwargs: Any) -> _FakeResponse:
            call_count[0] += 1
            if "pushover" in url:
                raise ConnectionError("pushover down")
            return _FakeResponse(status_code=200)

        monkeypatch.setattr(
            "src.research.notifications.httpx.post", split,
        )
        result = notify_operator(title="t", message="m")
        assert result.pushover_attempted
        assert not result.pushover_succeeded
        assert "ConnectionError" in result.pushover_error
        # Telegram still ran
        assert result.telegram_attempted and result.telegram_succeeded


class TestDispatchResult:
    def test_any_succeeded_prop(self) -> None:
        r = DispatchResult()
        assert not r.any_succeeded
        r.pushover_succeeded = True
        assert r.any_succeeded

    def test_any_attempted_prop(self) -> None:
        r = DispatchResult()
        assert not r.any_attempted
        r.telegram_attempted = True
        assert r.any_attempted
