"""Unit tests for notification dispatch (CL-0hr3 / CL-yta6 / CL-b92t).

Tests cover env-var gating + dispatch routing without making any real
network calls. We monkeypatch httpx.post on the notifications module
to record what would have been sent.

CL-b92t removed the Pushover channel entirely — Telegram is the sole
operator channel. ``TestPushoverRemoved`` greps the source tree to
keep it that way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from src.research.notifications import (
    DispatchResult,
    _html_to_plain,
    html_escape,
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


def _set_telegram_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "btok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")


# --------------------------------------------------------------------- #
# Pushover is gone (CL-b92t)
# --------------------------------------------------------------------- #


class TestPushoverRemoved:
    _ROOT = Path(__file__).resolve().parents[2]

    def test_no_pushover_env_reads_in_source_tree(self) -> None:
        """Grep-style guard: no PUSHOVER_* env var is read (or even
        named) anywhere under src/ or scripts/. The old channel must
        not creep back in."""
        offenders: list[str] = []
        for tree in ("src", "scripts"):
            for path in (self._ROOT / tree).rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                if "PUSHOVER_" in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(self._ROOT)))
        assert offenders == [], (
            f"PUSHOVER_* env vars referenced in: {offenders}"
        )

    def test_notifications_module_has_no_pushover_symbols(self) -> None:
        import src.research.notifications as mod

        assert not any("pushover" in name.lower() for name in vars(mod))

    def test_dispatch_result_has_no_pushover_fields(self) -> None:
        r = DispatchResult()
        assert not hasattr(r, "pushover_attempted")
        assert not hasattr(r, "pushover_succeeded")
        assert not hasattr(r, "pushover_error")

    def test_pushover_env_alone_dispatches_nothing(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        """Even with the legacy env vars set, no HTTP call happens —
        the channel is gone, not just gated off."""
        monkeypatch.setenv("PUSHOVER_API_TOKEN", "tok")
        monkeypatch.setenv("PUSHOVER_USER_KEY", "user")
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        result = notify_operator(title="t", message="m", priority=1)
        assert not result.any_attempted
        assert http_recorder == []


# --------------------------------------------------------------------- #
# Env-var gating
# --------------------------------------------------------------------- #


class TestGating:
    def test_no_env_vars_no_dispatch(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        result = notify_operator(title="t", message="m")
        assert not result.any_attempted
        assert http_recorder == []

    def test_telegram_configured_dispatches(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        _set_telegram_env(monkeypatch)
        result = notify_operator(title="GATE 1", message="hi")
        assert result.telegram_attempted and result.telegram_succeeded
        assert len(http_recorder) == 1
        call = http_recorder[0]
        assert call["url"] == "https://api.telegram.org/botbtok/sendMessage"
        assert call["data"]["chat_id"] == "12345"
        # Plain text — title prefixed, no Markdown formatting.
        # Markdown was a 400-magnet on slugs containing underscores.
        assert call["data"]["text"] == "GATE 1\n\nhi"
        assert "parse_mode" not in call["data"]

    def test_priority_accepted_but_ignored(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        """Legacy callers still pass priority (an old Pushover concept);
        the Telegram payload must not grow a priority field."""
        _set_telegram_env(monkeypatch)
        result = notify_operator(title="t", message="m", priority=2)
        assert result.telegram_succeeded
        assert "priority" not in http_recorder[0]["data"]

    def test_telegram_handles_underscores_in_message(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        """Smoke regression: an underscore-laden slug used to trigger a
        Telegram 400 because Markdown parse_mode interpreted ``_..._``
        as italic. Plain-text dispatch must accept this body cleanly."""
        _set_telegram_env(monkeypatch)
        result = notify_operator(
            title="GATE 1",
            message="Slug: regime_carry_underscores_in_slug\nNext line",
        )
        assert result.telegram_succeeded
        assert "regime_carry_underscores_in_slug" in (
            http_recorder[0]["data"]["text"]
        )

    def test_partial_env_no_dispatch(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "btok")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        result = notify_operator(title="t", message="m")
        assert not result.any_attempted
        assert http_recorder == []


class TestHtmlHelpers:
    def test_html_escape_hostile_text(self) -> None:
        # Headlines/slugs/reasons are interpolated into HTML bodies —
        # <, & and > must be escaped; underscores stay literal (the
        # whole reason HTML replaced Markdown, CL-frn7).
        assert html_escape("<b>x & y</b> a_slug_") == (
            "&lt;b&gt;x &amp; y&lt;/b&gt; a_slug_"
        )

    def test_html_escape_non_str(self) -> None:
        assert html_escape(42) == "42"

    def test_html_to_plain_strips_tags_and_unescapes(self) -> None:
        assert _html_to_plain(
            "<b>Trades:</b> EURUSD &amp; DXY\n<i>x &lt; y</i>",
        ) == "Trades: EURUSD & DXY\nx < y"


class TestHtmlMode:
    def test_telegram_gets_html_parse_mode_and_bold_title(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        _set_telegram_env(monkeypatch)
        result = notify_operator(
            title="GATE 1 — new trading hypothesis",
            message="<b>Trades:</b> EURUSD",
            html=True,
        )
        assert result.telegram_succeeded
        telegram = http_recorder[0]
        assert telegram["data"]["parse_mode"] == "HTML"
        assert telegram["data"]["text"] == (
            "<b>GATE 1 — new trading hypothesis</b>\n\n"
            "<b>Trades:</b> EURUSD"
        )

    def test_hostile_title_escaped_for_telegram(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        _set_telegram_env(monkeypatch)
        notify_operator(
            title="Paper <Q3> says risk & carry_trade", message="m", html=True,
        )
        assert (
            "<b>Paper &lt;Q3&gt; says risk &amp; carry_trade</b>"
            in http_recorder[0]["data"]["text"]
        )

    def test_html_400_falls_back_to_plain_text(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A malformed-tag 400 must never silence a gate alert: the
        dispatcher retries once with tags stripped, no parse_mode."""
        _set_telegram_env(monkeypatch)

        calls: list[dict[str, Any]] = []

        def flaky_post(url: str, **kwargs: Any) -> _FakeResponse:
            calls.append({"url": url, **kwargs})
            if kwargs["data"].get("parse_mode") == "HTML":
                return _FakeResponse(status_code=400)
            return _FakeResponse(status_code=200)

        monkeypatch.setattr(
            "src.research.notifications.httpx.post", flaky_post,
        )
        result = notify_operator(
            title="t", message="<b>broken<b> tags", html=True,
        )
        assert result.telegram_succeeded
        assert len(calls) == 2
        assert "parse_mode" not in calls[1]["data"]
        assert calls[1]["data"]["text"] == "t\n\nbroken tags"

    def test_plain_mode_unchanged_for_legacy_callers(
        self, monkeypatch: pytest.MonkeyPatch, http_recorder: list[Any],
    ) -> None:
        _set_telegram_env(monkeypatch)
        notify_operator(title="t", message="a_b <raw>")
        telegram = http_recorder[0]
        assert telegram["data"]["text"] == "t\n\na_b <raw>"
        assert "parse_mode" not in telegram["data"]


class TestFailureHandling:
    def test_transport_error_recorded_not_raised(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_telegram_env(monkeypatch)

        def boom(url: str, **kwargs: Any) -> _FakeResponse:
            raise ConnectionError("telegram down")

        monkeypatch.setattr(
            "src.research.notifications.httpx.post", boom,
        )
        result = notify_operator(title="t", message="m")
        assert result.telegram_attempted
        assert not result.telegram_succeeded
        assert "ConnectionError" in result.telegram_error
        assert not result.any_succeeded

    def test_http_400_recorded_in_result(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _set_telegram_env(monkeypatch)

        def bad(url: str, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse(status_code=400)

        monkeypatch.setattr(
            "src.research.notifications.httpx.post", bad,
        )
        result = notify_operator(title="t", message="m")
        assert result.telegram_attempted
        assert not result.telegram_succeeded
        assert "HTTPStatusError" in result.telegram_error


class TestHttpxLogSuppression:
    def test_httpx_info_silenced_at_import(self) -> None:
        """CL-wmn4: import-time setup pins httpx's logger to WARNING
        so successful requests don't log the URL (which contains the
        Telegram bot token in our case)."""
        # Re-import to be defensive against test-ordering effects
        import importlib
        import logging as _logging

        from src.research import notifications
        importlib.reload(notifications)
        assert _logging.getLogger("httpx").level == _logging.WARNING


class TestTokenScrubbing:
    def test_telegram_token_redacted_from_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """httpx echoes the request URL on HTTPStatusError, and that
        URL contains the bot token. Verify the dispatcher scrubs it
        before the token lands in DispatchResult.telegram_error."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-bot-token-xyz")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

        def boom_with_token_in_message(_url: str, **_kwargs: Any) -> Any:
            # Simulate httpx including the token-bearing URL in the
            # exception text (this is what real HTTPStatusError does).
            raise ConnectionError(
                "request to /botsecret-bot-token-xyz/sendMessage failed",
            )

        monkeypatch.setattr(
            "src.research.notifications.httpx.post",
            boom_with_token_in_message,
        )
        result = notify_operator(title="t", message="m")
        assert result.telegram_attempted
        assert not result.telegram_succeeded
        assert "secret-bot-token-xyz" not in result.telegram_error
        assert "[REDACTED]" in result.telegram_error


class TestDispatchResult:
    def test_any_succeeded_prop(self) -> None:
        r = DispatchResult()
        assert not r.any_succeeded
        r.telegram_succeeded = True
        assert r.any_succeeded

    def test_any_attempted_prop(self) -> None:
        r = DispatchResult()
        assert not r.any_attempted
        r.telegram_attempted = True
        assert r.any_attempted
