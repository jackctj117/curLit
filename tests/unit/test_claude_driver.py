"""Tests for the Claude driver's temperature-deprecation retry logic
(CL-xo0t).

Some Anthropic models (claude-opus-4-7 and reasoning variants) reject
the ``temperature`` request param with a 400 "temperature is
deprecated for this model" error. The driver catches that specific
error and retries the request without the param. Other API errors
propagate unchanged.

Tests mock the Anthropic SDK client to assert: (a) temperature is
included on the first call, (b) on the deprecation error the second
call drops it, (c) other errors don't trigger the retry.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from src.research.llm.client import Message, _ClaudeDriver


def _fake_response(text: str = "ok") -> Any:
    """Build an object that quacks like an Anthropic ``Messages.create``
    response — content blocks + usage."""
    text_block = MagicMock()
    text_block.text = text
    resp = MagicMock()
    resp.content = [text_block]
    resp.usage = MagicMock(input_tokens=10, output_tokens=20)
    return resp


def _build_driver_with_mock_client() -> tuple[_ClaudeDriver, MagicMock]:
    driver = _ClaudeDriver(api_key="fake")
    mock_client = MagicMock()
    driver._client = mock_client
    return driver, mock_client


class TestTemperatureDeprecationRetry:
    def test_first_call_includes_temperature(self) -> None:
        driver, mock_client = _build_driver_with_mock_client()
        mock_client.messages.create.return_value = _fake_response()

        driver.complete(
            messages=[Message(role="user", content="hi")],
            model="claude-sonnet-4-6",
            max_tokens=100,
            temperature=0.0,
        )
        assert mock_client.messages.create.call_count == 1
        kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" in kwargs
        assert kwargs["temperature"] == 0.0

    def test_retries_without_temperature_on_deprecation(self) -> None:
        driver, mock_client = _build_driver_with_mock_client()

        # First call: raise the canonical deprecation error.
        # Second call: succeed.
        deprecation_error = Exception(
            "Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', "
            "'message': '`temperature` is deprecated for this model.'}}",
        )
        mock_client.messages.create.side_effect = [
            deprecation_error,
            _fake_response("retried"),
        ]

        resp = driver.complete(
            messages=[Message(role="user", content="hi")],
            model="claude-opus-4-7",
            max_tokens=100,
            temperature=0.0,
        )
        assert resp.text == "retried"
        assert mock_client.messages.create.call_count == 2
        # First call included temperature
        first_kwargs = mock_client.messages.create.call_args_list[0].kwargs
        assert "temperature" in first_kwargs
        # Second (retry) did NOT include temperature
        second_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "temperature" not in second_kwargs

    def test_other_errors_propagate(self) -> None:
        driver, mock_client = _build_driver_with_mock_client()
        unrelated_error = Exception(
            "Error code: 429 - rate_limit_error: too many requests",
        )
        mock_client.messages.create.side_effect = unrelated_error

        try:
            driver.complete(
                messages=[Message(role="user", content="hi")],
                model="claude-sonnet-4-6",
                max_tokens=100,
                temperature=0.0,
            )
        except Exception as exc:
            assert "rate_limit" in str(exc)
        else:
            raise AssertionError("expected rate_limit error to propagate")

        # No retry should have happened
        assert mock_client.messages.create.call_count == 1
