"""Notification dispatch for the research pipeline (CL-0hr3, CL-yta6).

Two channels, both optional and env-var gated:

  * **Pushover**  — push alert to operator's phone via api.pushover.net.
                    Requires ``PUSHOVER_API_TOKEN`` + ``PUSHOVER_USER_KEY``.
                    Same channel ``scripts/run_edge_tests.py`` uses;
                    that pattern is mirrored here.
  * **Telegram**  — message to operator's bot chat. Requires
                    ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID``.

Either channel being unconfigured is a no-op for that channel — the
dispatcher logs that it was skipped and continues. This means dev
machines without either env-var run the loop end-to-end without
needing to wire anything; production sets the env-vars and gets
alerts.

Returned ``DispatchResult`` is purely informational — the loop
records whether each channel attempted to send so the per-run summary
can show "alerts dispatched" / "alerts skipped (env unset)" without
the loop having to care which channel.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


# HTTP timeout for notification dispatch. Both APIs are fast; this is
# a cap on how long the loop will block on a stuck request.
_HTTP_TIMEOUT_SEC: float = 10.0


@dataclass
class DispatchResult:
    """Per-channel send outcome."""

    pushover_attempted: bool = False
    pushover_succeeded: bool = False
    pushover_error: str = ""
    telegram_attempted: bool = False
    telegram_succeeded: bool = False
    telegram_error: str = ""

    @property
    def any_succeeded(self) -> bool:
        return self.pushover_succeeded or self.telegram_succeeded

    @property
    def any_attempted(self) -> bool:
        return self.pushover_attempted or self.telegram_attempted


def notify_operator(
    title: str,
    message: str,
    priority: int = 0,
) -> DispatchResult:
    """Send the same payload to every configured channel. Each channel
    runs independently; one failing doesn't block the other.

    ``priority``: Pushover priority (0=normal, 1=high, 2=emergency).
    Telegram has no notion of priority; we ignore it for that channel.
    """
    result = DispatchResult()
    _dispatch_pushover(title, message, priority, result)
    _dispatch_telegram(title, message, result)
    return result


def _dispatch_pushover(
    title: str, message: str, priority: int, result: DispatchResult,
) -> None:
    token = os.environ.get("PUSHOVER_API_TOKEN")
    user = os.environ.get("PUSHOVER_USER_KEY")
    if not token or not user:
        logger.info(
            "Pushover not configured (PUSHOVER_API_TOKEN / "
            "PUSHOVER_USER_KEY missing); skipping channel",
        )
        return
    result.pushover_attempted = True
    try:
        resp = httpx.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token,
                "user": user,
                "title": title,
                "message": message,
                "priority": priority,
            },
            timeout=_HTTP_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        result.pushover_succeeded = True
    except Exception as exc:
        logger.warning(
            "Pushover dispatch failed: %s: %s", type(exc).__name__, exc,
        )
        result.pushover_error = f"{type(exc).__name__}: {exc}"


def _dispatch_telegram(
    title: str, message: str, result: DispatchResult,
) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.info(
            "Telegram not configured (TELEGRAM_BOT_TOKEN / "
            "TELEGRAM_CHAT_ID missing); skipping channel",
        )
        return
    result.telegram_attempted = True
    # Plain text — no parse_mode. Markdown was a 400-magnet because
    # underscores in slugs (e.g. ``regime_carry_underscores``) are
    # italic toggles that don't pair up cleanly. HTML would also need
    # careful escaping. Plain text is the smallest safe surface.
    body = f"{title}\n\n{message}"
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": body,
            },
            timeout=_HTTP_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        result.telegram_succeeded = True
    except Exception as exc:
        # Scrub the bot token from any error message — httpx echoes
        # the request URL on HTTPStatusError and that contains the
        # token. Replace with a sentinel so logs + DispatchResult
        # don't leak credentials.
        sanitized = _scrub_token(str(exc), token)
        logger.warning(
            "Telegram dispatch failed: %s: %s", type(exc).__name__, sanitized,
        )
        result.telegram_error = f"{type(exc).__name__}: {sanitized}"


def _scrub_token(text: str, token: str) -> str:
    """Replace any occurrence of ``token`` in ``text`` with ``[REDACTED]``.
    Used to prevent the Telegram bot token from leaking into error
    messages / logs / DispatchResult fields when an HTTP exception
    echoes the request URL."""
    if not token:
        return text
    return text.replace(token, "[REDACTED]")
