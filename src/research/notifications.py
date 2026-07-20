"""Notification dispatch for the research pipeline (CL-0hr3, CL-yta6).

One channel, optional and env-var gated (CL-b92t removed the old
Pushover channel — Telegram is the sole operator channel now):

  * **Telegram**  — message to operator's bot chat. Requires
                    ``TELEGRAM_BOT_TOKEN`` + ``TELEGRAM_CHAT_ID``.

The channel being unconfigured is a no-op — the dispatcher logs that
it was skipped and continues. This means dev machines without the
env-vars run the loop end-to-end without needing to wire anything;
production sets the env-vars and gets alerts.

Formatting (CL-frn7): callers may pass ``html=True`` with a message
containing Telegram-HTML tags (``<b>``, ``<i>``, ``<code>``). Telegram
then gets ``parse_mode=HTML``. HTML was chosen over Markdown because
Markdown made underscores in slugs (``regime_carry``) unpaired italic
toggles → 400s; in HTML mode underscores are literal and only ``&``,
``<``, ``>`` need escaping — use :func:`html_escape` on every piece of
interpolated content (slugs, headlines, reasons).

Returned ``DispatchResult`` is purely informational — the loop
records whether the channel attempted to send so the per-run summary
can show "alerts dispatched" / "alerts skipped (env unset)".
"""

from __future__ import annotations

import html as _html
import logging
import os
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Silence httpx's INFO-level "HTTP Request: POST <url>" logging at
# import time (CL-wmn4). The Telegram URL contains the bot token —
# logging it on every successful send leaks the credential into any
# log aggregator (Loki, Cloudwatch, journald). httpx's WARNING and
# above still surface; if a request fails, the dispatcher's own
# exception handler logs it via _scrub_token.
logging.getLogger("httpx").setLevel(logging.WARNING)


# HTTP timeout for notification dispatch. The Telegram API is fast;
# this is a cap on how long the loop will block on a stuck request.
_HTTP_TIMEOUT_SEC: float = 10.0


# --------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------- #


def html_escape(text: str) -> str:
    """Escape interpolated content for a Telegram-HTML message body.

    Telegram's HTML parse mode only requires ``&``, ``<``, ``>`` to be
    escaped; quotes are left alone for readability. EVERY dynamic value
    (slug, headline, reason, path) MUST pass through this before being
    embedded next to formatting tags.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")


def _html_to_plain(text: str) -> str:
    """Telegram-HTML body → plain text (used for the Telegram
    plain-text fallback): strip tags, unescape entities."""
    return _html.unescape(_TAG_RE.sub("", text))


@dataclass
class DispatchResult:
    """Send outcome for the Telegram channel."""

    telegram_attempted: bool = False
    telegram_succeeded: bool = False
    telegram_error: str = ""

    @property
    def any_succeeded(self) -> bool:
        return self.telegram_succeeded

    @property
    def any_attempted(self) -> bool:
        return self.telegram_attempted


def notify_operator(
    title: str,
    message: str,
    priority: int = 0,  # kept for caller compatibility; ignored
    *,
    html: bool = False,
) -> DispatchResult:
    """Send ``title`` + ``message`` to the operator's Telegram chat.

    ``priority``: accepted for backwards compatibility with callers
    written against the old two-channel (Pushover) signature; Telegram
    has no notion of priority, so it is ignored.

    ``html=True``: ``message`` contains Telegram-HTML tags and its
    interpolated content is ALREADY escaped via :func:`html_escape`.
    Telegram renders it with ``parse_mode=HTML`` under a bolded
    (escaped) ``title`` header. ``html=False`` keeps the original
    plain-text behavior for existing callers.
    """
    result = DispatchResult()
    _dispatch_telegram(title, message, result, html=html)
    return result


def _dispatch_telegram(
    title: str,
    message: str,
    result: DispatchResult,
    html: bool = False,
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
    # HTML mode (CL-frn7): Markdown was a 400-magnet because
    # underscores in slugs (e.g. ``regime_carry_underscores``) are
    # italic toggles that don't pair up cleanly. In HTML mode
    # underscores are literal — only &/</> need escaping, which the
    # message builders do via html_escape(). Legacy callers (html=False)
    # keep plain text: smallest safe surface, zero escaping needed.
    if html:
        payload = {
            "chat_id": chat_id,
            "text": f"<b>{html_escape(title)}</b>\n\n{message}",
            "parse_mode": "HTML",
        }
    else:
        payload = {"chat_id": chat_id, "text": f"{title}\n\n{message}"}
    try:
        _telegram_send(token, payload)
        result.telegram_succeeded = True
    except Exception as exc:
        # An HTML body with a broken tag gets a 400 from Telegram.
        # Never let a formatting bug silence a gate alert: fall back
        # to a tag-stripped plain-text send once.
        if html:
            logger.warning(
                "Telegram HTML dispatch failed (%s: %s); retrying as "
                "plain text",
                type(exc).__name__, _scrub_token(str(exc), token),
            )
            try:
                _telegram_send(token, {
                    "chat_id": chat_id,
                    "text": f"{title}\n\n{_html_to_plain(message)}",
                })
                result.telegram_succeeded = True
                return
            except Exception as retry_exc:
                exc = retry_exc
        # Scrub the bot token from any error message — httpx echoes
        # the request URL on HTTPStatusError and that contains the
        # token. Replace with a sentinel so logs + DispatchResult
        # don't leak credentials.
        sanitized = _scrub_token(str(exc), token)
        logger.warning(
            "Telegram dispatch failed: %s: %s", type(exc).__name__, sanitized,
        )
        result.telegram_error = f"{type(exc).__name__}: {sanitized}"


def _telegram_send(token: str, payload: dict[str, str]) -> None:
    """One sendMessage POST; raises on transport or HTTP error."""
    resp = httpx.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        timeout=_HTTP_TIMEOUT_SEC,
    )
    resp.raise_for_status()


def _scrub_token(text: str, token: str) -> str:
    """Replace any occurrence of ``token`` in ``text`` with ``[REDACTED]``.
    Used to prevent the Telegram bot token from leaking into error
    messages / logs / DispatchResult fields when an HTTP exception
    echoes the request URL."""
    if not token:
        return text
    return text.replace(token, "[REDACTED]")
