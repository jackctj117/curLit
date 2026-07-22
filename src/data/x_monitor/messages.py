"""Telegram notification/message building for the X monitor (CL-3j86).

Split out of the former ``src/data/x_monitor.py`` monofile (CL-ikz2,
structural review 2026-07-21 §6.2.4). See the package ``__init__``
docstring for the full monitor overview.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from src.data.x_monitor.transport import Post
from src.data.x_monitor.watchlist import CATEGORY_FOOTERS, WatchAccount
from src.research.notifications import html_escape

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #

NOTIFY_TITLE = "X watchlist"

#: Max posts listed inside one combined (batched) message.
_BATCH_MAX_LISTED = 10


# --------------------------------------------------------------------- #
# Notification formatting
# --------------------------------------------------------------------- #


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _post_url(handle: str, post_id: str) -> str:
    return f"https://x.com/{handle}/status/{post_id}"


def build_post_message(
    account: WatchAccount,
    post: Post,
    text_limit: int = 500,
) -> str:
    """Telegram-HTML body for one new post. All interpolated content
    (tweet text is hostile input) goes through html_escape."""
    lines = [
        f"<b>@{html_escape(account.handle)}</b> "
        f"[{html_escape(account.category)}]",
    ]
    if post.text.strip():
        lines.append(html_escape(_truncate(post.text, text_limit)))
    lines.append(_post_url(account.handle, post.id))
    footer = CATEGORY_FOOTERS.get(account.category)
    if footer:
        lines.append(f"<i>{footer}</i>")
    return "\n".join(lines)


def build_batch_message(
    account: WatchAccount,
    posts: Sequence[Post],
    text_limit: int = 200,
) -> str:
    """One combined Telegram-HTML body for a burst of posts from a
    single account (>batch_threshold new posts in one cycle)."""
    lines = [
        f"<b>@{html_escape(account.handle)}</b> "
        f"[{html_escape(account.category)}] — {len(posts)} new posts",
    ]
    for post in posts[:_BATCH_MAX_LISTED]:
        text = _truncate(post.text, text_limit)
        if text:
            lines.append(f"• {html_escape(text)}")
        lines.append(_post_url(account.handle, post.id))
    if len(posts) > _BATCH_MAX_LISTED:
        lines.append(f"(+{len(posts) - _BATCH_MAX_LISTED} more)")
    footer = CATEGORY_FOOTERS.get(account.category)
    if footer:
        lines.append(f"<i>{footer}</i>")
    return "\n".join(lines)
