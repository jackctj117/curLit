"""X (Twitter) account watchlist monitor (CL-3j86).

Polls a curated watchlist (``configs/x_watchlist.yaml``) and relays
every NEW post to the operator's Telegram chat via
:func:`src.research.notifications.notify_operator`.

Ships DARK. The default backend is the official X API v2, which is a
paid product (basic tier) — until ``TWITTER_BEARER_TOKEN`` is
provisioned the monitor logs one clear idle line and does nothing.
It NEVER falls back to scraping: Nitter is dead, and scraping x.com
(or driving it with borrowed session cookies) violates the X ToS and
risks a ban on the account involved. If the operator wants posts
without buying the API, the CLI transport (below) exists — with the
risks stated loudly in its docstring — but it too ships dark.

Transports (selected via ``X_MONITOR_BACKEND`` = ``api`` | ``cli``):

* :class:`ApiTransport` — official v2 ``GET /2/users/:id/tweets`` with
  ``since_id``, reply/retweet exclusion, and a monthly READ-BUDGET
  GOVERNOR (default 9000 reads/month, ``X_MONITOR_MONTHLY_CAP``).
  Handle → user-id resolution happens once via ``GET /2/users/by`` and
  is cached to ``data/x_user_ids.json``.
* :class:`CliTransport` — runs an external, operator-supplied command
  per account (``X_MONITOR_CLI_CMD``) and parses a JSON array of posts
  from its stdout. See its docstring for the ToS / ban-risk honesty
  notes. No metered reads, so the budget governor does not apply;
  cadence tiers still do (politeness / detection-surface reduction).

Poll cadence by priority: ``high`` accounts every cycle, ``normal``
every 2nd, ``low`` every 4th. With the shipped watchlist that averages
~19 metered reads per cycle; the governor paces cycles so the monthly
cap survives the whole month (~90 min effective interval at defaults)
and HARD-STOPS polling — with a single Telegram warning — if the cap
is ever exhausted early.

Active-hours windows (CL-6t7v): an account may carry an OPTIONAL,
timezone-aware ``active_hours`` window; the monitor then only polls it
inside that window (see :func:`is_within_window`). This composes with
the cadence gate — both must pass. Accounts with no window are always-on
(the default), which is deliberate for breaking-news / flow / OSINT /
mining accounts. Only day-recap trader accounts get EOD windows.

State (``data/x_monitor_state.json``, atomic tmp+replace writes, same
pattern as ``src.research.telegram_approvals.save_offset``): per-handle
``since_id``, cycle counter, month spend, rate-limit backoff, and
per-account failure cooldowns. The first poll of an account is a
BASELINE — it records the newest post id without notifying, so a fresh
deploy doesn't spam the operator with old posts.

Package layout (decomposed from a single-module monofile, CL-ikz2 /
structural review 2026-07-21 §6.2.4 — no behavior change):

* :mod:`~src.data.x_monitor.watchlist` — accounts, categories, cadence
  tiers, active-hours windowing (CL-6t7v).
* :mod:`~src.data.x_monitor.transport` — :class:`ApiTransport`,
  :class:`CliTransport`, :func:`build_transport`, metered-read plumbing.
* :mod:`~src.data.x_monitor.messages` — Telegram notification bodies.
* :mod:`~src.data.x_monitor.monitor` — :class:`XWatchlistMonitor`,
  :class:`XMonitorConfig`, :class:`CycleSummary`, the poll cycle.

The entire public surface is re-exported here, so
``from src.data.x_monitor import X`` keeps working unchanged.
"""

from __future__ import annotations

from src.data.x_monitor.messages import (
    NOTIFY_TITLE,
    build_batch_message,
    build_post_message,
)
from src.data.x_monitor.monitor import (
    DEFAULT_STATE_PATH,
    CycleSummary,
    NotifyFn,
    XMonitorConfig,
    XWatchlistMonitor,
)
from src.data.x_monitor.transport import (
    DEFAULT_MONTHLY_CAP,
    DEFAULT_USER_IDS_PATH,
    IDLE_LINE_API,
    IDLE_LINE_CLI,
    X_API_BASE,
    ApiTransport,
    CliTransport,
    Post,
    RateLimitedError,
    Transport,
    TransportError,
    XApiGet,
    XApiResponse,
    build_transport,
)
from src.data.x_monitor.watchlist import (
    CADENCE,
    CATEGORY_FOOTERS,
    DEFAULT_WATCHLIST_PATH,
    KNOWN_CATEGORIES,
    ActiveHours,
    WatchAccount,
    is_within_window,
    load_watchlist,
)

__all__ = [
    "CADENCE",
    "CATEGORY_FOOTERS",
    "DEFAULT_MONTHLY_CAP",
    "DEFAULT_STATE_PATH",
    "DEFAULT_USER_IDS_PATH",
    "DEFAULT_WATCHLIST_PATH",
    "IDLE_LINE_API",
    "IDLE_LINE_CLI",
    "KNOWN_CATEGORIES",
    "NOTIFY_TITLE",
    "X_API_BASE",
    "ActiveHours",
    "ApiTransport",
    "CliTransport",
    "CycleSummary",
    "NotifyFn",
    "Post",
    "RateLimitedError",
    "Transport",
    "TransportError",
    "WatchAccount",
    "XApiGet",
    "XApiResponse",
    "XMonitorConfig",
    "XWatchlistMonitor",
    "build_batch_message",
    "build_post_message",
    "build_transport",
    "is_within_window",
    "load_watchlist",
]
