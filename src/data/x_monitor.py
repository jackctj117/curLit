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
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import re
import shlex
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import yaml

from src.research.notifications import html_escape, notify_operator

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #

DEFAULT_WATCHLIST_PATH = Path("configs/x_watchlist.yaml")
DEFAULT_STATE_PATH = Path("data/x_monitor_state.json")
DEFAULT_USER_IDS_PATH = Path("data/x_user_ids.json")

DEFAULT_MONTHLY_CAP = 9_000

X_API_BASE = "https://api.twitter.com/2"

#: The one clear line logged when the API backend has no token.
IDLE_LINE_API = (
    "X monitor idle: set TWITTER_BEARER_TOKEN — X API basic tier required"
)

#: The one clear line logged when the CLI backend has no command.
IDLE_LINE_CLI = (
    "X monitor idle: set X_MONITOR_CLI_CMD — cookie-based CLI tools "
    "violate X ToS and risk banning the account whose cookies they use; "
    "use a burner account"
)

#: Known watchlist categories. Adding a category is a deliberate act:
#: extend this set AND decide its footer policy in CATEGORY_FOOTERS.
KNOWN_CATEGORIES = frozenset(
    {"financial_flow", "conflict_osint", "africa_mining", "small_traders"},
)

#: Categories whose notifications carry a one-line honesty footer.
CATEGORY_FOOTERS: Mapping[str, str] = {
    "conflict_osint": "unverified — cross-check",
    "small_traders": "unverified performance claims",
}

#: Poll cadence per priority tier: poll when cycle % interval == 0.
CADENCE: Mapping[str, int] = {"high": 1, "normal": 2, "low": 4}

NOTIFY_TITLE = "X watchlist"

_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")

#: Max posts listed inside one combined (batched) message.
_BATCH_MAX_LISTED = 10

#: Cap on per-account failure cooldown (in cycles).
_MAX_FAILURE_COOLDOWN_CYCLES = 16

#: Weekday name → Python weekday() index (Mon=0 .. Sun=6). Accepts full
#: names and 3-letter abbreviations, case-insensitive.
_WEEKDAY_INDEX: Mapping[str, int] = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

#: Convenience day-set keywords for active_hours.days.
_WEEKDAYS = frozenset({0, 1, 2, 3, 4})  # Mon-Fri
_ALL_DAYS = frozenset(range(7))


# --------------------------------------------------------------------- #
# Watchlist config
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActiveHours:
    """An optional per-account polling window (CL-6t7v).

    When an account carries active_hours the monitor only polls it while
    "now" (converted to ``tz``) falls in ``[start, end)`` on an allowed
    weekday. A window is timezone-aware (``tz`` is an IANA name, so DST
    is handled by :mod:`zoneinfo`) and may cross midnight: when
    ``start > end`` the window is treated as overnight and the weekday
    check is applied to the day on which the window *starts*.

    KEY DESIGN: this is opt-in. Accounts with NO active_hours poll
    all-day (the always-on default) — that is deliberate for breaking-
    news / flow / OSINT / mining accounts. Only day-recap trader
    accounts get EOD windows. All fields here are already validated by
    :func:`load_watchlist`.
    """

    start: dt_time
    end: dt_time
    tz: ZoneInfo
    #: Allowed weekday indices (Mon=0 .. Sun=6). Default = every day.
    days: frozenset[int] = _ALL_DAYS

    @property
    def crosses_midnight(self) -> bool:
        return self.start > self.end


@dataclass(frozen=True)
class WatchAccount:
    """One watched X account from configs/x_watchlist.yaml."""

    handle: str
    category: str
    note: str
    priority: str  # high | normal | low
    keywords: tuple[str, ...] = ()  # empty = notify on all posts
    #: OPTIONAL polling window; None = always-on (all-day, the default).
    active_hours: ActiveHours | None = None


def _parse_hhmm(value: Any, ctx: str) -> dt_time:
    """Parse an 'HH:MM' string into a time. Fails loud (CL-6t7v)."""
    text = str(value).strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not m:
        raise ValueError(
            f"{ctx}: active_hours time {value!r} is not 'HH:MM'",
        )
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(
            f"{ctx}: active_hours time {value!r} out of range (00:00..23:59)",
        )
    return dt_time(hour=hh, minute=mm)


def _parse_days(value: Any, ctx: str) -> frozenset[int]:
    """Parse active_hours.days into a set of weekday indices (Mon=0).

    Accepts the keywords 'all' / 'weekdays', or a list of weekday names
    (full or 3-letter abbreviations). Fails loud on anything else.
    """
    if value is None:
        return _ALL_DAYS
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "all":
            return _ALL_DAYS
        if key in ("weekdays", "weekday"):
            return _WEEKDAYS
        # A bare day name is allowed as a convenience.
        if key in _WEEKDAY_INDEX:
            return frozenset({_WEEKDAY_INDEX[key]})
        raise ValueError(
            f"{ctx}: active_hours.days {value!r} unknown "
            "(use 'all', 'weekdays', or a list of weekday names)",
        )
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{ctx}: active_hours.days list is empty")
        days: set[int] = set()
        for item in value:
            key = str(item).strip().lower()
            if key not in _WEEKDAY_INDEX:
                raise ValueError(
                    f"{ctx}: active_hours.days entry {item!r} is not a "
                    "weekday name",
                )
            days.add(_WEEKDAY_INDEX[key])
        return frozenset(days)
    raise ValueError(
        f"{ctx}: active_hours.days must be a string keyword or a list",
    )


def _parse_active_hours(raw: Any, ctx: str) -> ActiveHours | None:
    """Parse the optional active_hours block for one account (CL-6t7v).

    Returns None when unset (always-on). Fails loud on a bad tz, a bad
    time format, or a missing start/end — an operator typo must not
    silently widen or narrow a window.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx}: active_hours must be a mapping")
    if "start" not in raw or "end" not in raw:
        raise ValueError(f"{ctx}: active_hours needs both 'start' and 'end'")
    if "tz" not in raw:
        raise ValueError(
            f"{ctx}: active_hours needs 'tz' (an IANA name, "
            "e.g. 'America/New_York')",
        )
    tz_name = str(raw["tz"]).strip()
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"{ctx}: active_hours tz {tz_name!r} is not a valid IANA "
            "timezone",
        ) from exc
    start = _parse_hhmm(raw["start"], ctx)
    end = _parse_hhmm(raw["end"], ctx)
    if start == end:
        raise ValueError(
            f"{ctx}: active_hours start and end are equal ({raw['start']!r}) "
            "— a zero-width window would never poll",
        )
    days = _parse_days(raw.get("days"), ctx)
    return ActiveHours(start=start, end=end, tz=tz, days=days)


def is_within_window(account: WatchAccount, now_utc: datetime) -> bool:
    """True if ``account`` may be polled at ``now_utc`` (CL-6t7v).

    An account with no active_hours is always-on → always True. Otherwise
    ``now_utc`` is converted into the window's timezone (so DST is honored
    by :mod:`zoneinfo`) and the local time must fall in ``[start, end)`` on
    an allowed weekday. Windows that cross midnight (``start > end``) are
    handled as overnight, with the weekday check applied to the day the
    window *opens*.

    ``now_utc`` should be timezone-aware; a naive datetime is assumed UTC.
    """
    window = account.active_hours
    if window is None:
        return True
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    local = now_utc.astimezone(window.tz)
    now_t = local.time()
    weekday = local.weekday()
    if window.crosses_midnight:
        # Overnight window, e.g. 22:00..02:00. Two half-open pieces:
        #   [start, midnight)  on the opening weekday
        #   [midnight, end)    on the following weekday
        if now_t >= window.start:
            return weekday in window.days
        if now_t < window.end:
            # We are past midnight; the window opened "yesterday".
            return ((weekday - 1) % 7) in window.days
        return False
    # Same-day window: [start, end) on an allowed weekday.
    return weekday in window.days and window.start <= now_t < window.end


def load_watchlist(path: Path | str = DEFAULT_WATCHLIST_PATH) -> list[WatchAccount]:
    """Load + validate the watchlist YAML. Fails loud on schema errors."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("categories"), dict):
        raise ValueError(f"{p}: expected a top-level 'categories' mapping")

    accounts: list[WatchAccount] = []
    seen: set[str] = set()
    for category, block in raw["categories"].items():
        if category not in KNOWN_CATEGORIES:
            raise ValueError(
                f"{p}: unknown category {category!r} — known: "
                f"{sorted(KNOWN_CATEGORIES)}; extend KNOWN_CATEGORIES "
                "deliberately (footer policy is category-keyed)",
            )
        entries = (block or {}).get("accounts")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{p}: category {category!r} has no accounts")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{p}: account entry in {category!r} not a mapping")
            handle = str(entry.get("handle", "")).strip()
            if not _HANDLE_RE.match(handle):
                raise ValueError(
                    f"{p}: invalid handle {handle!r} in {category!r} "
                    "(expected 1-15 chars of [A-Za-z0-9_], no @)",
                )
            if handle.lower() in seen:
                raise ValueError(f"{p}: duplicate handle {handle!r}")
            seen.add(handle.lower())
            note = str(entry.get("note", "")).strip()
            if not note:
                raise ValueError(f"{p}: @{handle} missing 'note'")
            priority = str(entry.get("priority", "")).strip()
            if priority not in CADENCE:
                raise ValueError(
                    f"{p}: @{handle} priority {priority!r} not in "
                    f"{sorted(CADENCE)}",
                )
            kw_raw = entry.get("keywords", [])
            if not isinstance(kw_raw, list):
                raise ValueError(f"{p}: @{handle} 'keywords' must be a list")
            keywords = tuple(str(k).strip() for k in kw_raw if str(k).strip())
            active_hours = _parse_active_hours(
                entry.get("active_hours"), f"{p}: @{handle}",
            )
            accounts.append(WatchAccount(
                handle=handle,
                category=category,
                note=note,
                priority=priority,
                keywords=keywords,
                active_hours=active_hours,
            ))
    return accounts


def _cap_from_env() -> int:
    raw = os.environ.get("X_MONITOR_MONTHLY_CAP", "").strip()
    if not raw:
        return DEFAULT_MONTHLY_CAP
    try:
        cap = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"X_MONITOR_MONTHLY_CAP={raw!r} is not an integer",
        ) from exc
    if cap <= 0:
        raise ValueError(f"X_MONITOR_MONTHLY_CAP must be positive, got {cap}")
    return cap


@dataclass
class XMonitorConfig:
    """Monitor knobs. Env-derived defaults resolve at construction."""

    watchlist_path: Path = DEFAULT_WATCHLIST_PATH
    state_path: Path = DEFAULT_STATE_PATH
    user_ids_path: Path = DEFAULT_USER_IDS_PATH
    monthly_cap: int = field(default_factory=_cap_from_env)
    exclude_replies: bool = True
    exclude_retweets: bool = True
    max_results: int = 5  # per-account tweets per poll (v2 minimum)
    batch_threshold: int = 3  # >N new posts for one account → combined msg
    text_limit: int = 500  # chars of tweet text in a single-post message
    batch_text_limit: int = 200  # chars per post inside a combined message
    paced: bool = True  # spread the monthly budget across the month

    # CLI-backend pacing (ignored by the metered API backend, which is
    # governed by the read budget instead). This is ordinary polite
    # rate-control — jittered, unsynchronized polling that avoids
    # hammering the endpoint on a fixed heartbeat. It is NOT a
    # detection-defeat mechanism: cookie-based access carries inherent
    # account risk (see CliTransport docstring) that no polling cadence
    # removes. Set the delay range to 0 to disable.
    cli_shuffle: bool = True  # randomize account order each cycle
    cli_min_gap_sec: float = 15.0  # min pause between accounts
    cli_max_gap_sec: float = 45.0  # max pause between accounts


# --------------------------------------------------------------------- #
# Posts + transports
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Post:
    """One fetched post, transport-agnostic."""

    id: str
    text: str
    created_at: str = ""


class TransportError(Exception):
    """Per-account fetch failure — triggers the failure cooldown."""


class RateLimitedError(TransportError):
    """HTTP 429 from the API — carries the reset time (epoch seconds)."""

    def __init__(self, reset_ts: float) -> None:
        super().__init__(f"rate limited until epoch {reset_ts:.0f}")
        self.reset_ts = reset_ts


class Transport(ABC):
    """How posts are fetched for one account. Two implementations:
    the official (paid) API and an external CLI tool. Both ship dark."""

    #: True when fetches consume the metered monthly read budget.
    metered: bool = False
    #: HTTP/API calls made so far (the monitor meters the delta).
    reads_made: int = 0
    #: The one clear line to log when the transport is not configured.
    idle_line: str = ""

    @abstractmethod
    def ready(self) -> bool:
        """True when the transport is configured and may be used."""

    def prepare(self, handles: Sequence[str]) -> None:  # noqa: B027
        """Optional pre-cycle setup (e.g. handle → id resolution).
        Deliberately a no-op default: only ApiTransport needs it."""

    @abstractmethod
    def fetch(self, account: WatchAccount, since_id: str | None) -> list[Post]:
        """Fetch recent posts for ``account``. Raises TransportError /
        RateLimitedError on failure. May return posts at or before
        ``since_id`` — the monitor re-filters client-side."""


@dataclass(frozen=True)
class XApiResponse:
    """Minimal HTTP response for the injectable API shim."""

    status_code: int
    body: str
    headers: Mapping[str, str] = field(default_factory=dict)


#: Injectable HTTP shim (url, headers, params) → response, in the same
#: spirit as src.research.ingest.HttpGet. Unit tests inject canned
#: responses; no live HTTP in unit tests.
XApiGet = Callable[[str, Mapping[str, str], Mapping[str, Any]], XApiResponse]


def _default_api_get(
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
) -> XApiResponse:
    resp = httpx.get(url, headers=dict(headers), params=dict(params), timeout=15.0)
    return XApiResponse(resp.status_code, resp.text, dict(resp.headers))


class ApiTransport(Transport):
    """Official X API v2 timeline fetcher (paid basic tier).

    Requires ``TWITTER_BEARER_TOKEN``; without it :meth:`ready` is
    False and the monitor idles. Every HTTP call (including 429s and
    the one-time ``GET /2/users/by`` handle resolution) increments
    ``reads_made`` so the monitor's budget governor can meter spend.
    Resolved user ids are cached to ``data/x_user_ids.json`` (atomic
    write) so resolution costs ~1 read per 100 handles, once ever.
    """

    metered = True
    idle_line = IDLE_LINE_API

    def __init__(
        self,
        user_ids_path: Path | str = DEFAULT_USER_IDS_PATH,
        api_get: XApiGet = _default_api_get,
        *,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
        max_results: int = 5,
    ) -> None:
        self.user_ids_path = Path(user_ids_path)
        self.api_get = api_get
        self.exclude_replies = exclude_replies
        self.exclude_retweets = exclude_retweets
        self.max_results = max_results
        self.reads_made = 0
        self._ids: dict[str, str] = self._load_id_cache()
        self._resolve_attempted: set[str] = set()

    # -- readiness ---------------------------------------------------- #

    @staticmethod
    def _token() -> str:
        return os.environ.get("TWITTER_BEARER_TOKEN", "").strip()

    def ready(self) -> bool:
        return bool(self._token())

    # -- id cache ----------------------------------------------------- #

    def _load_id_cache(self) -> dict[str, str]:
        if not self.user_ids_path.exists():
            return {}
        try:
            raw = json.loads(self.user_ids_path.read_text())
            return {str(k).lower(): str(v) for k, v in raw.items()}
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning(
                "user-id cache %s malformed (%s); will re-resolve",
                self.user_ids_path, type(exc).__name__,
            )
            return {}

    def _save_id_cache(self) -> None:
        _atomic_write_json(self.user_ids_path, self._ids)

    # -- API plumbing ------------------------------------------------- #

    def _call(self, url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        self.reads_made += 1
        headers = {"Authorization": f"Bearer {self._token()}"}
        try:
            resp = self.api_get(url, headers, params)
        except Exception as exc:
            raise TransportError(
                f"transport failure for {url}: {type(exc).__name__}",
            ) from exc
        if resp.status_code == 429:
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            try:
                reset_ts = float(resp_headers.get("x-rate-limit-reset", ""))
            except ValueError:
                reset_ts = time.time() + 900.0
            raise RateLimitedError(reset_ts)
        if resp.status_code != 200:
            raise TransportError(f"HTTP {resp.status_code} for {url}")
        try:
            data = json.loads(resp.body)
        except json.JSONDecodeError as exc:
            raise TransportError(f"invalid JSON from {url}") from exc
        if not isinstance(data, dict):
            raise TransportError(f"unexpected payload shape from {url}")
        return data

    # -- Transport interface ------------------------------------------ #

    def prepare(self, handles: Sequence[str]) -> None:
        """Resolve any not-yet-cached handles to user ids, ≤100 per
        ``GET /2/users/by`` call. Unresolvable handles (suspended,
        renamed) are warned about once per process, not retried."""
        missing = [
            h for h in handles
            if h.lower() not in self._ids
            and h.lower() not in self._resolve_attempted
        ]
        if not missing:
            return
        for i in range(0, len(missing), 100):
            chunk = missing[i:i + 100]
            data = self._call(
                f"{X_API_BASE}/users/by",
                {"usernames": ",".join(chunk)},
            )
            # Mark attempted only on a successful call: a 429 / outage
            # during resolution must not permanently orphan the handle.
            for h in chunk:
                self._resolve_attempted.add(h.lower())
            for user in data.get("data", []):
                username = str(user.get("username", "")).lower()
                uid = str(user.get("id", ""))
                if username and uid:
                    self._ids[username] = uid
        self._save_id_cache()
        for h in missing:
            if h.lower() not in self._ids:
                logger.warning(
                    "could not resolve @%s to a user id (suspended or "
                    "renamed?) — skipping until restart", h,
                )

    def fetch(self, account: WatchAccount, since_id: str | None) -> list[Post]:
        uid = self._ids.get(account.handle.lower())
        if uid is None:
            raise TransportError(f"no user id for @{account.handle}")
        params: dict[str, Any] = {
            "max_results": self.max_results,
            "tweet.fields": "created_at",
        }
        excludes = []
        if self.exclude_replies:
            excludes.append("replies")
        if self.exclude_retweets:
            excludes.append("retweets")
        if excludes:
            params["exclude"] = ",".join(excludes)
        if since_id:
            params["since_id"] = since_id
        data = self._call(f"{X_API_BASE}/users/{uid}/tweets", params)
        posts: list[Post] = []
        for t in data.get("data", []):
            if not isinstance(t, dict) or "id" not in t:
                continue
            posts.append(Post(
                id=str(t["id"]),
                text=str(t.get("text", "")),
                created_at=str(t.get("created_at", "")),
            ))
        return posts


class CliTransport(Transport):
    """External-CLI fetcher — OPERATOR-SUPPLIED, cookie-based tools.

    HONESTY, READ BEFORE ENABLING: cookie-based CLI tools drive X's
    private endpoints with a logged-in session and therefore VIOLATE
    the X Terms of Service. The ban risk attaches to the account whose
    cookies the tool uses — use a burner account, never the operator's
    real account. Install the tool via an auditable package manager
    (not curl|sh), and expect breakage whenever X changes its private
    endpoints. This monitor never touches cookies or credentials
    itself: authentication is entirely the external tool's problem.
    The subprocess environment is passed through untouched.

    Configuration: ``X_MONITOR_CLI_CMD`` is a command template run once
    per account with ``{handle}`` substituted, e.g.::

        X_MONITOR_CLI_CMD='bird user-tweets @{handle} -n 5 --json'

    The command must print a JSON array of posts on stdout. Parsing is
    defensive: ``id``/``id_str``/``rest_id`` and ``text``/``full_text``
    field spellings are accepted; malformed entries are skipped. A
    top-level ``{"data": [...]}`` wrapper is also accepted. Without
    ``X_MONITOR_CLI_CMD`` this transport idles (ships dark, like the
    API path). since_id filtering happens in the monitor, client-side.
    """

    metered = False  # no metered reads; budget governor does not apply
    idle_line = IDLE_LINE_CLI

    def __init__(
        self,
        cmd_template: str | None = None,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
        timeout_sec: float = 60.0,
    ) -> None:
        if cmd_template is None:
            cmd_template = os.environ.get("X_MONITOR_CLI_CMD", "").strip()
        self.cmd_template = cmd_template
        self.timeout_sec = timeout_sec
        self._runner = runner or self._default_runner
        self.reads_made = 0

    def _default_runner(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        # Operator-configured command; shell=False, argv fully tokenized.
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
            check=False,
        )

    def ready(self) -> bool:
        return bool(self.cmd_template)

    def fetch(self, account: WatchAccount, since_id: str | None) -> list[Post]:
        argv = [
            tok.replace("{handle}", account.handle)
            for tok in shlex.split(self.cmd_template)
        ]
        try:
            proc = self._runner(argv)
        except Exception as exc:
            raise TransportError(
                f"CLI command failed for @{account.handle}: "
                f"{type(exc).__name__}",
            ) from exc
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "").strip()[-200:]
            raise TransportError(
                f"CLI command exited {proc.returncode} for "
                f"@{account.handle}: {stderr_tail}",
            )
        return _parse_cli_posts(proc.stdout, account.handle)


def _parse_cli_posts(stdout: str, handle: str) -> list[Post]:
    """Defensively parse a JSON array of posts from CLI stdout."""
    try:
        payload = json.loads(stdout or "null")
    except json.JSONDecodeError as exc:
        raise TransportError(f"CLI stdout for @{handle} is not JSON") from exc
    if isinstance(payload, dict):
        payload = payload.get("data")
    if not isinstance(payload, list):
        raise TransportError(
            f"CLI stdout for @{handle} is not a JSON array of posts",
        )
    posts: list[Post] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue  # malformed entry — skip
        tid = entry.get("id") or entry.get("id_str") or entry.get("rest_id")
        if tid is None or not str(tid).strip():
            continue  # malformed entry — skip
        text = entry.get("text") or entry.get("full_text") or ""
        posts.append(Post(
            id=str(tid).strip(),
            text=str(text),
            created_at=str(entry.get("created_at") or ""),
        ))
    return posts


def build_transport(
    config: XMonitorConfig,
    *,
    api_get: XApiGet | None = None,
) -> Transport:
    """Select the transport from ``X_MONITOR_BACKEND`` (api | cli)."""
    backend = os.environ.get("X_MONITOR_BACKEND", "api").strip().lower() or "api"
    if backend == "api":
        return ApiTransport(
            user_ids_path=config.user_ids_path,
            api_get=api_get or _default_api_get,
            exclude_replies=config.exclude_replies,
            exclude_retweets=config.exclude_retweets,
            max_results=config.max_results,
        )
    if backend == "cli":
        return CliTransport()
    raise ValueError(
        f"unknown X_MONITOR_BACKEND {backend!r} (expected 'api' or 'cli')",
    )


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


# --------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------- #


@dataclass
class CycleSummary:
    """Outcome of one poll cycle — the daemon logs this."""

    idle: bool = False
    paced: bool = False
    rate_limited: bool = False
    budget_exhausted: bool = False
    polled: int = 0
    new_posts: int = 0
    notifications: int = 0
    reads_used: int = 0  # month-to-date after the cycle (metered only)


NotifyFn = Callable[..., Any]  # notify_operator-compatible


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """tmp + os.replace, mirroring telegram_approvals.save_offset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=0, sort_keys=True))
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _id_newer(candidate: str, since: str) -> bool:
    """Snowflake ids are numeric; compare numerically when possible."""
    try:
        return int(candidate) > int(since)
    except ValueError:
        return candidate != since


def _id_sort_key(post: Post) -> tuple[int, int | str]:
    try:
        return (0, int(post.id))
    except ValueError:
        return (1, post.id)


def _month_key(now: datetime) -> str:
    return now.strftime("%Y-%m")


def _month_end(now: datetime) -> datetime:
    if now.month == 12:
        return datetime(now.year + 1, 1, 1, tzinfo=UTC)
    return datetime(now.year, now.month + 1, 1, tzinfo=UTC)


def _fresh_state(now: datetime) -> dict[str, Any]:
    return {
        "month": _month_key(now),
        "reads_used": 0,
        "budget_warned": False,
        "cycle": 0,
        "last_cycle_ts": 0.0,
        "backoff_until": 0.0,
        "since_ids": {},
        "failures": {},
    }


class XWatchlistMonitor:
    """Cadenced watchlist poller + Telegram relay. See module docstring."""

    def __init__(
        self,
        config: XMonitorConfig | None = None,
        accounts: Sequence[WatchAccount] | None = None,
        transport: Transport | None = None,
        notify: NotifyFn = notify_operator,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep_fn: Callable[[float], None] = time.sleep,
        rand_fn: Callable[[float, float], float] | None = None,
    ) -> None:
        self.config = config or XMonitorConfig()
        self.accounts = list(
            accounts if accounts is not None
            else load_watchlist(self.config.watchlist_path)
        )
        self.transport = transport or build_transport(self.config)
        self.notify = notify
        self._now = now_fn
        # Injectable so tests are deterministic and never actually sleep.
        self._sleep = sleep_fn
        self._rand = rand_fn or random.uniform
        self._shuffle: Callable[[list[Any]], None] = random.shuffle
        self._idle_logged = False
        self.state = self._load_state()

    # -- state -------------------------------------------------------- #

    def _load_state(self) -> dict[str, Any]:
        p = self.config.state_path
        if not p.exists():
            return _fresh_state(self._now())
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            # Fail loud (project convention): silently re-baselining
            # would reset the month's spend accounting.
            raise RuntimeError(
                f"x_monitor state file {p} is corrupt ({exc}); delete it "
                "to reset (this re-baselines since_ids and month spend)",
            ) from exc
        state = _fresh_state(self._now())
        if isinstance(raw, dict):
            state.update(raw)
        return state

    def _save_state(self) -> None:
        _atomic_write_json(self.config.state_path, self.state)

    # -- budget governor ---------------------------------------------- #

    @property
    def _cap(self) -> int:
        return self.config.monthly_cap

    def _rollover(self, now: datetime) -> None:
        month = _month_key(now)
        if self.state.get("month") != month:
            logger.info(
                "month rollover %s → %s: read budget reset (cap %d)",
                self.state.get("month"), month, self._cap,
            )
            self.state["month"] = month
            self.state["reads_used"] = 0
            self.state["budget_warned"] = False
            self._save_state()

    def _budget_exhausted(self) -> bool:
        return int(self.state.get("reads_used", 0)) >= self._cap

    def _warn_budget_once(self) -> None:
        if self.state.get("budget_warned"):
            return
        self.state["budget_warned"] = True
        self._save_state()
        logger.warning(
            "X monitor: monthly read budget exhausted (%d reads); "
            "polling paused until next month", self._cap,
        )
        self.notify(
            NOTIFY_TITLE,
            f"Monthly X API read budget exhausted ({self._cap} reads). "
            "Polling is paused until the next calendar month (UTC). "
            "Raise X_MONITOR_MONTHLY_CAP only if the API plan allows it.",
        )

    def _avg_reads_per_cycle(self) -> float:
        return sum(1.0 / CADENCE[a.priority] for a in self.accounts)

    def _sustainable_interval_sec(self, now: datetime) -> float:
        """Seconds between poll cycles that makes the remaining budget
        last until month end. ~90 min with the shipped defaults."""
        remaining_reads = self._cap - int(self.state.get("reads_used", 0))
        if remaining_reads <= 0:
            return float("inf")
        avg = self._avg_reads_per_cycle()
        if avg <= 0:
            return 0.0
        cycles_left = remaining_reads / avg
        remaining_sec = (_month_end(now) - now).total_seconds()
        return max(0.0, remaining_sec / cycles_left)

    def _meter(self, before: int) -> None:
        delta = self.transport.reads_made - before
        if delta > 0:
            self.state["reads_used"] = (
                int(self.state.get("reads_used", 0)) + delta
            )

    # -- failure cooldown --------------------------------------------- #

    def _in_cooldown(self, handle: str, cycle: int) -> bool:
        entry = self.state.get("failures", {}).get(handle)
        return bool(entry) and cycle < int(entry.get("next_cycle", 0))

    def _record_failure(self, handle: str, cycle: int) -> None:
        failures = self.state.setdefault("failures", {})
        count = int(failures.get(handle, {}).get("count", 0)) + 1
        cooldown = min(2 ** count, _MAX_FAILURE_COOLDOWN_CYCLES)
        failures[handle] = {"count": count, "next_cycle": cycle + cooldown}
        logger.warning(
            "@%s poll failed (%d consecutive); cooling down %d cycle(s)",
            handle, count, cooldown,
        )

    def _record_success(self, handle: str) -> None:
        self.state.get("failures", {}).pop(handle, None)

    # -- polling ------------------------------------------------------ #

    def poll_cycle(self) -> CycleSummary:
        """One cadenced pass over the watchlist. Never raises for
        transport-level problems; always safe to call in a loop."""
        now = self._now()
        summary = CycleSummary()

        if not self.transport.ready():
            if not self._idle_logged:
                logger.info(self.transport.idle_line)
                self._idle_logged = True
            else:
                logger.debug(self.transport.idle_line)
            summary.idle = True
            return summary
        self._idle_logged = False

        metered = self.transport.metered
        if metered:
            self._rollover(now)
            summary.reads_used = int(self.state.get("reads_used", 0))
            if now.timestamp() < float(self.state.get("backoff_until", 0.0)):
                logger.info(
                    "rate-limit backoff active for another %.0fs; skipping",
                    float(self.state["backoff_until"]) - now.timestamp(),
                )
                summary.rate_limited = True
                return summary
            if self._budget_exhausted():
                self._warn_budget_once()
                summary.budget_exhausted = True
                return summary
            if self.config.paced:
                since_last = now.timestamp() - float(
                    self.state.get("last_cycle_ts", 0.0),
                )
                interval = self._sustainable_interval_sec(now)
                if float(self.state.get("last_cycle_ts", 0.0)) > 0 and (
                    since_last < interval
                ):
                    logger.debug(
                        "paced: %.0fs since last cycle < sustainable "
                        "%.0fs", since_last, interval,
                    )
                    summary.paced = True
                    return summary

        cycle = int(self.state.get("cycle", 0))
        due = [
            a for a in self.accounts
            if cycle % CADENCE[a.priority] == 0
            and not self._in_cooldown(a.handle, cycle)
        ]
        # Active-hours filter (CL-6t7v): drop accounts outside their
        # window. Windowless accounts are always-on and always pass, so
        # this composes with — never replaces — the cadence/cooldown
        # gate above. Only day-recap traders carry windows; breaking-news
        # accounts stay all-day.
        windowed_out = [a for a in due if not is_within_window(a, now)]
        if windowed_out:
            due = [a for a in due if is_within_window(a, now)]
            logger.debug(
                "active-hours: skipping %d account(s) outside their window: %s",
                len(windowed_out),
                ", ".join("@" + a.handle for a in windowed_out),
            )
        # CLI backend: randomize order so polling isn't a fixed sweep.
        if not metered and self.config.cli_shuffle and due:
            self._shuffle(due)

        # Handle → id resolution (API transport; ≤1 read per 100 handles).
        if due:
            before = self.transport.reads_made
            try:
                self.transport.prepare([a.handle for a in due])
            except RateLimitedError as rl:
                self._enter_backoff(rl, now, summary)
            except TransportError as exc:
                logger.warning("transport prepare failed: %s", exc)
            finally:
                if metered:
                    self._meter(before)

        for idx, account in enumerate(due):
            if summary.rate_limited:
                break
            if metered and self._budget_exhausted():
                self._warn_budget_once()
                summary.budget_exhausted = True
                break
            # CLI backend: jittered pause between accounts (skip before
            # the first). Ordinary rate-control; see config note.
            if not metered and idx > 0 and self.config.cli_max_gap_sec > 0:
                gap = self._rand(
                    self.config.cli_min_gap_sec,
                    self.config.cli_max_gap_sec,
                )
                if gap > 0:
                    self._sleep(gap)
            before = self.transport.reads_made
            since_id = self.state["since_ids"].get(account.handle)
            try:
                posts = self.transport.fetch(account, since_id)
            except RateLimitedError as rl:
                if metered:
                    self._meter(before)
                self._enter_backoff(rl, now, summary)
                break
            except TransportError as exc:
                if metered:
                    self._meter(before)
                logger.warning("fetch failed for @%s: %s", account.handle, exc)
                self._record_failure(account.handle, cycle)
                self._save_state()
                continue
            if metered:
                self._meter(before)
            self._record_success(account.handle)
            summary.polled += 1
            self._process_posts(account, since_id, posts, summary)
            self._save_state()

        self.state["cycle"] = cycle + 1
        self.state["last_cycle_ts"] = now.timestamp()
        self._save_state()
        if metered:
            summary.reads_used = int(self.state.get("reads_used", 0))
        return summary

    def _enter_backoff(
        self,
        rl: RateLimitedError,
        now: datetime,
        summary: CycleSummary,
    ) -> None:
        reset = max(rl.reset_ts, now.timestamp() + 60.0)
        self.state["backoff_until"] = reset
        self._save_state()
        summary.rate_limited = True
        logger.warning(
            "429 from X API; honoring x-rate-limit-reset — backing off "
            "until epoch %.0f (%.0fs)", reset, reset - now.timestamp(),
        )

    def _process_posts(
        self,
        account: WatchAccount,
        since_id: str | None,
        posts: list[Post],
        summary: CycleSummary,
    ) -> None:
        posts = sorted(posts, key=_id_sort_key)  # oldest → newest
        if since_id is None:
            # BASELINE: first sighting of this account — record the
            # newest id without notifying, so a fresh deploy doesn't
            # replay old posts at the operator.
            if posts:
                self.state["since_ids"][account.handle] = posts[-1].id
                logger.info(
                    "baseline @%s at post id %s (%d seen, not notified)",
                    account.handle, posts[-1].id, len(posts),
                )
            return

        new = [p for p in posts if _id_newer(p.id, since_id)]
        if not new:
            return
        self.state["since_ids"][account.handle] = new[-1].id
        summary.new_posts += len(new)

        if account.keywords:
            lowered = [k.lower() for k in account.keywords]
            new = [
                p for p in new
                if any(k in p.text.lower() for k in lowered)
            ]
            if not new:
                return

        if len(new) > self.config.batch_threshold:
            message = build_batch_message(
                account, new, self.config.batch_text_limit,
            )
            self.notify(NOTIFY_TITLE, message, html=True)
            summary.notifications += 1
        else:
            for post in new:
                message = build_post_message(
                    account, post, self.config.text_limit,
                )
                self.notify(NOTIFY_TITLE, message, html=True)
                summary.notifications += 1
