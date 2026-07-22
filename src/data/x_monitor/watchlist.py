"""Watchlist config for the X monitor (CL-3j86): accounts, categories,
priority cadence, and active-hours windowing (CL-6t7v).

Split out of the former ``src/data/x_monitor.py`` monofile (CL-ikz2,
structural review 2026-07-21 §6.2.4). See the package ``__init__``
docstring for the full monitor overview.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #

DEFAULT_WATCHLIST_PATH = Path("configs/x_watchlist.yaml")

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

_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")

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
