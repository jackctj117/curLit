"""The X watchlist monitor itself (CL-3j86): config knobs, the poll
cycle with its budget governor / pacing / failure cooldowns, and the
X→pipeline ingestion bridge (CL-esyo).

Split out of the former ``src/data/x_monitor.py`` monofile (CL-ikz2,
structural review 2026-07-21 §6.2.4). See the package ``__init__``
docstring for the full monitor overview.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.data.x_monitor.messages import (
    NOTIFY_TITLE,
    build_batch_message,
    build_post_message,
)
from src.data.x_monitor.transport import (
    DEFAULT_USER_IDS_PATH,
    Post,
    RateLimitedError,
    Transport,
    TransportError,
    _atomic_write_json,
    _cap_from_env,
    build_transport,
)
from src.data.x_monitor.watchlist import (
    CADENCE,
    DEFAULT_WATCHLIST_PATH,
    WatchAccount,
    is_within_window,
    load_watchlist,
)
from src.research.notifications import notify_operator

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------- #

DEFAULT_STATE_PATH = Path("data/x_monitor_state.json")

#: Cap on per-account failure cooldown (in cycles).
_MAX_FAILURE_COOLDOWN_CYCLES = 16


def _flag_env(name: str, *, default: bool) -> bool:
    """Parse a boolean env flag (CL-esyo). Accepts 1/0, true/false,
    yes/no, on/off (case-insensitive); unset → ``default``."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning(
        "%s=%r is not a recognized boolean; using default %s",
        name, raw, default,
    )
    return default


def _build_ingest_engine() -> Any | None:
    """Build a SQLAlchemy engine from POSTGRES_* env for X→pipeline
    ingestion (CL-esyo), mirroring scripts/event_pipeline._db_url and
    src.runtime.run_engine._build_db_engine. Returns None (with one
    warning) when SQLAlchemy or the DB URL can't be built — the monitor
    then runs with ingestion disabled but still polls (and can forward).
    """
    try:
        from sqlalchemy import create_engine  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "SQLAlchemy unavailable — X→pipeline ingestion disabled; "
            "monitor continues (forwarding only)",
        )
        return None
    db_url = os.environ.get(
        "DATABASE_URL",
        f"postgresql+psycopg2://{os.environ.get('POSTGRES_USER', 'fx')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'fx')}",
    )
    try:
        return create_engine(db_url)
    except Exception as exc:
        logger.warning(
            "could not build ingest DB engine (%s) — X→pipeline "
            "ingestion disabled; monitor continues", str(exc)[:120],
        )
        return None


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

    # Pipeline wiring (CL-esyo). The monitor's PURPOSE is to feed the
    # analysis pipeline: relevant posts become NEW geo_events rows the
    # impact agent assesses. Ingestion is ON by default; raw tweet
    # forwarding to Telegram is now OPT-IN (the operator wants ingestion
    # as the point, not a firehose relay). Env defaults resolve at
    # construction (X_MONITOR_INGEST / X_MONITOR_FORWARD).
    x_ingest_enabled: bool = field(default_factory=lambda: _flag_env(
        "X_MONITOR_INGEST", default=True,
    ))
    x_forward_enabled: bool = field(default_factory=lambda: _flag_env(
        "X_MONITOR_FORWARD", default=False,
    ))
    #: Max geo_events rows ingested per account per cycle (flood guard).
    x_ingest_cap: int = 10


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
    ingested: int = 0  # NEW geo_events rows created this cycle (CL-esyo)
    reads_used: int = 0  # month-to-date after the cycle (metered only)


NotifyFn = Callable[..., Any]  # notify_operator-compatible


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
    """Cadenced watchlist poller + Telegram relay. See package docstring."""

    def __init__(
        self,
        config: XMonitorConfig | None = None,
        accounts: Sequence[WatchAccount] | None = None,
        transport: Transport | None = None,
        notify: NotifyFn = notify_operator,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep_fn: Callable[[float], None] = time.sleep,
        rand_fn: Callable[[float, float], float] | None = None,
        engine: Any | None = None,
        playbooks: Any | None = None,
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
        # X→pipeline ingestion (CL-esyo). Relevant posts become NEW
        # geo_events rows for the impact agent. The engine + playbooks
        # are lazy-built (POSTGRES_* env; events playbook config) only
        # when ingestion is enabled — a DB-less monitor still polls and
        # can still forward. Both are injectable for tests.
        self._ingest_engine = engine
        self._playbooks = playbooks
        if self.config.x_ingest_enabled and self._ingest_engine is None:
            self._ingest_engine = _build_ingest_engine()
            if self._ingest_engine is None:
                logger.warning(
                    "X→pipeline ingestion enabled but no DB engine — "
                    "posts will NOT be ingested this run",
                )

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

        # X→pipeline ingestion (CL-esyo) — the monitor's PURPOSE. Every
        # NEW post (NOT keyword-filtered: the playbook watch-term match
        # inside ingest_posts is the real relevance gate) is offered to
        # the ingest bridge, which drops no-theme posts, the whole
        # small_traders category, and dedups on external_id. Fully
        # guarded: a failure here logs and never breaks the poll loop.
        if self.config.x_ingest_enabled and self._ingest_engine is not None:
            self._ingest_new_posts(account, new, summary)

        # Raw tweet forwarding to Telegram (CL-esyo) — now OPT-IN. The
        # operator wants ingestion as the point, so this only fires when
        # explicitly enabled. Keyword filtering still narrows the relay.
        if not self.config.x_forward_enabled:
            return

        forward = new
        if account.keywords:
            lowered = [k.lower() for k in account.keywords]
            forward = [
                p for p in forward
                if any(k in p.text.lower() for k in lowered)
            ]
            if not forward:
                return

        if len(forward) > self.config.batch_threshold:
            message = build_batch_message(
                account, forward, self.config.batch_text_limit,
            )
            self.notify(NOTIFY_TITLE, message, html=True)
            summary.notifications += 1
        else:
            for post in forward:
                message = build_post_message(
                    account, post, self.config.text_limit,
                )
                self.notify(NOTIFY_TITLE, message, html=True)
                summary.notifications += 1

    def _ingest_new_posts(
        self,
        account: WatchAccount,
        new: list[Post],
        summary: CycleSummary,
    ) -> None:
        """Feed NEW posts to the X→pipeline bridge (CL-esyo).

        Lazy-imports the events bridge so the monitor carries no hard
        events dependency. Any failure (import, DB, parse) is logged and
        swallowed — ingestion must never break the poll loop.
        """
        engine = self._ingest_engine
        if engine is None:  # narrowed for the type checker; guarded above
            return
        try:
            from src.events.playbooks import load_playbooks  # noqa: PLC0415
            from src.events.x_ingest import ingest_posts  # noqa: PLC0415

            if self._playbooks is None:
                self._playbooks = load_playbooks()
            result = ingest_posts(
                engine,
                account,
                new,
                self._playbooks,
                cap=self.config.x_ingest_cap,
            )
            summary.ingested += result.ingested
            if result.ingested or result.deduped:
                logger.info("%s", result.summary_line(account.handle))
            else:
                logger.debug("%s", result.summary_line(account.handle))
        except Exception:
            logger.exception(
                "X→pipeline ingestion failed for @%s; continuing",
                account.handle,
            )
