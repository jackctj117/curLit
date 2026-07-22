"""Tests for the X account watchlist monitor (CL-3j86).

All HTTP is mocked via the injectable XApiGet shim; the CLI transport
uses injected runners or local `cat` — no live X access anywhere.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from src.data.x_monitor import (
    CADENCE,
    CATEGORY_FOOTERS,
    IDLE_LINE_API,
    IDLE_LINE_CLI,
    KNOWN_CATEGORIES,
    ActiveHours,
    ApiTransport,
    CliTransport,
    Post,
    Transport,
    TransportError,
    WatchAccount,
    XApiResponse,
    XMonitorConfig,
    XWatchlistMonitor,
    build_batch_message,
    build_post_message,
    build_transport,
    is_within_window,
    load_watchlist,
)

REPO_WATCHLIST = Path(__file__).resolve().parents[2] / "configs" / "x_watchlist.yaml"


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _account(
    handle: str = "unusual_whales",
    category: str = "financial_flow",
    priority: str = "high",
    keywords: tuple[str, ...] = (),
    active_hours: ActiveHours | None = None,
) -> WatchAccount:
    return WatchAccount(
        handle=handle, category=category, note="test", priority=priority,
        keywords=keywords, active_hours=active_hours,
    )


class FakeApi:
    """Canned-response XApiGet shim; records every request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: user timelines keyed by user id → list of tweet dicts
        self.timelines: dict[str, list[dict[str, Any]]] = {}
        #: handle(lower) → id served by /users/by
        self.users: dict[str, str] = {}
        #: if set, every call returns this response instead
        self.force: XApiResponse | None = None

    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        params: Mapping[str, Any],
    ) -> XApiResponse:
        self.calls.append((url, dict(params)))
        assert "Authorization" in headers
        if self.force is not None:
            return self.force
        if url.endswith("/users/by"):
            names = str(params["usernames"]).split(",")
            data = [
                {"id": self.users[n.lower()], "username": n}
                for n in names if n.lower() in self.users
            ]
            return XApiResponse(200, json.dumps({"data": data}))
        uid = url.rsplit("/", 2)[-2]
        tweets = self.timelines.get(uid, [])
        since = params.get("since_id")
        if since is not None:
            tweets = [t for t in tweets if int(t["id"]) > int(since)]
        # v2 returns newest first
        tweets = sorted(tweets, key=lambda t: int(t["id"]), reverse=True)
        return XApiResponse(200, json.dumps({"data": tweets}))


class Recorder:
    """notify_operator-compatible sink recording every send."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, bool]] = []

    def __call__(
        self,
        title: str,
        message: str,
        priority: int = 0,
        *,
        html: bool = False,
    ) -> None:
        self.sent.append((title, message, html))


def _monitor(
    tmp_path: Path,
    accounts: list[WatchAccount],
    fake: Any,  # FakeApi, XApiGet-compatible callable, or a Transport
    *,
    cap: int = 9_000,
    paced: bool = False,
    now_fn: Any = None,
    x_forward_enabled: bool = True,
    x_ingest_enabled: bool = False,
    engine: Any = None,
    playbooks: Any = None,
    **config_kw: Any,
) -> tuple[XWatchlistMonitor, Recorder]:
    # CL-esyo: forwarding is opt-in in production (default off), but the
    # legacy notification-dispatch tests exercise forwarding — so this
    # helper defaults forwarding ON and ingestion OFF to isolate the
    # relay path. Wiring tests override these explicitly.
    config = XMonitorConfig(
        state_path=tmp_path / "state.json",
        user_ids_path=tmp_path / "ids.json",
        monthly_cap=cap,
        paced=paced,
        x_forward_enabled=x_forward_enabled,
        x_ingest_enabled=x_ingest_enabled,
        **config_kw,
    )
    if isinstance(fake, Transport):
        transport = fake
    else:
        transport = ApiTransport(
            user_ids_path=config.user_ids_path,
            api_get=fake,
            exclude_replies=config.exclude_replies,
            exclude_retweets=config.exclude_retweets,
            max_results=config.max_results,
        )
    recorder = Recorder()
    kwargs: dict[str, Any] = {}
    if now_fn is not None:
        kwargs["now_fn"] = now_fn
    monitor = XWatchlistMonitor(
        config=config,
        accounts=accounts,
        transport=transport,
        notify=recorder,
        engine=engine,
        playbooks=playbooks,
        **kwargs,
    )
    return monitor, recorder


@pytest.fixture
def token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test-token")


# --------------------------------------------------------------------- #
# Watchlist YAML
# --------------------------------------------------------------------- #


class TestWatchlistConfig:
    def test_repo_watchlist_loads(self) -> None:
        accounts = load_watchlist(REPO_WATCHLIST)
        assert len(accounts) >= 30

    def test_handles_unique_case_insensitive(self) -> None:
        accounts = load_watchlist(REPO_WATCHLIST)
        lowered = [a.handle.lower() for a in accounts]
        assert len(lowered) == len(set(lowered))

    def test_categories_known_and_all_present(self) -> None:
        accounts = load_watchlist(REPO_WATCHLIST)
        categories = {a.category for a in accounts}
        assert categories == KNOWN_CATEGORIES

    def test_priorities_valid_and_notes_present(self) -> None:
        for a in load_watchlist(REPO_WATCHLIST):
            assert a.priority in CADENCE, a.handle
            assert a.note.strip(), a.handle

    def test_small_traders_all_low_priority(self) -> None:
        smalls = [
            a for a in load_watchlist(REPO_WATCHLIST)
            if a.category == "small_traders"
        ]
        assert len(smalls) == 6
        assert all(a.priority == "low" for a in smalls)

    def test_expected_high_priority_set(self) -> None:
        highs = {
            a.handle for a in load_watchlist(REPO_WATCHLIST)
            if a.priority == "high"
        }
        assert highs == {
            "unusual_whales", "DeItaone", "HindenburgRes",
            "sentdefender", "Osinttechnical",
            "Africa_In_EN", "robert_ivanhoe",
        }

    def test_rejects_duplicate_handle(self, tmp_path: Path) -> None:
        p = tmp_path / "w.yaml"
        p.write_text(
            "categories:\n  financial_flow:\n    accounts:\n"
            "      - {handle: a, note: x, priority: high}\n"
            "      - {handle: A, note: x, priority: low}\n",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_watchlist(p)

    def test_rejects_unknown_category(self, tmp_path: Path) -> None:
        p = tmp_path / "w.yaml"
        p.write_text(
            "categories:\n  crypto_bros:\n    accounts:\n"
            "      - {handle: a, note: x, priority: high}\n",
        )
        with pytest.raises(ValueError, match="unknown category"):
            load_watchlist(p)

    def test_rejects_bad_priority(self, tmp_path: Path) -> None:
        p = tmp_path / "w.yaml"
        p.write_text(
            "categories:\n  financial_flow:\n    accounts:\n"
            "      - {handle: a, note: x, priority: urgent}\n",
        )
        with pytest.raises(ValueError, match="priority"):
            load_watchlist(p)

    def test_rejects_bad_handle(self, tmp_path: Path) -> None:
        p = tmp_path / "w.yaml"
        p.write_text(
            "categories:\n  financial_flow:\n    accounts:\n"
            "      - {handle: '@bad handle!', note: x, priority: low}\n",
        )
        with pytest.raises(ValueError, match="invalid handle"):
            load_watchlist(p)


# --------------------------------------------------------------------- #
# Id resolution + cache
# --------------------------------------------------------------------- #


@pytest.mark.usefixtures("token_env")
class TestIdResolution:
    def test_resolution_and_cache_roundtrip(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"unusual_whales": "111"}
        fake.timelines["111"] = [{"id": "5", "text": "hi"}]
        monitor, _ = _monitor(tmp_path, [_account()], fake)
        monitor.poll_cycle()

        cache = json.loads((tmp_path / "ids.json").read_text())
        assert cache == {"unusual_whales": "111"}
        by_calls = [c for c in fake.calls if c[0].endswith("/users/by")]
        assert len(by_calls) == 1

        # Fresh transport: cache hit → no second /users/by call
        fake2 = FakeApi()
        fake2.timelines["111"] = [{"id": "6", "text": "again"}]
        monitor2, _ = _monitor(tmp_path, [_account()], fake2)
        monitor2.poll_cycle()
        assert all(not c[0].endswith("/users/by") for c in fake2.calls)

    def test_unresolvable_handle_not_retried(self, tmp_path: Path) -> None:
        fake = FakeApi()  # knows no users
        monitor, recorder = _monitor(tmp_path, [_account()], fake)
        monitor.poll_cycle()
        monitor.poll_cycle()
        by_calls = [c for c in fake.calls if c[0].endswith("/users/by")]
        assert len(by_calls) == 1  # warned once, not hammered
        assert recorder.sent == []


# --------------------------------------------------------------------- #
# since_id + state persistence
# --------------------------------------------------------------------- #


@pytest.mark.usefixtures("token_env")
class TestSinceIdAndState:
    def test_first_poll_baselines_without_notifying(
        self, tmp_path: Path,
    ) -> None:
        fake = FakeApi()
        fake.users = {"unusual_whales": "111"}
        fake.timelines["111"] = [
            {"id": "100", "text": "old"}, {"id": "99", "text": "older"},
        ]
        monitor, recorder = _monitor(tmp_path, [_account()], fake)
        summary = monitor.poll_cycle()
        assert recorder.sent == []
        assert summary.notifications == 0
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["since_ids"]["unusual_whales"] == "100"

    def test_since_id_advances_and_is_sent_to_api(
        self, tmp_path: Path,
    ) -> None:
        fake = FakeApi()
        fake.users = {"unusual_whales": "111"}
        fake.timelines["111"] = [{"id": "100", "text": "old"}]
        monitor, recorder = _monitor(tmp_path, [_account()], fake)
        monitor.poll_cycle()  # baseline at 100

        fake.timelines["111"].append({"id": "101", "text": "fresh"})
        summary = monitor.poll_cycle()
        assert summary.new_posts == 1
        assert len(recorder.sent) == 1
        timeline_calls = [
            c for c in fake.calls if c[0].endswith("/tweets")
        ]
        assert timeline_calls[-1][1]["since_id"] == "100"
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["since_ids"]["unusual_whales"] == "101"

    def test_state_survives_restart(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"unusual_whales": "111"}
        fake.timelines["111"] = [{"id": "100", "text": "old"}]
        monitor, _ = _monitor(tmp_path, [_account()], fake)
        monitor.poll_cycle()

        fake2 = FakeApi()
        fake2.timelines["111"] = [{"id": "100", "text": "old"}]
        monitor2, recorder2 = _monitor(tmp_path, [_account()], fake2)
        monitor2.poll_cycle()
        # Restart neither re-baselines nor re-notifies post 100
        assert recorder2.sent == []
        timeline_calls = [c for c in fake2.calls if c[0].endswith("/tweets")]
        assert timeline_calls[0][1]["since_id"] == "100"

    def test_snowflake_ids_beyond_2_53_compare_as_ints(
        self, tmp_path: Path,
    ) -> None:
        # 64-bit snowflakes lose precision as floats; must compare as
        # ints. 9007199254740993 == 2**53 + 1 (float-equal to 2**53).
        base = str(2 ** 53)  # 9007199254740992
        nxt = str(2 ** 53 + 1)
        fake = FakeApi()
        fake.users = {"unusual_whales": "111"}
        fake.timelines["111"] = [{"id": base, "text": "old"}]
        monitor, recorder = _monitor(tmp_path, [_account()], fake)
        monitor.poll_cycle()  # baseline

        fake.timelines["111"].append({"id": nxt, "text": "fresh"})
        summary = monitor.poll_cycle()
        assert summary.new_posts == 1
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["since_ids"]["unusual_whales"] == nxt

    def test_corrupt_state_fails_loud(self, tmp_path: Path) -> None:
        (tmp_path / "state.json").write_text("{not json")
        fake = FakeApi()
        with pytest.raises(RuntimeError, match="corrupt"):
            _monitor(tmp_path, [_account()], fake)


# --------------------------------------------------------------------- #
# Budget governor
# --------------------------------------------------------------------- #


@pytest.mark.usefixtures("token_env")
class TestBudgetGovernor:
    def test_spend_counting(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"a": "1", "b": "2"}
        fake.timelines = {"1": [], "2": []}
        accounts = [_account("a"), _account("b")]
        monitor, _ = _monitor(tmp_path, accounts, fake)
        monitor.poll_cycle()
        # 1 users/by call + 2 timeline calls = 3 reads
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["reads_used"] == 3

    def test_month_rollover_resets_spend(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"a": "1"}
        fake.timelines = {"1": []}
        clock = {"now": datetime(2026, 6, 30, 23, 0, tzinfo=UTC)}
        monitor, recorder = _monitor(
            tmp_path, [_account("a")], fake, cap=2,
            now_fn=lambda: clock["now"],
        )
        monitor.poll_cycle()  # spends 2 (users/by + timeline) → cap hit
        summary = monitor.poll_cycle()
        assert summary.budget_exhausted
        assert len(recorder.sent) == 1  # the exhaustion warning

        clock["now"] = datetime(2026, 7, 1, 0, 5, tzinfo=UTC)
        summary = monitor.poll_cycle()
        assert not summary.budget_exhausted
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["month"] == "2026-07"
        assert not state["budget_warned"]

    def test_exhaustion_stops_polling_and_warns_once(
        self, tmp_path: Path,
    ) -> None:
        fake = FakeApi()
        fake.users = {"a": "1"}
        fake.timelines = {"1": []}
        monitor, recorder = _monitor(tmp_path, [_account("a")], fake, cap=2)
        monitor.poll_cycle()  # 2 reads → exhausted
        calls_after_spend = len(fake.calls)

        s1 = monitor.poll_cycle()
        s2 = monitor.poll_cycle()
        assert s1.budget_exhausted and s2.budget_exhausted
        assert len(fake.calls) == calls_after_spend  # hard stop: no HTTP
        warnings = [m for m in recorder.sent if "budget exhausted" in m[1]]
        assert len(warnings) == 1  # single Telegram warning

    def test_mid_cycle_exhaustion_stops_remaining_accounts(
        self, tmp_path: Path,
    ) -> None:
        fake = FakeApi()
        fake.users = {"a": "1", "b": "2", "c": "3"}
        fake.timelines = {"1": [], "2": [], "3": []}
        accounts = [_account("a"), _account("b"), _account("c")]
        # cap 2: users/by (1) + first timeline (2) → exhausted before b/c
        monitor, recorder = _monitor(tmp_path, accounts, fake, cap=2)
        summary = monitor.poll_cycle()
        assert summary.budget_exhausted
        assert summary.polled == 1
        timeline_calls = [c for c in fake.calls if c[0].endswith("/tweets")]
        assert len(timeline_calls) == 1
        assert any("budget exhausted" in m[1] for m in recorder.sent)

    def test_pacing_spreads_budget(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"a": "1"}
        fake.timelines = {"1": []}
        clock = {"now": datetime(2026, 7, 10, 12, 0, tzinfo=UTC)}
        monitor, _ = _monitor(
            tmp_path, [_account("a")], fake, paced=True,
            now_fn=lambda: clock["now"],
        )
        assert not monitor.poll_cycle().paced  # first cycle runs
        assert monitor.poll_cycle().paced  # immediate rerun is paced
        # Far enough in the future → allowed again
        clock["now"] = datetime(2026, 7, 10, 14, 0, tzinfo=UTC)
        assert not monitor.poll_cycle().paced

    def test_cap_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("X_MONITOR_MONTHLY_CAP", "123")
        config = XMonitorConfig(
            state_path=tmp_path / "s.json",
            user_ids_path=tmp_path / "i.json",
        )
        assert config.monthly_cap == 123
        monkeypatch.setenv("X_MONITOR_MONTHLY_CAP", "bogus")
        with pytest.raises(ValueError, match="X_MONITOR_MONTHLY_CAP"):
            XMonitorConfig(
                state_path=tmp_path / "s.json",
                user_ids_path=tmp_path / "i.json",
            )


# --------------------------------------------------------------------- #
# Priority cadence
# --------------------------------------------------------------------- #


@pytest.mark.usefixtures("token_env")
class TestCadence:
    def test_high_every_cycle_normal_2nd_low_4th(
        self, tmp_path: Path,
    ) -> None:
        fake = FakeApi()
        fake.users = {"hi": "1", "nor": "2", "lo": "3"}
        fake.timelines = {"1": [], "2": [], "3": []}
        accounts = [
            _account("hi", priority="high"),
            _account("nor", priority="normal"),
            _account("lo", priority="low"),
        ]
        monitor, _ = _monitor(tmp_path, accounts, fake)

        polled_per_cycle: list[set[str]] = []
        uid_to_handle = {"1": "hi", "2": "nor", "3": "lo"}
        for _ in range(4):
            before = len(fake.calls)
            monitor.poll_cycle()
            cycle_calls = fake.calls[before:]
            polled = {
                uid_to_handle[url.rsplit("/", 2)[-2]]
                for url, _params in cycle_calls
                if url.endswith("/tweets")
            }
            polled_per_cycle.append(polled)

        assert polled_per_cycle == [
            {"hi", "nor", "lo"},  # cycle 0
            {"hi"},               # cycle 1
            {"hi", "nor"},        # cycle 2
            {"hi"},               # cycle 3
        ]


# --------------------------------------------------------------------- #
# Reply / RT exclusion params
# --------------------------------------------------------------------- #


@pytest.mark.usefixtures("token_env")
class TestExclusionParams:
    def _params(self, tmp_path: Path, **config_kw: Any) -> dict[str, Any]:
        fake = FakeApi()
        fake.users = {"unusual_whales": "111"}
        fake.timelines = {"111": []}
        monitor, _ = _monitor(tmp_path, [_account()], fake, **config_kw)
        monitor.poll_cycle()
        return next(
            params for url, params in fake.calls if url.endswith("/tweets")
        )

    def test_default_excludes_replies_and_retweets(
        self, tmp_path: Path,
    ) -> None:
        params = self._params(tmp_path)
        assert params["exclude"] == "replies,retweets"
        assert params["tweet.fields"] == "created_at"
        assert params["max_results"] == 5

    def test_exclusion_configurable(self, tmp_path: Path) -> None:
        params = self._params(tmp_path, exclude_replies=False)
        assert params["exclude"] == "retweets"
        params = self._params(
            tmp_path, exclude_replies=False, exclude_retweets=False,
        )
        assert "exclude" not in params


# --------------------------------------------------------------------- #
# Notification formatting
# --------------------------------------------------------------------- #


class TestFormatting:
    def test_hostile_text_is_escaped(self) -> None:
        account = _account()
        post = Post(id="1", text='<script>alert("&")</script> <b>x</b>')
        msg = build_post_message(account, post)
        assert "<script>" not in msg
        assert "&lt;script&gt;" in msg
        assert "&amp;" in msg
        # The only tags present are ours
        assert msg.startswith("<b>@unusual_whales</b> [financial_flow]")

    def test_link_format(self) -> None:
        msg = build_post_message(_account(), Post(id="987", text="hi"))
        assert "https://x.com/unusual_whales/status/987" in msg

    def test_truncates_to_limit(self) -> None:
        long = "x" * 2000
        msg = build_post_message(_account(), Post(id="1", text=long))
        body_line = msg.splitlines()[1]
        assert len(body_line) <= 501
        assert body_line.endswith("…")

    def test_conflict_osint_footer(self) -> None:
        account = _account("sentdefender", category="conflict_osint")
        msg = build_post_message(account, Post(id="1", text="boom"))
        assert msg.endswith("<i>unverified — cross-check</i>")

    def test_small_traders_footer(self) -> None:
        account = _account("Fun_Trades1", category="small_traders")
        msg = build_post_message(account, Post(id="1", text="+400% again"))
        assert msg.endswith("<i>unverified performance claims</i>")

    def test_financial_flow_has_no_footer(self) -> None:
        msg = build_post_message(_account(), Post(id="1", text="flow"))
        assert "<i>" not in msg

    def test_batch_message_contains_all_posts_and_one_footer(self) -> None:
        account = _account("sentdefender", category="conflict_osint")
        posts = [Post(id=str(i), text=f"post {i} <&>") for i in range(1, 6)]
        msg = build_batch_message(account, posts)
        assert "— 5 new posts" in msg
        for i in range(1, 6):
            assert f"https://x.com/sentdefender/status/{i}" in msg
        assert msg.count(CATEGORY_FOOTERS["conflict_osint"]) == 1
        assert "&lt;&amp;&gt;" in msg


@pytest.mark.usefixtures("token_env")
class TestNotificationDispatch:
    def test_batching_over_threshold_sends_one_message(
        self, tmp_path: Path,
    ) -> None:
        fake = FakeApi()
        fake.users = {"unusual_whales": "111"}
        fake.timelines["111"] = [{"id": "10", "text": "seed"}]
        monitor, recorder = _monitor(tmp_path, [_account()], fake)
        monitor.poll_cycle()  # baseline
        fake.timelines["111"] += [
            {"id": str(i), "text": f"burst {i}"} for i in range(11, 16)
        ]
        summary = monitor.poll_cycle()
        assert summary.new_posts == 5
        assert len(recorder.sent) == 1  # one combined message
        title, message, html = recorder.sent[0]
        assert html
        assert "— 5 new posts" in message

    def test_at_or_below_threshold_sends_individual_messages(
        self, tmp_path: Path,
    ) -> None:
        fake = FakeApi()
        fake.users = {"unusual_whales": "111"}
        fake.timelines["111"] = [{"id": "10", "text": "seed"}]
        monitor, recorder = _monitor(tmp_path, [_account()], fake)
        monitor.poll_cycle()  # baseline
        fake.timelines["111"] += [
            {"id": "11", "text": "a"}, {"id": "12", "text": "b"},
        ]
        monitor.poll_cycle()
        assert len(recorder.sent) == 2
        assert all(html for _t, _m, html in recorder.sent)

    def test_keyword_filter(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"robert_ivanhoe": "9"}
        fake.timelines["9"] = [{"id": "10", "text": "seed"}]
        account = _account(
            "robert_ivanhoe", category="africa_mining",
            keywords=("copper", "kamoa"),
        )
        monitor, recorder = _monitor(tmp_path, [account], fake)
        monitor.poll_cycle()  # baseline
        fake.timelines["9"] += [
            {"id": "11", "text": "Great dinner tonight"},
            {"id": "12", "text": "Kamoa-Kakula copper output up"},
        ]
        monitor.poll_cycle()
        assert len(recorder.sent) == 1
        assert "Kamoa" in recorder.sent[0][1]
        # since_id still advanced past the filtered post
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["since_ids"]["robert_ivanhoe"] == "12"


# --------------------------------------------------------------------- #
# Tokenless idle path
# --------------------------------------------------------------------- #


class TestIdlePath:
    def test_tokenless_poll_is_idle_no_http(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
        fake = FakeApi()
        monitor, recorder = _monitor(tmp_path, [_account()], fake)
        with caplog.at_level("INFO"):
            summary = monitor.poll_cycle()
        assert summary.idle
        assert fake.calls == []
        assert recorder.sent == []
        assert IDLE_LINE_API in caplog.text
        assert not (tmp_path / "state.json").exists()

    def test_idle_line_logged_once_at_info(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
        monitor, _ = _monitor(tmp_path, [_account()], FakeApi())
        with caplog.at_level("INFO"):
            monitor.poll_cycle()
            monitor.poll_cycle()
            monitor.poll_cycle()
        info_lines = [
            r for r in caplog.records
            if r.levelname == "INFO" and IDLE_LINE_API in r.getMessage()
        ]
        assert len(info_lines) == 1

    def test_exact_idle_line_wording(self) -> None:
        assert IDLE_LINE_API == (
            "X monitor idle: set TWITTER_BEARER_TOKEN — X API basic tier "
            "required"
        )


# --------------------------------------------------------------------- #
# 429 backoff
# --------------------------------------------------------------------- #


@pytest.mark.usefixtures("token_env")
class TestRateLimitBackoff:
    def test_429_sets_backoff_honoring_reset_header(
        self, tmp_path: Path,
    ) -> None:
        clock = {"now": datetime(2026, 7, 10, 12, 0, tzinfo=UTC)}
        reset_ts = clock["now"].timestamp() + 3600
        fake = FakeApi()
        fake.force = XApiResponse(
            429, "", {"X-Rate-Limit-Reset": str(int(reset_ts))},
        )
        monitor, _ = _monitor(
            tmp_path, [_account()], fake, now_fn=lambda: clock["now"],
        )
        summary = monitor.poll_cycle()
        assert summary.rate_limited
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["backoff_until"] == pytest.approx(reset_ts)

        # Before reset: no HTTP at all
        calls_before = len(fake.calls)
        clock["now"] = datetime(2026, 7, 10, 12, 30, tzinfo=UTC)
        assert monitor.poll_cycle().rate_limited
        assert len(fake.calls) == calls_before

        # After reset: polls again
        fake.force = None
        fake.users = {"unusual_whales": "111"}
        fake.timelines = {"111": []}
        clock["now"] = datetime(2026, 7, 10, 13, 5, tzinfo=UTC)
        summary = monitor.poll_cycle()
        assert not summary.rate_limited
        assert len(fake.calls) > calls_before

    def test_failure_cooldown_per_account(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"a": "1", "b": "2"}
        fake.timelines = {"2": []}

        real_call = fake.__call__

        def flaky(
            url: str, headers: Mapping[str, str], params: Mapping[str, Any],
        ) -> XApiResponse:
            if "/users/1/" in url:
                fake.calls.append((url, dict(params)))
                return XApiResponse(500, "boom")
            return real_call(url, headers, params)

        monitor, _ = _monitor(tmp_path, [_account("a"), _account("b")], flaky)
        monitor.poll_cycle()  # a fails → cooldown; b polls fine
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["failures"]["a"]["count"] == 1
        before = len(fake.calls)
        monitor.poll_cycle()  # a still cooling down
        a_calls = [c for c in fake.calls[before:] if "/users/1/" in c[0]]
        assert a_calls == []


# --------------------------------------------------------------------- #
# CLI transport
# --------------------------------------------------------------------- #


class TestCliTransport:
    def test_backend_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = XMonitorConfig(
            state_path=tmp_path / "s.json", user_ids_path=tmp_path / "i.json",
        )
        monkeypatch.delenv("X_MONITOR_BACKEND", raising=False)
        assert isinstance(build_transport(config), ApiTransport)
        monkeypatch.setenv("X_MONITOR_BACKEND", "cli")
        assert isinstance(build_transport(config), CliTransport)
        monkeypatch.setenv("X_MONITOR_BACKEND", "webscrape")
        with pytest.raises(ValueError, match="X_MONITOR_BACKEND"):
            build_transport(config)

    def test_missing_cmd_idles_dark(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("X_MONITOR_CLI_CMD", raising=False)
        transport = CliTransport()
        assert not transport.ready()
        monitor, recorder = _monitor(tmp_path, [_account()], transport)
        with caplog.at_level("INFO"):
            summary = monitor.poll_cycle()
        assert summary.idle
        assert IDLE_LINE_CLI in caplog.text
        assert "ToS" in IDLE_LINE_CLI  # honesty requirement
        assert "burner" in IDLE_LINE_CLI
        assert recorder.sent == []

    def test_command_template_substitution(self) -> None:
        captured: list[list[str]] = []

        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            captured.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="[]", stderr="")

        transport = CliTransport(
            cmd_template="bird user-tweets @{handle} -n 5 --json",
            runner=runner,
        )
        transport.fetch(_account("DeItaone"), None)
        assert captured == [
            ["bird", "user-tweets", "@DeItaone", "-n", "5", "--json"],
        ]

    def test_defensive_parse_field_spellings_and_malformed_skip(
        self,
    ) -> None:
        payload = json.dumps([
            {"id": 101, "text": "plain id+text"},
            {"id_str": "102", "full_text": "id_str+full_text"},
            {"rest_id": "103", "text": "rest_id"},
            {"text": "no id — malformed, skipped"},
            "not-a-dict",
            {"id": "", "text": "blank id — skipped"},
        ])

        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, stdout=payload, stderr="")

        transport = CliTransport(cmd_template="x {handle}", runner=runner)
        posts = transport.fetch(_account(), None)
        assert [p.id for p in posts] == ["101", "102", "103"]
        assert posts[1].text == "id_str+full_text"

    def test_data_wrapper_accepted(self) -> None:
        payload = json.dumps({"data": [{"id": "7", "text": "wrapped"}]})

        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, stdout=payload, stderr="")

        transport = CliTransport(cmd_template="x {handle}", runner=runner)
        assert [p.id for p in transport.fetch(_account(), None)] == ["7"]

    def test_nonzero_exit_raises_transport_error(self) -> None:
        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                argv, 1, stdout="", stderr="Not logged in",
            )

        transport = CliTransport(cmd_template="x {handle}", runner=runner)
        with pytest.raises(TransportError, match="exited 1"):
            transport.fetch(_account(), None)

    def test_non_json_stdout_raises_transport_error(self) -> None:
        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                argv, 0, stdout="rate limited, try later", stderr="",
            )

        transport = CliTransport(cmd_template="x {handle}", runner=runner)
        with pytest.raises(TransportError, match="not JSON"):
            transport.fetch(_account(), None)

    def test_since_id_filtering_applied_by_monitor(
        self, tmp_path: Path,
    ) -> None:
        # CLI tools don't take since_id — the monitor must filter.
        tweets = [{"id": "100", "text": "old"}, {"id": "101", "text": "mid"}]

        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps(tweets), stderr="",
            )

        transport = CliTransport(cmd_template="x {handle}", runner=runner)
        monitor, recorder = _monitor(tmp_path, [_account()], transport)
        monitor.poll_cycle()  # baseline at 101
        assert recorder.sent == []
        tweets.append({"id": "102", "text": "fresh"})
        summary = monitor.poll_cycle()
        assert summary.new_posts == 1
        assert len(recorder.sent) == 1
        assert "status/102" in recorder.sent[0][1]

    def test_cli_end_to_end_with_real_subprocess(
        self, tmp_path: Path,
    ) -> None:
        # Local `cat` stands in for the external tool — no network.
        fixture = tmp_path / "posts.json"
        fixture.write_text(json.dumps([
            {"id": "200", "text": "hello from the fixture"},
        ]))
        transport = CliTransport(cmd_template=f"cat {fixture}")
        posts = transport.fetch(_account(), None)
        assert posts == [Post(id="200", text="hello from the fixture")]

    def test_cli_backend_is_unmetered(self, tmp_path: Path) -> None:
        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, stdout="[]", stderr="")

        transport = CliTransport(cmd_template="x {handle}", runner=runner)
        assert transport.metered is False
        monitor, _ = _monitor(tmp_path, [_account()], transport, cap=1)
        for _ in range(5):  # would exceed cap=1 if metered
            summary = monitor.poll_cycle()
            assert not summary.budget_exhausted
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["reads_used"] == 0


class TestCliPacing:
    """CLI-backend polite pacing: jittered inter-account gaps + shuffle.
    Ordinary rate-control (deterministic here via injected sleep/rand/
    shuffle) — NOT a detection-defeat claim; see the config note."""

    def _cli_monitor(
        self, tmp_path: Path, accounts: list[WatchAccount],
        sleeps: list[float], **cfg: Any,
    ) -> tuple[XWatchlistMonitor, Recorder]:
        payload = json.dumps([{"id": "500", "text": "hi"}])

        def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(argv, 0, stdout=payload, stderr="")

        transport = CliTransport(cmd_template="x {handle}", runner=runner)
        config = XMonitorConfig(
            state_path=tmp_path / "state.json",
            user_ids_path=tmp_path / "ids.json",
            **cfg,
        )
        recorder = Recorder()
        monitor = XWatchlistMonitor(
            config=config, accounts=accounts, transport=transport,
            notify=recorder,
            sleep_fn=sleeps.append,          # record instead of sleeping
            rand_fn=lambda lo, hi: (lo + hi) / 2.0,  # deterministic midpoint
        )
        # deterministic, no-op shuffle so order assertions are stable
        monitor._shuffle = lambda seq: None
        return monitor, recorder

    def test_gap_between_accounts_not_before_first(
        self, tmp_path: Path,
    ) -> None:
        accts = [_account("a"), _account("b"), _account("c")]
        sleeps: list[float] = []
        monitor, _ = self._cli_monitor(
            tmp_path, accts, sleeps,
            cli_min_gap_sec=15.0, cli_max_gap_sec=45.0,
        )
        monitor.poll_cycle()
        # 3 accounts → 2 inter-account gaps (none before the first).
        assert sleeps == [30.0, 30.0]  # midpoint of 15..45

    def test_zero_gap_disables_pacing(self, tmp_path: Path) -> None:
        accts = [_account("a"), _account("b")]
        sleeps: list[float] = []
        monitor, _ = self._cli_monitor(
            tmp_path, accts, sleeps,
            cli_min_gap_sec=0.0, cli_max_gap_sec=0.0,
        )
        monitor.poll_cycle()
        assert sleeps == []

    def test_shuffle_invoked_for_cli_backend(self, tmp_path: Path) -> None:
        accts = [_account("a"), _account("b")]
        sleeps: list[float] = []
        monitor, _ = self._cli_monitor(
            tmp_path, accts, sleeps, cli_shuffle=True,
            cli_min_gap_sec=0.0, cli_max_gap_sec=0.0,
        )
        shuffled: list[Any] = []
        monitor._shuffle = lambda seq: shuffled.append(list(seq))
        monitor.poll_cycle()
        assert len(shuffled) == 1  # order was randomized once this cycle


# --------------------------------------------------------------------- #
# Active-hours windows (CL-6t7v)
# --------------------------------------------------------------------- #


def _hours(
    start: str, end: str, tz: str = "America/New_York",
    days: frozenset[int] = frozenset(range(7)),
) -> ActiveHours:
    sh, sm = (int(x) for x in start.split(":"))
    eh, em = (int(x) for x in end.split(":"))
    return ActiveHours(
        start=dt_time(sh, sm), end=dt_time(eh, em),
        tz=ZoneInfo(tz), days=days,
    )


class TestIsWithinWindow:
    def test_no_window_is_always_on(self) -> None:
        acct = _account()  # no active_hours
        # Any instant, including one far outside any plausible window.
        assert is_within_window(acct, datetime(2026, 7, 20, 3, 0, tzinfo=UTC))

    def test_inside_same_day_window(self) -> None:
        # 17:30-20:30 ET; 2026-07-20 is a Monday. 18:00 ET = 22:00 UTC (EDT).
        acct = _account(active_hours=_hours("17:30", "20:30"))
        now = datetime(2026, 7, 20, 22, 0, tzinfo=UTC)  # 18:00 EDT
        assert is_within_window(acct, now)

    def test_outside_same_day_window(self) -> None:
        acct = _account(active_hours=_hours("17:30", "20:30"))
        # 12:00 UTC = 08:00 EDT — well before the window opens.
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        assert not is_within_window(acct, now)

    def test_end_is_exclusive(self) -> None:
        acct = _account(active_hours=_hours("17:30", "20:30"))
        # Exactly 20:30 EDT = 00:30 UTC next day → excluded ([start, end)).
        at_end = datetime(2026, 7, 21, 0, 30, tzinfo=UTC)
        assert not is_within_window(acct, at_end)
        # One minute before end is included.
        before_end = datetime(2026, 7, 21, 0, 29, tzinfo=UTC)
        assert is_within_window(acct, before_end)

    def test_start_is_inclusive(self) -> None:
        acct = _account(active_hours=_hours("17:30", "20:30"))
        # Exactly 17:30 EDT = 21:30 UTC → included.
        at_start = datetime(2026, 7, 20, 21, 30, tzinfo=UTC)
        assert is_within_window(acct, at_start)

    def test_midnight_crossing_before_and_after(self) -> None:
        # Overnight window 22:00-02:00 ET (start > end).
        acct = _account(active_hours=_hours("22:00", "02:00"))
        w = acct.active_hours
        assert w is not None and w.crosses_midnight
        # 23:00 EDT Mon = 03:00 UTC Tue → inside (opening leg).
        assert is_within_window(
            acct, datetime(2026, 7, 21, 3, 0, tzinfo=UTC),
        )
        # 01:00 EDT Tue = 05:00 UTC Tue → inside (past-midnight leg).
        assert is_within_window(
            acct, datetime(2026, 7, 21, 5, 0, tzinfo=UTC),
        )
        # 12:00 EDT Tue = 16:00 UTC → outside.
        assert not is_within_window(
            acct, datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        )

    def test_midnight_crossing_weekday_applies_to_opening_day(self) -> None:
        # Overnight window only on Fridays (opening day). Fri 2026-07-24.
        acct = _account(active_hours=_hours(
            "22:00", "02:00", days=frozenset({4}),
        ))
        # Fri 23:00 EDT = Sat 03:00 UTC → inside (opened Friday).
        assert is_within_window(
            acct, datetime(2026, 7, 25, 3, 0, tzinfo=UTC),
        )
        # Sat 01:00 EDT = Sat 05:00 UTC → still inside (opened Friday).
        assert is_within_window(
            acct, datetime(2026, 7, 25, 5, 0, tzinfo=UTC),
        )
        # Sat 23:00 EDT = Sun 03:00 UTC → outside (Saturday not allowed).
        assert not is_within_window(
            acct, datetime(2026, 7, 26, 3, 0, tzinfo=UTC),
        )

    def test_weekday_filter_excludes_weekend(self) -> None:
        acct = _account(active_hours=_hours(
            "17:30", "20:30", days=frozenset({0, 1, 2, 3, 4}),
        ))
        # 2026-07-25 is a Saturday. 18:00 ET local, inside time-of-day.
        sat = datetime(2026, 7, 25, 22, 0, tzinfo=UTC)  # 18:00 EDT Sat
        assert not is_within_window(acct, sat)
        # 2026-07-24 Friday at the same wall-clock time → allowed.
        fri = datetime(2026, 7, 24, 22, 0, tzinfo=UTC)
        assert is_within_window(acct, fri)

    def test_tz_conversion_et_vs_utc(self) -> None:
        # A window expressed in ET must be evaluated in ET, not UTC.
        acct = _account(active_hours=_hours("17:30", "20:30"))
        # 18:00 UTC would be inside a naive UTC read, but it is 14:00 EDT
        # — outside the ET window. Confirms tz conversion actually runs.
        assert not is_within_window(
            acct, datetime(2026, 7, 20, 18, 0, tzinfo=UTC),
        )

    def test_dst_boundary_sanity_winter_vs_summer(self) -> None:
        # Same ET wall-clock (18:00) maps to different UTC across DST.
        acct = _account(active_hours=_hours("17:30", "20:30"))
        # Summer (EDT, UTC-4): 18:00 EDT = 22:00 UTC.
        assert is_within_window(
            acct, datetime(2026, 7, 20, 22, 0, tzinfo=UTC),
        )
        # Winter (EST, UTC-5): 18:00 EST = 23:00 UTC. zoneinfo shifts the
        # offset, so the SAME 22:00 UTC is now 17:00 EST — before the
        # window — while 23:00 UTC is inside. 2026-01-20 is a Tuesday.
        assert not is_within_window(
            acct, datetime(2026, 1, 20, 22, 0, tzinfo=UTC),
        )
        assert is_within_window(
            acct, datetime(2026, 1, 20, 23, 0, tzinfo=UTC),
        )

    def test_naive_datetime_assumed_utc(self) -> None:
        acct = _account(active_hours=_hours("17:30", "20:30"))
        naive = datetime(2026, 7, 20, 22, 0)  # no tzinfo → treated as UTC
        assert is_within_window(acct, naive)


@pytest.mark.usefixtures("token_env")
class TestPollCycleWindowFilter:
    def test_out_of_window_account_dropped_windowless_kept(
        self, tmp_path: Path,
    ) -> None:
        fake = FakeApi()
        fake.users = {"news": "1", "trader": "2"}
        fake.timelines = {"1": [], "2": []}
        # 12:00 UTC = 08:00 EDT — trader window (17:30-20:30 ET) is closed.
        clock = {"now": datetime(2026, 7, 20, 12, 0, tzinfo=UTC)}
        news = _account("news", priority="high")
        trader = _account(
            "trader", category="small_traders", priority="high",
            active_hours=_hours("17:30", "20:30"),
        )
        monitor, _ = _monitor(
            tmp_path, [news, trader], fake, now_fn=lambda: clock["now"],
        )
        monitor.poll_cycle()
        timeline_calls = [c for c in fake.calls if c[0].endswith("/tweets")]
        polled_uids = {c[0].rsplit("/", 2)[-2] for c in timeline_calls}
        assert "1" in polled_uids   # windowless news polled
        assert "2" not in polled_uids  # out-of-window trader skipped

    def test_in_window_account_is_polled(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"trader": "2"}
        fake.timelines = {"2": []}
        # 22:00 UTC = 18:00 EDT — inside 17:30-20:30 ET.
        clock = {"now": datetime(2026, 7, 20, 22, 0, tzinfo=UTC)}
        trader = _account(
            "trader", category="small_traders", priority="high",
            active_hours=_hours("17:30", "20:30"),
        )
        monitor, _ = _monitor(
            tmp_path, [trader], fake, now_fn=lambda: clock["now"],
        )
        monitor.poll_cycle()
        timeline_calls = [c for c in fake.calls if c[0].endswith("/tweets")]
        assert len(timeline_calls) == 1  # polled while in-window

    def test_windowless_accounts_never_filtered(self, tmp_path: Path) -> None:
        fake = FakeApi()
        fake.users = {"a": "1", "b": "2"}
        fake.timelines = {"1": [], "2": []}
        # An hour that is outside every plausible trader window.
        clock = {"now": datetime(2026, 7, 20, 8, 0, tzinfo=UTC)}
        accts = [_account("a"), _account("b")]  # both windowless
        monitor, _ = _monitor(
            tmp_path, accts, fake, now_fn=lambda: clock["now"],
        )
        monitor.poll_cycle()
        timeline_calls = [c for c in fake.calls if c[0].endswith("/tweets")]
        assert len(timeline_calls) == 2  # both all-day accounts polled

    def test_skip_logged_at_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        fake = FakeApi()
        fake.users = {"trader": "2"}
        fake.timelines = {"2": []}
        clock = {"now": datetime(2026, 7, 20, 12, 0, tzinfo=UTC)}  # closed
        trader = _account(
            "trader", category="small_traders", priority="high",
            active_hours=_hours("17:30", "20:30"),
        )
        monitor, _ = _monitor(
            tmp_path, [trader], fake, now_fn=lambda: clock["now"],
        )
        with caplog.at_level("DEBUG"):
            monitor.poll_cycle()
        assert any(
            "active-hours" in r.getMessage() and "@trader" in r.getMessage()
            for r in caplog.records
        )


# --------------------------------------------------------------------- #
# active_hours YAML parsing (CL-6t7v)
# --------------------------------------------------------------------- #


class TestActiveHoursParsing:
    def _load_one(self, tmp_path: Path, active_hours_block: str) -> WatchAccount:
        p = tmp_path / "w.yaml"
        p.write_text(
            "categories:\n  small_traders:\n    accounts:\n"
            "      - handle: t\n        note: x\n        priority: low\n"
            + active_hours_block,
        )
        accts = load_watchlist(p)
        assert len(accts) == 1
        return accts[0]

    def test_parses_full_block(self, tmp_path: Path) -> None:
        acct = self._load_one(
            tmp_path,
            "        active_hours:\n"
            "          days: weekdays\n"
            "          start: \"17:30\"\n"
            "          end: \"20:30\"\n"
            "          tz: America/New_York\n",
        )
        w = acct.active_hours
        assert w is not None
        assert w.start == dt_time(17, 30)
        assert w.end == dt_time(20, 30)
        assert w.tz == ZoneInfo("America/New_York")
        assert w.days == frozenset({0, 1, 2, 3, 4})

    def test_days_default_is_all(self, tmp_path: Path) -> None:
        acct = self._load_one(
            tmp_path,
            "        active_hours:\n"
            "          start: \"09:00\"\n"
            "          end: \"17:00\"\n"
            "          tz: UTC\n",
        )
        assert acct.active_hours is not None
        assert acct.active_hours.days == frozenset(range(7))

    def test_days_list_of_names(self, tmp_path: Path) -> None:
        acct = self._load_one(
            tmp_path,
            "        active_hours:\n"
            "          days: [Mon, wednesday, FRI]\n"
            "          start: \"09:00\"\n"
            "          end: \"17:00\"\n"
            "          tz: UTC\n",
        )
        assert acct.active_hours is not None
        assert acct.active_hours.days == frozenset({0, 2, 4})

    def test_no_active_hours_is_none(self, tmp_path: Path) -> None:
        acct = self._load_one(tmp_path, "")
        assert acct.active_hours is None

    def test_bad_tz_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a valid IANA"):
            self._load_one(
                tmp_path,
                "        active_hours:\n"
                "          start: \"09:00\"\n"
                "          end: \"17:00\"\n"
                "          tz: Mars/Olympus_Mons\n",
            )

    def test_bad_time_format_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not 'HH:MM'"):
            self._load_one(
                tmp_path,
                "        active_hours:\n"
                "          start: \"9am\"\n"
                "          end: \"17:00\"\n"
                "          tz: UTC\n",
            )

    def test_out_of_range_time_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="out of range"):
            self._load_one(
                tmp_path,
                "        active_hours:\n"
                "          start: \"25:00\"\n"
                "          end: \"17:00\"\n"
                "          tz: UTC\n",
            )

    def test_missing_start_or_end_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="both 'start' and 'end'"):
            self._load_one(
                tmp_path,
                "        active_hours:\n"
                "          start: \"09:00\"\n"
                "          tz: UTC\n",
            )

    def test_missing_tz_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs 'tz'"):
            self._load_one(
                tmp_path,
                "        active_hours:\n"
                "          start: \"09:00\"\n"
                "          end: \"17:00\"\n",
            )

    def test_bad_weekday_name_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a weekday name"):
            self._load_one(
                tmp_path,
                "        active_hours:\n"
                "          days: [Mon, Funday]\n"
                "          start: \"09:00\"\n"
                "          end: \"17:00\"\n"
                "          tz: UTC\n",
            )

    def test_equal_start_end_fails_loud(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="equal"):
            self._load_one(
                tmp_path,
                "        active_hours:\n"
                "          start: \"09:00\"\n"
                "          end: \"09:00\"\n"
                "          tz: UTC\n",
            )


class TestRepoWatchlistWindows:
    """The shipped watchlist: only the 6 traders are windowed (CL-6t7v)."""

    def test_only_small_traders_are_windowed(self) -> None:
        accts = load_watchlist(REPO_WATCHLIST)
        windowed = [a for a in accts if a.active_hours is not None]
        assert {a.handle for a in windowed} == {
            "Fun_Trades1", "timeframeking", "prosperousguy",
            "StocksniperTeam", "Simply0DTE", "TT_stocks_",
        }
        assert all(a.category == "small_traders" for a in windowed)

    def test_news_accounts_are_windowless(self) -> None:
        accts = load_watchlist(REPO_WATCHLIST)
        non_traders = [a for a in accts if a.category != "small_traders"]
        # 31 original + 7 analysts/think-tanks added 2026-07-21 (Brad_Setser,
        # PhillipsPOBrien, oryxspioenkop, CSIS, RUSI_org, CFR_org,
        # AtlanticCouncil). The invariant under test: news/flow/OSINT accounts
        # must NEVER carry an active_hours window (breaking news is 24h).
        assert len(non_traders) == 38
        assert all(a.active_hours is None for a in non_traders)

    def test_seeded_trader_windows_match_operator_table(self) -> None:
        accts = {a.handle: a for a in load_watchlist(REPO_WATCHLIST)}
        et = ZoneInfo("America/New_York")
        for h in ("timeframeking", "prosperousguy", "StocksniperTeam",
                  "TT_stocks_"):
            w = accts[h].active_hours
            assert w is not None, h
            assert w.tz == et
            assert w.start == dt_time(17, 30)
            assert w.end == dt_time(20, 30)
            assert w.days == frozenset({0, 1, 2, 3, 4})
        fun = accts["Fun_Trades1"].active_hours
        assert fun is not None
        assert fun.tz == ZoneInfo("Europe/Berlin")
        assert fun.start == dt_time(20, 0) and fun.end == dt_time(23, 59)
        odte = accts["Simply0DTE"].active_hours
        assert odte is not None
        assert odte.tz == ZoneInfo("Europe/Rome")
        assert odte.start == dt_time(19, 0) and odte.end == dt_time(23, 0)


# --------------------------------------------------------------------- #
# X → pipeline ingestion wiring (CL-esyo)
# --------------------------------------------------------------------- #


class _SpyEngine:
    """A stand-in DB engine that need never be touched — ingest is
    spied at the ingest_posts boundary, so the engine is just a marker."""


class _IngestSpy:
    """Records ingest_posts(...) calls; returns a canned IngestResult."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, engine: Any, account: Any, posts: list[Post],
        playbooks: Any, cap: int = 10,
    ) -> Any:
        from src.events.x_ingest import IngestResult

        self.calls.append({
            "engine": engine, "account": account, "posts": list(posts),
            "cap": cap,
        })
        return IngestResult(ingested=len(posts))


@pytest.mark.usefixtures("token_env")
class TestPipelineWiring:
    """Ingestion default ON, forwarding default OFF, both independently
    toggleable, baseline still silent, DB-absent safe."""

    def _fresh_account(self, fake: FakeApi) -> None:
        fake.users = {"unusual_whales": "111"}
        fake.timelines["111"] = [{"id": "100", "text": "baseline"}]

    def _new_post(
        self, fake: FakeApi,
        text_in: str = "Iran closes the Strait of Hormuz",
    ) -> None:
        fake.timelines["111"].append({"id": "101", "text": text_in})

    def test_ingest_called_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _IngestSpy()
        monkeypatch.setattr("src.events.x_ingest.ingest_posts", spy)
        fake = FakeApi()
        self._fresh_account(fake)
        monitor, recorder = _monitor(
            tmp_path, [_account()], fake,
            x_ingest_enabled=True, x_forward_enabled=False,
            engine=_SpyEngine(), playbooks={},
        )
        monitor.poll_cycle()  # baseline — no ingest, no forward
        assert spy.calls == []
        assert recorder.sent == []

        self._new_post(fake)
        summary = monitor.poll_cycle()
        assert len(spy.calls) == 1
        assert [p.id for p in spy.calls[0]["posts"]] == ["101"]
        assert spy.calls[0]["cap"] == monitor.config.x_ingest_cap
        assert summary.ingested == 1
        # Forwarding is off by default → no Telegram sends.
        assert recorder.sent == []

    def test_forward_suppressed_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _IngestSpy()
        monkeypatch.setattr("src.events.x_ingest.ingest_posts", spy)
        fake = FakeApi()
        self._fresh_account(fake)
        # Production defaults: ingest on, forward off.
        monitor, recorder = _monitor(
            tmp_path, [_account()], fake,
            x_ingest_enabled=True, x_forward_enabled=False,
            engine=_SpyEngine(), playbooks={},
        )
        monitor.poll_cycle()
        self._new_post(fake)
        monitor.poll_cycle()
        assert recorder.sent == []
        assert len(spy.calls) == 1

    def test_forward_only_no_ingest(self, tmp_path: Path) -> None:
        # Forwarding on, ingestion off → notify fires, no engine built.
        fake = FakeApi()
        self._fresh_account(fake)
        monitor, recorder = _monitor(
            tmp_path, [_account()], fake,
            x_ingest_enabled=False, x_forward_enabled=True,
        )
        assert monitor._ingest_engine is None
        monitor.poll_cycle()
        self._new_post(fake)
        summary = monitor.poll_cycle()
        assert len(recorder.sent) == 1
        assert summary.ingested == 0

    def test_both_toggle_independently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Both on: ingest AND forward.
        spy = _IngestSpy()
        monkeypatch.setattr("src.events.x_ingest.ingest_posts", spy)
        fake = FakeApi()
        self._fresh_account(fake)
        monitor, recorder = _monitor(
            tmp_path, [_account()], fake,
            x_ingest_enabled=True, x_forward_enabled=True,
            engine=_SpyEngine(), playbooks={},
        )
        monitor.poll_cycle()
        self._new_post(fake)
        monitor.poll_cycle()
        assert len(spy.calls) == 1
        assert len(recorder.sent) == 1

    def test_baseline_silent_no_ingest_no_forward(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _IngestSpy()
        monkeypatch.setattr("src.events.x_ingest.ingest_posts", spy)
        fake = FakeApi()
        self._fresh_account(fake)
        monitor, recorder = _monitor(
            tmp_path, [_account()], fake,
            x_ingest_enabled=True, x_forward_enabled=True,
            engine=_SpyEngine(), playbooks={},
        )
        summary = monitor.poll_cycle()  # first sighting = baseline
        assert spy.calls == []
        assert recorder.sent == []
        assert summary.ingested == 0
        # since_id was recorded so the next cycle is live.
        state = json.loads((tmp_path / "state.json").read_text())
        assert state["since_ids"]["unusual_whales"] == "100"

    def test_db_absent_is_safe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Ingestion enabled but no engine (DB unreachable) → no crash,
        # no ingest attempt; monitor still polls (and could forward).
        spy = _IngestSpy()
        monkeypatch.setattr("src.events.x_ingest.ingest_posts", spy)
        fake = FakeApi()
        self._fresh_account(fake)
        monitor, recorder = _monitor(
            tmp_path, [_account()], fake,
            x_ingest_enabled=True, x_forward_enabled=True,
            engine=None, playbooks={},
        )
        # No engine was injected and none could be lazily built in the
        # test env for a definitely-unreachable DB is possible, so force
        # the None to model the DB-absent branch deterministically.
        monitor._ingest_engine = None
        monitor.poll_cycle()
        self._new_post(fake)
        summary = monitor.poll_cycle()
        assert spy.calls == []  # ingestion skipped — engine is None
        assert summary.ingested == 0
        # Forwarding still worked.
        assert len(recorder.sent) == 1

    def test_ingest_failure_never_breaks_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("ingest exploded")

        monkeypatch.setattr("src.events.x_ingest.ingest_posts", _boom)
        fake = FakeApi()
        self._fresh_account(fake)
        monitor, recorder = _monitor(
            tmp_path, [_account()], fake,
            x_ingest_enabled=True, x_forward_enabled=True,
            engine=_SpyEngine(), playbooks={},
        )
        monitor.poll_cycle()
        self._new_post(fake)
        # Must not raise — the poll loop survives an ingest failure.
        summary = monitor.poll_cycle()
        assert summary.ingested == 0
        assert len(recorder.sent) == 1  # forwarding still happened

    def test_small_traders_full_post_list_offered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The monitor offers ALL new posts to ingest_posts (the category
        # / theme gate lives inside ingest_posts, not the monitor). Verify
        # a keyword'd account still hands the full unfiltered list over.
        spy = _IngestSpy()
        monkeypatch.setattr("src.events.x_ingest.ingest_posts", spy)
        fake = FakeApi()
        self._fresh_account(fake)
        acct = _account(keywords=("hormuz",))
        monitor, _ = _monitor(
            tmp_path, [acct], fake,
            x_ingest_enabled=True, x_forward_enabled=False,
            engine=_SpyEngine(), playbooks={},
        )
        monitor.poll_cycle()
        fake.timelines["111"].append({"id": "101", "text": "unrelated chatter"})
        monitor.poll_cycle()
        # 'unrelated chatter' fails the forward keyword filter, but is
        # still handed to ingest (theme gate decides relevance there).
        assert len(spy.calls) == 1
        assert [p.id for p in spy.calls[0]["posts"]] == ["101"]
