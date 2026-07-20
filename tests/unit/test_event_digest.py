"""Unit tests for the event-scan digest (CL-b92t).

Covers digest formatting (threshold filter, instrument dedup, cap,
HTML escaping of hostile GDELT headlines), the never-spam rule
(empty cycle / nothing above threshold → no message), the
scripts/event_pipeline.py flag wiring (--digest/--no-digest +
EVENT_DIGEST_MIN_URGENCY), and the CL-mgcp enrichment (price/change
tokens, per-event age, Ideas:/Fade: sections, token line wrapping).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.events.digest import (
    DEFAULT_MIN_URGENCY,
    MAX_EVENTS,
    MAX_IDEAS,
    TOKENS_PER_LINE,
    build_digest,
    send_digest,
)
from src.events.impact_agent import AssessmentResult
from src.research.notifications import DispatchResult

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _res(
    event_id: int = 1,
    headline: str = "Strait of Hormuz closed",
    theme: str | None = "energy_chokepoint",
    status: str = "ASSESSED",
    urgency: Any = 7,
    affected: list[dict[str, str]] | None = None,
) -> AssessmentResult:
    return AssessmentResult(
        event_id=event_id,
        headline=headline,
        theme=theme,
        status=status,
        assessment={
            "core_event": "x",
            "direction": "bearish",
            "urgency": urgency,
            "horizon": "hours",
            "confidence": 0.8,
            "affected": affected if affected is not None else [],
            "rationale": "r",
        },
    )


def _aff(instrument: str, kind: str = "oanda", direction: str = "long") -> dict[str, str]:
    return {
        "instrument": instrument, "kind": kind,
        "direction": direction, "reason": "why",
    }


@pytest.fixture
def notify_recorder(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record src.events.digest.notify_operator calls; no network."""
    calls: list[dict[str, Any]] = []

    def fake_notify(title: str, message: str, priority: int = 0,
                    *, html: bool = False) -> DispatchResult:
        calls.append({
            "title": title, "message": message,
            "priority": priority, "html": html,
        })
        return DispatchResult(telegram_attempted=True, telegram_succeeded=True)

    monkeypatch.setattr("src.events.digest.notify_operator", fake_notify)
    return calls


# --------------------------------------------------------------------- #
# Threshold filter + never-spam
# --------------------------------------------------------------------- #


class TestThreshold:
    def test_empty_cycle_builds_nothing(self) -> None:
        assert build_digest([]) is None

    def test_all_below_threshold_builds_nothing(self) -> None:
        assert build_digest([_res(urgency=4), _res(event_id=2, urgency=1)]) is None

    def test_dismissed_events_excluded(self) -> None:
        dismissed = AssessmentResult(
            event_id=9, headline="junk", theme=None,
            status="DISMISSED", assessment={"rationale": "parse failure"},
        )
        assert build_digest([dismissed]) is None

    def test_threshold_is_inclusive(self) -> None:
        built = build_digest([_res(urgency=5)], min_urgency=5)
        assert built is not None
        assert "<b>5/10</b>" in built[1]

    def test_custom_threshold_filters(self) -> None:
        results = [_res(event_id=1, urgency=6), _res(event_id=2, urgency=8,
                                                     headline="big one")]
        built = build_digest(results, min_urgency=7)
        assert built is not None
        title, message = built
        assert "1 event," in title
        assert "big one" in message
        assert "Hormuz" not in message

    def test_garbage_urgency_treated_as_zero(self) -> None:
        assert build_digest([_res(urgency="high")]) is None

    def test_send_digest_silent_on_empty(
        self, notify_recorder: list[dict[str, Any]],
    ) -> None:
        assert send_digest([]) is None
        assert send_digest([_res(urgency=2)]) is None
        assert notify_recorder == []


# --------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------- #


class TestFormatting:
    def test_header_counts_events_and_tradables(self) -> None:
        results = [
            _res(event_id=1, urgency=7,
                 affected=[_aff("BCO_USD"), _aff("XAU_USD")]),
            _res(event_id=2, urgency=6,
                 affected=[_aff("BCO_USD"), _aff("NVDA", kind="equity_watch",
                                                 direction="watch")]),
        ]
        built = build_digest(results)
        assert built is not None
        title, _ = built
        assert title == "Event scan — 2 events, 2 tradable instruments"

    def test_groups_by_theme_with_urgency_lines(self) -> None:
        results = [
            _res(event_id=1, theme="energy_chokepoint", urgency=9,
                 headline="Hormuz shut"),
            _res(event_id=2, theme="cb_surprise", urgency=6,
                 headline="SNB shock cut"),
            _res(event_id=3, theme="energy_chokepoint", urgency=5,
                 headline="Tanker seized"),
        ]
        built = build_digest(results)
        assert built is not None
        _, message = built
        lines = message.split("\n")
        # Theme headers italic; most urgent theme first; events sorted
        # urgency-desc inside their group.
        assert lines[0] == "<i>energy_chokepoint</i>"
        assert lines[1] == "<b>9/10</b> Hormuz shut"
        assert lines[2] == "<b>5/10</b> Tanker seized"
        assert "<i>cb_surprise</i>" in lines
        assert "<b>6/10</b> SNB shock cut" in lines

    def test_missing_theme_grouped_as_other(self) -> None:
        built = build_digest([_res(theme=None)])
        assert built is not None
        assert "<i>other</i>" in built[1]

    def test_headline_truncated(self) -> None:
        long_headline = "X" * 200
        built = build_digest([_res(headline=long_headline)])
        assert built is not None
        _, message = built
        event_line = next(
            ln for ln in message.split("\n") if ln.startswith("<b>7/10</b>")
        )
        assert len(event_line) < 120
        assert event_line.endswith("…")

    def test_hostile_headline_html_escaped(self) -> None:
        built = build_digest([
            _res(headline='<script>alert("pwn")</script> oil & gas <b>up</b>'),
        ])
        assert built is not None
        _, message = built
        assert "<script>" not in message
        assert "<b>up</b>" not in message
        assert "&lt;script&gt;" in message
        assert "oil &amp; gas" in message

    def test_hostile_theme_and_instrument_escaped(self) -> None:
        built = build_digest([
            _res(theme="a<&>b", affected=[_aff("EUR<USD", direction="long")]),
        ])
        assert built is not None
        _, message = built
        assert "<i>a&lt;&amp;&gt;b</i>" in message
        assert "EUR&lt;USD" in message


class TestInstrumentLines:
    def test_tradable_line_deduped_with_arrows(self) -> None:
        results = [
            _res(event_id=1, urgency=7,
                 affected=[_aff("BCO_USD", direction="long"),
                           _aff("USD_JPY", kind="fx", direction="short")]),
            _res(event_id=2, urgency=6,
                 affected=[_aff("BCO_USD", direction="long")]),
        ]
        built = build_digest(results)
        assert built is not None
        _, message = built
        tradable_line = next(
            ln for ln in message.split("\n") if ln.startswith("<b>Tradable:</b>")
        )
        assert tradable_line == "<b>Tradable:</b> BCO_USD↑ USD_JPY↓"
        assert tradable_line.count("BCO_USD") == 1

    def test_conflicting_directions_drop_arrow(self) -> None:
        results = [
            _res(event_id=1, urgency=7, affected=[_aff("BCO_USD", direction="long")]),
            _res(event_id=2, urgency=6, affected=[_aff("BCO_USD", direction="short")]),
        ]
        built = build_digest(results)
        assert built is not None
        assert "<b>Tradable:</b> BCO_USD\n" in built[1] + "\n"

    def test_watch_line_unions_watch_only_instruments(self) -> None:
        results = [
            _res(event_id=1, urgency=7, affected=[
                _aff("FRO", kind="equity_watch", direction="watch"),
                _aff("STNG", kind="equity_watch", direction="watch"),
                _aff("EUR_NOK", kind="fx", direction="watch"),  # demoted tradable
            ]),
            _res(event_id=2, urgency=6, affected=[
                _aff("FRO", kind="equity_watch", direction="watch"),
            ]),
        ]
        built = build_digest(results)
        assert built is not None
        _, message = built
        watch_line = next(
            ln for ln in message.split("\n") if ln.startswith("<b>Watch:</b>")
        )
        assert watch_line == "<b>Watch:</b> FRO STNG EUR_NOK"

    def test_tradable_wins_over_watch_for_same_instrument(self) -> None:
        results = [
            _res(event_id=1, urgency=7, affected=[_aff("XAU_USD", direction="long")]),
            _res(event_id=2, urgency=6,
                 affected=[_aff("XAU_USD", direction="watch")]),
        ]
        built = build_digest(results)
        assert built is not None
        _, message = built
        assert "<b>Tradable:</b> XAU_USD↑" in message
        assert "<b>Watch:</b>" not in message

    def test_no_instrument_lines_when_no_affected(self) -> None:
        built = build_digest([_res(affected=[])])
        assert built is not None
        _, message = built
        assert "Tradable:" not in message
        assert "Watch:" not in message

    def test_malformed_affected_ignored(self) -> None:
        r = _res()
        r.assessment["affected"] = "not-a-list"
        built = build_digest([r])
        assert built is not None
        assert "Tradable:" not in built[1]


class TestCap:
    def test_event_lines_capped_with_more_note(self) -> None:
        results = [
            _res(event_id=i, urgency=5 + (i % 5), headline=f"event number {i}")
            for i in range(MAX_EVENTS + 4)
        ]
        built = build_digest(results)
        assert built is not None
        title, message = built
        event_lines = [
            ln for ln in message.split("\n") if ln.startswith("<b>") and "/10</b>" in ln
        ]
        assert len(event_lines) == MAX_EVENTS
        assert f"+4 more above urgency {DEFAULT_MIN_URGENCY}" in message
        # Header still counts every qualifying event.
        assert f"{MAX_EVENTS + 4} events" in title

    def test_instrument_union_covers_elided_events(self) -> None:
        # The lowest-urgency event is elided from the lines but its
        # ticker still makes the Tradable feed.
        results = [
            _res(event_id=i, urgency=9, headline=f"e{i}")
            for i in range(MAX_EVENTS)
        ]
        results.append(
            _res(event_id=99, urgency=5, headline="elided",
                 affected=[_aff("NATGAS_USD", direction="long")]),
        )
        built = build_digest(results)
        assert built is not None
        _, message = built
        assert "elided" not in message
        assert "<b>Tradable:</b> NATGAS_USD↑" in message


# --------------------------------------------------------------------- #
# Enrichment (CL-mgcp): prices, age, Ideas/Fade, wrapping
# --------------------------------------------------------------------- #


def _idea(**overrides: Any) -> dict[str, Any]:
    idea = {
        "ticker": "TSM",
        "action": "buy_puts",
        "direction": "bearish",
        "confidence": 0.7,
        "rationale": "advanced-node concentration risk",
        "time_horizon": "short",
        "holding_period_days": "2-6",
        "time_stop_days": 5,
        # Concrete levels (CL-jiqq) — percentages the grounder converts
        # to dollar stop/targets/R:R when a price is present.
        "stop_loss_pct": 0.07,
        "target_pct": [0.10, 0.18],
        "entry_trigger": "on confirmed blockade language",
        "invalidation": "official denial",
        "suggested_entry": "",
        "preferred_instrument": "",
        "notes": "",
    }
    idea.update(overrides)
    return idea


_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)


class TestPriceAnnotations:
    def test_tradable_token_gains_price_change_arrow(self) -> None:
        built = build_digest(
            [_res(urgency=7, affected=[_aff("BCO_USD", direction="long")])],
            prices={"BCO_USD": {"price": 78.4, "change_pct": 2.08}},
        )
        assert built is not None
        assert "<b>Tradable:</b> BCO_USD 78.4 (+2.1%)↑" in built[1]

    def test_watch_token_price_and_rvol_mark(self) -> None:
        built = build_digest(
            [_res(urgency=7, affected=[
                _aff("FRO", kind="equity_watch", direction="watch"),
            ])],
            volume_marks={"FRO": 3.2},
            prices={"FRO": {"price": 24.1, "change_pct": 3.2}},
        )
        assert built is not None
        assert "<b>Watch:</b> FRO $24.10 (+3.2%)×3.2" in built[1]

    def test_unpriced_ticker_renders_bare(self) -> None:
        built = build_digest(
            [_res(urgency=7, affected=[_aff("BCO_USD", direction="long"),
                                       _aff("XAU_USD", direction="short")])],
            prices={"BCO_USD": {"price": 78.4, "change_pct": 2.08}},
        )
        assert built is not None
        assert "BCO_USD 78.4 (+2.1%)↑" in built[1]
        assert "XAU_USD↓" in built[1]

    def test_token_lines_wrap_above_limit(self) -> None:
        affected = [
            _aff(f"T{i}", kind="equity_watch", direction="watch")
            for i in range(TOKENS_PER_LINE + 2)
        ]
        built = build_digest([_res(urgency=7, affected=affected)])
        assert built is not None
        lines = built[1].split("\n")
        watch_idx = next(
            i for i, ln in enumerate(lines) if ln.startswith("<b>Watch:</b>")
        )
        assert lines[watch_idx].count(" ") == TOKENS_PER_LINE  # label + 6 tokens
        assert lines[watch_idx + 1] == f"T{TOKENS_PER_LINE} T{TOKENS_PER_LINE + 1}"


class TestEventAge:
    def test_age_appended_when_seen_at_known(self) -> None:
        built = build_digest(
            [_res(event_id=7, urgency=7)],
            seen_ats={7: _NOW - timedelta(hours=2)},
            now=_NOW,
        )
        assert built is not None
        assert "<b>7/10</b> Strait of Hormuz closed (2h ago)" in built[1]

    def test_unknown_seen_at_renders_no_age(self) -> None:
        built = build_digest([_res(event_id=7, urgency=7)], seen_ats={}, now=_NOW)
        assert built is not None
        assert "ago)" not in built[1]

    def test_string_seen_at_parsed(self) -> None:
        built = build_digest(
            [_res(event_id=7, urgency=7)],
            seen_ats={7: (_NOW - timedelta(days=3)).isoformat()},
            now=_NOW,
        )
        assert built is not None
        assert "(3d ago)" in built[1]


class TestIdeasSection:
    def _with_ideas(self, *ideas: dict[str, Any], **kw: Any) -> str:
        r = _res(urgency=7)
        r.assessment["trade_ideas"] = list(ideas)
        built = build_digest([r], **kw)
        assert built is not None
        return built[1]

    def test_idea_line_full_format(self) -> None:
        # CL-jiqq: the block carries GROUNDED dollar levels computed from
        # the LLM percentages + the real price. buy_puts is bearish →
        # stop ABOVE spot, targets BELOW. Multi-line: bold ticker+price+
        # action, full entry trigger, then the number segments.
        message = self._with_ideas(
            _idea(),
            prices={"TSM": {"price": 172.4, "change_pct": -1.8}},
        )
        assert "<b>Ideas:</b>" in message
        head = next(
            ln for ln in message.split("\n") if ln.startswith("<b>TSM</b>")
        )
        assert head.startswith("<b>TSM</b> $172.40 (-1.8%) — BUY PUTS 1-3wk")
        assert "entry: on confirmed blockade" in message
        assert "stop $186" in message            # 172.4 × 1.08 (option underlying)
        assert "tgt $155/$141" in message         # 172.4 × 0.90 / 0.82
        assert "R:R" in message
        assert "5d stop" in message

    def test_idea_line_without_price(self) -> None:
        # No live price → no dollar levels, but action / DTE / trigger /
        # time stop still render (the honest %-only card). Multi-line
        # block: bold ticker + action on line 1, segments on line 3.
        message = self._with_ideas(_idea())
        head = next(
            ln for ln in message.split("\n") if ln.startswith("<b>TSM</b>")
        )
        assert head.startswith("<b>TSM</b> — BUY PUTS 1-3wk")
        assert "stop $" not in message            # no price → no dollar stop
        assert "5d stop" in message

    def test_long_entry_trigger_shown_in_full(self) -> None:
        # CL-jiqq follow-up: entry conditions are the operator's action
        # signal — they must NOT be truncated mid-sentence. A 200-char
        # trigger (under the 240 bound) renders whole, no ellipsis.
        trigger = "Enter only on confirmed blockade language " + "R" * 150
        message = self._with_ideas(_idea(entry_trigger=trigger))
        assert f"entry: {trigger}" in message
        assert "…" not in message

    def test_hostile_idea_fields_escaped(self) -> None:
        message = self._with_ideas(_idea(
            ticker="<TSM&>", entry_trigger='<script>alert("x")</script>',
        ))
        assert "<script>" not in message
        assert "&lt;TSM&amp;&gt;" in message
        assert "&lt;script&gt;" in message

    def test_ideas_deduped_across_events(self) -> None:
        r1 = _res(event_id=1, urgency=9)
        r1.assessment["trade_ideas"] = [_idea(ticker="DUP")]
        r2 = _res(event_id=2, urgency=6)
        r2.assessment["trade_ideas"] = [_idea(ticker="DUP"),
                                        _idea(ticker="OTHER")]
        built = build_digest([r1, r2])
        assert built is not None
        message = built[1]
        assert message.count("<b>DUP</b> —") == 1  # deduped on (ticker, action)
        assert "<b>OTHER</b> —" in message

    def test_ideas_capped_with_more_note(self) -> None:
        message = self._with_ideas(
            *[_idea(ticker=f"AA{i}") for i in range(MAX_IDEAS + 2)],
        )
        idea_lines = [ln for ln in message.split("\n") if ln.startswith("<b>AA")]
        assert len(idea_lines) == MAX_IDEAS
        assert "+2 more ideas" in message

    def test_no_ideas_no_section(self) -> None:
        built = build_digest([_res(urgency=7)])
        assert built is not None
        assert "Ideas:" not in built[1]


class TestFadeSection:
    def test_fade_lines(self) -> None:
        r = _res(urgency=7)
        r.assessment["fade_candidates"] = [
            {"ticker": "NVDA", "action": "fade the spike",
             "reason": "routine drills, priced in"},
        ]
        built = build_digest([r])
        assert built is not None
        assert "<b>Fade:</b>" in built[1]
        # Multi-line fade block: bold ticker + action, full reason on line 2.
        assert "<b>NVDA</b> — fade the spike" in built[1]
        assert "routine drills, priced in" in built[1]

    def test_hostile_fade_escaped(self) -> None:
        r = _res(urgency=7)
        r.assessment["fade_candidates"] = [
            {"ticker": "NVDA", "action": "<b>fade</b>", "reason": "a & b"},
        ]
        built = build_digest([r])
        assert built is not None
        assert "<b>fade</b>" not in built[1]
        assert "&lt;b&gt;fade&lt;/b&gt;" in built[1]
        assert "a &amp; b" in built[1]

    def test_no_fades_no_section(self) -> None:
        built = build_digest([_res(urgency=7)])
        assert built is not None
        assert "Fade:" not in built[1]


class TestSendDigestEnrichmentForwarding:
    def test_prices_and_seen_ats_reach_the_message(
        self, notify_recorder: list[dict[str, Any]],
    ) -> None:
        r = _res(event_id=5, urgency=7,
                 affected=[_aff("BCO_USD", direction="long")])
        result = send_digest(
            [r],
            prices={"BCO_USD": {"price": 78.4, "change_pct": 2.08}},
            seen_ats={5: datetime.now(UTC) - timedelta(hours=2)},
        )
        assert result is not None and result.telegram_succeeded
        message = notify_recorder[0]["message"]
        assert "BCO_USD 78.4 (+2.1%)↑" in message
        assert "(2h ago)" in message


# --------------------------------------------------------------------- #
# send_digest dispatch
# --------------------------------------------------------------------- #


class TestSendDigest:
    def test_sends_html_via_notify_operator(
        self, notify_recorder: list[dict[str, Any]],
    ) -> None:
        result = send_digest([
            _res(urgency=7, affected=[_aff("BCO_USD", direction="long")]),
        ])
        assert result is not None and result.telegram_succeeded
        assert len(notify_recorder) == 1
        call = notify_recorder[0]
        assert call["html"] is True
        assert call["title"].startswith("Event scan — 1 event, 1 tradable")
        assert "<b>7/10</b>" in call["message"]
        assert "<b>Tradable:</b> BCO_USD↑" in call["message"]

    def test_min_urgency_forwarded(
        self, notify_recorder: list[dict[str, Any]],
    ) -> None:
        assert send_digest([_res(urgency=6)], min_urgency=7) is None
        assert notify_recorder == []


# --------------------------------------------------------------------- #
# scripts/event_pipeline.py wiring
# --------------------------------------------------------------------- #


class _FakeAgent:
    """Stands in for EventImpactAgent inside _cycle; no DB, no LLM."""

    results: list[AssessmentResult] = []

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def assess_new_events(self, limit: int = 20) -> list[AssessmentResult]:
        return list(self.results)[:limit]


class _NoopScanner:
    """Stands in for RelativeVolumeScanner inside _cycle (--scan is
    default-on since CL-i4sr); no yfinance, no DB."""

    def __init__(self, _db_url: str, **_kw: Any) -> None:
        pass

    def scan(self, persist: bool = True) -> list[Any]:
        return []


@pytest.fixture
def pipeline_mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    from scripts import event_pipeline as mod

    monkeypatch.setattr(
        "src.events.impact_agent.EventImpactAgent", _FakeAgent,
    )
    monkeypatch.setattr(
        "src.scanners.relative_volume.RelativeVolumeScanner", _NoopScanner,
    )
    monkeypatch.setattr("sqlalchemy.create_engine", lambda _url: None)
    return mod


class TestPipelineWiring:
    def test_digest_flag_default_on(self, pipeline_mod: Any) -> None:
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        assert args.digest is True
        assert args.digest_min_urgency is None  # resolved later vs env

    def test_no_digest_flag(self, pipeline_mod: Any) -> None:
        args = pipeline_mod._build_parser().parse_args(
            ["--assess", "--no-digest"],
        )
        assert args.digest is False

    def test_digest_min_urgency_cli(self, pipeline_mod: Any) -> None:
        args = pipeline_mod._build_parser().parse_args(
            ["--assess", "--digest-min-urgency", "8"],
        )
        assert args.digest_min_urgency == 8

    def test_resolve_min_urgency_cli_beats_env(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EVENT_DIGEST_MIN_URGENCY", "3")
        assert pipeline_mod._resolve_digest_min_urgency(8) == 8

    def test_resolve_min_urgency_env(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EVENT_DIGEST_MIN_URGENCY", "3")
        assert pipeline_mod._resolve_digest_min_urgency(None) == 3

    def test_resolve_min_urgency_default(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("EVENT_DIGEST_MIN_URGENCY", raising=False)
        assert pipeline_mod._resolve_digest_min_urgency(None) == DEFAULT_MIN_URGENCY

    def test_resolve_min_urgency_garbage_env_falls_back(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EVENT_DIGEST_MIN_URGENCY", "loud")
        assert pipeline_mod._resolve_digest_min_urgency(None) == DEFAULT_MIN_URGENCY

    def _run_cycle(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        results: list[AssessmentResult],
    ) -> list[dict[str, Any]]:
        _FakeAgent.results = results
        sent: list[dict[str, Any]] = []

        def fake_send(res: Any, min_urgency: int = 5, **kw: Any) -> DispatchResult:
            sent.append({"results": list(res), "min_urgency": min_urgency})
            return DispatchResult(telegram_attempted=True, telegram_succeeded=True)

        monkeypatch.setattr("src.events.digest.send_digest", fake_send)
        args = pipeline_mod._build_parser().parse_args(argv)
        args.digest_min_urgency = pipeline_mod._resolve_digest_min_urgency(
            args.digest_min_urgency,
        )
        pipeline_mod._cycle(args)
        return sent

    def test_assess_cycle_calls_send_digest(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("EVENT_DIGEST_MIN_URGENCY", raising=False)
        results = [_res(urgency=7)]
        sent = self._run_cycle(
            pipeline_mod, monkeypatch,
            ["--assess", "--digest-min-urgency", "6"], results,
        )
        assert len(sent) == 1
        assert sent[0]["results"] == results
        assert sent[0]["min_urgency"] == 6

    def test_no_digest_skips_send(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sent = self._run_cycle(
            pipeline_mod, monkeypatch,
            ["--assess", "--no-digest"], [_res(urgency=9)],
        )
        assert sent == []

    def test_digest_failure_does_not_crash_cycle(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _FakeAgent.results = [_res(urgency=9)]

        def boom(*_a: Any, **_kw: Any) -> DispatchResult:
            raise RuntimeError("telegram exploded")

        monkeypatch.setattr("src.events.digest.send_digest", boom)
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        args.digest_min_urgency = 5
        pipeline_mod._cycle(args)  # must not raise


class TestPipelineIdeaWiring:
    """CL-mgcp: _cycle persists ideas, expires stale ones, and feeds
    the one-per-cycle price batch into the digest."""

    def _wire(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        calls: dict[str, Any] = {"persist": [], "expire": 0, "sent": []}

        def fake_get_prices(tickers: Any, engine: Any = None, **_kw: Any) -> dict:
            calls["price_tickers"] = sorted(tickers)
            return {"TSM": {"price": 172.4, "change_pct": -1.8}}

        def fake_persist(engine: Any, eid: int, assessment: Any,
                         prices: Any = None, now: Any = None) -> int:
            calls["persist"].append((eid, prices))
            return 1

        def fake_expire(engine: Any, now: Any = None) -> int:
            calls["expire"] += 1
            return 0

        def fake_send(res: Any, min_urgency: int = 5, **kw: Any) -> DispatchResult:
            calls["sent"].append(kw)
            return DispatchResult(telegram_attempted=True, telegram_succeeded=True)

        monkeypatch.setattr("src.events.prices.get_prices", fake_get_prices)
        monkeypatch.setattr("src.events.idea_ledger.persist_ideas", fake_persist)
        monkeypatch.setattr("src.events.idea_ledger.expire_stale", fake_expire)
        monkeypatch.setattr("src.events.digest.send_digest", fake_send)
        return calls

    def _result_with_idea(self, event_id: int = 7) -> AssessmentResult:
        r = _res(event_id=event_id, urgency=7,
                 affected=[_aff("BCO_USD", direction="long")])
        r.assessment["trade_ideas"] = [
            {"ticker": "TSM", "action": "buy_puts", "time_stop_days": 5},
        ]
        return r

    def test_assessed_events_persisted_with_batch_prices(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._wire(monkeypatch)
        _FakeAgent.results = [self._result_with_idea()]
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        args.digest_min_urgency = 5
        pipeline_mod._cycle(args)
        # One price batch for affected + idea tickers, shared everywhere.
        assert calls["price_tickers"] == ["BCO_USD", "TSM"]
        assert calls["persist"] == [
            (7, {"TSM": {"price": 172.4, "change_pct": -1.8}}),
        ]
        assert calls["expire"] == 1  # stale sweep runs every cycle
        assert calls["sent"][0]["prices"] == {
            "TSM": {"price": 172.4, "change_pct": -1.8},
        }

    def test_dismissed_events_not_persisted(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._wire(monkeypatch)
        _FakeAgent.results = [AssessmentResult(
            event_id=9, headline="junk", theme=None,
            status="DISMISSED", assessment={"rationale": "parse failure"},
        )]
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        args.digest_min_urgency = 5
        pipeline_mod._cycle(args)
        assert calls["persist"] == []
        assert calls["expire"] == 1

    def test_ledger_failure_never_kills_cycle_or_digest(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._wire(monkeypatch)

        def boom(*_a: Any, **_kw: Any) -> int:
            raise RuntimeError("db down")

        monkeypatch.setattr("src.events.idea_ledger.persist_ideas", boom)
        _FakeAgent.results = [self._result_with_idea()]
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        args.digest_min_urgency = 5
        pipeline_mod._cycle(args)  # must not raise
        assert len(calls["sent"]) == 1  # digest still went out


# --------------------------------------------------------------------- #
# Prediction-market corroboration line (CL-r1ep)
# --------------------------------------------------------------------- #


class _FakePolySignal:
    """Stub PolymarketSignal exposing only latest_prob_for_theme."""

    def __init__(self, by_theme: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.by_theme = by_theme

    def latest_prob_for_theme(self, theme: str) -> dict[str, dict[str, Any]]:
        return self.by_theme.get(theme, {})


class TestPolyCorroboration:
    def test_line_rendered_for_theme(self) -> None:
        poly = _FakePolySignal({
            "energy_chokepoint": {
                "hormuz-closure-2026": {
                    "question": "Hormuz closed?", "yes_prob": 0.18,
                    "rising": True,
                },
            },
        })
        built = build_digest([_res(urgency=7)], poly_signal=poly)
        assert built is not None
        _, msg = built
        assert "Prediction mkt:" in msg
        assert "hormuz-closure-2026 18% ↑" in msg

    def test_falling_arrow(self) -> None:
        poly = _FakePolySignal({
            "energy_chokepoint": {
                "s": {"question": "q", "yes_prob": 0.40, "rising": False},
            },
        })
        _, msg = build_digest([_res(urgency=7)], poly_signal=poly)  # type: ignore[misc]
        assert "s 40% ↓" in msg

    def test_no_arrow_when_direction_unknown(self) -> None:
        poly = _FakePolySignal({
            "energy_chokepoint": {
                "s": {"question": "q", "yes_prob": 0.25, "rising": None},
            },
        })
        _, msg = build_digest([_res(urgency=7)], poly_signal=poly)  # type: ignore[misc]
        assert "s 25%" in msg
        assert "s 25% ↑" not in msg and "s 25% ↓" not in msg

    def test_highest_prob_market_cited(self) -> None:
        poly = _FakePolySignal({
            "energy_chokepoint": {
                "low": {"question": "q", "yes_prob": 0.10, "rising": None},
                "high": {"question": "q", "yes_prob": 0.55, "rising": True},
            },
        })
        _, msg = build_digest([_res(urgency=7)], poly_signal=poly)  # type: ignore[misc]
        assert "high 55%" in msg
        assert "low 10%" not in msg

    def test_absent_signal_no_line(self) -> None:
        _, msg = build_digest([_res(urgency=7)], poly_signal=None)  # type: ignore[misc]
        assert "Prediction mkt" not in msg

    def test_lookup_failure_fail_soft(self) -> None:
        class Boom:
            def latest_prob_for_theme(self, theme: str) -> dict[str, Any]:
                raise RuntimeError("db down")

        built = build_digest([_res(urgency=7)], poly_signal=Boom())
        assert built is not None  # digest still builds
        _, msg = built
        assert "Prediction mkt" not in msg
