"""Unit tests for the event-scan digest (CL-b92t).

Covers digest formatting (threshold filter, instrument dedup, cap,
HTML escaping of hostile GDELT headlines), the never-spam rule
(empty cycle / nothing above threshold → no message), and the
scripts/event_pipeline.py flag wiring (--digest/--no-digest +
EVENT_DIGEST_MIN_URGENCY).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.events.digest import (
    DEFAULT_MIN_URGENCY,
    MAX_EVENTS,
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


@pytest.fixture
def pipeline_mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    from scripts import event_pipeline as mod

    monkeypatch.setattr(
        "src.events.impact_agent.EventImpactAgent", _FakeAgent,
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
