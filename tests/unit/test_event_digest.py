"""Unit tests for the event-scan digest (CL-b92t).

Covers digest formatting (threshold filter, instrument dedup, cap,
HTML escaping of hostile GDELT headlines), the never-spam rule
(empty cycle / nothing above threshold → no message), the
scripts/event_pipeline.py flag wiring (--digest/--no-digest +
EVENT_DIGEST_MIN_URGENCY), and the CL-mgcp enrichment (price/change
tokens, per-event age, Ideas:/Fade: sections, token line wrapping).
"""

from __future__ import annotations

import threading
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
        "instrument": instrument,
        "kind": kind,
        "direction": direction,
        "reason": "why",
    }


@pytest.fixture
def notify_recorder(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record src.events.digest.notify_operator calls; no network."""
    calls: list[dict[str, Any]] = []

    def fake_notify(
        title: str, message: str, priority: int = 0, *, html: bool = False
    ) -> DispatchResult:
        calls.append(
            {
                "title": title,
                "message": message,
                "priority": priority,
                "html": html,
            }
        )
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
            event_id=9,
            headline="junk",
            theme=None,
            status="DISMISSED",
            assessment={"rationale": "parse failure"},
        )
        assert build_digest([dismissed]) is None

    def test_threshold_is_inclusive(self) -> None:
        built = build_digest([_res(urgency=5)], min_urgency=5)
        assert built is not None
        assert "<b>5/10</b>" in built[1]

    def test_custom_threshold_filters(self) -> None:
        results = [_res(event_id=1, urgency=6), _res(event_id=2, urgency=8, headline="big one")]
        built = build_digest(results, min_urgency=7)
        assert built is not None
        title, message = built
        assert "1 event," in title
        assert "big one" in message
        assert "Hormuz" not in message

    def test_garbage_urgency_treated_as_zero(self) -> None:
        assert build_digest([_res(urgency="high")]) is None

    def test_send_digest_silent_on_empty(
        self,
        notify_recorder: list[dict[str, Any]],
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
            _res(event_id=1, urgency=7, affected=[_aff("BCO_USD"), _aff("XAU_USD")]),
            _res(
                event_id=2,
                urgency=6,
                affected=[_aff("BCO_USD"), _aff("NVDA", kind="equity_watch", direction="watch")],
            ),
        ]
        built = build_digest(results)
        assert built is not None
        title, _ = built
        assert title == "Event scan — 2 events, 2 tradable instruments"

    def test_groups_by_theme_with_urgency_lines(self) -> None:
        results = [
            _res(event_id=1, theme="energy_chokepoint", urgency=9, headline="Hormuz shut"),
            _res(event_id=2, theme="cb_surprise", urgency=6, headline="SNB shock cut"),
            _res(event_id=3, theme="energy_chokepoint", urgency=5, headline="Tanker seized"),
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
        event_line = next(ln for ln in message.split("\n") if ln.startswith("<b>7/10</b>"))
        assert len(event_line) < 120
        assert event_line.endswith("…")

    def test_hostile_headline_html_escaped(self) -> None:
        built = build_digest(
            [
                _res(headline='<script>alert("pwn")</script> oil & gas <b>up</b>'),
            ]
        )
        assert built is not None
        _, message = built
        assert "<script>" not in message
        assert "<b>up</b>" not in message
        assert "&lt;script&gt;" in message
        assert "oil &amp; gas" in message

    def test_hostile_theme_and_instrument_escaped(self) -> None:
        built = build_digest(
            [
                _res(theme="a<&>b", affected=[_aff("EUR<USD", direction="long")]),
            ]
        )
        assert built is not None
        _, message = built
        assert "<i>a&lt;&amp;&gt;b</i>" in message
        assert "EUR&lt;USD" in message


class TestInstrumentLines:
    def test_tradable_line_deduped_with_arrows(self) -> None:
        results = [
            _res(
                event_id=1,
                urgency=7,
                affected=[
                    _aff("BCO_USD", direction="long"),
                    _aff("USD_JPY", kind="fx", direction="short"),
                ],
            ),
            _res(event_id=2, urgency=6, affected=[_aff("BCO_USD", direction="long")]),
        ]
        built = build_digest(results)
        assert built is not None
        _, message = built
        tradable_line = next(ln for ln in message.split("\n") if ln.startswith("<b>Tradable:</b>"))
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
            _res(
                event_id=1,
                urgency=7,
                affected=[
                    _aff("FRO", kind="equity_watch", direction="watch"),
                    _aff("STNG", kind="equity_watch", direction="watch"),
                    _aff("EUR_NOK", kind="fx", direction="watch"),  # demoted tradable
                ],
            ),
            _res(
                event_id=2,
                urgency=6,
                affected=[
                    _aff("FRO", kind="equity_watch", direction="watch"),
                ],
            ),
        ]
        built = build_digest(results)
        assert built is not None
        _, message = built
        watch_line = next(ln for ln in message.split("\n") if ln.startswith("<b>Watch:</b>"))
        assert watch_line == "<b>Watch:</b> FRO STNG EUR_NOK"

    def test_tradable_wins_over_watch_for_same_instrument(self) -> None:
        results = [
            _res(event_id=1, urgency=7, affected=[_aff("XAU_USD", direction="long")]),
            _res(event_id=2, urgency=6, affected=[_aff("XAU_USD", direction="watch")]),
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
        event_lines = [ln for ln in message.split("\n") if ln.startswith("<b>") and "/10</b>" in ln]
        assert len(event_lines) == MAX_EVENTS
        assert f"+4 more above urgency {DEFAULT_MIN_URGENCY}" in message
        # Header still counts every qualifying event.
        assert f"{MAX_EVENTS + 4} events" in title

    def test_instrument_union_covers_elided_events(self) -> None:
        # The lowest-urgency event is elided from the lines but its
        # ticker still makes the Tradable feed.
        results = [_res(event_id=i, urgency=9, headline=f"e{i}") for i in range(MAX_EVENTS)]
        results.append(
            _res(
                event_id=99,
                urgency=5,
                headline="elided",
                affected=[_aff("NATGAS_USD", direction="long")],
            ),
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
            [
                _res(
                    urgency=7,
                    affected=[
                        _aff("FRO", kind="equity_watch", direction="watch"),
                    ],
                )
            ],
            volume_marks={"FRO": 3.2},
            prices={"FRO": {"price": 24.1, "change_pct": 3.2}},
        )
        assert built is not None
        assert "<b>Watch:</b> FRO $24.10 (+3.2%)×3.2" in built[1]

    def test_unpriced_ticker_renders_bare(self) -> None:
        built = build_digest(
            [
                _res(
                    urgency=7,
                    affected=[
                        _aff("BCO_USD", direction="long"),
                        _aff("XAU_USD", direction="short"),
                    ],
                )
            ],
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
        watch_idx = next(i for i, ln in enumerate(lines) if ln.startswith("<b>Watch:</b>"))
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
        head = next(ln for ln in message.split("\n") if ln.startswith("<b>TSM</b>"))
        assert head.startswith("<b>TSM</b> $172.40 (-1.8%) — BUY PUTS 1-3wk")
        assert "entry: on confirmed blockade" in message
        assert "stop $186" in message  # 172.4 × 1.08 (option underlying)
        assert "tgt $155/$141" in message  # 172.4 × 0.90 / 0.82
        assert "R:R" in message
        assert "5d stop" in message

    def test_idea_line_without_price(self) -> None:
        # No live price → no dollar levels, but action / DTE / trigger /
        # time stop still render (the honest %-only card). Multi-line
        # block: bold ticker + action on line 1, segments on line 3.
        message = self._with_ideas(_idea())
        head = next(ln for ln in message.split("\n") if ln.startswith("<b>TSM</b>"))
        assert head.startswith("<b>TSM</b> — BUY PUTS 1-3wk")
        assert "stop $" not in message  # no price → no dollar stop
        assert "5d stop" in message

    def test_rh_proxy_subline_commodity(self) -> None:
        # CL-vowz: every idea block carries an indented Robinhood-proxy
        # sub-line. A LONG XAU_USD idea → the gold ETFs.
        message = self._with_ideas(
            _idea(ticker="XAU_USD", action="long", direction="bullish"),
        )
        assert "  RH: GLD/IAU" in message

    def test_rh_proxy_subline_index_short(self) -> None:
        # A SHORT index idea surfaces the inverse ETFs on the sub-line.
        message = self._with_ideas(
            _idea(ticker="SPX500_USD", action="short", direction="bearish"),
        )
        assert "  RH: SPY→short via SH/SDS" in message

    def test_rh_proxy_subline_fx(self) -> None:
        message = self._with_ideas(
            _idea(ticker="USD_JPY", action="short", direction="bearish"),
        )
        assert "  RH: FX — n/a" in message

    def test_rh_proxy_subline_equity_direct(self) -> None:
        # A plain equity idea (the default _idea ticker is TSM) → direct.
        message = self._with_ideas(_idea())
        assert "  RH: trades directly" in message

    def test_rh_proxy_subline_escaped(self) -> None:
        # A hostile ticker cannot break the sub-line; it renders 'n/a'
        # (unknown) and is escaped like every interpolated value.
        message = self._with_ideas(_idea(ticker="<X&>"))
        assert "<script>" not in message
        # unknown instrument → n/a proxy line still present.
        assert "  RH: n/a" in message

    def test_long_entry_trigger_shown_in_full(self) -> None:
        # CL-jiqq follow-up: entry conditions are the operator's action
        # signal — they must NOT be truncated mid-sentence. A 200-char
        # trigger (under the 240 bound) renders whole, no ellipsis.
        trigger = "Enter only on confirmed blockade language " + "R" * 150
        message = self._with_ideas(_idea(entry_trigger=trigger))
        assert f"entry: {trigger}" in message
        assert "…" not in message

    def test_hostile_idea_fields_escaped(self) -> None:
        message = self._with_ideas(
            _idea(
                ticker="<TSM&>",
                entry_trigger='<script>alert("x")</script>',
            )
        )
        assert "<script>" not in message
        assert "&lt;TSM&amp;&gt;" in message
        assert "&lt;script&gt;" in message

    def test_ideas_deduped_across_events(self) -> None:
        r1 = _res(event_id=1, urgency=9)
        r1.assessment["trade_ideas"] = [_idea(ticker="DUP")]
        r2 = _res(event_id=2, urgency=6)
        r2.assessment["trade_ideas"] = [_idea(ticker="DUP"), _idea(ticker="OTHER")]
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


class TestNicheTag(TestIdeasSection):
    """CL-u2ph: a merged niche idea renders a compact '🎯 niche (N hops,
    asym X)' tag with the torque reason, plus a size-small/check-spread
    caveat when the liquidity floor was tripped. All LLM fields escaped."""

    def _niche_idea(self, **over: Any) -> dict[str, Any]:
        idea = _idea(
            ticker="ABC",
            action="buy_calls",
            direction="bullish",
            niche=True,
            hop_count=4,
            torque_reason="single-asset junior, high operating leverage",
            asymmetry_score=0.71,
            liquidity_flag=False,
        )
        idea.update(over)
        return idea

    def test_niche_tag_renders(self) -> None:
        message = self._with_ideas(self._niche_idea())
        assert "🎯 niche (4 hops, asym 0.71)" in message
        assert "single-asset junior" in message

    def test_singular_hop_label(self) -> None:
        message = self._with_ideas(self._niche_idea(hop_count=1))
        assert "🎯 niche (1 hop, asym" in message  # "hop", not "hops"

    def test_illiquid_caveat_when_flagged(self) -> None:
        message = self._with_ideas(self._niche_idea(liquidity_flag=True))
        assert "⚠ small/illiquid — size small, check spread" in message

    def test_no_caveat_when_liquid(self) -> None:
        message = self._with_ideas(self._niche_idea(liquidity_flag=False))
        assert "size small, check spread" not in message

    def test_non_niche_idea_has_no_tag(self) -> None:
        message = self._with_ideas(_idea())  # plain, no niche key
        assert "🎯 niche" not in message

    def test_torque_reason_html_escaped(self) -> None:
        message = self._with_ideas(
            self._niche_idea(torque_reason="<script>alert(1)</script> levered"),
        )
        assert "<script>" not in message
        assert "&lt;script&gt;" in message

    def test_missing_asymmetry_renders_question_mark(self) -> None:
        message = self._with_ideas(self._niche_idea(asymmetry_score=None))
        assert "asym ?" in message


class TestNicheConsolidation(TestIdeasSection):
    """CL-u2ph: a niche idea corroborated across TWO events still
    consolidates through the existing (ticker, action) dedup — the merged
    niche idea is a normal trade_ideas entry, so redundancy reads as
    conviction, not two duplicate niche lines."""

    def test_niche_idea_consolidates_across_events(self) -> None:
        niche = {
            "ticker": "ABC",
            "action": "buy_calls",
            "direction": "bullish",
            "niche": True,
            "hop_count": 3,
            "torque_reason": "sole supplier",
            "asymmetry_score": 0.68,
            "liquidity_flag": False,
            "rationale": "chain",
            "time_horizon": "short",
            "time_stop_days": 10,
        }
        r1 = _res(event_id=1, urgency=8, theme="critical_minerals")
        r1.assessment["trade_ideas"] = [dict(niche)]
        r2 = _res(event_id=2, urgency=7, theme="taiwan_strait")
        r2.assessment["trade_ideas"] = [dict(niche)]
        built = build_digest([r1, r2])
        assert built is not None
        message = built[1]
        # ONE consolidated ABC idea block (not two), carrying both the
        # niche tag and the cross-event corroboration note.
        assert message.count("<b>ABC</b>") == 1
        assert "🎯 niche (3 hops" in message
        assert "corroborated by 2 events" in message


class TestCorroboration:
    """CL-5mkf: cross-event duplicate ideas are CONVICTION, not noise.
    The kept idea carries a 'corroborated by N events (themes)' note."""

    def _gold_idea(self, **kw: Any) -> dict[str, Any]:
        return _idea(
            ticker="XAU_USD",
            action="long",
            direction="bullish",
            rationale="risk-off bid",
            **kw,
        )

    def test_two_events_same_idea_corroborated_note(self) -> None:
        r1 = _res(event_id=1, theme="energy_chokepoint", urgency=9)
        r1.assessment["trade_ideas"] = [self._gold_idea()]
        r2 = _res(event_id=2, theme="russia_ukraine", urgency=6)
        r2.assessment["trade_ideas"] = [self._gold_idea()]
        built = build_digest([r1, r2])
        assert built is not None
        message = built[1]
        # ONE consolidated idea, with a corroboration note naming BOTH
        # themes (most-urgent-first order).
        assert message.count("<b>XAU_USD</b>") == 1
        assert "✓ corroborated by 2 events" in message
        assert "energy_chokepoint" in message
        assert "russia_ukraine" in message

    def test_single_event_idea_has_no_note(self) -> None:
        r = _res(event_id=1, theme="energy_chokepoint", urgency=8)
        r.assessment["trade_ideas"] = [self._gold_idea()]
        built = build_digest([r])
        assert built is not None
        assert "corroborated by" not in built[1]

    def test_theme_cap_and_escaping(self) -> None:
        # Four distinct themes proposing the SAME idea → 3 shown + "+1
        # more"; a hostile theme is HTML-escaped in the note.
        themes = ["energy_chokepoint", "russia_ukraine", "a<&>b", "sahel_coup"]
        results = []
        for i, th in enumerate(themes, 1):
            r = _res(event_id=i, theme=th, urgency=10 - i)
            r.assessment["trade_ideas"] = [self._gold_idea()]
            results.append(r)
        built = build_digest(results)
        assert built is not None
        message = built[1]
        assert "corroborated by 4 events" in message
        assert "+1 more" in message
        # The hostile theme is escaped wherever it lands.
        assert "a<&>b" not in message
        # First three themes (urgency-desc) are the ones shown inline.
        assert "energy_chokepoint" in message
        assert "russia_ukraine" in message


class TestConcentrationNote:
    """CL-wbmw (generalizes CL-5mkf): a displayed reminder when the
    advisory ideas pile into ANY instrument — corroborated by >1 event OR
    repeated across >= 2 distinct ideas — plus a gold/silver CLUSTER
    variant. Additive display only; never removes/hides an idea."""

    def _gold(self, action: str = "long") -> dict[str, Any]:
        return _idea(ticker="XAU_USD", action=action, direction="bullish", rationale="risk-off")

    def _silver(self) -> dict[str, Any]:
        return _idea(ticker="XAG_USD", action="long", direction="bullish", rationale="risk-off")

    # ---- general per-instrument note (now fires for ANY instrument) ----

    def test_note_when_corroborated_general_instrument(self) -> None:
        # A NON-gold instrument (BCO_USD) corroborated by 2 events → the
        # generalized per-instrument note fires (proves it's no longer
        # gold-only).
        oil = _idea(ticker="BCO_USD", action="long", direction="bullish")
        r1 = _res(event_id=1, theme="energy_chokepoint", urgency=9)
        r1.assessment["trade_ideas"] = [oil]
        r2 = _res(event_id=2, theme="russia_ukraine", urgency=7)
        r2.assessment["trade_ideas"] = [oil]
        built = build_digest([r1, r2])
        assert built is not None
        message = built[1]
        assert "⚠️ already exposed to BCO_USD via 2 ideas — watch concentration" in message
        # Rendered once, in the Ideas section, before the idea blocks.
        assert message.count("watch concentration") == 1
        assert message.index("watch concentration") < message.index("<b>BCO_USD</b>")

    def test_note_when_two_distinct_oil_ideas(self) -> None:
        # Two DISTINCT ideas on the same instrument (different actions) →
        # flagged by the >= 2-ideas rule even without corroboration.
        r = _res(event_id=1, theme="energy_chokepoint", urgency=9)
        r.assessment["trade_ideas"] = [
            _idea(ticker="BCO_USD", action="long"),
            _idea(ticker="BCO_USD", action="buy_calls"),
        ]
        built = build_digest([r])
        assert built is not None
        assert "already exposed to BCO_USD" in built[1]

    def test_escaping_and_line_cap(self) -> None:
        # Four distinct over-weight instruments (each corroborated by 2
        # events) → 3 lines + "+1 more"; a hostile ticker is HTML-escaped.
        tickers = ["BCO_USD", "USD_CAD", "a<&>b", "NAS100_USD"]
        r1 = _res(event_id=1, urgency=9)
        r2 = _res(event_id=2, urgency=8)
        r1.assessment["trade_ideas"] = [_idea(ticker=t, action="long") for t in tickers]
        r2.assessment["trade_ideas"] = [_idea(ticker=t, action="long") for t in tickers]
        built = build_digest([r1, r2])
        assert built is not None
        message = built[1]
        note_lines = [ln for ln in message.split("\n") if "watch concentration" in ln]
        assert len(note_lines) == 3
        assert "+1 more over-weight instruments" in message
        assert "a<&>b" not in message  # raw hostile ticker escaped

    # ---- haven-cluster variant ----

    def test_cluster_note_when_multiple_distinct_havens(self) -> None:
        # Gold + silver, one idea each (neither individually over-weight) →
        # the haven-cluster line fires for the group.
        r = _res(event_id=1, theme="broad_riskoff", urgency=9)
        r.assessment["trade_ideas"] = [self._gold(), self._silver()]
        built = build_digest([r])
        assert built is not None
        assert "⚠️ watch gold/silver (haven) concentration" in built[1]

    def test_prefer_specific_over_cluster(self) -> None:
        # Gold corroborated by 2 events (individually over-weight) + one
        # silver idea. Gold gets its OWN specific line; the remaining lone
        # silver is NOT enough for a cluster line → no double-warn.
        r1 = _res(event_id=1, theme="russia_ukraine", urgency=9)
        r1.assessment["trade_ideas"] = [self._gold(), self._silver()]
        r2 = _res(event_id=2, theme="energy_chokepoint", urgency=7)
        r2.assessment["trade_ideas"] = [self._gold()]
        built = build_digest([r1, r2])
        assert built is not None
        message = built[1]
        assert "already exposed to XAU_USD" in message
        # Only silver remains for the cluster, which alone isn't over-weight.
        assert "watch gold/silver (haven) concentration" not in message

    def test_no_note_for_single_haven_idea(self) -> None:
        r = _res(event_id=1, theme="broad_riskoff", urgency=9)
        r.assessment["trade_ideas"] = [self._gold()]
        built = build_digest([r])
        assert built is not None
        assert "watch concentration" not in built[1]

    def test_no_note_for_single_distinct_idea(self) -> None:
        r1 = _res(event_id=1, urgency=9)
        r1.assessment["trade_ideas"] = [_idea(ticker="BCO_USD", action="long")]
        built = build_digest([r1])
        assert built is not None
        assert "watch concentration" not in built[1]

    def test_variety_preserved_distinct_niche_ideas(self) -> None:
        # GUARDRAIL: 5 DISTINCT niche ideas (all count=1, distinct tickers)
        # render in FULL with NO consolidation and NO concentration note.
        # The variety of niche puts/shorts/longs is the alpha — the
        # concentration work must never reduce it.
        r = _res(event_id=1, theme="pharma_api_supply", urgency=9)
        r.assessment["trade_ideas"] = [
            _idea(ticker="TEVA", action="buy_puts", direction="bearish"),
            _idea(ticker="RDY", action="buy_puts", direction="bearish"),
            _idea(ticker="DHT", action="buy_calls", direction="bullish"),
            _idea(ticker="FRO", action="buy_calls", direction="bullish"),
            _idea(ticker="NVDA", action="short", direction="bearish"),
        ]
        built = build_digest([r])
        assert built is not None
        message = built[1]
        # All five distinct tickers present, each rendered once.
        for t in ("TEVA", "RDY", "DHT", "FRO", "NVDA"):
            assert message.count(f"<b>{t}</b>") == 1
        # No consolidation, no concentration note, no idea elided.
        assert "watch concentration" not in message
        assert "corroborated by" not in message
        assert "more ideas" not in message


class TestFadeSection:
    def test_fade_lines(self) -> None:
        r = _res(urgency=7)
        r.assessment["fade_candidates"] = [
            {"ticker": "NVDA", "action": "fade the spike", "reason": "routine drills, priced in"},
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
        self,
        notify_recorder: list[dict[str, Any]],
    ) -> None:
        r = _res(event_id=5, urgency=7, affected=[_aff("BCO_USD", direction="long")])
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
        self,
        notify_recorder: list[dict[str, Any]],
    ) -> None:
        result = send_digest(
            [
                _res(urgency=7, affected=[_aff("BCO_USD", direction="long")]),
            ]
        )
        assert result is not None and result.telegram_succeeded
        assert len(notify_recorder) == 1
        call = notify_recorder[0]
        assert call["html"] is True
        assert call["title"].startswith("Event scan — 1 event, 1 tradable")
        assert "<b>7/10</b>" in call["message"]
        assert "<b>Tradable:</b> BCO_USD↑" in call["message"]

    def test_min_urgency_forwarded(
        self,
        notify_recorder: list[dict[str, Any]],
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
        "src.events.impact_agent.EventImpactAgent",
        _FakeAgent,
    )
    monkeypatch.setattr(
        "src.scanners.relative_volume.RelativeVolumeScanner",
        _NoopScanner,
    )
    monkeypatch.setattr("sqlalchemy.create_engine", lambda _url: None)
    # The niche pass (CL-u2ph, default-on) burns a live LLM call — no-op
    # it in the generic wiring fixture so digest tests stay hermetic. The
    # dedicated TestPipelineNicheWiring exercises the real step with mocks
    # (it grabs the original off ``_real_niche_step`` below).
    mod._real_niche_step = mod._niche_step  # type: ignore[attr-defined]
    monkeypatch.setattr(mod, "_niche_step", lambda *_a, **_kw: 0)
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
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EVENT_DIGEST_MIN_URGENCY", "3")
        assert pipeline_mod._resolve_digest_min_urgency(8) == 8

    def test_resolve_min_urgency_env(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EVENT_DIGEST_MIN_URGENCY", "3")
        assert pipeline_mod._resolve_digest_min_urgency(None) == 3

    def test_resolve_min_urgency_default(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("EVENT_DIGEST_MIN_URGENCY", raising=False)
        assert pipeline_mod._resolve_digest_min_urgency(None) == DEFAULT_MIN_URGENCY

    def test_resolve_min_urgency_garbage_env_falls_back(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
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
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("EVENT_DIGEST_MIN_URGENCY", raising=False)
        results = [_res(urgency=7)]
        sent = self._run_cycle(
            pipeline_mod,
            monkeypatch,
            ["--assess", "--digest-min-urgency", "6"],
            results,
        )
        assert len(sent) == 1
        assert sent[0]["results"] == results
        assert sent[0]["min_urgency"] == 6

    def test_no_digest_skips_send(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sent = self._run_cycle(
            pipeline_mod,
            monkeypatch,
            ["--assess", "--no-digest"],
            [_res(urgency=9)],
        )
        assert sent == []

    def test_digest_failure_does_not_crash_cycle(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
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

        def fake_persist(
            engine: Any, eid: int, assessment: Any, prices: Any = None, now: Any = None
        ) -> int:
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
        r = _res(event_id=event_id, urgency=7, affected=[_aff("BCO_USD", direction="long")])
        r.assessment["trade_ideas"] = [
            {"ticker": "TSM", "action": "buy_puts", "time_stop_days": 5},
        ]
        return r

    def test_assessed_events_persisted_with_batch_prices(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
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
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = self._wire(monkeypatch)
        _FakeAgent.results = [
            AssessmentResult(
                event_id=9,
                headline="junk",
                theme=None,
                status="DISMISSED",
                assessment={"rationale": "parse failure"},
            )
        ]
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        args.digest_min_urgency = 5
        pipeline_mod._cycle(args)
        assert calls["persist"] == []
        assert calls["expire"] == 1

    def test_ledger_failure_never_kills_cycle_or_digest(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
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


class TestPipelineNicheWiring:
    """CL-u2ph: --niche flag + the _niche_step quota gate + fail-soft.
    Uses the real _niche_step (the generic fixture no-ops it) but mocks
    the NicheAgent/SymbolUniverse so no LLM/DB is touched."""

    def test_niche_flag_default_on(self, pipeline_mod: Any) -> None:
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        assert args.niche is True

    def test_no_niche_flag(self, pipeline_mod: Any) -> None:
        args = pipeline_mod._build_parser().parse_args(["--assess", "--no-niche"])
        assert args.niche is False

    def test_resolve_niche_min_urgency_env(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NICHE_MIN_URGENCY", "9")
        assert pipeline_mod._resolve_niche_min_urgency() == 9

    def test_resolve_niche_min_urgency_default(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.events.niche_agent import DEFAULT_MIN_URGENCY as NICHE_DEFAULT

        monkeypatch.delenv("NICHE_MIN_URGENCY", raising=False)
        assert pipeline_mod._resolve_niche_min_urgency() == NICHE_DEFAULT

    def _run_niche_step(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
        results: list[AssessmentResult],
        min_urgency: int = 7,
        universe_factory: Any = None,
    ) -> list[int]:
        """Drive the REAL _niche_step directly (bypassing _cycle's no-op)
        with a mocked NicheAgent / SymbolUniverse; returns the event_ids
        the agent's run() was actually called on (the quota-gate probe)."""
        ran_on: list[int] = []

        class _FakeNicheAgent:
            def __init__(self, **_kw: Any) -> None:
                pass

            def run(self, row: dict[str, Any], playbook: Any = None) -> list[Any]:
                ran_on.append(int(row["id"]))
                return []  # no ideas → no merge/persist

            def merge_into_assessment(self, *_a: Any, **_kw: Any) -> int:
                return 0

        monkeypatch.setattr(
            "src.data.symbols.SymbolUniverse",
            universe_factory or (lambda _e: object()),
        )
        monkeypatch.setattr("src.events.niche_agent.NicheAgent", _FakeNicheAgent)
        # _niche_step calls _json.dumps + engine.begin() only when ideas
        # merge; with no ideas it never touches the (None) engine. Use the
        # ORIGINAL _niche_step (the fixture no-op'd the module attribute).
        pipeline_mod._real_niche_step(None, results, min_urgency)
        return ran_on

    def test_niche_runs_on_high_urgency_assessed(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ran = self._run_niche_step(
            pipeline_mod,
            monkeypatch,
            [_res(event_id=1, urgency=8)],
        )
        assert ran == [1]

    def test_niche_skips_low_urgency(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # urgency 5 < min 7 → the niche pass never touches it (quota gate).
        ran = self._run_niche_step(
            pipeline_mod,
            monkeypatch,
            [_res(event_id=2, urgency=5)],
        )
        assert ran == []

    def test_niche_skips_dismissed(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dismissed = AssessmentResult(
            event_id=3,
            headline="junk",
            theme=None,
            status="DISMISSED",
            assessment={"urgency": 9},
        )
        ran = self._run_niche_step(pipeline_mod, monkeypatch, [dismissed])
        assert ran == []

    def test_niche_failure_never_kills_cycle(
        self,
        pipeline_mod: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # _niche_step raising must be swallowed by _cycle's try/except.
        # The generic fixture already no-ops _niche_step, so re-point it
        # at a raising stub and confirm _cycle survives.
        def boom(*_a: Any, **_kw: Any) -> int:
            raise RuntimeError("niche exploded")

        monkeypatch.setattr(pipeline_mod, "_niche_step", boom)
        monkeypatch.setattr(
            "src.events.digest.send_digest",
            lambda *_a, **_kw: DispatchResult(
                telegram_attempted=True,
                telegram_succeeded=True,
            ),
        )
        _FakeAgent.results = [_res(event_id=1, urgency=9)]
        args = pipeline_mod._build_parser().parse_args(["--assess"])
        args.digest_min_urgency = 5
        args.niche_min_urgency = 7
        pipeline_mod._cycle(args)  # must not raise


# --------------------------------------------------------------------- #
# Niche discovery fan-out (CL-818b)
# --------------------------------------------------------------------- #


class _FakeUniverse:
    def __init__(self, _engine: Any) -> None:
        pass

    def exists(self, _ticker: str) -> bool:  # warmed by _niche_step up front
        return True


class _RecordingEngine:
    """Captures the event ids persisted via begin()/execute() UPDATE."""

    def __init__(self) -> None:
        self.persisted: list[int] = []
        self._lock = threading.Lock()

    def begin(self) -> Any:
        engine = self

        class _Ctx:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_a: Any) -> bool:
                return False

            def execute(self, _stmt: Any, params: dict[str, Any]) -> None:
                with engine._lock:
                    engine.persisted.append(int(params["id"]))

        return _Ctx()


class TestNicheStepConcurrency:
    """CL-818b: the niche loop DISCOVERS across events concurrently (bounded)
    then MERGES + PERSISTS serially. Correctness must match the old serial
    loop: every qualifying event discovered exactly once, all ideas merged,
    the count summed, and one event's failure is fail-soft."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch, agent_cls: Any) -> None:
        monkeypatch.setattr("src.data.symbols.SymbolUniverse", _FakeUniverse)
        monkeypatch.setattr("src.events.niche_agent.NicheAgent", agent_cls)

    def _agent_cls(self, ran: list[int], lock: threading.Lock, fail_on: int | None = None) -> Any:
        class _Agent:
            def __init__(self, **_kw: Any) -> None:
                pass

            def run(self, row: dict[str, Any], playbook: Any = None) -> list[Any]:
                with lock:
                    ran.append(int(row["id"]))
                if fail_on is not None and int(row["id"]) == fail_on:
                    raise RuntimeError(f"boom on event {fail_on}")
                return [f"idea-{row['id']}"]

            def merge_into_assessment(
                self, assessment: dict[str, Any], ideas: list[Any], **_kw: Any
            ) -> int:
                assessment.setdefault("trade_ideas", []).extend(ideas)
                return len(ideas)

        return _Agent

    def test_all_events_discovered_once_and_merged(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NICHE_MAX_CONCURRENCY", "4")
        ran: list[int] = []
        self._patch(monkeypatch, self._agent_cls(ran, threading.Lock()))
        engine = _RecordingEngine()
        results = [_res(event_id=i, urgency=8) for i in (1, 2, 3, 4, 5)]

        surfaced = pipeline_mod._real_niche_step(engine, results, 7)

        assert surfaced == 5
        assert sorted(ran) == [1, 2, 3, 4, 5]  # each discovered exactly once
        assert sorted(engine.persisted) == [1, 2, 3, 4, 5]  # each persisted once
        for r in results:  # merges landed on the right assessment
            assert r.assessment["trade_ideas"] == [f"idea-{r.event_id}"]

    def test_one_event_failure_is_fail_soft(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NICHE_MAX_CONCURRENCY", "4")
        ran: list[int] = []
        self._patch(monkeypatch, self._agent_cls(ran, threading.Lock(), fail_on=2))
        engine = _RecordingEngine()
        results = [_res(event_id=i, urgency=8) for i in (1, 2, 3)]

        surfaced = pipeline_mod._real_niche_step(engine, results, 7)

        assert surfaced == 2  # 1 and 3 survived; 2 failed soft
        assert sorted(ran) == [1, 2, 3]  # all three were attempted
        assert sorted(engine.persisted) == [1, 3]

    def test_serial_path_when_concurrency_one(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NICHE_MAX_CONCURRENCY", "1")
        ran: list[int] = []
        self._patch(monkeypatch, self._agent_cls(ran, threading.Lock()))
        engine = _RecordingEngine()
        results = [_res(event_id=i, urgency=8) for i in (1, 2, 3)]

        surfaced = pipeline_mod._real_niche_step(engine, results, 7)

        assert surfaced == 3
        assert ran == [1, 2, 3]  # serial → deterministic call order

    def test_merge_persist_failure_is_fail_soft_per_event(
        self, pipeline_mod: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ultra-review nit: CL-818b moved merge+persist out of the per-event
        try/except, so a transient DB error on event N dropped the merges for
        N+1..M — and those events are already ASSESSED, so the next cycle
        never retries them. Each event's merge/persist must fail soft."""
        monkeypatch.setenv("NICHE_MAX_CONCURRENCY", "4")
        ran: list[int] = []
        self._patch(monkeypatch, self._agent_cls(ran, threading.Lock()))

        class _FailingEngine(_RecordingEngine):
            """Raises on the persist for event 2 only."""

            def begin(self) -> Any:
                outer = super().begin()
                engine = self

                class _Ctx:
                    def __enter__(self) -> Any:
                        return self

                    def __exit__(self, *_a: Any) -> bool:
                        return False

                    def execute(self, stmt: Any, params: dict[str, Any]) -> None:
                        if int(params["id"]) == 2:
                            raise RuntimeError("transient DB blip")
                        with engine._lock:
                            engine.persisted.append(int(params["id"]))

                del outer
                return _Ctx()

        engine = _FailingEngine()
        results = [_res(event_id=i, urgency=8) for i in (1, 2, 3)]

        surfaced = pipeline_mod._real_niche_step(engine, results, 7)

        # Event 2's persist blew up, but 1 and 3 still persisted — the loop
        # did not abort on the first failure.
        assert sorted(engine.persisted) == [1, 3]
        assert surfaced == 3  # merges counted for all three

    def test_max_concurrency_env_parse_and_clamp(self, pipeline_mod: Any) -> None:
        import os

        prev = os.environ.get("NICHE_MAX_CONCURRENCY")
        try:
            for raw, expected in [("4", 4), ("1", 1), ("999", 8), ("0", 1), ("junk", 4), ("", 4)]:
                os.environ["NICHE_MAX_CONCURRENCY"] = raw
                assert pipeline_mod._niche_max_concurrency() == expected
        finally:
            if prev is None:
                os.environ.pop("NICHE_MAX_CONCURRENCY", None)
            else:
                os.environ["NICHE_MAX_CONCURRENCY"] = prev


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
        poly = _FakePolySignal(
            {
                "energy_chokepoint": {
                    "hormuz-closure-2026": {
                        "question": "Hormuz closed?",
                        "yes_prob": 0.18,
                        "rising": True,
                    },
                },
            }
        )
        built = build_digest([_res(urgency=7)], poly_signal=poly)
        assert built is not None
        _, msg = built
        assert "Prediction mkt:" in msg
        assert "hormuz-closure-2026 18% ↑" in msg

    def test_falling_arrow(self) -> None:
        poly = _FakePolySignal(
            {
                "energy_chokepoint": {
                    "s": {"question": "q", "yes_prob": 0.40, "rising": False},
                },
            }
        )
        _, msg = build_digest([_res(urgency=7)], poly_signal=poly)  # type: ignore[misc]
        assert "s 40% ↓" in msg

    def test_no_arrow_when_direction_unknown(self) -> None:
        poly = _FakePolySignal(
            {
                "energy_chokepoint": {
                    "s": {"question": "q", "yes_prob": 0.25, "rising": None},
                },
            }
        )
        _, msg = build_digest([_res(urgency=7)], poly_signal=poly)  # type: ignore[misc]
        assert "s 25%" in msg
        assert "s 25% ↑" not in msg and "s 25% ↓" not in msg

    def test_highest_prob_market_cited(self) -> None:
        poly = _FakePolySignal(
            {
                "energy_chokepoint": {
                    "low": {"question": "q", "yes_prob": 0.10, "rising": None},
                    "high": {"question": "q", "yes_prob": 0.55, "rising": True},
                },
            }
        )
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
