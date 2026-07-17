"""Unit tests — scripts/scorecard.py pure parts (CL-4nnr).

Log parsing (Health equity, kill-switch), equity/P&L math, research-run
summarization, spend ledger windowing, threshold checks and the
GREEN/AMBER/RED verdict. Fixture-driven — no DB, network or streamlit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from scripts.scorecard import (
    Metric,
    build_metric,
    check_threshold,
    count_kill_switch_triggers,
    engine_log_files,
    equity_stats,
    parse_health_equity,
    sum_debate_spend,
    summarize_research_runs,
    week_verdict,
)


def _jsonl(msg: str, ts: str, **extra) -> str:
    return json.dumps({"ts": ts, "level": "DEBUG", "msg": msg, **extra})


T0 = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


# =============================================================================
# Health-line parsing
# =============================================================================


class TestParseHealthEquity:
    def test_extracts_ts_and_equity(self):
        lines = [
            _jsonl("Health: equity=100000.00", "2026-07-13T12:00:00+00:00"),
            _jsonl("Health: equity=100250.55", "2026-07-13T13:00:00+00:00"),
        ]
        points = parse_health_equity(lines)
        assert [(p[0].hour, p[1]) for p in points] == [(12, 100000.0), (13, 100250.55)]

    def test_ignores_non_health_and_malformed(self):
        lines = [
            "not json at all {",
            _jsonl("Reconciliation error", "2026-07-13T12:00:00+00:00"),
            _jsonl("Health check error", "2026-07-13T12:01:00+00:00"),
            json.dumps(["Health: equity=1.0"]),  # non-dict record
            _jsonl("Health: equity=99500.00", "2026-07-13T12:02:00+00:00"),
            "",
        ]
        points = parse_health_equity(lines)
        assert len(points) == 1
        assert points[0][1] == 99500.0

    def test_sorted_by_timestamp(self):
        lines = [
            _jsonl("Health: equity=2.0", "2026-07-13T14:00:00+00:00"),
            _jsonl("Health: equity=1.0", "2026-07-13T12:00:00+00:00"),
        ]
        assert [p[1] for p in parse_health_equity(lines)] == [1.0, 2.0]

    def test_naive_timestamp_coerced_to_utc(self):
        lines = [_jsonl("Health: equity=5.0", "2026-07-13T12:00:00")]
        (ts, _eq), = parse_health_equity(lines)
        assert ts.tzinfo is not None


# =============================================================================
# Kill-switch counting
# =============================================================================


class TestKillSwitchCount:
    def test_counts_only_kill_switch_msgs(self):
        lines = [
            _jsonl(
                "KILL SWITCH: daily_loss triggered — action=flatten context={}",
                "2026-07-13T12:00:00+00:00",
            ),
            _jsonl("Kill switch daily_loss: OK (value=0.001)",
                   "2026-07-13T12:01:00+00:00"),
            _jsonl("Health: equity=1.0", "2026-07-13T12:02:00+00:00"),
        ]
        assert count_kill_switch_triggers(lines) == 1

    def test_window_filtering(self):
        mk = lambda ts: _jsonl("KILL SWITCH: x triggered — action=halt context={}", ts)  # noqa: E731
        lines = [
            mk("2026-07-10T00:00:00+00:00"),
            mk("2026-07-13T12:00:00+00:00"),
            mk("2026-07-20T00:00:00+00:00"),
        ]
        n = count_kill_switch_triggers(
            lines, start=T0 - timedelta(days=1), end=T0 + timedelta(days=1),
        )
        assert n == 1

    def test_marker_in_payload_but_not_msg_not_counted(self):
        lines = [_jsonl("routine note", "2026-07-13T12:00:00+00:00",
                        context="KILL SWITCH history")]
        assert count_kill_switch_triggers(lines) == 0


# =============================================================================
# Equity stats
# =============================================================================


class TestEquityStats:
    def _points(self):
        return [
            (T0, 100000.0),
            (T0 + timedelta(hours=4), 100500.0),          # day 1 close
            (T0 + timedelta(days=1), 100200.0),           # day 2 close
            (T0 + timedelta(days=2), 101000.0),           # day 3 close
        ]

    def test_weekly_pnl_and_pct(self):
        stats = equity_stats(self._points(), T0 - timedelta(days=1),
                             T0 + timedelta(days=7))
        assert stats["weekly_pnl"] == pytest.approx(1000.0)
        assert stats["weekly_pnl_pct"] == pytest.approx(1.0)
        assert stats["n_samples"] == 4

    def test_daily_close_takes_last_sample_of_day(self):
        stats = equity_stats(self._points(), T0 - timedelta(days=1),
                             T0 + timedelta(days=7))
        assert stats["daily_close"]["2026-07-13"] == 100500.0

    def test_daily_pnl_diffs(self):
        stats = equity_stats(self._points(), T0 - timedelta(days=1),
                             T0 + timedelta(days=7))
        assert stats["daily_pnl"]["2026-07-14"] == pytest.approx(-300.0)
        assert stats["daily_pnl"]["2026-07-15"] == pytest.approx(800.0)

    def test_out_of_window_points_excluded(self):
        stats = equity_stats(self._points(), T0 + timedelta(days=1, hours=-1),
                             T0 + timedelta(days=7))
        assert stats["first_equity"] == 100200.0

    def test_empty_window_returns_empty_dict(self):
        assert equity_stats(self._points(), T0 + timedelta(days=30),
                            T0 + timedelta(days=37)) == {}


# =============================================================================
# Research summaries + spend
# =============================================================================


class TestResearchSummaries:
    def test_sums_verdicts_and_errors(self):
        runs = [
            {"debates_run": 2, "verdicts_promote": 1, "verdicts_reject": 1,
             "verdicts_escalate": 0, "errors": ["boom"]},
            {"debates_run": 1, "verdicts_promote": 0, "verdicts_reject": 1,
             "verdicts_escalate": 1, "errors": []},
        ]
        s = summarize_research_runs(runs)
        assert s["n_runs"] == 2
        assert s["debates_run"] == 3
        assert s["verdicts_promote"] == 1
        assert s["verdicts_reject"] == 2
        assert s["verdicts_escalate"] == 1
        assert s["errors"] == 1

    def test_missing_keys_default_zero(self):
        s = summarize_research_runs([{}])
        assert s["debates_run"] == 0
        assert s["errors"] == 0

    def test_empty_list(self):
        assert summarize_research_runs([])["n_runs"] == 0


class TestDebateSpend:
    def _lines(self):
        return [
            json.dumps({"timestamp": "2026-07-13T12:00:00+00:00", "usd_cost": 0.25}),
            json.dumps({"timestamp": "2026-07-14T12:00:00+00:00", "usd_cost": 1.75}),
            json.dumps({"timestamp": "2026-06-01T12:00:00+00:00", "usd_cost": 99.0}),
            "garbage line",
            json.dumps({"timestamp": "2026-07-14T13:00:00+00:00"}),  # no cost
        ]

    def test_sums_within_window(self):
        total = sum_debate_spend(
            self._lines(), start=T0 - timedelta(days=1), end=T0 + timedelta(days=7),
        )
        assert total == pytest.approx(2.0)

    def test_no_window_sums_all_valid(self):
        assert sum_debate_spend(self._lines()) == pytest.approx(101.0)


# =============================================================================
# Thresholds, metrics, verdict
# =============================================================================


class TestThresholds:
    def test_max_mode(self):
        assert check_threshold(0, "max", 0)
        assert not check_threshold(1, "max", 0)

    def test_min_mode(self):
        assert check_threshold(-0.5, "min", -1.0)
        assert not check_threshold(-2.0, "min", -1.0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown threshold mode"):
            check_threshold(1, "between", 0)


class TestBuildMetric:
    def test_pass_and_fail(self):
        ok = build_metric("kill_switch_triggers", "Kill switches", 0)
        bad = build_metric("kill_switch_triggers", "Kill switches", 2)
        assert ok.passed is True
        assert bad.passed is False
        assert bad.critical is True

    def test_none_value_is_unknown(self):
        m = build_metric("trade_count", "Orders", None)
        assert m.passed is None
        assert m.display == "n/a"


def _metric(key: str, passed: bool | None, critical: bool = False) -> Metric:
    return Metric(key=key, name=key, value=0, display="0", target="t",
                  passed=passed, critical=critical)


class TestWeekVerdict:
    def test_all_pass_green(self):
        assert week_verdict([_metric("a", True), _metric("b", True)]) == "GREEN"

    def test_one_noncritical_fail_amber(self):
        ms = [_metric("a", True), _metric("b", False)]
        assert week_verdict(ms) == "AMBER"

    def test_unknown_source_amber(self):
        ms = [_metric("a", True), _metric("b", None)]
        assert week_verdict(ms) == "AMBER"

    def test_critical_fail_red(self):
        ms = [_metric("a", True), _metric("kill", False, critical=True)]
        assert week_verdict(ms) == "RED"

    def test_two_fails_red(self):
        ms = [_metric("a", False), _metric("b", False), _metric("c", True)]
        assert week_verdict(ms) == "RED"

    def test_empty_metric_list_green(self):
        assert week_verdict([]) == "GREEN"


# =============================================================================
# Engine log file selection
# =============================================================================


class TestEngineLogFiles:
    def test_main_file_and_recent_rotations(self, tmp_path):
        (tmp_path / "live_engine.jsonl").write_text("")
        (tmp_path / "live_engine.jsonl.2026-07-12").write_text("")
        (tmp_path / "live_engine.jsonl.2026-06-01").write_text("")
        (tmp_path / "live_engine.jsonl.bogus").write_text("")
        (tmp_path / "other.jsonl").write_text("")
        files = engine_log_files(tmp_path, start=T0 - timedelta(days=2))
        names = sorted(p.name for p in files)
        assert names == ["live_engine.jsonl", "live_engine.jsonl.2026-07-12"]

    def test_empty_dir(self, tmp_path):
        assert engine_log_files(tmp_path, start=T0) == []
