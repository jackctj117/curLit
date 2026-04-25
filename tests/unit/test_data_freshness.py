"""Unit tests — monitoring.data_freshness (CL-zfe0)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.monitoring.data_freshness import (
    Criticality,
    DataFreshnessMonitor,
    FeedDescriptor,
    FreshnessGate,
)

# Anchor time so tests are deterministic.
T0 = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Construction + registration
# =============================================================================


class TestRegistration:
    def test_register_via_constructor(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="US_2Y", max_age_seconds=3600),
        ])
        assert monitor.get_descriptor("US_2Y") is not None

    def test_register_after_construction(self) -> None:
        monitor = DataFreshnessMonitor()
        monitor.register(FeedDescriptor(name="EURUSD", max_age_seconds=60))
        assert monitor.get_descriptor("EURUSD") is not None

    def test_register_idempotent(self) -> None:
        monitor = DataFreshnessMonitor()
        d = FeedDescriptor(name="US_2Y", max_age_seconds=3600)
        monitor.register(d)
        monitor.register(d)
        assert monitor.get_descriptor("US_2Y") is d

    def test_invalid_descriptor_rejected(self) -> None:
        with pytest.raises(AssertionError):
            FeedDescriptor(name="", max_age_seconds=60)
        with pytest.raises(AssertionError):
            FeedDescriptor(name="x", max_age_seconds=0)


# =============================================================================
# Initial state
# =============================================================================


class TestInitialState:
    def test_never_updated_is_not_fresh(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="US_2Y", max_age_seconds=3600),
        ])
        assert monitor.is_fresh("US_2Y", now=T0) is False

    def test_unregistered_feed_is_not_fresh(self) -> None:
        monitor = DataFreshnessMonitor()
        assert monitor.is_fresh("missing", now=T0) is False


# =============================================================================
# Heartbeats
# =============================================================================


class TestHeartbeats:
    def test_record_update_makes_fresh(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="US_2Y", max_age_seconds=3600),
        ])
        monitor.record_update("US_2Y", ts=T0)
        assert monitor.is_fresh("US_2Y", now=T0)

    def test_within_max_age_is_fresh(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="US_2Y", max_age_seconds=3600),
        ])
        monitor.record_update("US_2Y", ts=T0)
        # 30 minutes later — still within 1h window.
        assert monitor.is_fresh("US_2Y", now=T0 + timedelta(minutes=30))

    def test_past_max_age_not_fresh(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="US_2Y", max_age_seconds=3600),
        ])
        monitor.record_update("US_2Y", ts=T0)
        # 2 hours later — exceeds 1h window.
        assert not monitor.is_fresh("US_2Y", now=T0 + timedelta(hours=2))

    def test_failed_attempt_does_not_reset_freshness(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="US_2Y", max_age_seconds=3600),
        ])
        monitor.record_update("US_2Y", ts=T0)
        # Some time later, an attempt fails.
        monitor.record_attempt("US_2Y", success=False, ts=T0 + timedelta(minutes=5))
        # last_updated is still T0; staleness measured against that.
        state = monitor.get_state("US_2Y")
        assert state is not None
        assert state.last_updated == T0
        assert state.consecutive_failures == 1
        # Still fresh (5min < 1h).
        assert monitor.is_fresh("US_2Y", now=T0 + timedelta(minutes=5))

    def test_successful_attempt_resets_failures(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="US_2Y", max_age_seconds=3600),
        ])
        monitor.record_attempt("US_2Y", success=False, ts=T0)
        monitor.record_attempt("US_2Y", success=False, ts=T0 + timedelta(seconds=30))
        state = monitor.get_state("US_2Y")
        assert state is not None and state.consecutive_failures == 2
        monitor.record_attempt("US_2Y", success=True, ts=T0 + timedelta(minutes=1))
        assert monitor.get_state("US_2Y").consecutive_failures == 0


# =============================================================================
# Auto-registration
# =============================================================================


class TestAutoRegistration:
    def test_record_update_auto_registers_with_default(self) -> None:
        monitor = DataFreshnessMonitor()
        monitor.record_update("never_pre_registered", ts=T0)
        assert monitor.get_descriptor("never_pre_registered") is not None
        # Default max-age is 1 hour.
        assert monitor.is_fresh("never_pre_registered", now=T0)
        # Past 1h is stale (default).
        assert not monitor.is_fresh(
            "never_pre_registered", now=T0 + timedelta(hours=2),
        )


# =============================================================================
# Reports
# =============================================================================


class TestStalenessReport:
    def test_report_partitions_correctly(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="A", max_age_seconds=60),
            FeedDescriptor(name="B", max_age_seconds=60, criticality=Criticality.CRITICAL),
            FeedDescriptor(name="C", max_age_seconds=3600),
        ])
        # A was updated recently, B never, C stale.
        monitor.record_update("A", ts=T0)
        monitor.record_update("C", ts=T0 - timedelta(hours=2))
        report = monitor.staleness_report(now=T0)
        assert "A" in report.fresh
        assert "B" in report.stale
        assert "C" in report.stale
        # B is critical-stale; C is just stale.
        assert "B" in report.critical_stale
        assert "C" not in report.critical_stale

    def test_report_to_dict_keys(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="A", max_age_seconds=60),
        ])
        monitor.record_update("A", ts=T0)
        d = monitor.staleness_report(now=T0).to_dict()
        for key in (
            "ts", "fresh_count", "stale_count", "critical_stale_count",
            "fresh", "stale", "critical_stale",
        ):
            assert key in d


# =============================================================================
# Fallback gate
# =============================================================================


class TestFallback:
    def test_returns_primary_when_fresh(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(
                name="US_2Y", max_age_seconds=3600,
                fallback_feed="US_2Y_BACKUP",
            ),
            FeedDescriptor(name="US_2Y_BACKUP", max_age_seconds=3600),
        ])
        monitor.record_update("US_2Y", ts=T0)
        gate = FreshnessGate(monitor)
        assert gate.preferred_fresh_feed("US_2Y", now=T0) == "US_2Y"

    def test_returns_fallback_when_primary_stale(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(
                name="US_2Y", max_age_seconds=3600,
                fallback_feed="US_2Y_BACKUP",
            ),
            FeedDescriptor(name="US_2Y_BACKUP", max_age_seconds=3600),
        ])
        # Primary has never updated; fallback has.
        monitor.record_update("US_2Y_BACKUP", ts=T0)
        gate = FreshnessGate(monitor)
        assert (
            gate.preferred_fresh_feed("US_2Y", now=T0)
            == "US_2Y_BACKUP"
        )

    def test_returns_none_when_both_stale(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(
                name="US_2Y", max_age_seconds=3600,
                fallback_feed="US_2Y_BACKUP",
            ),
            FeedDescriptor(name="US_2Y_BACKUP", max_age_seconds=3600),
        ])
        # Neither has been updated.
        gate = FreshnessGate(monitor)
        assert gate.preferred_fresh_feed("US_2Y", now=T0) is None

    def test_returns_none_when_no_fallback_configured(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(name="US_2Y", max_age_seconds=3600),
        ])
        gate = FreshnessGate(monitor)
        assert gate.preferred_fresh_feed("US_2Y", now=T0) is None


# =============================================================================
# Criticality
# =============================================================================


class TestCriticality:
    def test_critical_appears_in_critical_stale(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(
                name="price_stream", max_age_seconds=10,
                criticality=Criticality.CRITICAL,
            ),
        ])
        monitor.record_update("price_stream", ts=T0)
        report = monitor.staleness_report(now=T0 + timedelta(seconds=30))
        assert "price_stream" in report.critical_stale

    def test_warn_does_not_appear_in_critical_stale(self) -> None:
        monitor = DataFreshnessMonitor([
            FeedDescriptor(
                name="research", max_age_seconds=60,
                criticality=Criticality.WARN,
            ),
        ])
        monitor.record_update("research", ts=T0)
        report = monitor.staleness_report(now=T0 + timedelta(minutes=5))
        assert "research" in report.stale
        assert "research" not in report.critical_stale
