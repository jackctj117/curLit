"""Runtime data staleness detection + vendor failover (CL-zfe0).

Great Expectations validates *structure* — schema, ranges, anomalies — but
none of that matters if the data is simply old. A FRED ingester that
silently fails for 6 hours leaves the rate-diff strategy reading yesterday's
yields, which produces orphaned signals that persist until somebody notices.

This module gives the live engine a single place to ask "is feed X fresh
enough to act on?". Three pieces:

    FeedDescriptor       — config: name, max_age_seconds, criticality.

    DataFreshnessMonitor — heartbeat store. Ingesters call record_update()
                           on success and record_attempt() on every attempt
                           (success or failure). Tracks last_updated_ts +
                           last_attempted_ts per feed; computes staleness;
                           emits the fx_data_freshness_seconds Prometheus
                           gauge.

    FreshnessGate        — `is_fresh(name)` for strategy code to gate
                           decisions, with fallback-feed promotion: if the
                           primary source is stale but a configured fallback
                           is fresh, the gate transparently switches.

Wire into ingesters: import the singleton-ish DataFreshnessMonitor (or
inject one), call record_update/record_attempt at the right moment. Wire
into strategies: query is_fresh before generating signals; skip the cycle
if the feed they depend on is stale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.monitoring.metrics import data_freshness_seconds

logger = logging.getLogger(__name__)


# Default conservative thresholds when a feed isn't explicitly described.
# 1 hour is generous for FRED (which updates daily) and tight for tick streams.
_DEFAULT_MAX_AGE_SECONDS: int = 3600


class Criticality(Enum):
    """How urgent it is to act on staleness for this feed."""

    CRITICAL = "critical"  # halt new trades when stale (e.g., live price stream)
    WARN = "warn"          # alert + log; strategies decide whether to skip
    INFO = "info"          # log only


@dataclass
class FeedDescriptor:
    """Static config for one feed."""

    name: str
    max_age_seconds: int
    criticality: Criticality = Criticality.WARN
    fallback_feed: str | None = None  # name of an alternate feed to try

    def __post_init__(self) -> None:
        assert self.name, "feed name must be non-empty"
        assert self.max_age_seconds > 0, (
            f"max_age_seconds must be positive, got {self.max_age_seconds}"
        )


@dataclass
class FeedState:
    """Mutable runtime state for one feed."""

    name: str
    last_updated: datetime | None = None
    last_attempted: datetime | None = None
    consecutive_failures: int = 0

    def age_seconds(self, now: datetime) -> float:
        """Seconds since last_updated (inf when never updated)."""
        if self.last_updated is None:
            return float("inf")
        return (now - self.last_updated).total_seconds()


@dataclass
class StalenessReport:
    """One-shot snapshot of all feed states."""

    ts: datetime
    fresh: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    critical_stale: list[str] = field(default_factory=list)
    states: dict[str, FeedState] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "fresh_count": len(self.fresh),
            "stale_count": len(self.stale),
            "critical_stale_count": len(self.critical_stale),
            "fresh": list(self.fresh),
            "stale": list(self.stale),
            "critical_stale": list(self.critical_stale),
        }


# =============================================================================
# Monitor
# =============================================================================


class DataFreshnessMonitor:
    """Heartbeat store + staleness query + Prometheus emission.

    Thread-unsafe at the level of internal dict mutation — the live engine
    is single-threaded for ingestion (one task per source). If multiple
    threads share a monitor, wrap calls in a Lock.
    """

    def __init__(self, descriptors: list[FeedDescriptor] | None = None) -> None:
        self._descriptors: dict[str, FeedDescriptor] = {}
        self._states: dict[str, FeedState] = {}
        if descriptors:
            for d in descriptors:
                self.register(d)

    def register(self, descriptor: FeedDescriptor) -> None:
        """Register a feed for monitoring. Idempotent."""
        self._descriptors[descriptor.name] = descriptor
        self._states.setdefault(
            descriptor.name, FeedState(name=descriptor.name),
        )
        logger.info(
            "Registered feed %s (max_age=%ds, criticality=%s)",
            descriptor.name, descriptor.max_age_seconds,
            descriptor.criticality.value,
        )

    def record_update(self, name: str, ts: datetime | None = None) -> None:
        """Mark a feed as freshly updated. Resets consecutive_failures."""
        ts = ts or datetime.now(UTC)
        self._ensure_state(name)
        state = self._states[name]
        state.last_updated = ts
        state.last_attempted = ts
        state.consecutive_failures = 0
        self._publish_metric(name, age_seconds=0.0)

    def record_attempt(self, name: str, success: bool, ts: datetime | None = None) -> None:
        """Mark an attempt — sets last_attempted; increments failures on miss."""
        ts = ts or datetime.now(UTC)
        self._ensure_state(name)
        state = self._states[name]
        state.last_attempted = ts
        if success:
            state.last_updated = ts
            state.consecutive_failures = 0
            self._publish_metric(name, age_seconds=0.0)
        else:
            state.consecutive_failures += 1
            # Don't reset last_updated — staleness still measured against last good update.
            self._publish_metric(name, age_seconds=state.age_seconds(ts))

    def is_fresh(self, name: str, now: datetime | None = None) -> bool:
        """Returns True iff feed is registered and within its max-age window."""
        if name not in self._descriptors:
            # Never-registered feeds are conservatively NOT fresh.
            return False
        now = now or datetime.now(UTC)
        state = self._states[name]
        return state.age_seconds(now) <= self._descriptors[name].max_age_seconds

    def staleness_report(self, now: datetime | None = None) -> StalenessReport:
        """Build a report of fresh/stale feeds at `now`."""
        now = now or datetime.now(UTC)
        fresh: list[str] = []
        stale: list[str] = []
        critical_stale: list[str] = []
        for name, descriptor in self._descriptors.items():
            state = self._states[name]
            age = state.age_seconds(now)
            self._publish_metric(name, age_seconds=age)
            if age <= descriptor.max_age_seconds:
                fresh.append(name)
            else:
                stale.append(name)
                if descriptor.criticality == Criticality.CRITICAL:
                    critical_stale.append(name)
        return StalenessReport(
            ts=now,
            fresh=fresh,
            stale=stale,
            critical_stale=critical_stale,
            states=dict(self._states),
        )

    def get_state(self, name: str) -> FeedState | None:
        return self._states.get(name)

    def get_descriptor(self, name: str) -> FeedDescriptor | None:
        return self._descriptors.get(name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_state(self, name: str) -> None:
        if name not in self._states:
            self._states[name] = FeedState(name=name)
        if name not in self._descriptors:
            # Auto-register with conservative defaults so callers can record
            # heartbeats without pre-declaring (logged for visibility).
            logger.warning(
                "Feed %s not pre-registered; auto-registering with default %ds max-age",
                name, _DEFAULT_MAX_AGE_SECONDS,
            )
            self._descriptors[name] = FeedDescriptor(
                name=name, max_age_seconds=_DEFAULT_MAX_AGE_SECONDS,
            )

    def _publish_metric(self, name: str, age_seconds: float) -> None:
        try:
            # `data_freshness_seconds` was originally labeled (symbol, source)
            # for tick streams. We reuse the same gauge for general feeds with
            # name == symbol and source == "feed_monitor" so dashboards can
            # filter consistently.
            data_freshness_seconds.labels(symbol=name, source="feed_monitor").set(
                # Inf is not a valid Prometheus value; clip to a large finite.
                min(age_seconds, 1e9),
            )
        except Exception:
            logger.exception("data_freshness_seconds metric set failed")


# =============================================================================
# Gate (with fallback)
# =============================================================================


class FreshnessGate:
    """Strategy-facing query layer — returns the freshest available feed.

    Use:
        gate = FreshnessGate(monitor)
        feed = gate.preferred_fresh_feed("US_2Y")
        if feed is None:
            return  # skip this signal cycle
        # use `feed` to fetch data — it's either the primary or a fallback.
    """

    def __init__(self, monitor: DataFreshnessMonitor) -> None:
        self.monitor = monitor

    def is_fresh(self, name: str, now: datetime | None = None) -> bool:
        return self.monitor.is_fresh(name, now=now)

    def preferred_fresh_feed(
        self, primary: str, now: datetime | None = None,
    ) -> str | None:
        """Return primary if fresh; otherwise its registered fallback if fresh.

        Returns None when neither is available — callers must handle this
        (typically: skip the signal cycle).
        """
        if self.monitor.is_fresh(primary, now=now):
            return primary
        descriptor = self.monitor.get_descriptor(primary)
        if descriptor is None or descriptor.fallback_feed is None:
            return None
        if self.monitor.is_fresh(descriptor.fallback_feed, now=now):
            logger.warning(
                "Primary feed %s stale; using fallback %s",
                primary, descriptor.fallback_feed,
            )
            return descriptor.fallback_feed
        return None
