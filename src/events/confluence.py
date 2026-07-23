"""Event confluence — confirmation layer for the current-events pipeline (CL-mnhw).

Consumes ``geo_events`` rows the producer half (GDELT ingest + Event
Impact Agent, CL-6iu7) has advanced to ASSESSED, and decides — inside a
confirmation window measured from ``seen_at`` — whether the market is
actually reacting the way the LLM assessment predicted:

  * **Gate A (quality)**   — assessment urgency >= ``min_urgency`` AND
    confidence >= ``min_confidence``. Failing Gate A never confirms; the
    row rides out the window and expires (so high-urgency/low-confidence
    events still surface via the expired-unconfirmed operator alert).
  * **Gate B (market)**    — per tradable affected instrument (kind
    ``oanda``/``fx``): price must have moved in the assessed direction
    since ``seen_at`` by at least ``confirm_move_frac`` of that
    instrument's 20d realized DAILY vol (default 0.25 = a quarter-sigma
    day). Live broker ticks are preferred for the current price;
    DataProvider closes are the fallback for both legs. ``equity_watch``
    / ``polymarket`` entries never gate — alert-only.

>= ``min_confirmed_instruments`` Gate-B passes → ASSESSED → CONFIRMED.
Past ``confirm_window_max_minutes`` without confirmation → EXPIRED.
Status transitions are guarded single UPDATEs (``WHERE status = :old``)
so status + status_updated_at move together atomically and concurrent
writers cannot double-advance a row.

Self-contained on purpose: the sibling owns the rest of ``src/events``;
this module only shares the ``geo_events`` schema with it.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

#: Assessment `kind` values that can gate confirmation (and be traded).
TRADABLE_KINDS = ("oanda", "fx")
#: Assessment `direction` values that imply an actual position.
TRADE_DIRECTIONS = ("long", "short")

_TRADING_DAYS_PER_YEAR = 252.0


def mid_price(tick: Any) -> float | None:
    """Mid from a live-price tick dict ({"bid": x, "ask": y}); None if absent."""
    if not isinstance(tick, dict):
        return None
    bid, ask = tick.get("bid"), tick.get("ask")
    if bid is None or ask is None:
        return None
    try:
        return (float(bid) + float(ask)) / 2.0
    except (TypeError, ValueError):
        return None


def as_utc(value: Any) -> datetime | None:
    """Normalize DB timestamps (aware/naive datetime or ISO string) to UTC.

    Postgres hands back aware datetimes for timestamptz; sqlite test
    fixtures hand back ISO strings. Naive values are assumed UTC.
    """
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass
class ConfluenceConfig:
    # ---- Gate A (assessment quality) --------------------------------
    # urgency is the LLM's 1-10 "how fast does this move markets"; 7
    # keeps only clearly market-moving events.
    min_urgency: int = 7
    # confidence is the LLM's 0-1 self-assessed conviction; 0.75 drops
    # hedged / ambiguous assessments.
    min_confidence: float = 0.75
    # ---- Confirmation window (minutes from seen_at) -----------------
    # Before min: too early to trust the move (knee-jerk spike risk).
    # After max: the edge is gone — expire rather than chase.
    confirm_window_min_minutes: int = 30
    confirm_window_max_minutes: int = 120
    # ---- Gate B (market confirmation) -------------------------------
    # Required move since seen_at, as a fraction of the instrument's
    # 20d realized DAILY vol. 0.25 = a quarter-sigma move in the
    # assessed direction.
    confirm_move_frac: float = 0.25
    # How many tradable instruments must pass Gate B to CONFIRM.
    min_confirmed_instruments: int = 1
    # Window (daily returns) for the baseline realized vol.
    realized_vol_window: int = 20
    # Optional extra check: short-window realized vol must exceed
    # vol_spike_ratio x the baseline vol (the event actually moved
    # vol, not just drifted price). Off by default.
    vol_spike_check_enabled: bool = False
    vol_spike_window: int = 5
    vol_spike_ratio: float = 1.5
    # Intraday quote staleness bound (CL-dz71): a quote from intraday_quotes
    # only counts as the reference/current price if it is within this many
    # minutes of the lookup time. Keeps a stalled poller from feeding an old
    # price as "now", and requires a real quote near seen_at. Beyond it, the
    # daily close is used (pre-CL-dz71 behavior).
    intraday_max_staleness_minutes: int = 15


@dataclass
class PollCache:
    """Per-poll-cycle batched price/vol cache for Gate B (CL-9ts9 / CL-8s2a).

    Confluence Gate B used to look mid/vol up PER event × PER instrument,
    each call opening a NEW DB connection → hundreds of PG round-trips per
    tick as the ASSESSED set grew. This cache holds the values that are
    IDENTICAL across every event in a single tick — the current mid (at
    ``now``) and the realized daily vol (at ``now``) per instrument —
    prefetched ONCE via the DataProvider ``*_batch`` helpers, plus a
    lazily-filled per-(symbol, seen_at) reference-price memo so a repeated
    seen_at isn't re-queried within the cycle.

    STRICTLY single-cycle: a fresh cache is built each poll (built by
    :meth:`EventConfluence.build_poll_cache`), so there is NO cross-tick
    staleness — a stale value can never outlive the tick that fetched it.
    When ``None`` is passed to evaluate_and_transition, the confluence
    layer falls straight back to the byte-identical per-symbol reads.
    """

    #: symbol → current mid at ``now`` (intraday-at-now, daily-close
    #: fallback) — replaces the ``_provider_price(symbol, now)`` fallback
    #: leg. Absent key = the batched lookup found nothing (→ per-call path).
    mid_now: dict[str, float] = field(default_factory=dict)
    #: symbol → DAILY realized vol at ``now`` (already de-annualized).
    #: Absent key = no vol data batched (→ per-call path).
    daily_vol_now: dict[str, float] = field(default_factory=dict)
    #: (symbol, seen_at) → reference price memo, filled lazily by the
    #: reference-price leg so a shared seen_at is queried at most once.
    ref_price: dict[tuple[str, datetime], float | None] = field(default_factory=dict)
    #: True once mid_now/daily_vol_now have been prefetched — so a cache
    #: object built but never populated (empty instrument set) still routes
    #: through the per-call path rather than treating "" as "no data".
    prefetched: bool = False
    #: Diagnostics for the one-line timing log (CL-3lga): how many symbols
    #: were batched and how many DB queries the batch prefetch issued.
    instruments_looked_up: int = 0
    batch_queries: int = 0


@dataclass
class InstrumentCheck:
    """Gate-B outcome for a single affected instrument."""

    instrument: str  # assessment instrument id
    symbol: str      # market symbol used for price/vol lookups
    kind: str
    direction: str
    confirmed: bool = False
    move_frac: float | None = None       # observed move since seen_at
    threshold_frac: float | None = None  # required |move|
    reason: str = ""


@dataclass
class ConfluenceResult:
    event_id: Any
    outcome: str  # "confirmed" | "expired" | "pending"
    quality_passed: bool = False
    #: True when OUR guarded UPDATE advanced the row (we "won" the
    #: transition). False for pending outcomes or lost races.
    transitioned: bool = False
    urgency: int = 0
    confidence: float = 0.0
    checks: list[InstrumentCheck] = field(default_factory=list)
    #: Cross-asset corroboration read (CL-6mzn) — a
    #: :class:`src.events.cross_asset.CrossAssetResult`, or None when the
    #: layer isn't configured / the event didn't reach confirmation. This
    #: is a DISPLAY annotation, not a gate: it never changes ``outcome``.
    cross_asset: Any = None


class EventConfluence:
    """Evaluates one ASSESSED ``geo_events`` row against Gates A+B and
    writes the resulting status transition atomically.

    ``instrument_map`` maps assessment instrument ids to market/broker
    symbols for price lookups; unmapped ids fall through unchanged here
    (the *strategy* is where unknown instruments are refused for
    trading — confirmation may still read them from the DataProvider).
    """

    def __init__(
        self,
        config: ConfluenceConfig | None = None,
        data_provider: Any = None,
        db_engine: Any = None,
        instrument_map: dict[str, str] | None = None,
        cross_asset_config: Any = None,
    ) -> None:
        self.config = config or ConfluenceConfig()
        self.data = data_provider
        self.db = db_engine
        self.instrument_map = dict(instrument_map or {})
        #: Cross-asset corroboration config (CL-6mzn), a
        #: :class:`src.events.cross_asset.CrossAssetConfig`. Optional —
        #: when None the cross-asset annotation is simply not computed.
        self.cross_asset_config = cross_asset_config

    # ------------------------------------------------------------------
    # Assessment parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_assessment(raw: Any) -> dict[str, Any] | None:
        """assessment arrives as dict (psycopg2 JSONB) or JSON string
        (sqlite fixtures). None when absent/unparseable."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_and_transition(
        self,
        event: dict[str, Any],
        prices: dict[str, Any] | None = None,
        now: datetime | None = None,
        poll_cache: PollCache | None = None,
    ) -> ConfluenceResult:
        """Evaluate one ASSESSED row; write CONFIRMED/EXPIRED atomically.

        ``prices`` is the live tick dict the engine hands strategies
        ({symbol: {"bid":…, "ask":…}}); used for the current-price leg
        when available, DataProvider closes otherwise.

        ``poll_cache`` (CL-9ts9 / CL-8s2a) is an optional per-tick
        :class:`PollCache` of pre-batched current-mid / daily-vol reads
        shared across all events in a poll cycle. It never changes any
        gate decision — the same value the per-instrument path would have
        read is served from the batch instead — so a batched verdict is
        byte-identical to the per-call verdict. None → per-call reads.
        """
        now = now or datetime.now(UTC)
        event_id = event.get("id")
        result = ConfluenceResult(event_id=event_id, outcome="pending")

        seen_at = as_utc(event.get("seen_at"))
        if seen_at is None:
            logger.warning(
                "geo_event id=%s has unparseable seen_at %r — leaving row untouched",
                event_id, event.get("seen_at"),
            )
            return result

        elapsed = now - seen_at
        window_min = timedelta(minutes=self.config.confirm_window_min_minutes)
        window_max = timedelta(minutes=self.config.confirm_window_max_minutes)

        assessment = self.parse_assessment(event.get("assessment"))
        if assessment is not None:
            try:
                result.urgency = int(assessment.get("urgency") or 0)
            except (TypeError, ValueError):
                result.urgency = 0
            try:
                result.confidence = float(assessment.get("confidence") or 0.0)
            except (TypeError, ValueError):
                result.confidence = 0.0

        # Past the window: expire regardless of anything else. A move
        # that shows up only after max_minutes is too stale to chase.
        if elapsed > window_max:
            result.outcome = "expired"
            result.transitioned = self.transition(event_id, "ASSESSED", "EXPIRED")
            return result

        if assessment is None:
            logger.warning(
                "geo_event id=%s has no parseable assessment — will expire at window end",
                event_id,
            )
            return result

        # Gate A — quality. Failures stay pending and ride out the
        # window to EXPIRED (not DISMISSED) so big-but-unconfirmable
        # events still reach the operator via the expired alert.
        result.quality_passed = (
            result.urgency >= self.config.min_urgency
            and result.confidence >= self.config.min_confidence
        )
        if not result.quality_passed or elapsed < window_min:
            return result

        # Gate B — market confirmation per tradable instrument.
        confirmed_count = 0
        affected = assessment.get("affected") or []
        if not isinstance(affected, list):
            affected = []
        for aff in affected:
            if not isinstance(aff, dict):
                continue
            check = self._check_instrument(
                aff, seen_at, now, prices or {}, poll_cache,
            )
            result.checks.append(check)
            if check.confirmed:
                confirmed_count += 1

        if confirmed_count >= self.config.min_confirmed_instruments:
            result.outcome = "confirmed"
            result.transitioned = self.transition(event_id, "ASSESSED", "CONFIRMED")
            # Cross-asset corroboration (CL-6mzn) — a DISPLAY annotation,
            # never a gate. Compute for confirmed events so the operator
            # can SEE whether the theme's related commodity/asset is
            # corroborating (or NOT — fade risk). Best-effort: any error
            # leaves cross_asset None and the confirmation stands.
            result.cross_asset = self._cross_asset(event.get("theme"), seen_at, now)
        return result

    def _cross_asset(self, theme: Any, since: datetime, now: datetime) -> Any:
        """Compute the theme's cross-asset read, or None when the layer
        isn't configured / anything goes wrong (annotation only)."""
        if self.cross_asset_config is None or self.data is None:
            return None
        try:
            from src.events.cross_asset import cross_asset_confirmation  # noqa: PLC0415

            return cross_asset_confirmation(
                self.data, theme, since, self.cross_asset_config, now=now,
            )
        except Exception:
            logger.debug("cross-asset confirmation failed", exc_info=True)
            return None

    def _check_instrument(
        self,
        aff: dict[str, Any],
        seen_at: datetime,
        now: datetime,
        prices: dict[str, Any],
        poll_cache: PollCache | None = None,
    ) -> InstrumentCheck:
        instrument = str(aff.get("instrument") or "")
        kind = str(aff.get("kind") or "")
        direction = str(aff.get("direction") or "")
        symbol = self.instrument_map.get(instrument, instrument)
        check = InstrumentCheck(
            instrument=instrument, symbol=symbol, kind=kind, direction=direction,
        )

        if kind not in TRADABLE_KINDS:
            # equity_watch / polymarket: alert-only, never gates.
            check.reason = "not_tradable_kind"
            return check
        if direction not in TRADE_DIRECTIONS:
            check.reason = "watch_only"
            return check

        # Current price: live tick first, provider close fallback. The
        # fallback leg (and only that leg) is served from the per-tick
        # batch cache when present — the same value _provider_price(now)
        # would have read (CL-9ts9). Live-tick priority is unchanged.
        p1 = mid_price(prices.get(symbol))
        if p1 is None:
            p1 = self._current_price(symbol, now, poll_cache)
        # Reference price at (or just before) seen_at: provider closes,
        # memoized per (symbol, seen_at) within the cycle.
        p0 = self._reference_price(symbol, seen_at, poll_cache)
        if p0 is None or p1 is None or p0 <= 0:
            check.reason = "no_price_data"
            return check

        daily_vol = self._cached_daily_vol(
            symbol, self.config.realized_vol_window, now, poll_cache,
        )
        if daily_vol is None or daily_vol <= 0:
            check.reason = "no_vol_data"
            return check

        move = (p1 - p0) / p0
        threshold = self.config.confirm_move_frac * daily_vol
        check.move_frac = move
        check.threshold_frac = threshold

        # Bullish thesis needs a +move, bearish needs a -move, each at
        # least a quarter-sigma (confirm_move_frac) of daily vol.
        signed = move if direction == "long" else -move
        if signed < threshold:
            check.reason = "move_below_threshold"
            return check

        if self.config.vol_spike_check_enabled:
            spike_vol = self._daily_vol(symbol, self.config.vol_spike_window, now)
            if spike_vol is None or spike_vol < self.config.vol_spike_ratio * daily_vol:
                check.reason = "no_vol_spike"
                return check

        check.confirmed = True
        check.reason = "confirmed"
        return check

    # ------------------------------------------------------------------
    # Per-tick batch cache (CL-9ts9 / CL-8s2a)
    # ------------------------------------------------------------------

    @staticmethod
    def collect_instruments(
        events: list[dict[str, Any]],
        instrument_map: dict[str, str],
    ) -> list[str]:
        """The unique set of Gate-B market symbols across ``events`` —
        only tradable-kind, tradable-direction affected instruments, since
        those are the only ones that hit price/vol reads. Symbols are the
        instrument_map-resolved ids the DataProvider batch helpers key on.
        Non-dict / malformed rows are skipped exactly as Gate B skips them."""
        symbols: set[str] = set()
        for event in events:
            assessment = EventConfluence.parse_assessment(event.get("assessment"))
            if assessment is None:
                continue
            affected = assessment.get("affected") or []
            if not isinstance(affected, list):
                continue
            for aff in affected:
                if not isinstance(aff, dict):
                    continue
                if str(aff.get("kind") or "") not in TRADABLE_KINDS:
                    continue
                if str(aff.get("direction") or "") not in TRADE_DIRECTIONS:
                    continue
                instrument = str(aff.get("instrument") or "")
                symbols.add(instrument_map.get(instrument, instrument))
        return sorted(symbols)

    def build_poll_cache(
        self, events: list[dict[str, Any]], now: datetime,
    ) -> PollCache:
        """Prefetch the current-mid and daily-vol for the WHOLE instrument
        set of a poll cycle in one batch each (CL-9ts9 / CL-8s2a), so Gate
        B reads them from memory instead of a per-event × per-instrument DB
        round-trip. The reference-price-at-seen_at leg stays per-call
        (seen_at differs per event) but is memoized within the cycle.

        Best-effort: any batch error leaves the corresponding map empty and
        Gate B transparently falls back to the byte-identical per-symbol
        reads — never breaks a poll. No provider → an empty (unprefetched)
        cache, i.e. the per-call path throughout."""
        cache = PollCache()
        if self.data is None:
            return cache
        symbols = self.collect_instruments(events, self.instrument_map)
        cache.instruments_looked_up = len(symbols)
        if not symbols:
            cache.prefetched = True
            return cache

        # Current mid @ now: intraday-at-now (within staleness) first, daily
        # close fallback for whatever intraday didn't cover — mirrors the
        # per-symbol _provider_price(now) order exactly.
        mid: dict[str, float] = {}
        getter = getattr(self.data, "get_intraday_values_batch", None)
        if getter is not None:
            try:
                mid.update(getter(
                    symbols, now, self.config.intraday_max_staleness_minutes,
                ))
                cache.batch_queries += 1
            except Exception:
                logger.debug("intraday batch prefetch failed", exc_info=True)
        missing_mid = [s for s in symbols if s not in mid]
        latest_getter = getattr(self.data, "get_latest_values_batch", None)
        if missing_mid and latest_getter is not None:
            try:
                mid.update(latest_getter(missing_mid, now))
                cache.batch_queries += 1
            except Exception:
                logger.debug("latest-value batch prefetch failed", exc_info=True)
        cache.mid_now = mid

        # Daily realized vol @ now (de-annualized, matching _daily_vol).
        vol_getter = getattr(self.data, "get_realized_vols_batch", None)
        if vol_getter is not None:
            try:
                annualized = vol_getter(
                    symbols, self.config.realized_vol_window, now,
                )
                cache.batch_queries += 1
                cache.daily_vol_now = {
                    sym: float(v) / math.sqrt(_TRADING_DAYS_PER_YEAR)
                    for sym, v in annualized.items()
                    if v is not None
                }
            except Exception:
                logger.debug("realized-vol batch prefetch failed", exc_info=True)
        cache.prefetched = True
        return cache

    def _current_price(
        self, symbol: str, now: datetime, poll_cache: PollCache | None,
    ) -> float | None:
        """Current-price fallback leg — served from the per-tick batch when
        the cache prefetched it, else the byte-identical per-symbol read."""
        if poll_cache is not None and poll_cache.prefetched:
            cached = poll_cache.mid_now.get(symbol)
            if cached is not None:
                return cached
            # Absent from the batch means the batched read found nothing —
            # the per-call read would find nothing too. But fall through so
            # a partial/failed batch (empty map) still self-heals per call.
            if poll_cache.mid_now:
                return None
        return self._provider_price(symbol, now)

    def _reference_price(
        self, symbol: str, seen_at: datetime, poll_cache: PollCache | None,
    ) -> float | None:
        """Reference price at ``seen_at`` — per-call (seen_at differs per
        event) but memoized per (symbol, seen_at) within the cycle so a
        shared seen_at is queried at most once.

        CL-hn0t (P1): the baseline MUST be an INTRADAY quote at/near seen_at
        (within ``intraday_max_staleness_minutes``) — NOT the daily-close
        fallback. A daily close has no freshness bound relative to seen_at, so
        a stale close (last night's / Friday's) as the baseline miscounted a
        normal pre-event overnight/weekend gap as post-headline confirmation.
        Gate B is explicitly an intraday confirmation gate; with no intraday
        reference the leg reports ``no_price_data`` and does NOT confirm rather
        than confirm on a gap that happened before the headline."""
        if poll_cache is None:
            return self._intraday_price(symbol, seen_at)
        key = (symbol, seen_at)
        if key not in poll_cache.ref_price:
            poll_cache.ref_price[key] = self._intraday_price(symbol, seen_at)
        return poll_cache.ref_price[key]

    def _cached_daily_vol(
        self, symbol: str, window: int, now: datetime,
        poll_cache: PollCache | None,
    ) -> float | None:
        """Daily realized vol at ``now`` — from the per-tick batch when
        prefetched (default window only), else the per-symbol read. A
        non-default window (never used in the live path) always reads per
        call so the batched value can't be served for the wrong window."""
        if (
            poll_cache is not None
            and poll_cache.prefetched
            and window == self.config.realized_vol_window
        ):
            cached = poll_cache.daily_vol_now.get(symbol)
            if cached is not None:
                return cached
            if poll_cache.daily_vol_now:
                return None
        return self._daily_vol(symbol, window, now)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _provider_price(self, symbol: str, as_of: datetime) -> float | None:
        if self.data is None:
            return None
        # Intraday quote store first (CL-dz71) — a real, timestamped price at
        # this instant, so the seen_at reference and the current price are BOTH
        # event-timescale rather than a stale (possibly days-old) daily close.
        intraday = self._intraday_price(symbol, as_of)
        if intraday is not None:
            return intraday
        try:
            value = self.data.get_latest_value(symbol, as_of)
        except Exception as exc:
            logger.debug(
                "Provider price lookup failed for %s @ %s: %s: %s",
                symbol, as_of, type(exc).__name__, exc,
            )
            return None
        return float(value) if value is not None else None

    def _intraday_price(self, symbol: str, as_of: datetime) -> float | None:
        """Nearest intraday quote at/before ``as_of`` within the staleness
        bound, or None. Guarded by getattr so providers without the intraday
        method (older fixtures) simply skip straight to the daily fallback."""
        getter = getattr(self.data, "get_intraday_value", None)
        if getter is None:
            return None
        try:
            value = getter(
                symbol, as_of, self.config.intraday_max_staleness_minutes,
            )
        except Exception as exc:
            logger.debug(
                "Intraday price lookup failed for %s @ %s: %s: %s",
                symbol, as_of, type(exc).__name__, exc,
            )
            return None
        return float(value) if value is not None else None

    def _daily_vol(self, symbol: str, window: int, as_of: datetime) -> float | None:
        """Realized DAILY vol (fraction) — get_realized_vol is annualized."""
        if self.data is None:
            return None
        try:
            annualized = self.data.get_realized_vol(symbol, window=window, as_of=as_of)
        except Exception as exc:
            logger.debug(
                "Realized vol lookup failed for %s: %s: %s",
                symbol, type(exc).__name__, exc,
            )
            return None
        if annualized is None:
            return None
        return float(annualized) / math.sqrt(_TRADING_DAYS_PER_YEAR)

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def transition(self, event_id: Any, old: str, new: str) -> bool:
        """Guarded atomic transition old → new.

        status and status_updated_at move together in ONE UPDATE inside
        one transaction, and ``WHERE status = :old`` means exactly one
        writer wins a race. Returns True iff this call advanced the row.
        """
        if self.db is None or event_id is None:
            return False
        try:
            with self.db.begin() as conn:
                res = conn.execute(
                    text(
                        "UPDATE geo_events "
                        "SET status = :new, status_updated_at = :ts "
                        "WHERE id = :id AND status = :old"
                    ),
                    {
                        "new": new,
                        # ISO string binds portably (psycopg2 casts to
                        # timestamptz; sqlite fixtures store TEXT).
                        "ts": datetime.now(UTC).isoformat(),
                        "id": event_id,
                        "old": old,
                    },
                )
            return bool(res.rowcount == 1)
        except Exception:
            logger.exception(
                "geo_events transition %s → %s failed for id=%s", old, new, event_id,
            )
            return False
