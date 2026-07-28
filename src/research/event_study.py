"""Event study (CL-z95p) — does the event-driven pipeline actually have edge?

Measures what the market DID after every assessed ``geo_events`` row, using
only data already in the live DB (``geo_events`` + ``intraday_quotes``), and
cuts the result by every dimension the pipeline gates on (urgency, confidence,
theme, playbook tier, gate outcome, direction). Nothing here places, sizes, or
suggests a trade — it is measurement only.

The two questions it answers
----------------------------
1. **Is there a move to catch at all?** Signed returns from the HEADLINE
   (``seen_at``) at 30 / 60 / 120 / 240 / 1440 minutes, per tradable affected
   leg, signed by the assessment's own direction (+1 long / −1 short).
2. **Does the confirmation gate help or hurt?** The same returns re-measured
   from the CONFIRMATION entry, reported side by side with the headline
   numbers over the identical event set, plus the confirmation-latency
   distribution. If confirmation-entry returns are systematically worse than
   headline-entry returns over the same events, the gate is buying its
   selectivity by paying away the move.

Measurement rules (the parts that make or break an event study)
---------------------------------------------------------------
* **NO LOOKAHEAD.** The entry price is the FIRST quote with ``ts >= anchor``
  (anchor = ``seen_at`` for the headline entry, ``status_updated_at`` for the
  confirmation entry), within ``quote_match_tolerance_min``. A quote stamped
  even one second BEFORE the anchor is never eligible as an entry — that price
  was not tradable when the news landed. Horizon prices, by contrast, use the
  NEAREST quote to ``anchor + h`` on either side, which is symmetric noise
  around a point far past the entry and cannot leak the entry.
* **Missing data is COUNTED, never imputed.** A leg with no in-tolerance entry
  quote is excluded and counted; a horizon with no in-tolerance quote is
  dropped for that leg and counted. Every exclusion reason appears in the
  report. There is no forward-fill, no last-known-price, no interpolation.
* **The confirmation timestamp is an APPROXIMATION.** ``geo_events`` stores no
  dedicated confirmation time; ``status_updated_at`` is the moment the row was
  last transitioned. For a CONFIRMED or TRADED row that transition IS the
  confirmation/trade event (the confluence layer moves ASSESSED → CONFIRMED and
  the strategy moves CONFIRMED → TRADED in the same guarded UPDATE that stamps
  ``status_updated_at``), so it is a good proxy — but a row re-transitioned
  later would carry the later stamp. Treat confirmation-entry numbers as
  "entry at approximately the gate's decision time".
* **Costs are quoted, not netted.** Returns are GROSS. The half-spread
  fraction at entry ``(ask − bid) / 2 / mid`` is reported alongside so the
  reader can net it themselves; when the quote row carries no bid/ask the cost
  is reported as unavailable rather than assumed zero.
* **``intraday_quotes`` is a ROLLING BUFFER.** ``src/data/intraday_pricer.py``
  prunes to a short retention horizon (24h by default) every cycle. The actual
  ``min(ts)``/``max(ts)`` coverage is measured and printed at the top of the
  report; a ``--days`` window wider than the buffer is flagged loudly, because
  every event older than the buffer is unmeasurable, not flat.

Options attribution (CL-d44a caveat)
------------------------------------
:func:`run_options_attribution` joins ``alpaca_option_orders`` →
``trade_ideas`` → ``geo_events`` and cuts realized option P&L by theme,
urgency, idea confidence, niche / red-team flags and entry spread. Rows whose
``entry_mid`` IS NULL predate migration 018: their ``pnl_pct`` was marked at
the BID against an ASK fill, so it is spread-corrupted and systematically
negative. Those rows are reported as a SEPARATE "legacy bid-marked era" cohort
and are never pooled with post-018 mid-marked rows.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import DateTime, bindparam, text
from sqlalchemy.engine import Engine

from src.events.playbooks import Playbook
from src.events.prices import parse_ts
from src.strategies.event_driven_config import _default_instrument_map

logger = logging.getLogger(__name__)

#: Assessment ``kind`` values that are actually tradable (mirrors
#: :mod:`src.events.confluence` — kept local so this module never imports the
#: live confirmation layer).
TRADABLE_KINDS = frozenset({"oanda", "fx"})

#: Assessment ``direction`` values that carry a sign (``watch`` does not).
TRADE_DIRECTIONS = frozenset({"long", "short"})

#: Statuses whose assessment has a tradable read worth measuring. DISMISSED is
#: deliberately absent — a triaged-out row has no tradable assessment — but it
#: IS counted in the census so the funnel adds up.
ANALYZED_STATUSES = ("ASSESSED", "CONFIRMED", "TRADED", "EXPIRED")

#: Statuses whose ``status_updated_at`` approximates a confirmation/trade time.
CONFIRMED_STATUSES = ("CONFIRMED", "TRADED")

#: Every status pulled for the census (analyzed set + the dismissed pile).
CENSUS_STATUSES = (*ANALYZED_STATUSES, "DISMISSED")

#: Entry-basis labels.
HEADLINE = "headline"
CONFIRMATION = "confirmation"

#: Theme label used when a theme has no matching playbook.
UNMATCHED_TIER = "unmatched"

_URGENCY_BUCKETS = (">=7", "5-6", "<5", "unknown")
_CONFIDENCE_BUCKETS = (">=0.75", "0.55-0.749", "<0.55", "unknown")

_MISSING = "n/a"


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EventStudyConfig:
    """Measurement knobs. Frozen — a study is reproducible from its config."""

    #: Forward horizons (minutes from the entry anchor) to price.
    horizons_minutes: tuple[int, ...] = (30, 60, 120, 240, 1440)
    #: A quote further than this from its target timestamp is NOT a match.
    quote_match_tolerance_min: int = 15
    #: Window (minutes from entry) over which MFE/MAE excursions are scanned.
    mfe_mae_horizon_min: int = 240
    #: Cells with fewer observations than this are flagged LOW-N in the report.
    min_bucket_n: int = 20
    #: Assessment instrument id → OANDA quote symbol. Reuses the live
    #: strategy's map so the study measures exactly what the strategy could
    #: have traded; anything unmapped is excluded and counted, never guessed.
    instrument_map: Mapping[str, str] = field(default_factory=_default_instrument_map)


# ----------------------------------------------------------------------
# Quote plumbing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    ts: datetime
    mid: float
    bid: float | None
    ask: float | None

    @property
    def half_spread_frac(self) -> float | None:
        """``(ask − bid) / 2 / mid`` — the one-way cost of crossing at this
        quote. ``None`` (reported as unavailable) when the feed carried no
        two-sided quote or a non-positive mid."""
        if self.bid is None or self.ask is None or self.mid <= 0:
            return None
        return (self.ask - self.bid) / 2.0 / self.mid


class QuoteSeries:
    """Time-sorted quotes for one symbol with the three lookups the study
    needs. Pure in-memory — no DB access, so every match rule is unit
    testable without a database."""

    def __init__(self, quotes: Iterable[Quote]) -> None:
        self.quotes: list[Quote] = sorted(quotes, key=lambda q: q.ts)
        self._ts: list[datetime] = [q.ts for q in self.quotes]

    def __len__(self) -> int:
        return len(self.quotes)

    def first_at_or_after(self, target: datetime, tolerance_min: float) -> Quote | None:
        """The ENTRY rule: earliest quote at/after ``target``, within
        tolerance. Never returns a quote before ``target`` — that is the
        no-lookahead guarantee."""
        idx = bisect.bisect_left(self._ts, target)
        if idx >= len(self.quotes):
            return None
        candidate = self.quotes[idx]
        if (candidate.ts - target) > timedelta(minutes=tolerance_min):
            return None
        return candidate

    def nearest(self, target: datetime, tolerance_min: float) -> Quote | None:
        """The HORIZON rule: closest quote on either side of ``target``,
        within tolerance. Safe because horizon targets sit far past the entry
        (the shortest horizon exceeds the tolerance)."""
        if not self.quotes:
            return None
        idx = bisect.bisect_left(self._ts, target)
        best: Quote | None = None
        best_gap: timedelta | None = None
        for cand_idx in (idx - 1, idx):
            if cand_idx < 0 or cand_idx >= len(self.quotes):
                continue
            cand = self.quotes[cand_idx]
            gap = abs(cand.ts - target)
            if best_gap is None or gap < best_gap:
                best, best_gap = cand, gap
        if best is None or best_gap is None:
            return None
        if best_gap > timedelta(minutes=tolerance_min):
            return None
        return best

    def between(self, lo: datetime, hi: datetime) -> list[Quote]:
        """Quotes in the half-open interval ``(lo, hi]`` — the MFE/MAE scan."""
        start = bisect.bisect_right(self._ts, lo)
        end = bisect.bisect_right(self._ts, hi)
        return self.quotes[start:end]


def _to_float(value: Any) -> float | None:
    """Numeric columns arrive as Decimal (psycopg2 NUMERIC), float (sqlite) or
    None. Anything else is data corruption → None (counted, never guessed)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_quotes(
    engine: Engine,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
) -> dict[str, QuoteSeries]:
    """Bulk-load ``intraday_quotes`` for ``symbols`` in ``[start, end]``.

    One query for the whole study: the buffer is small (24h × 19 symbols ×
    30/hour) and per-leg queries would be thousands of round trips.
    """
    if not symbols:
        return {}
    stmt = text(
        "SELECT symbol, ts, bid, ask, mid FROM intraday_quotes "
        "WHERE symbol IN :symbols AND ts >= :start AND ts <= :end"
    ).bindparams(
        bindparam("symbols", expanding=True),
        bindparam("start", type_=DateTime(timezone=True)),
        bindparam("end", type_=DateTime(timezone=True)),
    )
    buckets: dict[str, list[Quote]] = {}
    with engine.connect() as conn:
        rows = conn.execute(
            stmt, {"symbols": list(dict.fromkeys(symbols)), "start": start, "end": end}
        ).mappings()
        for row in rows:
            ts = parse_ts(row["ts"])
            mid = _to_float(row["mid"])
            if ts is None or mid is None or mid <= 0:
                continue
            buckets.setdefault(str(row["symbol"]), []).append(
                Quote(ts=ts, mid=mid, bid=_to_float(row["bid"]), ask=_to_float(row["ask"]))
            )
    return {sym: QuoteSeries(qs) for sym, qs in buckets.items()}


@dataclass(frozen=True)
class QuoteCoverage:
    """What the rolling buffer ACTUALLY holds — printed before any result so a
    thin buffer is never mistaken for a flat market."""

    min_ts: datetime | None
    max_ts: datetime | None
    rows: int
    symbols: int

    @property
    def span_hours(self) -> float | None:
        if self.min_ts is None or self.max_ts is None:
            return None
        return (self.max_ts - self.min_ts).total_seconds() / 3600.0


def load_quote_coverage(engine: Engine) -> QuoteCoverage:
    with engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT min(ts) AS min_ts, max(ts) AS max_ts, count(*) AS rows_n, "
                    "count(DISTINCT symbol) AS sym_n FROM intraday_quotes"
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        return QuoteCoverage(None, None, 0, 0)
    return QuoteCoverage(
        min_ts=parse_ts(row["min_ts"]),
        max_ts=parse_ts(row["max_ts"]),
        rows=int(row["rows_n"] or 0),
        symbols=int(row["sym_n"] or 0),
    )


# ----------------------------------------------------------------------
# Assessment parsing / bucketing
# ----------------------------------------------------------------------


def parse_assessment(raw: Any) -> dict[str, Any] | None:
    """assessment arrives as a dict (psycopg2 JSONB) or a JSON string (sqlite
    fixtures). ``None`` when absent/unparseable — mirrors
    ``EventConfluence.parse_assessment`` so the study and the live gate read
    the same rows the same way."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _assessment_int(assessment: Mapping[str, Any], key: str) -> int | None:
    value = assessment.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _assessment_float(assessment: Mapping[str, Any], key: str) -> float | None:
    value = assessment.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bucket_urgency(urgency: int | None) -> str:
    """Urgency buckets. Boundaries match the live Gate A default
    (``min_urgency=7``) so the top bucket is "what the machine would trade"."""
    if urgency is None:
        return "unknown"
    if urgency >= 7:
        return ">=7"
    if urgency >= 5:
        return "5-6"
    return "<5"


def bucket_confidence(confidence: float | None) -> str:
    """Confidence buckets. ``>=0.75`` is the live ``min_confidence`` default;
    the middle band starts at the stale-fact ceiling neighbourhood (0.55)."""
    if confidence is None:
        return "unknown"
    if confidence >= 0.75:
        return ">=0.75"
    if confidence >= 0.55:
        return "0.55-0.749"
    return "<0.55"


def bucket_spread(spread: float | None) -> str:
    if spread is None:
        return "null"
    if spread < 0.2:
        return "<0.2"
    if spread < 0.35:
        return "0.2-0.35"
    if spread <= 0.5:
        return "0.35-0.5"
    return ">0.5"


# ----------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LegObservation:
    """One (event, tradable affected leg, entry basis) measurement."""

    event_id: int
    status: str
    theme: str
    tier: str
    urgency: int | None
    confidence: float | None
    instrument: str
    symbol: str
    direction: str
    entry_basis: str
    anchor: datetime
    entry_ts: datetime
    entry_price: float
    half_spread_frac: float | None
    #: horizon minutes → signed return (fraction). Missing horizons absent.
    returns: Mapping[int, float]
    #: horizons that had no in-tolerance quote (counted, never imputed).
    missing_horizons: tuple[int, ...]
    mfe: float | None
    mae: float | None

    @property
    def sign(self) -> int:
        return 1 if self.direction == "long" else -1


@dataclass(frozen=True)
class Cell:
    """One aggregation cell. ``t_stat`` is mean / standard error."""

    n: int
    hit_rate: float | None
    mean: float | None
    median: float | None
    std: float | None
    t_stat: float | None


def summarize(values: Sequence[float]) -> Cell:
    """n / hit-rate / mean / median / std / t on a sample. Degenerate samples
    yield ``None`` rather than a fabricated statistic (a 1-observation 'std of
    0' would read as certainty)."""
    n = len(values)
    if n == 0:
        return Cell(0, None, None, None, None, None)
    hit = sum(1 for v in values if v > 0) / n
    mean = statistics.fmean(values)
    median = statistics.median(values)
    if n < 2:
        return Cell(n, hit, mean, median, None, None)
    std = statistics.stdev(values)
    se = std / math.sqrt(n)
    t = mean / se if se > 0 else None
    return Cell(n, hit, mean, median, std, t)


@dataclass(frozen=True)
class EventStudyResult:
    config: EventStudyConfig
    days: int
    window_start: datetime
    window_end: datetime
    coverage: QuoteCoverage
    #: status → row count inside the window (includes DISMISSED).
    status_counts: Mapping[str, int]
    headline_legs: tuple[LegObservation, ...]
    confirmation_legs: tuple[LegObservation, ...]
    #: exclusion reason → count. Every dropped row lands in exactly one.
    exclusions: Mapping[str, int]
    #: unmapped assessment instrument id → leg count.
    unmapped_instruments: Mapping[str, int]
    #: horizon minutes → legs that had no in-tolerance quote at that horizon.
    missing_horizon_counts: Mapping[int, int]
    #: (status_updated_at − seen_at) minutes for CONFIRMED/TRADED rows.
    latency_minutes: tuple[float, ...]
    events_with_legs: int
    events_with_confirmation_legs: int


# ----------------------------------------------------------------------
# The study
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _PendingLeg:
    """A leg to be priced, resolved in pass 1 so pass 2 can bulk-load quotes
    for every symbol in ONE query instead of a round trip per leg."""

    event_id: int
    status: str
    theme: str
    tier: str
    urgency: int | None
    confidence: float | None
    instrument: str
    symbol: str
    direction: str
    entry_basis: str
    anchor: datetime


def _theme_tiers(playbooks: Mapping[str, Playbook] | None) -> dict[str, str]:
    return {key: pb.tier for key, pb in (playbooks or {}).items()}


def _fetch_events(
    engine: Engine, window_start: datetime, window_end: datetime
) -> list[dict[str, Any]]:
    stmt = text(
        "SELECT id, seen_at, status, status_updated_at, theme, assessment "
        "FROM geo_events "
        "WHERE seen_at >= :start AND seen_at <= :end AND status IN :statuses "
        "ORDER BY seen_at"
    ).bindparams(
        bindparam("statuses", expanding=True),
        bindparam("start", type_=DateTime(timezone=True)),
        bindparam("end", type_=DateTime(timezone=True)),
    )
    with engine.connect() as conn:
        rows = conn.execute(
            stmt,
            {"start": window_start, "end": window_end, "statuses": list(CENSUS_STATUSES)},
        ).mappings()
        return [dict(r) for r in rows]


def _tradable_legs(assessment: Mapping[str, Any]) -> list[dict[str, str]]:
    affected = assessment.get("affected")
    if not isinstance(affected, list):
        return []
    legs: list[dict[str, str]] = []
    for entry in affected:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "")
        direction = str(entry.get("direction") or "")
        instrument = str(entry.get("instrument") or "").strip()
        if kind not in TRADABLE_KINDS or direction not in TRADE_DIRECTIONS or not instrument:
            continue
        legs.append({"instrument": instrument, "direction": direction})
    return legs


def measure_leg(
    *,
    event_id: int,
    status: str,
    theme: str,
    tier: str,
    urgency: int | None,
    confidence: float | None,
    instrument: str,
    symbol: str,
    direction: str,
    entry_basis: str,
    anchor: datetime,
    series: QuoteSeries,
    config: EventStudyConfig,
) -> tuple[LegObservation | None, list[int]]:
    """Price one leg from ``anchor``. Returns ``(observation, missing_horizons)``;
    ``observation`` is ``None`` when no in-tolerance ENTRY quote exists (the
    whole leg is then unmeasurable and gets counted as such by the caller)."""
    tolerance = float(config.quote_match_tolerance_min)
    entry = series.first_at_or_after(anchor, tolerance)
    if entry is None:
        return None, []

    sign = 1 if direction == "long" else -1
    returns: dict[int, float] = {}
    missing: list[int] = []
    for horizon in config.horizons_minutes:
        quote = series.nearest(anchor + timedelta(minutes=horizon), tolerance)
        if quote is None:
            missing.append(horizon)
            continue
        returns[horizon] = (quote.mid / entry.mid - 1.0) * sign

    excursion_end = entry.ts + timedelta(minutes=config.mfe_mae_horizon_min)
    path = [(q.mid / entry.mid - 1.0) * sign for q in series.between(entry.ts, excursion_end)]
    mfe = max(path) if path else None
    mae = min(path) if path else None

    return (
        LegObservation(
            event_id=event_id,
            status=status,
            theme=theme,
            tier=tier,
            urgency=urgency,
            confidence=confidence,
            instrument=instrument,
            symbol=symbol,
            direction=direction,
            entry_basis=entry_basis,
            anchor=anchor,
            entry_ts=entry.ts,
            entry_price=entry.mid,
            half_spread_frac=entry.half_spread_frac,
            returns=returns,
            missing_horizons=tuple(missing),
            mfe=mfe,
            mae=mae,
        ),
        missing,
    )


def run_event_study(
    engine: Engine,
    config: EventStudyConfig | None = None,
    *,
    days: int = 30,
    playbooks: Mapping[str, Playbook] | None = None,
    now: datetime | None = None,
) -> EventStudyResult:
    """Read-only. Fetches assessed events in the ``days`` window, prices every
    tradable leg from the headline and (where the row confirmed/traded) from
    the confirmation, and returns the raw observations plus a full accounting
    of everything that could NOT be measured."""
    cfg = config or EventStudyConfig()
    window_end = (now or datetime.now(UTC)).astimezone(UTC)
    window_start = window_end - timedelta(days=days)

    coverage = load_quote_coverage(engine)
    events = _fetch_events(engine, window_start, window_end)

    status_counts: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    missing_horizons: Counter[int] = Counter()
    latency: list[float] = []
    tiers = _theme_tiers(playbooks)

    # ---- pass 1: parse, count, collect the symbols/anchors we must price ---
    pending: list[_PendingLeg] = []
    max_horizon = max(cfg.horizons_minutes) if cfg.horizons_minutes else 0
    scan_horizon = max(max_horizon, cfg.mfe_mae_horizon_min)

    for event in events:
        status = str(event.get("status") or "")
        status_counts[status] += 1
        if status not in ANALYZED_STATUSES:
            continue

        seen_at = parse_ts(event.get("seen_at"))
        if seen_at is None:
            exclusions["events_bad_seen_at"] += 1
            continue

        assessment = parse_assessment(event.get("assessment"))
        if assessment is None:
            exclusions["events_no_assessment"] += 1
            continue

        legs = _tradable_legs(assessment)
        if not legs:
            exclusions["events_no_tradable_legs"] += 1
            continue

        event_id = int(event.get("id") or 0)
        theme = str(event.get("theme") or "") or "(none)"
        tier = tiers.get(theme, UNMATCHED_TIER)
        urgency = _assessment_int(assessment, "urgency")
        confidence = _assessment_float(assessment, "confidence")

        status_updated_at = parse_ts(event.get("status_updated_at"))
        confirm_anchor: datetime | None = None
        if status in CONFIRMED_STATUSES and status_updated_at is not None:
            delta_min = (status_updated_at - seen_at).total_seconds() / 60.0
            latency.append(delta_min)
            # A stamp at/before the headline can't be a confirmation time.
            if delta_min > 0:
                confirm_anchor = status_updated_at
            else:
                exclusions["confirmation_anchor_not_after_headline"] += 1
        elif status in CONFIRMED_STATUSES:
            exclusions["confirmation_anchor_missing"] += 1

        for leg in legs:
            instrument = leg["instrument"]
            symbol = cfg.instrument_map.get(instrument)
            if symbol is None:
                exclusions["legs_unmapped_instrument"] += 1
                unmapped[instrument] += 1
                continue
            pending.append(
                _PendingLeg(
                    event_id=event_id,
                    status=status,
                    theme=theme,
                    tier=tier,
                    urgency=urgency,
                    confidence=confidence,
                    instrument=instrument,
                    symbol=symbol,
                    direction=leg["direction"],
                    entry_basis=HEADLINE,
                    anchor=seen_at,
                )
            )
            if confirm_anchor is not None:
                pending.append(
                    _PendingLeg(
                        event_id=event_id,
                        status=status,
                        theme=theme,
                        tier=tier,
                        urgency=urgency,
                        confidence=confidence,
                        instrument=instrument,
                        symbol=symbol,
                        direction=leg["direction"],
                        entry_basis=CONFIRMATION,
                        anchor=confirm_anchor,
                    )
                )

    # ---- pass 2: one bulk quote load, then price every pending leg --------
    quote_symbols = sorted({p.symbol for p in pending})
    quotes: dict[str, QuoteSeries] = {}
    if pending:
        anchors = [p.anchor for p in pending]
        quotes = load_quotes(
            engine,
            quote_symbols,
            min(anchors) - timedelta(minutes=cfg.quote_match_tolerance_min),
            max(anchors) + timedelta(minutes=scan_horizon + cfg.quote_match_tolerance_min),
        )

    headline: list[LegObservation] = []
    confirmation: list[LegObservation] = []
    for job in pending:
        series = quotes.get(job.symbol)
        if series is None or not len(series):
            exclusions["legs_symbol_absent_from_quote_feed"] += 1
            continue
        obs, missing = measure_leg(
            event_id=job.event_id,
            status=job.status,
            theme=job.theme,
            tier=job.tier,
            urgency=job.urgency,
            confidence=job.confidence,
            instrument=job.instrument,
            symbol=job.symbol,
            direction=job.direction,
            entry_basis=job.entry_basis,
            anchor=job.anchor,
            series=series,
            config=cfg,
        )
        if obs is None:
            # Split the "no entry" bucket three ways against the buffer's TRUE
            # extent — "the rolling buffer had already pruned this event" is a
            # COVERAGE limit, not a feed gap, and conflating them hides why the
            # sample is small.
            if coverage.min_ts is not None and job.anchor < coverage.min_ts:
                exclusions["legs_anchor_before_quote_buffer"] += 1
            elif coverage.max_ts is not None and job.anchor > coverage.max_ts:
                exclusions["legs_anchor_after_quote_buffer"] += 1
            else:
                exclusions["legs_no_entry_quote_in_tolerance"] += 1
            continue
        for horizon in missing:
            missing_horizons[horizon] += 1
        if job.entry_basis == HEADLINE:
            headline.append(obs)
        else:
            confirmation.append(obs)

    logger.info(
        "event study: %d events in window, %d headline legs, %d confirmation legs, %d exclusions",
        sum(status_counts.values()),
        len(headline),
        len(confirmation),
        sum(exclusions.values()),
    )

    return EventStudyResult(
        config=cfg,
        days=days,
        window_start=window_start,
        window_end=window_end,
        coverage=coverage,
        status_counts=dict(status_counts),
        headline_legs=tuple(headline),
        confirmation_legs=tuple(confirmation),
        exclusions=dict(exclusions),
        unmapped_instruments=dict(unmapped),
        missing_horizon_counts=dict(missing_horizons),
        latency_minutes=tuple(latency),
        events_with_legs=len({o.event_id for o in headline}),
        events_with_confirmation_legs=len({o.event_id for o in confirmation}),
    )


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def aggregate(
    legs: Sequence[LegObservation],
    key: str,
    horizons: Sequence[int],
) -> dict[str, dict[int, Cell]]:
    """Group ``legs`` by one dimension and summarize each horizon within it.

    ``key`` ∈ {overall, urgency, confidence, theme, tier, status, direction,
    symbol}.
    """
    grouped: dict[str, dict[int, list[float]]] = {}
    for leg in legs:
        bucket = _bucket_of(leg, key)
        slot = grouped.setdefault(bucket, {h: [] for h in horizons})
        for horizon in horizons:
            value = leg.returns.get(horizon)
            if value is not None:
                slot[horizon].append(value)
    return {
        bucket: {h: summarize(values) for h, values in per_horizon.items()}
        for bucket, per_horizon in grouped.items()
    }


def _bucket_of(leg: LegObservation, key: str) -> str:
    if key == "overall":
        return "ALL"
    if key == "urgency":
        return bucket_urgency(leg.urgency)
    if key == "confidence":
        return bucket_confidence(leg.confidence)
    if key == "theme":
        return leg.theme
    if key == "tier":
        return leg.tier
    if key == "status":
        return leg.status
    if key == "direction":
        return leg.direction
    if key == "symbol":
        return leg.symbol
    msg = f"unknown aggregation key {key!r}"
    raise ValueError(msg)


def _bucket_order(key: str, buckets: Iterable[str]) -> list[str]:
    """Stable, meaningful bucket order: fixed for ordinal dimensions, then
    by descending n (approximated by alphabetical) for open sets."""
    present = list(buckets)
    fixed: tuple[str, ...]
    if key == "urgency":
        fixed = _URGENCY_BUCKETS
    elif key == "confidence":
        fixed = _CONFIDENCE_BUCKETS
    elif key == "status":
        fixed = ("TRADED", "CONFIRMED", "EXPIRED", "ASSESSED")
    elif key == "direction":
        fixed = ("long", "short")
    elif key == "tier":
        fixed = ("specific", "generic", UNMATCHED_TIER)
    else:
        return sorted(present)
    ordered = [b for b in fixed if b in present]
    ordered.extend(sorted(b for b in present if b not in fixed))
    return ordered


# ----------------------------------------------------------------------
# Options attribution (CL-d44a legacy split)
# ----------------------------------------------------------------------

#: Rows with entry_mid IS NULL predate migration 018 — pnl_pct was marked at
#: the BID against an ASK fill and is spread-corrupted.
LEGACY_ERA = "legacy bid-marked era (pre-018, entry_mid NULL)"
POST_018_ERA = "post-018 mid-marked"


@dataclass(frozen=True)
class OptionOutcome:
    idea_id: str
    ticker: str
    pnl_pct: float
    era: str
    exit_reason: str
    theme: str
    urgency: int | None
    confidence: float | None
    niche: bool
    red_team: bool
    entry_spread_pct: float | None
    attributed: bool


@dataclass(frozen=True)
class OptionsAttributionResult:
    days: int
    window_start: datetime
    window_end: datetime
    min_bucket_n: int
    outcomes: tuple[OptionOutcome, ...]
    #: rows with pnl_pct but no matching trade_idea / geo_event.
    unattributed: int
    #: total alpaca_option_orders rows in the window (any pnl_pct).
    total_rows: int
    #: rows with pnl_pct NULL (still open / never a real position).
    unrealized_rows: int


def _has_marker(notes: Any, marker: str) -> bool:
    """Case-insensitive substring test on ``trade_ideas.notes`` — the portable
    equivalent of ``notes ILIKE '%marker%'`` (sqlite has no ILIKE)."""
    if not isinstance(notes, str):
        return False
    return marker.lower() in notes.lower()


def run_options_attribution(
    engine: Engine,
    *,
    days: int = 30,
    now: datetime | None = None,
    min_bucket_n: int = 20,
) -> OptionsAttributionResult:
    """Read-only. Realized option P&L joined back to the originating idea and
    event. The legacy/post-018 split is applied HERE, not in the renderer, so
    no aggregation can accidentally pool the two eras."""
    window_end = (now or datetime.now(UTC)).astimezone(UTC)
    window_start = window_end - timedelta(days=days)

    stmt = text(
        "SELECT o.idea_id AS idea_id, o.ticker AS ticker, o.pnl_pct AS pnl_pct, "
        "o.exit_reason AS exit_reason, o.entry_mid AS entry_mid, "
        "o.entry_spread_pct AS entry_spread_pct, "
        "i.idea_id AS matched_idea, i.confidence AS confidence, i.notes AS notes, "
        "g.id AS event_id, g.theme AS theme, g.assessment AS assessment "
        "FROM alpaca_option_orders o "
        "LEFT JOIN trade_ideas i ON i.idea_id = o.idea_id "
        "LEFT JOIN geo_events g ON g.id = i.geo_event_id "
        "WHERE o.submitted_at >= :start AND o.submitted_at <= :end"
    ).bindparams(
        bindparam("start", type_=DateTime(timezone=True)),
        bindparam("end", type_=DateTime(timezone=True)),
    )

    outcomes: list[OptionOutcome] = []
    unattributed = 0
    total_rows = 0
    unrealized = 0
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"start": window_start, "end": window_end}).mappings()
        for row in rows:
            total_rows += 1
            pnl = _to_float(row["pnl_pct"])
            if pnl is None:
                unrealized += 1
                continue
            assessment = parse_assessment(row["assessment"]) or {}
            attributed = row["matched_idea"] is not None and row["event_id"] is not None
            if not attributed:
                unattributed += 1
            outcomes.append(
                OptionOutcome(
                    idea_id=str(row["idea_id"]),
                    ticker=str(row["ticker"] or ""),
                    pnl_pct=pnl,
                    era=LEGACY_ERA if row["entry_mid"] is None else POST_018_ERA,
                    exit_reason=str(row["exit_reason"] or "(none)"),
                    theme=str(row["theme"] or "(unattributed)"),
                    urgency=_assessment_int(assessment, "urgency"),
                    confidence=_to_float(row["confidence"]),
                    niche=_has_marker(row["notes"], "niche"),
                    red_team=_has_marker(row["notes"], "red-team"),
                    entry_spread_pct=_to_float(row["entry_spread_pct"]),
                    attributed=attributed,
                )
            )

    logger.info(
        "options attribution: %d rows in window, %d realized, %d unattributed",
        total_rows,
        len(outcomes),
        unattributed,
    )
    return OptionsAttributionResult(
        days=days,
        window_start=window_start,
        window_end=window_end,
        min_bucket_n=min_bucket_n,
        outcomes=tuple(outcomes),
        unattributed=unattributed,
        total_rows=total_rows,
        unrealized_rows=unrealized,
    )


def aggregate_options(outcomes: Sequence[OptionOutcome], key: str) -> dict[str, Cell]:
    grouped: dict[str, list[float]] = {}
    for outcome in outcomes:
        grouped.setdefault(_option_bucket_of(outcome, key), []).append(outcome.pnl_pct)
    return {bucket: summarize(values) for bucket, values in grouped.items()}


def _option_bucket_of(outcome: OptionOutcome, key: str) -> str:
    if key == "overall":
        return "ALL"
    if key == "theme":
        return outcome.theme
    if key == "urgency":
        return bucket_urgency(outcome.urgency)
    if key == "confidence":
        return bucket_confidence(outcome.confidence)
    if key == "niche":
        return "niche" if outcome.niche else "not-niche"
    if key == "red_team":
        return "red-teamed" if outcome.red_team else "not-red-teamed"
    if key == "spread":
        return bucket_spread(outcome.entry_spread_pct)
    if key == "exit_reason":
        return outcome.exit_reason
    if key == "ticker":
        return outcome.ticker
    msg = f"unknown options aggregation key {key!r}"
    raise ValueError(msg)


# ----------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 2, scale: float = 1.0) -> str:
    if value is None:
        return _MISSING
    return f"{value * scale:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return _MISSING
    return f"{value * 100:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_(no rows)_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(lines)


def _percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile on a sorted copy. ``None`` when empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _cell_row(label: str, horizon: int, cell: Cell, min_n: int) -> list[str]:
    flag = "LOW-N" if cell.n < min_n else ""
    return [
        label,
        f"{horizon}m",
        str(cell.n),
        _fmt_pct(cell.hit_rate),
        _fmt(cell.mean, 2, 10_000.0),
        _fmt(cell.median, 2, 10_000.0),
        _fmt(cell.std, 2, 10_000.0),
        _fmt(cell.t_stat, 2),
        flag,
    ]


_RETURN_HEADERS = (
    "bucket",
    "horizon",
    "n",
    "hit",
    "mean bps",
    "median bps",
    "std bps",
    "t",
    "flag",
)


def _render_dimension(
    title: str,
    legs: Sequence[LegObservation],
    key: str,
    config: EventStudyConfig,
) -> str:
    agg = aggregate(legs, key, config.horizons_minutes)
    rows: list[list[str]] = []
    for bucket in _bucket_order(key, agg.keys()):
        for horizon in config.horizons_minutes:
            cell = agg[bucket][horizon]
            if cell.n == 0:
                continue
            rows.append(_cell_row(bucket, horizon, cell, config.min_bucket_n))
    return f"### {title}\n\n" + _table(_RETURN_HEADERS, rows)


def _render_coverage(result: EventStudyResult) -> str:
    cov = result.coverage
    lines = [
        "## 1. Data coverage — READ THIS FIRST",
        "",
        f"* study window requested: **{result.days}d** "
        f"({result.window_start:%Y-%m-%d %H:%M} → {result.window_end:%Y-%m-%d %H:%M} UTC)",
    ]
    if cov.min_ts is None or cov.max_ts is None:
        lines.append("* `intraday_quotes`: **EMPTY** — nothing is measurable.")
        return "\n".join(lines)

    span = cov.span_hours or 0.0
    lines.append(
        f"* `intraday_quotes` actual coverage: **{cov.min_ts:%Y-%m-%d %H:%M} → "
        f"{cov.max_ts:%Y-%m-%d %H:%M} UTC** ({span:.1f}h, {cov.rows:,} rows, "
        f"{cov.symbols} symbols)"
    )
    requested_hours = result.days * 24.0
    if span < requested_hours * 0.98:
        lines.extend(
            [
                "",
                f"> **WARNING — the quote buffer ({span:.1f}h) is far shorter than the "
                f"requested {requested_hours:.0f}h window.** `intraday_quotes` is a "
                "ROLLING confirmation buffer pruned to a short retention horizon by "
                "`src/data/intraday_pricer.py`, not a historical archive. Every event "
                f"seen before {cov.min_ts:%Y-%m-%d %H:%M} UTC is UNMEASURABLE — it is "
                "counted in the exclusions below, never scored as flat. Treat the "
                "sample sizes here as the true, small sample they are.",
            ]
        )

    # Horizon feasibility: a horizon h can only be priced for an anchor at
    # least h minutes before the END of the buffer (plus the tolerance). With a
    # short buffer the long horizons collapse onto a sliver of wall-clock time,
    # so their "sample" is one market window replayed across instruments — a
    # large n there is NOT a large sample.
    span_min = span * 60.0
    tol = float(result.config.quote_match_tolerance_min)
    feasible_rows = []
    for horizon in result.config.horizons_minutes:
        eligible = max(0.0, span_min - horizon + tol)
        feasible_rows.append(
            [
                f"{horizon}m",
                f"{eligible:.0f}",
                f"{(eligible / span_min * 100.0) if span_min > 0 else 0.0:.0f}%",
            ]
        )
    lines.extend(
        [
            "",
            "**Horizon feasibility given this buffer** — the wall-clock span of "
            "anchors that can possibly be priced at each horizon:",
            "",
            _table(
                ("horizon", "eligible anchor span (min)", "% of buffer"),
                feasible_rows,
            ),
            "",
            "> **INDEPENDENCE CAVEAT — every t-stat below is an UPPER BOUND on "
            "significance.** Observations are legs, not independent trials: one "
            "event contributes a leg per affected instrument, and events cluster "
            "into the same minutes on the same instruments. The effective sample "
            "is closer to the number of distinct events × distinct market windows "
            "than to the leg count. A horizon whose eligible anchor span above is "
            "a small % of the buffer is measuring ONE market window across many "
            "instruments — read it as an anecdote with a big n, not as evidence.",
        ]
    )
    return "\n".join(lines)


def _render_census(result: EventStudyResult) -> str:
    lines = ["## 2. Event census and exclusions", ""]
    census_rows = [
        [status, f"{result.status_counts.get(status, 0):,}"] for status in CENSUS_STATUSES
    ]
    census_rows.append(["**total in window**", f"{sum(result.status_counts.values()):,}"])
    lines.append(_table(("status", "events"), census_rows))
    lines.extend(
        [
            "",
            "DISMISSED rows are counted but never scored — a triaged-out row carries "
            "no tradable assessment.",
            "",
            "**Exclusions (every unmeasured row lands in exactly one bucket):**",
            "",
        ]
    )
    excl_rows = [[reason, f"{count:,}"] for reason, count in sorted(result.exclusions.items())]
    lines.append(_table(("exclusion reason", "count"), excl_rows))
    lines.extend(
        [
            "",
            f"* events contributing >=1 measured headline leg: **{result.events_with_legs:,}**",
            f"* events contributing >=1 measured confirmation leg: "
            f"**{result.events_with_confirmation_legs:,}**",
            f"* measured headline legs: **{len(result.headline_legs):,}**",
            f"* measured confirmation legs: **{len(result.confirmation_legs):,}**",
        ]
    )
    if result.unmapped_instruments:
        lines.extend(["", "**Unmapped assessment instruments (excluded, not guessed):**", ""])
        lines.append(
            _table(
                ("instrument", "legs"),
                [
                    [inst, f"{n:,}"]
                    for inst, n in sorted(
                        result.unmapped_instruments.items(), key=lambda kv: -kv[1]
                    )
                ],
            )
        )
    if result.missing_horizon_counts:
        lines.extend(["", "**Horizons dropped for lack of an in-tolerance quote:**", ""])
        lines.append(
            _table(
                ("horizon", "legs missing"),
                [
                    [f"{h}m", f"{result.missing_horizon_counts.get(h, 0):,}"]
                    for h in result.config.horizons_minutes
                ],
            )
        )
    return "\n".join(lines)


def _render_costs(result: EventStudyResult) -> str:
    spreads = [
        leg.half_spread_frac for leg in result.headline_legs if leg.half_spread_frac is not None
    ]
    lines = ["## 3. Entry cost (half-spread at entry)", ""]
    if not spreads:
        lines.append(
            "Cost estimate **unavailable** — no entry quote carried both a bid and an "
            "ask. Returns below are GROSS and cannot be netted from this data."
        )
        return "\n".join(lines)
    lines.extend(
        [
            f"* legs with a two-sided entry quote: **{len(spreads):,}** of "
            f"{len(result.headline_legs):,}",
            f"* mean one-way half-spread: **{statistics.fmean(spreads) * 10_000:.2f} bps**",
            f"* median: **{statistics.median(spreads) * 10_000:.2f} bps** "
            f"(round-trip ≈ {statistics.median(spreads) * 2 * 10_000:.2f} bps)",
            "",
            "All returns below are GROSS. A signal must beat the ROUND-TRIP cost "
            "(2 × half-spread) before it is worth trading.",
        ]
    )
    per_symbol = [
        [
            symbol,
            str(len(values)),
            f"{statistics.median(values) * 10_000:.2f}",
        ]
        for symbol, values in sorted(_group_spreads(result.headline_legs).items())
    ]
    lines.extend(["", _table(("symbol", "legs", "median half-spread bps"), per_symbol)])
    return "\n".join(lines)


def _group_spreads(legs: Sequence[LegObservation]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for leg in legs:
        if leg.half_spread_frac is not None:
            grouped.setdefault(leg.symbol, []).append(leg.half_spread_frac)
    return grouped


def _render_latency(result: EventStudyResult) -> str:
    lines = ["## 4. Confirmation latency (status_updated_at − seen_at)", ""]
    if not result.latency_minutes:
        lines.append("No CONFIRMED/TRADED rows in the window — no latency to report.")
        return "\n".join(lines)
    values = list(result.latency_minutes)
    lines.extend(
        [
            f"* n = **{len(values)}** CONFIRMED/TRADED rows",
            f"* p25 = **{_fmt(_percentile(values, 0.25), 1)} min**",
            f"* median = **{_fmt(_percentile(values, 0.50), 1)} min**",
            f"* p75 = **{_fmt(_percentile(values, 0.75), 1)} min**",
            "",
            "`status_updated_at` is an APPROXIMATION of the confirmation/trade time — "
            "it is the moment of the last status transition, which for a CONFIRMED or "
            "TRADED row is that confirmation. A row transitioned again later would "
            "carry the later stamp.",
        ]
    )
    return "\n".join(lines)


def _render_entry_comparison(result: EventStudyResult) -> str:
    cfg = result.config
    confirmed_ids = {leg.event_id for leg in result.confirmation_legs}
    headline_all = result.headline_legs
    headline_subset = [leg for leg in headline_all if leg.event_id in confirmed_ids]

    rows: list[list[str]] = []
    for horizon in cfg.horizons_minutes:
        a = summarize([leg.returns[horizon] for leg in headline_all if horizon in leg.returns])
        b = summarize([leg.returns[horizon] for leg in headline_subset if horizon in leg.returns])
        c = summarize(
            [leg.returns[horizon] for leg in result.confirmation_legs if horizon in leg.returns]
        )
        rows.append(
            [
                f"{horizon}m",
                str(a.n),
                _fmt(a.mean, 2, 10_000.0),
                _fmt_pct(a.hit_rate),
                str(b.n),
                _fmt(b.mean, 2, 10_000.0),
                _fmt_pct(b.hit_rate),
                str(c.n),
                _fmt(c.mean, 2, 10_000.0),
                _fmt_pct(c.hit_rate),
                _fmt(c.t_stat, 2),
                "LOW-N" if min(a.n, b.n, c.n) < cfg.min_bucket_n else "",
            ]
        )
    headers = (
        "horizon",
        "n (HL all)",
        "mean bps",
        "hit",
        "n (HL conf-subset)",
        "mean bps",
        "hit",
        "n (CONF entry)",
        "mean bps",
        "hit",
        "t (CONF)",
        "flag",
    )
    return (
        "## 5. THE PRIMARY QUESTION — headline entry vs confirmation entry\n\n"
        "`HL all` = every measured leg entered at the headline. "
        "`HL conf-subset` = headline entry restricted to the SAME events that later "
        "confirmed/traded (the apples-to-apples comparison). "
        "`CONF entry` = the same events entered at the approximate confirmation time. "
        "If `CONF entry` is materially worse than `HL conf-subset`, the gate is paying "
        "away the move it selected for.\n\n" + _table(headers, rows)
    )


def _render_excursions(result: EventStudyResult) -> str:
    cfg = result.config
    mfe = [leg.mfe for leg in result.headline_legs if leg.mfe is not None]
    mae = [leg.mae for leg in result.headline_legs if leg.mae is not None]
    lines = [f"## 6. Excursions within {cfg.mfe_mae_horizon_min}m of the headline entry", ""]
    if not mfe or not mae:
        lines.append("No measurable excursion paths.")
        return "\n".join(lines)
    rows = [
        [
            "MFE (max favorable, signed)",
            str(len(mfe)),
            _fmt(statistics.fmean(mfe), 2, 10_000.0),
            _fmt(statistics.median(mfe), 2, 10_000.0),
            _fmt(_percentile(mfe, 0.9), 2, 10_000.0),
        ],
        [
            "MAE (max adverse, signed)",
            str(len(mae)),
            _fmt(statistics.fmean(mae), 2, 10_000.0),
            _fmt(statistics.median(mae), 2, 10_000.0),
            _fmt(_percentile(mae, 0.1), 2, 10_000.0),
        ],
    ]
    lines.append(_table(("measure", "n", "mean bps", "median bps", "tail bps"), rows))
    lines.extend(
        [
            "",
            "Both are SIGNED excursions of the assessment's own direction. MFE is the "
            "best mark-to-market the trade ever saw; MAE is the worst. If |MAE| is "
            "comparable to MFE, a 1% hard stop is a coin flip on noise, not on thesis.",
        ]
    )
    return "\n".join(lines)


_OPTION_HEADERS = ("bucket", "n", "win rate", "mean %", "median %", "std %", "t", "flag")


def _option_rows(agg: Mapping[str, Cell], min_n: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for bucket, cell in sorted(agg.items(), key=lambda kv: -kv[1].n):
        rows.append(
            [
                bucket,
                str(cell.n),
                _fmt_pct(cell.hit_rate),
                _fmt(cell.mean, 1, 100.0),
                _fmt(cell.median, 1, 100.0),
                _fmt(cell.std, 1, 100.0),
                _fmt(cell.t_stat, 2),
                "LOW-N" if cell.n < min_n else "",
            ]
        )
    return rows


def build_options_report(result: OptionsAttributionResult) -> str:
    lines = [
        "## 8. Options attribution (realized Alpaca paper P&L)",
        "",
        f"Window: {result.window_start:%Y-%m-%d %H:%M} → "
        f"{result.window_end:%Y-%m-%d %H:%M} UTC ({result.days}d), on "
        "`alpaca_option_orders.submitted_at`.",
        "",
        f"* rows in window: **{result.total_rows:,}**",
        f"* realized (pnl_pct NOT NULL): **{len(result.outcomes):,}**",
        f"* unrealized / non-positions (pnl_pct NULL): **{result.unrealized_rows:,}**",
        f"* unattributed (no matching trade_idea or geo_event): **{result.unattributed:,}**",
    ]
    if not result.outcomes:
        lines.extend(["", "_No realized option P&L in this window._"])
        return "\n".join(lines)

    by_era: dict[str, list[OptionOutcome]] = {}
    for outcome in result.outcomes:
        by_era.setdefault(outcome.era, []).append(outcome)

    lines.extend(
        [
            "",
            "> **CL-d44a — the two eras are NEVER pooled.** Rows with `entry_mid IS NULL` "
            "predate migration 018: they were marked at the BID against an ASK fill, so "
            "their `pnl_pct` carries the full option spread as a phantom loss. Those "
            "numbers measure the SPREAD, not the thesis. Only the post-018 mid-marked "
            "cohort is an honest read on whether the ideas worked.",
            "",
        ]
    )

    for era in (POST_018_ERA, LEGACY_ERA):
        rows = by_era.get(era, [])
        lines.extend(["", f"### Era: {era} — n = {len(rows)}", ""])
        if not rows:
            lines.append("_(no rows in this cohort)_")
            continue
        for title, key in (
            ("Overall", "overall"),
            ("By theme", "theme"),
            ("By urgency bucket (from the event assessment)", "urgency"),
            ("By idea confidence bucket", "confidence"),
            ("By niche flag (notes contains 'niche')", "niche"),
            ("By red-team flag (notes contains 'red-team')", "red_team"),
            ("By entry_spread_pct bucket", "spread"),
            ("By exit reason", "exit_reason"),
        ):
            lines.extend(
                [
                    "",
                    f"**{title}**",
                    "",
                    _table(
                        _OPTION_HEADERS,
                        _option_rows(aggregate_options(rows, key), result.min_bucket_n),
                    ),
                ]
            )
    return "\n".join(lines)


def build_report(
    result: EventStudyResult,
    options: OptionsAttributionResult | None = None,
) -> str:
    """Render the whole study as GitHub-flavoured markdown (also readable as
    plain text in a terminal or on a phone)."""
    cfg = result.config
    sections = [
        "# curLit event study (CL-z95p)",
        "",
        f"Generated {result.window_end:%Y-%m-%d %H:%M} UTC. Returns are GROSS, in "
        "basis points, SIGNED by the assessment's own direction (+1 long / −1 short). "
        f"Cells with n < {cfg.min_bucket_n} are flagged **LOW-N** — they are noise, "
        "not evidence.",
        "",
        f"Config: horizons={list(cfg.horizons_minutes)}m, "
        f"tolerance={cfg.quote_match_tolerance_min}m, "
        f"MFE/MAE window={cfg.mfe_mae_horizon_min}m, min_bucket_n={cfg.min_bucket_n}.",
        "",
        "Entry is the FIRST quote at or after the anchor (no lookahead — a quote "
        "before the anchor is never an entry). Horizon prices are the NEAREST quote "
        "to anchor+h within tolerance; a horizon with no match is DROPPED and counted.",
        "",
        _render_coverage(result),
        "",
        _render_census(result),
        "",
        _render_costs(result),
        "",
        _render_latency(result),
        "",
        _render_entry_comparison(result),
        "",
        _render_excursions(result),
        "",
        "## 7. Headline-entry returns by dimension",
        "",
    ]
    for title, key in (
        ("Overall", "overall"),
        ("By urgency bucket", "urgency"),
        ("By confidence bucket", "confidence"),
        ("By gate outcome (status)", "status"),
        ("By direction", "direction"),
        ("By playbook tier", "tier"),
        ("By theme", "theme"),
        ("By instrument", "symbol"),
    ):
        sections.extend([_render_dimension(title, result.headline_legs, key, cfg), ""])

    if result.confirmation_legs:
        sections.extend(["## 7b. Confirmation-entry returns by dimension", ""])
        for title, key in (
            ("Overall", "overall"),
            ("By urgency bucket", "urgency"),
            ("By confidence bucket", "confidence"),
            ("By gate outcome (status)", "status"),
            ("By direction", "direction"),
            ("By theme", "theme"),
        ):
            sections.extend([_render_dimension(title, result.confirmation_legs, key, cfg), ""])

    if options is not None:
        sections.extend([build_options_report(options), ""])

    return "\n".join(sections).rstrip() + "\n"
