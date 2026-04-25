"""Paper-vs-live divergence detector (G4 / CL-yq5) — separate signal from execution.

When a strategy's paper Sharpe is 0.8 but live Sharpe is 0.2, the question is:
is the signal still good and execution is eating the edge, or has the signal
itself decayed? Running the paper broker alongside the live broker on the same
intents lets us compare *execution quality directly*:

    price divergence (bps)   live_fill_price vs paper_fill_price for the same intent
    latency (ms)             live_fill_ts    vs paper_fill_ts
    match rate               fraction of intents where both brokers actually filled

This module owns the matching + reporting; the concurrent paper broker
operation is wired by the LiveEngine (separate ticket) so each intent is
submitted to both brokers and the resulting fills get tagged with a shared
intent_id for matching.

Reference: reference/14_edge_testing.md, plus G1+G2 inform the statistical
significance layer (this file just produces the raw divergence metrics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.execution.broker import Fill

logger = logging.getLogger(__name__)


# Default flagging threshold per acceptance criteria — execution price drift
# beyond this is "concerning" enough to surface in the divergence report.
_DEFAULT_PRICE_DIVERGENCE_FLAG_BPS: float = 2.0

# Default match tolerance window for matching paper and live fills by
# timestamp. 60 s accommodates broker-side queueing variance without admitting
# unrelated fills as matches. Tunable per deployment.
_DEFAULT_MATCH_TOLERANCE_SEC: float = 60.0


@dataclass
class MatchedPair:
    """One paper-live fill match with computed divergence metrics."""

    intent_id: str
    symbol: str
    side: str
    quantity_paper: float
    quantity_live: float
    price_paper: float
    price_live: float
    ts_paper: datetime
    ts_live: datetime
    price_diff_bps: float
    latency_ms: float
    flagged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity_paper": self.quantity_paper,
            "quantity_live": self.quantity_live,
            "price_paper": self.price_paper,
            "price_live": self.price_live,
            "ts_paper": self.ts_paper.isoformat(),
            "ts_live": self.ts_live.isoformat(),
            "price_diff_bps": self.price_diff_bps,
            "latency_ms": self.latency_ms,
            "flagged": self.flagged,
        }


@dataclass
class DivergenceReport:
    """Aggregated paper-vs-live divergence summary across many fills."""

    n_paper_fills: int
    n_live_fills: int
    n_matched: int
    avg_price_diff_bps: float
    avg_abs_price_diff_bps: float
    avg_latency_ms: float
    signal_match_rate: float
    n_flagged: int
    flag_threshold_bps: float
    matched_pairs: list[MatchedPair] = field(repr=False, default_factory=list)
    unmatched_paper_intents: list[str] = field(repr=False, default_factory=list)
    unmatched_live_intents: list[str] = field(repr=False, default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_paper_fills": self.n_paper_fills,
            "n_live_fills": self.n_live_fills,
            "n_matched": self.n_matched,
            "avg_price_diff_bps": self.avg_price_diff_bps,
            "avg_abs_price_diff_bps": self.avg_abs_price_diff_bps,
            "avg_latency_ms": self.avg_latency_ms,
            "signal_match_rate": self.signal_match_rate,
            "n_flagged": self.n_flagged,
            "flag_threshold_bps": self.flag_threshold_bps,
            "n_unmatched_paper": len(self.unmatched_paper_intents),
            "n_unmatched_live": len(self.unmatched_live_intents),
        }


def _price_diff_bps(price_paper: float, price_live: float, side: str) -> float:
    """Signed price divergence in bps from the perspective of execution cost.

    A buy that fills at a HIGHER live price than paper costs us money → +bps.
    A sell that fills at a LOWER live price than paper costs us money → +bps.
    Negative bps means live executed at a more favorable price than paper.
    """
    if price_paper <= 0:
        return 0.0
    sign = 1.0 if side == "buy" else -1.0
    return sign * (price_live - price_paper) / price_paper * 10_000.0


class PaperLiveDivergence:
    """Match paper fills against live fills and compute execution divergence.

    Usage:
        detector = PaperLiveDivergence(flag_threshold_bps=2.0)
        report = detector.compare(paper_fills, live_fills)
        if report.avg_abs_price_diff_bps > some_alarm:
            handle_alarm(report)
    """

    def __init__(
        self,
        flag_threshold_bps: float = _DEFAULT_PRICE_DIVERGENCE_FLAG_BPS,
        match_tolerance_seconds: float = _DEFAULT_MATCH_TOLERANCE_SEC,
    ) -> None:
        assert flag_threshold_bps > 0, (
            f"flag_threshold_bps must be positive, got {flag_threshold_bps}"
        )
        assert match_tolerance_seconds > 0, (
            f"match_tolerance_seconds must be positive, got {match_tolerance_seconds}"
        )
        self.flag_threshold_bps = flag_threshold_bps
        self.match_tolerance = timedelta(seconds=match_tolerance_seconds)

    def compare(
        self,
        paper_fills: list[Fill],
        live_fills: list[Fill],
    ) -> DivergenceReport:
        """Match paper-live fills by (symbol, side, timestamp window) and report.

        Each paper fill is matched to the closest unused live fill on the same
        symbol+side within `match_tolerance`. A live fill can match at most one
        paper fill (and vice versa). Unmatched intents on either side are
        recorded for the signal_match_rate metric.
        """
        matched: list[MatchedPair] = []
        used_live_idx: set[int] = set()

        # Sort live fills by ts for fast nearest-neighbor lookup.
        live_with_idx: list[tuple[int, Fill]] = sorted(
            enumerate(live_fills), key=lambda x: x[1].timestamp,
        )

        unmatched_paper: list[str] = []

        for paper in paper_fills:
            best_idx: int | None = None
            best_dt = self.match_tolerance + timedelta(seconds=1)
            for idx, live in live_with_idx:
                if idx in used_live_idx:
                    continue
                if live.symbol != paper.symbol or live.side != paper.side:
                    continue
                dt = abs(live.timestamp - paper.timestamp)
                if dt > self.match_tolerance:
                    continue
                if dt < best_dt:
                    best_dt = dt
                    best_idx = idx

            if best_idx is None:
                unmatched_paper.append(paper.fill_id)
                continue

            used_live_idx.add(best_idx)
            live = live_fills[best_idx]
            price_diff_bps = _price_diff_bps(paper.price, live.price, paper.side)
            latency_ms = (
                live.timestamp - paper.timestamp
            ).total_seconds() * 1000.0
            matched.append(
                MatchedPair(
                    intent_id=paper.order_id,
                    symbol=paper.symbol,
                    side=paper.side,
                    quantity_paper=paper.quantity,
                    quantity_live=live.quantity,
                    price_paper=paper.price,
                    price_live=live.price,
                    ts_paper=paper.timestamp,
                    ts_live=live.timestamp,
                    price_diff_bps=price_diff_bps,
                    latency_ms=latency_ms,
                    flagged=abs(price_diff_bps) > self.flag_threshold_bps,
                )
            )

        unmatched_live = [
            f.fill_id for i, f in enumerate(live_fills) if i not in used_live_idx
        ]

        n_paper = len(paper_fills)
        n_live = len(live_fills)
        n_matched = len(matched)

        if n_matched == 0:
            avg_diff = 0.0
            avg_abs_diff = 0.0
            avg_latency = 0.0
        else:
            avg_diff = sum(p.price_diff_bps for p in matched) / n_matched
            avg_abs_diff = sum(abs(p.price_diff_bps) for p in matched) / n_matched
            avg_latency = sum(p.latency_ms for p in matched) / n_matched

        # Signal match rate: fraction of paper intents that found a live match.
        # If no paper fills, rate is undefined → return 0.0 with note in caller.
        match_rate = (n_matched / n_paper) if n_paper > 0 else 0.0
        n_flagged = sum(1 for p in matched if p.flagged)

        report = DivergenceReport(
            n_paper_fills=n_paper,
            n_live_fills=n_live,
            n_matched=n_matched,
            avg_price_diff_bps=avg_diff,
            avg_abs_price_diff_bps=avg_abs_diff,
            avg_latency_ms=avg_latency,
            signal_match_rate=match_rate,
            n_flagged=n_flagged,
            flag_threshold_bps=self.flag_threshold_bps,
            matched_pairs=matched,
            unmatched_paper_intents=unmatched_paper,
            unmatched_live_intents=unmatched_live,
        )
        logger.info(
            "Paper-live divergence: %d/%d matched, avg |diff|=%.2f bps, "
            "avg latency=%.0f ms, %d flagged at >%g bps",
            n_matched, n_paper, avg_abs_diff, avg_latency, n_flagged,
            self.flag_threshold_bps,
        )
        return report
