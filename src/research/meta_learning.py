"""Research meta-learning (CL-fsj).

Closes the feedback loop between research-loop output and live-trading
outcome. The question this module answers: *which sources / authors /
topics actually produce strategies that survive promotion to live, and
which produce noise we waste extractor + debate budget on?*

Pipeline:

  1. Read all research_papers rows that have an associated implemented
     strategy (papers.implementation_priority > 0 OR a foreign-key into
     strategy_fills).
  2. For each, look up the strategy's live P&L from PnLAttributor.
  3. Roll up by source_label, author, and detected topic (keyword
     overlap with predefined topic buckets).
  4. Persist the rollup to research_papers.evaluation_data JSONB and
     emit a MetaLearningWeights overlay that RelevanceScorer reads.

The overlay is intentionally a multiplier on existing keyword weights,
not a wholesale replacement — small adjustments compound, big swaps
react too aggressively to small samples.

Quarterly cadence (per CL-fsj spec): the orchestrator re-runs after
strategies have had at least one full quarter of live data. Below that
horizon the per-source sample is too small to be informative.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Topic buckets — coarse partition of FX research themes. The bucket a
# paper falls into is the bucket whose keyword set has the most matches
# in the paper's title + abstract. Ties broken alphabetically (stable).
# --------------------------------------------------------------------- #
_TOPIC_BUCKETS: dict[str, tuple[str, ...]] = {
    "carry": ("carry trade", "currency carry", "interest rate parity"),
    "momentum": ("fx momentum", "currency momentum", "trend"),
    "value": ("real exchange rate", "purchasing power", "ppp"),
    "volatility": ("volatility risk premium", "fx volatility", "vix"),
    "sentiment": ("central bank", "monetary policy", "fomc", "ecb"),
    "microstructure": ("order flow", "fx microstructure", "limit order"),
}

# Multiplier bounds. A source that produced 100% winners shouldn't get
# its weights tripled — sample sizes are small. A source that produced
# 100% losers shouldn't be zeroed out either; some signals are slow.
_MIN_MULTIPLIER: float = 0.5
_MAX_MULTIPLIER: float = 1.5


@dataclass
class SourceOutcome:
    source_label: str
    n_papers: int
    n_implemented: int
    n_promoted: int  # made it to live trading (CL-6vv)
    total_pnl_usd: float
    multiplier: float = 1.0


@dataclass
class TopicOutcome:
    topic: str
    n_papers: int
    total_pnl_usd: float


@dataclass
class MetaLearningReport:
    sources: dict[str, SourceOutcome] = field(default_factory=dict)
    topics: dict[str, TopicOutcome] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _detect_topic(title: str, abstract: str) -> str:
    """Return the topic bucket whose keywords have the most matches.
    Returns ``"other"`` if no bucket has any match."""
    haystack = f"{title}\n{abstract}".lower()
    scores: dict[str, int] = defaultdict(int)
    for topic, kws in _TOPIC_BUCKETS.items():
        for kw in kws:
            if kw in haystack:
                scores[topic] += 1
    if not scores:
        return "other"
    return max(sorted(scores), key=lambda t: scores[t])


def _compute_multiplier(implemented: int, total_pnl: float) -> float:
    """Map (n_implemented, total_pnl) → multiplier.

    Rules:
      - 0 implemented → 1.0 (no signal yet)
      - implemented but 0 / negative pnl → 0.85 (slight downweight)
      - positive pnl, scaled by magnitude / count
    """
    if implemented == 0:
        return 1.0
    if total_pnl <= 0:
        return 0.85
    # Cap the upweight at +50% even for outsized winners so a single
    # 10× strategy doesn't drown out the rest of the universe.
    avg_pnl = total_pnl / implemented
    # Scale: $1k per implemented paper → 1.05 mult. $10k → 1.50 mult.
    scaled = 1.0 + min((avg_pnl / 20_000.0), 0.5)
    return max(_MIN_MULTIPLIER, min(_MAX_MULTIPLIER, scaled))


class MetaLearner:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def gather(
        self,
        since: datetime | None = None,
    ) -> MetaLearningReport:
        """Read papers + strategy fills, build the rollup. Returns the
        in-memory report; persistence is opt-in via .persist()."""
        since = since or datetime.now(UTC) - timedelta(days=365)

        report = MetaLearningReport()

        with self.engine.connect() as conn:
            # Pull all papers ingested in the lookback window. Use the
            # caller's `since` directly as the floor; rows without
            # ingested_at are treated as "now" (i.e. always included).
            paper_rows = conn.execute(
                text("""
                SELECT paper_id, title, abstract, source, implementation_priority
                FROM research_papers
                WHERE ingested_at IS NULL OR ingested_at >= :since
            """),
                {"since": since},
            ).fetchall()

            if not paper_rows:
                return report

            # Aggregate per-strategy P&L from strategy_fills (CL-8dq).
            # Map paper → strategy via paper.implementation_priority +
            # the convention that strategy_id = paper_id when promoted.
            # For papers with no matching strategy_id, P&L = 0.
            pnl_rows = conn.execute(
                text("""
                SELECT strategy_id,
                       COALESCE(SUM(quantity * fill_price), 0) AS pnl
                FROM strategy_fills
                GROUP BY strategy_id
            """)
            ).fetchall()
            pnl_by_strategy = {sid: float(p) for sid, p in pnl_rows}

        sources: dict[str, SourceOutcome] = {}
        topics: dict[str, list[float]] = defaultdict(list)

        for paper_id, title, abstract, source, prio in paper_rows:
            source_label = str(source or "unknown")
            implemented = bool(prio and int(prio) > 0)
            promoted = paper_id in pnl_by_strategy
            paper_pnl = pnl_by_strategy.get(paper_id, 0.0)

            if source_label not in sources:
                sources[source_label] = SourceOutcome(
                    source_label=source_label,
                    n_papers=0,
                    n_implemented=0,
                    n_promoted=0,
                    total_pnl_usd=0.0,
                )
            s = sources[source_label]
            s.n_papers += 1
            if implemented:
                s.n_implemented += 1
            if promoted:
                s.n_promoted += 1
            s.total_pnl_usd += paper_pnl

            topic = _detect_topic(title or "", abstract or "")
            topics[topic].append(paper_pnl)

        # Compute multipliers + finalize.
        for s in sources.values():
            s.multiplier = _compute_multiplier(s.n_implemented, s.total_pnl_usd)
        report.sources = sources
        report.topics = {
            t: TopicOutcome(topic=t, n_papers=len(pnls), total_pnl_usd=sum(pnls))
            for t, pnls in topics.items()
        }
        return report

    def persist_and_emit_overlay(
        self,
        report: MetaLearningReport,
    ) -> dict[str, dict[str, float]]:
        """Write report into research_papers.evaluation_data and return
        the overlay dict consumable by RelevanceScorer.

        Overlay shape: {source_label: {keyword: multiplier}}. We apply
        the source's overall multiplier to ALL keywords for that source —
        per-keyword nuance would need orders of magnitude more data.
        """
        overlay: dict[str, dict[str, float]] = {}
        for source_label, outcome in report.sources.items():
            # Apply the source-level multiplier to every keyword in the
            # criteria. RelevanceScorer.score() looks up source_label in
            # the overlay and reads per-keyword multipliers.
            overlay[source_label] = {"*": outcome.multiplier}

        # Persist a per-source row in research_papers.evaluation_data —
        # we use a synthetic "meta:learning:<source>" paper_id so the
        # rollup survives without polluting actual paper rows.
        try:
            with self.engine.begin() as conn:
                for source_label, outcome in report.sources.items():
                    payload = {
                        "kind": "meta_learning_rollup",
                        "source_label": source_label,
                        "n_papers": outcome.n_papers,
                        "n_implemented": outcome.n_implemented,
                        "n_promoted": outcome.n_promoted,
                        "total_pnl_usd": outcome.total_pnl_usd,
                        "multiplier": outcome.multiplier,
                        "generated_at": report.generated_at.isoformat(),
                    }
                    pid = f"meta:learning:{source_label.replace(' ', '_').lower()}"
                    dialect = self.engine.dialect.name
                    if dialect == "postgresql":
                        stmt = text("""
                            INSERT INTO research_papers
                                (paper_id, source, evaluation_data)
                            VALUES (:pid, :source, CAST(:data AS JSONB))
                            ON CONFLICT (paper_id) DO UPDATE
                                SET evaluation_data = EXCLUDED.evaluation_data
                        """)
                    else:
                        stmt = text(
                            "INSERT OR REPLACE INTO research_papers "
                            "(paper_id, source, evaluation_data) "
                            "VALUES (:pid, :source, :data)",
                        )
                    conn.execute(
                        stmt,
                        {
                            "pid": pid,
                            "source": source_label,
                            "data": json.dumps(payload),
                        },
                    )
        except Exception:
            logger.exception("meta-learning persist failed")
        return overlay


def update_scorer_overlay(
    scorer: Any,
    learner: MetaLearner,
) -> int:
    """Convenience: gather → persist → push the overlay onto an existing
    RelevanceScorer. Returns the number of source-labels updated."""
    report = learner.gather()
    overlay = learner.persist_and_emit_overlay(report)
    scorer.meta_overlay = overlay
    return len(overlay)
