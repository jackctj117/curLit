"""RelevanceScorer for paper triage (CL-366).

Scores each ingested paper on three dimensions and writes a single
``relevance_score`` to ``research_papers``. Scores feed the dashboard's
triage view; the LLM idea agent then drinks from the top-N filtered
list rather than the raw firehose.

Scoring dimensions:

  1. Keyword score  — sum of weights for matched FX/macro terms in
                      title/abstract minus weights for matched
                      red-flag terms.
  2. Author score   — flat bonus per matched preferred author (Sarno,
                      Lustig, Kelly, Menkhoff, etc).
  3. Category score — flat bonus when arXiv category / source label
                      matches a preferred set.

The scorer is configurable via RelevanceCriteria; defaults baked in
from the CL-1i7 evaluation rubric and observed productive sources.

CL-fsj (research meta-learning) adjusts these weights based on which
sources / authors / topics produced strategies that survived to live
trading. The MetaLearningWeights overlay is loaded at score time when
present.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Default keyword weights — sourced from the CL-1i7 paper evaluation
# rubric and observed productive FX-research topics. Each weight is the
# contribution to the keyword score per match; matches across multiple
# fields (title + abstract) count once each, capped at field count.
# --------------------------------------------------------------------- #
_DEFAULT_POSITIVE_KEYWORDS: dict[str, float] = {
    # FX-specific factor literature (highest signal)
    "carry trade": 3.5,
    "fx momentum": 3.0,
    "currency carry": 3.5,
    "exchange rate": 2.0,
    "currency factor": 2.5,
    "interest rate parity": 2.5,
    "uncovered interest": 2.5,
    # Volatility-conditioned strategies (Menkhoff 2012 family)
    "volatility risk premium": 3.0,
    "fx volatility": 2.5,
    "global volatility": 2.0,
    # Microstructure / liquidity
    "order flow": 2.0,
    "fx microstructure": 2.5,
    "limit order book": 1.5,
    # Macro plumbing
    "central bank": 1.5,
    "monetary policy": 1.5,
    "real exchange rate": 1.5,
    "purchasing power": 1.5,
    # Methodologies that travel well to FX
    "regime switching": 2.0,
    "out-of-sample": 1.5,
    "walk-forward": 1.5,
    "transaction cost": 1.5,
}

_DEFAULT_NEGATIVE_KEYWORDS: dict[str, float] = {
    # Pure-theory papers that don't backtest
    "theoretical model": 1.0,
    "stylized model": 1.0,
    "general equilibrium": 1.0,
    # Single-stock or non-FX topics that get flagged as macro
    "earnings announcement": 2.0,
    "options pricing": 2.0,
    "merger arbitrage": 2.0,
    # ML papers that tend to overfit small samples
    "deep learning": 1.0,
    "transformer": 1.5,
    "neural network": 1.0,
}

_DEFAULT_PREFERRED_AUTHORS: tuple[str, ...] = (
    # FX factor / carry literature
    "sarno",
    "lustig",
    "menkhoff",
    "verdelhan",
    "schmeling",
    # Asset pricing + alpha
    "kelly",
    "asness",
    "moskowitz",
    "cochrane",
    # Microstructure
    "hasbrouck",
    "easley",
    # Macro + monetary
    "gourinchas",
    "rey",
)

_DEFAULT_PREFERRED_CATEGORIES: tuple[str, ...] = (
    "q-fin.PM",
    "q-fin.TR",
    "q-fin.ST",
    "NBER",
    "BIS Working Papers",
    "FRBSF Economic Letter",
)

_AUTHOR_BONUS: float = 2.0
_CATEGORY_BONUS: float = 1.5


@dataclass
class RelevanceCriteria:
    positive_keywords: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_POSITIVE_KEYWORDS),
    )
    negative_keywords: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_NEGATIVE_KEYWORDS),
    )
    preferred_authors: tuple[str, ...] = _DEFAULT_PREFERRED_AUTHORS
    preferred_categories: tuple[str, ...] = _DEFAULT_PREFERRED_CATEGORIES
    author_bonus: float = _AUTHOR_BONUS
    category_bonus: float = _CATEGORY_BONUS


@dataclass
class ScoreBreakdown:
    keyword_score: float
    author_score: float
    category_score: float

    @property
    def total(self) -> float:
        return self.keyword_score + self.author_score + self.category_score


class RelevanceScorer:
    def __init__(
        self,
        criteria: RelevanceCriteria | None = None,
        meta_overlay: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.criteria = criteria or RelevanceCriteria()
        # CL-fsj overlay: {source_label: {keyword: multiplier}}.
        # Multipliers stack on top of the base keyword weights at scoring
        # time. Default 1.0 = no adjustment; >1.0 = upweight (this source
        # produced winners), <1.0 = downweight (this source produced
        # losers).
        self.meta_overlay = meta_overlay or {}

    def score(
        self,
        title: str,
        abstract: str,
        authors: Iterable[str] = (),
        category: str = "",
        source_label: str = "",
    ) -> ScoreBreakdown:
        text_lower = f"{title}\n{abstract}".lower()
        overlay = self.meta_overlay.get(source_label, {})

        keyword_score = 0.0
        for kw, weight in self.criteria.positive_keywords.items():
            if kw in text_lower:
                multiplier = overlay.get(kw, 1.0)
                keyword_score += weight * multiplier
        for kw, weight in self.criteria.negative_keywords.items():
            if kw in text_lower:
                keyword_score -= weight

        # Authors: tokenize on commas/semicolons, lowercase, last-name match.
        author_score = 0.0
        author_lower = " ".join(authors).lower()
        for preferred in self.criteria.preferred_authors:
            # Word-boundary so "kelly" doesn't match "berkeley".
            if re.search(rf"\b{re.escape(preferred)}\b", author_lower):
                author_score += self.criteria.author_bonus

        # Category bonus: arXiv categories live in source_label like
        # "arXiv q-fin.PM"; NBER/BIS use the source label directly.
        category_score = 0.0
        cat_haystack = f"{category} {source_label}".lower()
        for preferred in self.criteria.preferred_categories:
            if preferred.lower() in cat_haystack:
                category_score += self.criteria.category_bonus
                break  # one bonus per paper, not per category match

        return ScoreBreakdown(
            keyword_score=keyword_score,
            author_score=author_score,
            category_score=category_score,
        )

    def score_paper(self, paper: Any) -> ScoreBreakdown:
        """Score a Paper or Paper-like object."""
        return self.score(
            title=getattr(paper, "title", "") or "",
            abstract=getattr(paper, "abstract", "") or "",
            authors=getattr(paper, "authors", ()) or (),
            category=getattr(paper, "category", "") or "",
            source_label=getattr(paper, "source_label", "") or "",
        )

    def score_and_update(
        self,
        engine: Any,
        paper_id: str,
        paper: Any,
    ) -> float:
        """Score and persist to research_papers.relevance_score. Returns the
        total."""
        breakdown = self.score_paper(paper)
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                    UPDATE research_papers
                    SET relevance_score = :s
                    WHERE paper_id = :pid
                """),
                    {"s": breakdown.total, "pid": paper_id},
                )
        except Exception:
            logger.exception("score_and_update DB write failed for %s", paper_id)
        return breakdown.total
