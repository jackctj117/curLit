"""Tests for RelevanceScorer (CL-366)."""

from __future__ import annotations

import pytest

from src.research.relevance_scorer import RelevanceCriteria, RelevanceScorer


class _FakePaper:
    def __init__(self, **kw) -> None:  # type: ignore[no-untyped-def]
        for k, v in kw.items():
            setattr(self, k, v)


class TestKeywordScore:
    def test_high_score_for_fx_paper(self) -> None:
        s = RelevanceScorer()
        out = s.score(
            title="Carry Trades and Global Foreign Exchange Volatility",
            abstract=(
                "We document a strong link between carry trade returns and "
                "global FX volatility. Using daily exchange rate data..."
            ),
        )
        assert out.total > 5.0

    def test_low_score_for_irrelevant_paper(self) -> None:
        s = RelevanceScorer()
        out = s.score(
            title="Earnings Announcement Reactions in U.S. Equities",
            abstract=(
                "We analyze stock returns around earnings announcements "
                "using a deep learning model on transformer architectures."
            ),
        )
        # Earnings announcement (-2), deep learning (-1), transformer (-1.5)
        assert out.total < 0


class TestAuthorBonus:
    def test_preferred_author_adds_bonus(self) -> None:
        s = RelevanceScorer()
        no_author = s.score(
            title="Random Paper",
            abstract="some text",
            authors=("J. Smith",),
        ).total
        with_author = s.score(
            title="Random Paper",
            abstract="some text",
            authors=("L. Sarno",),
        ).total
        assert with_author > no_author
        assert with_author - no_author == pytest.approx(2.0)

    def test_word_boundary(self) -> None:
        # "kelly" is a preferred author but shouldn't match "berkeley"
        s = RelevanceScorer()
        out = s.score(
            title="t",
            abstract="a",
            authors=("UC Berkeley Press",),
        )
        assert out.author_score == 0


class TestCategoryBonus:
    def test_arxiv_qfin_pm_bonus(self) -> None:
        s = RelevanceScorer()
        no_cat = s.score(title="t", abstract="a").total
        with_cat = s.score(
            title="t",
            abstract="a",
            source_label="arXiv q-fin.PM",
        ).total
        assert with_cat > no_cat


class TestMetaOverlay:
    def test_overlay_applies_per_source(self) -> None:
        # Source-level multiplier upweights keyword scoring 1.5×.
        overlay = {"arXiv q-fin.PM": {"carry trade": 1.5}}
        s = RelevanceScorer(meta_overlay=overlay)

        baseline = s.score(
            title="Carry Trade Returns",
            abstract="x",
            source_label="other",
        ).keyword_score
        boosted = s.score(
            title="Carry Trade Returns",
            abstract="x",
            source_label="arXiv q-fin.PM",
        ).keyword_score

        assert boosted > baseline
        # 3.5 × 1.5 = 5.25 vs 3.5 baseline
        assert boosted == pytest.approx(5.25)


class TestScoreAndUpdate:
    def test_persists_to_db(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from sqlalchemy import create_engine, text

        engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE research_papers ("
                    "  paper_id TEXT PRIMARY KEY, relevance_score REAL DEFAULT 0)",
                )
            )
            conn.execute(
                text(
                    "INSERT INTO research_papers (paper_id) VALUES ('paper:1')",
                )
            )

        s = RelevanceScorer()
        paper = _FakePaper(
            title="Carry Trade Returns",
            abstract="we test carry trade momentum",
            authors=("Sarno",),
            source_label="arXiv q-fin.PM",
        )
        score = s.score_and_update(engine, "paper:1", paper)
        assert score > 0

        with engine.connect() as conn:
            v = conn.execute(
                text(
                    "SELECT relevance_score FROM research_papers WHERE paper_id='paper:1'",
                )
            ).scalar()
            assert v == pytest.approx(score)


class TestCustomCriteria:
    def test_user_can_swap_criteria(self) -> None:
        # Operator wants to track papers about silver markets specifically.
        criteria = RelevanceCriteria(
            positive_keywords={"silver": 5.0},
            negative_keywords={},
            preferred_authors=(),
            preferred_categories=(),
        )
        s = RelevanceScorer(criteria=criteria)
        out = s.score(title="Silver Market Anomalies", abstract="silver squeeze")
        # "silver" appears in both title and abstract — counted once
        # because we lowercase-substring match the combined haystack.
        assert out.keyword_score == pytest.approx(5.0)
