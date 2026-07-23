"""Unit tests — NLP: lexicon scorer, diff analyzer."""

from src.nlp.diff import StatementDiffer
from src.nlp.lexicon_scorer import LexiconScorer


class TestLexiconScorer:
    def test_hawkish_sentence_scores_positive(self) -> None:
        scorer = LexiconScorer()
        result = scorer.score_text("inflation remains elevated and persistent")
        assert result.net_score > 0

    def test_dovish_sentence_scores_negative(self) -> None:
        scorer = LexiconScorer()
        result = scorer.score_text("inflation has moderated and the committee is patient")
        assert result.net_score < 0

    def test_neutral_sentence_scores_near_zero(self) -> None:
        scorer = LexiconScorer()
        result = scorer.score_text("inflation was 3.1 percent in March")
        assert abs(result.net_score) < 0.5


class TestStatementDiffer:
    def test_diff_detects_added_hawkish(self) -> None:
        differ = StatementDiffer()
        prev = ["economic activity is growing."]
        curr = ["economic activity is growing.", "inflation remains elevated."]
        diff = differ.diff(curr, prev)
        assert diff.net_shift > 0

    def test_diff_detects_removed_dovish(self) -> None:
        differ = StatementDiffer()
        prev = ["inflation has moderated.", "the committee is patient."]
        curr = ["inflation has moderated."]
        diff = differ.diff(curr, prev)
        assert diff.net_shift > 0
