"""Lexicon-based hawkish/dovish sentiment scoring."""

import re
from dataclasses import dataclass


HAWKISH_TERMS: set[str] = {
    "additional firming", "further tightening", "more restrictive",
    "higher for longer", "committed to returning", "combat inflation",
    "firmly anchored", "determined", "vigilant", "forceful",
    "inflation remains elevated", "persistent", "stubborn", "sticky",
    "upside risks", "tight labor", "wage pressures", "robust",
    "strong demand", "overheating", "raise", "increase", "hike",
    "tightening", "restrictive stance", "above neutral",
    "sufficiently restrictive",
}

DOVISH_TERMS: set[str] = {
    "rate cut", "easing", "accommodate", "support growth",
    "downside risks", "economic slack", "disinflation",
    "inflation has moderated", "labor market has cooled",
    "patient", "gradual", "careful", "monitor", "data-dependent",
    "balanced", "cumulative", "lags of monetary policy",
    "transmission", "softened", "eased", "lower", "reduce",
    "cut", "normalize", "toward neutral", "less restrictive",
}

UNCERTAINTY_TERMS: set[str] = {
    "uncertain", "uncertainty", "risks", "elevated uncertainty",
    "difficult to assess", "highly uncertain", "could", "might",
    "perhaps", "possibly",
}

INTENSIFIERS: set[str] = {
    "strongly", "firmly", "significantly", "substantially",
    "materially", "notably", "considerably",
}

HEDGES: set[str] = {
    "somewhat", "modestly", "slightly", "marginally",
    "to some degree", "appears", "seems",
}


@dataclass
class LexiconScores:
    hawkish_count: int
    dovish_count: int
    uncertainty_count: int
    intensifier_count: int
    hedge_count: int
    net_score: float
    intensity: float


class LexiconScorer:
    def __init__(self) -> None:
        self._hawkish_re = self._compile(HAWKISH_TERMS)
        self._dovish_re = self._compile(DOVISH_TERMS)
        self._uncertainty_re = self._compile(UNCERTAINTY_TERMS)
        self._intensifier_re = self._compile(INTENSIFIERS)
        self._hedge_re = self._compile(HEDGES)

    @staticmethod
    def _compile(terms: set[str]) -> re.Pattern:
        return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)

    def score_text(self, text: str) -> LexiconScores:
        wc = max(len(text.split()), 1)
        h = len(self._hawkish_re.findall(text))
        d = len(self._dovish_re.findall(text))
        u = len(self._uncertainty_re.findall(text))
        i = len(self._intensifier_re.findall(text))
        hg = len(self._hedge_re.findall(text))
        return LexiconScores(h, d, u, i, hg, (h - d) / (h + d + 1), (i - hg) / wc)

    def score_sentences(self, sentences: list[str]) -> list[LexiconScores]:
        return [self.score_text(s) for s in sentences]
