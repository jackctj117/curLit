"""Statement diff analyzer — redline comparison with net hawkish shift scoring."""

import difflib
from dataclasses import dataclass
from typing import Any


@dataclass
class StatementDiff:
    added_sentences: list[str]
    removed_sentences: list[str]
    modified_pairs: list[tuple[str, str]]
    added_hawkish_score: float
    removed_hawkish_score: float
    net_shift: float
    raw_diff_ratio: float


class StatementDiffer:
    def __init__(
        self,
        lexicon_scorer: Any = None,
        similarity_threshold: float = 0.6,
    ) -> None:
        if lexicon_scorer is None:
            from src.nlp.lexicon_scorer import LexiconScorer

            lexicon_scorer = LexiconScorer()
        self.lex = lexicon_scorer
        self.sim_threshold = similarity_threshold

    def diff(self, current: list[str], previous: list[str]) -> StatementDiff:
        matcher = difflib.SequenceMatcher(None, previous, current)
        added: list[str] = []
        removed: list[str] = []
        modified: list[tuple[str, str]] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "insert":
                added.extend(current[j1:j2])
            elif tag == "delete":
                removed.extend(previous[i1:i2])
            elif tag == "replace":
                matched_new = set()
                for old in previous[i1:i2]:
                    best_match, best_ratio = None, 0.0
                    for new in current[j1:j2]:
                        r = difflib.SequenceMatcher(None, old, new).ratio()
                        if r > best_ratio and r > self.sim_threshold and new not in matched_new:
                            best_match, best_ratio = new, r
                    if best_match:
                        modified.append((old, best_match))
                        matched_new.add(best_match)
                    else:
                        removed.append(old)
                for new in current[j1:j2]:
                    if new not in matched_new:
                        added.append(new)

        added_score = self.lex.score_text(" ".join(added)).net_score if added else 0.0
        removed_score = self.lex.score_text(" ".join(removed)).net_score if removed else 0.0
        net_shift = added_score - removed_score

        return StatementDiff(
            added,
            removed,
            modified,
            added_score,
            removed_score,
            net_shift,
            matcher.ratio(),
        )
