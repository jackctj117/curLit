"""Unit tests for instrument extraction (CL-frn7).

Covers both notification sources: candidate strategy code (GATE 2,
wraps the backtest runner's canonical symbol read) and hypothesis
briefs (GATE 1, parses the '## Data requirements' section). Both must
fail SAFE — empty results plus a warning, never an exception — because
they run inside the notification path of a live research loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.research.instruments import (
    extract_brief_instruments,
    extract_candidate_instruments,
)

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


CANDIDATE_WITH_EXECUTION_SYMBOL = '''\
"""Fake candidate: cross-symbol signals, single execution pair."""


class Strategy:
    symbols = ["DXY", "EURUSD"]
    execution_symbol = "EURUSD"

    def fit(self, data):
        return self

    def generate_signals(self, data):
        return None
'''

CANDIDATE_WITH_EXECUTION_SYMBOLS_PLURAL = """\
class Strategy:
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    execution_symbols = ["GBPUSD", "USDJPY"]

    def fit(self, data):
        return self

    def generate_signals(self, data):
        return None
"""

CANDIDATE_ANNOTATED_STYLE = '''\
class Strategy:
    """Mirrors the real Implementer output: annotated class attrs."""

    symbols: list[str] = ["EURUSD"]

    def fit(self, data):
        return self

    def generate_signals(self, data):
        return None
'''

BRIEF_WITH_DATA_REQUIREMENTS = """\
# Hypothesis: SCI regime filter beats baseline momentum

## Change to baseline
None — new strategy class.

## Data requirements
- `prices.{symbol}` — daily close for `EURUSD` and GBPUSD
- Macro inputs: FRED `DGS10` and `VIXCLS` (10Y yield, VIX)
- Prediction market: POLY:fed-rate-cut-2026
- If any are absent, flag as `NOT YET INGESTED` before implementation.

## References
- USDJPY appears here but is OUTSIDE the section and must be ignored.
"""

BRIEF_WITHOUT_SECTION = """\
# Hypothesis: no data requirements heading in this one

Some prose mentioning EURUSD as the traded pair.
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


# --------------------------------------------------------------------- #
# Candidate strategy code
# --------------------------------------------------------------------- #


class TestExtractCandidateInstruments:
    def test_execution_symbol_first_then_signal_symbols(
        self,
        tmp_path: Path,
    ) -> None:
        code = _write(tmp_path / "cand.py", CANDIDATE_WITH_EXECUTION_SYMBOL)
        assert extract_candidate_instruments(code) == ["EURUSD", "DXY"]

    def test_execution_symbols_plural_supported(self, tmp_path: Path) -> None:
        code = _write(
            tmp_path / "cand.py",
            CANDIDATE_WITH_EXECUTION_SYMBOLS_PLURAL,
        )
        assert extract_candidate_instruments(code) == [
            "GBPUSD",
            "USDJPY",
            "EURUSD",
        ]

    def test_annotated_class_attr_style(self, tmp_path: Path) -> None:
        # The real Implementer writes ``symbols: list[str] = [...]`` —
        # exactly what backtest_runner._read_symbols consumes.
        code = _write(tmp_path / "cand.py", CANDIDATE_ANNOTATED_STYLE)
        assert extract_candidate_instruments(code) == ["EURUSD"]

    def test_no_symbols_attr_uses_backtest_default(
        self,
        tmp_path: Path,
    ) -> None:
        # _read_symbols fail-opens to EURUSD; the notification must
        # show what the backtest would actually have traded.
        code = _write(
            tmp_path / "cand.py",
            "class Strategy:\n"
            "    def fit(self, data):\n        return self\n"
            "    def generate_signals(self, data):\n        return None\n",
        )
        assert extract_candidate_instruments(code) == ["EURUSD"]

    def test_missing_file_returns_empty_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            out = extract_candidate_instruments(tmp_path / "nope.py")
        assert out == []
        assert any("could not extract instruments" in r.message for r in caplog.records)

    def test_broken_code_returns_empty(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        code = _write(tmp_path / "cand.py", "def broken(:\n")
        with caplog.at_level(logging.WARNING):
            assert extract_candidate_instruments(code) == []

    def test_repeated_calls_do_not_leak_modules(self, tmp_path: Path) -> None:
        import sys

        code = _write(tmp_path / "cand.py", CANDIDATE_WITH_EXECUTION_SYMBOL)
        before = set(sys.modules)
        extract_candidate_instruments(code)
        extract_candidate_instruments(code)
        leaked = {m for m in set(sys.modules) - before if m.startswith("_research_strategy_")}
        assert leaked == set()


# --------------------------------------------------------------------- #
# Hypothesis briefs
# --------------------------------------------------------------------- #


class TestExtractBriefInstruments:
    def test_tradable_and_inputs_split(self, tmp_path: Path) -> None:
        brief = _write(tmp_path / "brief.md", BRIEF_WITH_DATA_REQUIREMENTS)
        tradable, inputs = extract_brief_instruments(brief)
        assert tradable == ["POLY:fed-rate-cut-2026", "EURUSD", "GBPUSD"]
        assert inputs == ["DGS10", "VIXCLS"]

    def test_symbols_outside_section_ignored(self, tmp_path: Path) -> None:
        brief = _write(tmp_path / "brief.md", BRIEF_WITH_DATA_REQUIREMENTS)
        tradable, _inputs = extract_brief_instruments(brief)
        assert "USDJPY" not in tradable  # only in '## References'

    def test_prose_uppercase_not_misread_as_series(
        self,
        tmp_path: Path,
    ) -> None:
        _tradable, inputs = extract_brief_instruments(
            _write(tmp_path / "brief.md", BRIEF_WITH_DATA_REQUIREMENTS),
        )
        # `NOT YET INGESTED` is backticked prose, not a series id.
        assert "NOT" not in inputs
        assert "INGESTED" not in inputs

    def test_missing_section_falls_back_to_whole_doc(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        brief = _write(tmp_path / "brief.md", BRIEF_WITHOUT_SECTION)
        with caplog.at_level(logging.WARNING):
            tradable, inputs = extract_brief_instruments(brief)
        assert tradable == ["EURUSD"]
        assert inputs == []
        assert any("no '## Data requirements'" in r.message for r in caplog.records)

    def test_missing_file_returns_empty_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            out = extract_brief_instruments(tmp_path / "nope.md")
        assert out == ([], [])
        assert any("could not read hypothesis brief" in r.message for r in caplog.records)

    def test_six_letter_word_is_not_an_fx_pair(self, tmp_path: Path) -> None:
        brief = _write(
            tmp_path / "brief.md",
            "# Hypothesis: x\n\n## Data requirements\n"
            "- SIGNAL and OUTPUT are uppercase words, not pairs\n"
            "- EURUSD is the traded pair\n",
        )
        tradable, _ = extract_brief_instruments(brief)
        assert tradable == ["EURUSD"]
