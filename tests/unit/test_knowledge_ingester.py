"""Unit tests for the knowledge-archive ingester (CL-jg43).

Exercises the deterministic paths (corpus loading, notes-mode chunking,
code-fence stripping, source_id stability) without DB or LLM calls. The
LLM-extract path is integration-tested separately when keys are
available — here we just confirm JSON parsing handles common LLM output
quirks.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the script under test as a module since it lives under scripts/.
_SPEC = importlib.util.spec_from_file_location(
    "seed_knowledge_archive",
    Path(__file__).parents[2] / "scripts" / "seed_knowledge_archive.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["seed_knowledge_archive"] = _MOD
_SPEC.loader.exec_module(_MOD)


CorpusEntry = _MOD.CorpusEntry
_load_corpus = _MOD._load_corpus
_extract_notes_file = _MOD._extract_notes_file
_strip_code_fence = _MOD._strip_code_fence


# =============================================================================
# CorpusEntry
# =============================================================================


@pytest.fixture
def sample_entry() -> CorpusEntry:
    return CorpusEntry(
        title="Test Title",
        author="A. Author",
        year=2024,
        source_type="book",
        importance="P1",
        topic_tags=["t1", "t2"],
        rationale="…",
        extract_strategy="chapter-summary",
        extract_source="llm-prior-knowledge",
        chunks_per_source=10,
    )


class TestCorpusEntry:
    def test_source_id_stable_across_construction(
        self, sample_entry: CorpusEntry,
    ) -> None:
        """source_id must be deterministic so reruns are idempotent —
        recomputing it on an identical entry must yield the same hash."""
        again = CorpusEntry(
            title="Test Title", author="A. Author", year=2024,
            source_type="book", importance="P1",
            topic_tags=["t1", "t2"], rationale="…",
            extract_strategy="chapter-summary",
            extract_source="llm-prior-knowledge",
            chunks_per_source=10,
        )
        assert sample_entry.source_id == again.source_id
        assert len(sample_entry.source_id) == 64  # sha256 hex

    def test_source_id_changes_with_year(
        self, sample_entry: CorpusEntry,
    ) -> None:
        """A later edition with a different year must get a different
        source_id — different work, different chunks."""
        new_edition = CorpusEntry(
            title="Test Title", author="A. Author", year=2025,
            source_type="book", importance="P1",
            topic_tags=["t1", "t2"], rationale="…",
            extract_strategy="chapter-summary",
            extract_source="llm-prior-knowledge",
            chunks_per_source=10,
        )
        assert new_edition.source_id != sample_entry.source_id

    def test_citation_format(self, sample_entry: CorpusEntry) -> None:
        assert sample_entry.citation == "A. Author (2024). Test Title."


# =============================================================================
# Corpus YAML loading
# =============================================================================


class TestLoadCorpus:
    def test_load_real_corpus(self) -> None:
        """The committed knowledge_corpus.yaml must always load — that's
        the contract between the operator's edits and the ingester."""
        path = Path("docs/research/knowledge_corpus.yaml")
        entries = _load_corpus(path)
        assert len(entries) >= 1
        # All entries must round-trip through the dataclass
        for e in entries:
            assert e.title and e.author
            assert e.importance in ("P1", "P2", "P3")
            assert e.chunks_per_source > 0

    def test_missing_top_level_key_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not_sources:\n  - foo\n")
        with pytest.raises(ValueError, match="missing 'sources'"):
            _load_corpus(bad)


# =============================================================================
# Notes-mode chunking
# =============================================================================


class TestNotesExtraction:
    def test_three_sections_produce_three_chunks(
        self, tmp_path: Path, sample_entry: CorpusEntry,
    ) -> None:
        notes = tmp_path / "notes.md"
        notes.write_text(
            "## Section A\n"
            "Body A line one.\n"
            "Body A line two.\n"
            "\n"
            "## Section B\n"
            "Body B.\n"
            "\n"
            "## Section C\n"
            "Body C only.\n",
        )
        chunks = _extract_notes_file(sample_entry, notes)
        assert len(chunks) == 3
        assert chunks[0].page_ref == "Section A"
        assert "Body A line one." in chunks[0].chunk_text
        assert "Body A line two." in chunks[0].chunk_text
        assert chunks[1].page_ref == "Section B"
        assert chunks[2].page_ref == "Section C"

    def test_empty_sections_skipped(
        self, tmp_path: Path, sample_entry: CorpusEntry,
    ) -> None:
        notes = tmp_path / "notes.md"
        notes.write_text(
            "## Empty section\n"
            "\n"
            "## Real section\n"
            "Real content here.\n",
        )
        chunks = _extract_notes_file(sample_entry, notes)
        # Empty section dropped; only real section remains
        assert len(chunks) == 1
        assert chunks[0].page_ref == "Real section"

    def test_missing_file_raises(
        self, tmp_path: Path, sample_entry: CorpusEntry,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="notes file not found"):
            _extract_notes_file(sample_entry, tmp_path / "nope.md")

    def test_chunk_idx_zero_indexed_and_contiguous(
        self, tmp_path: Path, sample_entry: CorpusEntry,
    ) -> None:
        notes = tmp_path / "notes.md"
        notes.write_text(
            "## A\nbody A\n## B\nbody B\n## C\nbody C\n",
        )
        chunks = _extract_notes_file(sample_entry, notes)
        assert [c.chunk_idx for c in chunks] == [0, 1, 2]


# =============================================================================
# Code-fence stripping
# =============================================================================


class TestCodeFenceStripping:
    def test_plain_json_unchanged(self) -> None:
        assert _strip_code_fence('[{"a": 1}]') == '[{"a": 1}]'

    def test_json_fence_stripped(self) -> None:
        assert _strip_code_fence('```json\n[{"a": 1}]\n```') == '[{"a": 1}]'

    def test_unlabeled_fence_stripped(self) -> None:
        assert _strip_code_fence('```\n[{"a": 1}]\n```') == '[{"a": 1}]'

    def test_whitespace_around_fence_handled(self) -> None:
        wrapped = "  \n```json\n[1, 2]\n```\n  "
        assert _strip_code_fence(wrapped) == "[1, 2]"

    def test_fence_only_at_start_and_end(self) -> None:
        # Mid-string ``` should be preserved
        text = "[\"line1\", \"line2 with ``` mid\", \"line3\"]"
        assert _strip_code_fence(text) == text
