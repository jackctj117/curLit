"""Unit tests for the Obsidian knowledge-graph exporter (CL-uuy0).

Pure, offline, tmp_path only — no network, no DB. Exercises the mini-fixture
graph (two overlapping themes incl. a watch-only theme) and the fail-loud paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.export_knowledge_graph as ekg
import yaml

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------------- #
# Fixtures — a mini 2-theme playbook. `alpha` and `beta` share FRO, STNG, XOM
# (>= 3 equity_watch names) so overlap fires. `beta` is all equity_watch, so
# it is WATCH-ONLY (tradable_count == 0) and SINGLE-KIND. `alpha` adds its own
# oanda + fx tradable legs. A symbol's kind is consistent across themes (the
# real-data invariant the exporter enforces), so the shared names are
# equity_watch in BOTH themes.
# ------------------------------------------------------------------------- #

_MINI_PLAYBOOK = {
    "themes": {
        "alpha": {
            "name": "Alpha energy theme",
            "description": "A tradable energy theme.",
            "watch_terms": ["oil shock", "hormuz"],
            "instruments": [
                {"instrument": "BCO_USD", "kind": "oanda", "direction": "long",
                 "rationale": "Brent | benchmark"},
                {"instrument": "WTICO_USD", "kind": "oanda", "direction": "long",
                 "rationale": "WTI follows"},
                {"instrument": "XAU_USD", "kind": "oanda", "direction": "long",
                 "rationale": "safe haven"},
                {"instrument": "USD_CAD", "kind": "fx", "direction": "short",
                 "rationale": "petro fx"},
                {"instrument": "FRO", "kind": "equity_watch", "direction": "watch",
                 "rationale": "tanker (shared 1)"},
                {"instrument": "STNG", "kind": "equity_watch", "direction": "watch",
                 "rationale": "product tanker (shared 2)"},
                {"instrument": "XOM", "kind": "equity_watch", "direction": "watch",
                 "rationale": "oil major (shared 3)"},
            ],
        },
        "beta": {
            "name": "Beta watch-only theme",
            "description": "An advisory-only equity theme.",
            "watch_terms": ["equities"],
            "instruments": [
                {"instrument": "FRO", "kind": "equity_watch", "direction": "watch",
                 "rationale": "shared 1"},
                {"instrument": "STNG", "kind": "equity_watch", "direction": "watch",
                 "rationale": "shared 2"},
                {"instrument": "XOM", "kind": "equity_watch", "direction": "watch",
                 "rationale": "shared 3"},
                {"instrument": "OXY", "kind": "equity_watch", "direction": "watch",
                 "rationale": "oil major"},
            ],
        },
    }
}

_MINI_CROSS = {
    "themes": {
        "alpha": {
            "instruments": [
                {"instrument": "BCO_USD", "expected_direction": "up", "weight": 2.0},
                {"instrument": "USD_CAD", "expected_direction": "down", "weight": 0.5},
            ]
        }
    }
}


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def mini_playbook(tmp_path: Path) -> Path:
    return _write_yaml(tmp_path / "playbooks.yaml", _MINI_PLAYBOOK)


@pytest.fixture
def mini_cross(tmp_path: Path) -> Path:
    return _write_yaml(tmp_path / "cross.yaml", _MINI_CROSS)


@pytest.fixture
def seed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "seed"
    d.mkdir()
    (d / "Crude.md").write_text(
        "# Crude\nLinks [[alpha]] and [[BCO_USD]] and [[Freight]].\n",
        encoding="utf-8",
    )
    (d / "Freight.md").write_text(
        "# Freight\nLinks the tanker [[FRO]] and concept [[Crude]].\n",
        encoding="utf-8",
    )
    return d


def _export(
    tmp_path: Path,
    playbook: Path,
    cross: Path | None = None,
    seed: Path | None = None,
) -> tuple[Path, ekg.ExportResult]:
    out = tmp_path / "vault"
    result = ekg.export_vault(
        out_dir=out,
        seed_dir=seed or (tmp_path / "no_seed"),
        playbooks_path=playbook,
        cross_asset_path=cross,
        retail_path=None,
        manifest_path=tmp_path / ".manifest.json",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    return out, result


# ------------------------------------------------------------------------- #
# Structure / files.
# ------------------------------------------------------------------------- #


def test_theme_and_instrument_files_created(tmp_path, mini_playbook):
    out, result = _export(tmp_path, mini_playbook)
    assert (out / "Themes" / "alpha.md").exists()
    assert (out / "Themes" / "beta.md").exists()
    # union of instruments across both themes: BCO_USD, WTICO_USD, XAU_USD,
    # USD_CAD, FRO, STNG, XOM (shared) + OXY = 8 unique
    for sym in ("BCO_USD", "WTICO_USD", "XAU_USD", "USD_CAD", "FRO", "STNG", "XOM", "OXY"):
        assert (out / "Instruments" / f"{sym}.md").exists(), sym
    assert result.theme_count == 2
    assert result.instrument_count == 8


def test_coverage_and_readme_written(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    assert (out / "Coverage.md").exists()
    assert (out / "README.md").exists()
    readme = (out / "README.md").read_text()
    assert "NOT part of the live fleet" in readme or "not part of the live fleet" in readme.lower()
    assert "source of truth" in readme.lower()


def test_banner_present_in_generated_notes(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    theme = (out / "Themes" / "alpha.md").read_text()
    assert "NOT live state" in theme
    assert "make knowledge-vault" in theme


# ------------------------------------------------------------------------- #
# Wikilinks.
# ------------------------------------------------------------------------- #


def test_instrument_cells_are_wikilinks(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    theme = (out / "Themes" / "alpha.md").read_text()
    assert "| [[BCO_USD]] | oanda | long |" in theme
    assert "[[USD_CAD]]" in theme


def test_wikilinks_are_well_formed(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    theme = (out / "Themes" / "alpha.md").read_text()
    links = ekg.extract_wikilinks(theme)
    assert "BCO_USD" in links
    # every link target must resolve to a real generated note
    for target in links:
        candidates = [
            out / "Instruments" / f"{target}.md",
            out / "Themes" / f"{target}.md",
            out / "Concepts" / f"{target}.md",
        ]
        assert any(c.exists() for c in candidates), target


def test_backlink_note_in_instrument(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    inst = (out / "Instruments" / "FRO.md").read_text()
    assert "backlinks" in inst.lower()
    # both themes watch FRO; the direction table must show both
    assert "[[alpha]]" in inst
    assert "[[beta]]" in inst


# ------------------------------------------------------------------------- #
# Overlaps.
# ------------------------------------------------------------------------- #


def test_overlap_section_links_sibling_theme(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    alpha = (out / "Themes" / "alpha.md").read_text()
    beta = (out / "Themes" / "beta.md").read_text()
    assert "## Overlapping themes" in alpha
    assert "[[beta]]" in alpha  # alpha overlaps beta (3 shared)
    assert "[[alpha]]" in beta  # symmetric


def test_no_overlap_below_threshold(tmp_path):
    data = {
        "themes": {
            "one": {
                "name": "One", "description": "d", "watch_terms": ["x"],
                "instruments": [
                    {"instrument": "BCO_USD", "kind": "oanda", "direction": "long",
                     "rationale": "r"},
                    {"instrument": "WTICO_USD", "kind": "oanda", "direction": "long",
                     "rationale": "r"},
                ],
            },
            "two": {
                "name": "Two", "description": "d", "watch_terms": ["y"],
                "instruments": [
                    {"instrument": "BCO_USD", "kind": "oanda", "direction": "long",
                     "rationale": "r"},
                    {"instrument": "XAU_USD", "kind": "oanda", "direction": "long",
                     "rationale": "r"},
                ],
            },
        }
    }
    pb = _write_yaml(tmp_path / "pb.yaml", data)
    out, _ = _export(tmp_path, pb)
    one = (out / "Themes" / "one.md").read_text()
    # only 1 shared instrument < threshold of 3 => no overlap link
    assert "[[two]]" not in one


# ------------------------------------------------------------------------- #
# Coverage counts + flags.
# ------------------------------------------------------------------------- #


def test_coverage_counts_and_watch_only_flag(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    cov = (out / "Coverage.md").read_text()
    # beta is all equity_watch => WATCH-ONLY and SINGLE-KIND, tradable 0
    beta_line = next(ln for ln in cov.splitlines() if "[[beta]]" in ln)
    assert "WATCH-ONLY" in beta_line
    assert "SINGLE-KIND" in beta_line
    assert "0 oanda / 0 fx / 4 equity_watch / 0 pm" in beta_line
    # alpha has 3 oanda + 1 fx = 4 tradable (plus 3 equity_watch shared names)
    alpha_line = next(ln for ln in cov.splitlines() if "[[alpha]]" in ln)
    assert "WATCH-ONLY" not in alpha_line


def test_coverage_sorts_watch_only_first(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    cov = (out / "Coverage.md").read_text()
    lines = cov.splitlines()
    beta_idx = next(i for i, ln in enumerate(lines) if "[[beta]]" in ln)
    alpha_idx = next(i for i, ln in enumerate(lines) if "[[alpha]]" in ln)
    # watch-only beta must sort above alpha
    assert beta_idx < alpha_idx


def test_coverage_has_readonly_sql(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    cov = (out / "Coverage.md").read_text()
    assert "```sql" in cov
    assert "geo_events" in cov
    assert "urgency" in cov
    # the exporter must NOT have executed anything — purity is structural, but
    # assert the SQL is present verbatim as a string for the operator.
    assert ekg.URGENCY_SQL in cov


def test_thin_flag(tmp_path):
    data = {
        "themes": {
            "small": {
                "name": "Small", "description": "d", "watch_terms": ["x"],
                "instruments": [
                    {"instrument": "BCO_USD", "kind": "oanda", "direction": "long",
                     "rationale": "r"},
                    {"instrument": "USD_CAD", "kind": "fx", "direction": "short",
                     "rationale": "r"},
                ],
            },
        }
    }
    pb = _write_yaml(tmp_path / "pb.yaml", data)
    out, _ = _export(tmp_path, pb)
    cov = (out / "Coverage.md").read_text()
    small_line = next(ln for ln in cov.splitlines() if "[[small]]" in ln)
    assert "THIN" in small_line  # 2 < 8


# ------------------------------------------------------------------------- #
# Cross-asset + retail enrichment.
# ------------------------------------------------------------------------- #


def test_cross_asset_corroborators_rendered(tmp_path, mini_playbook, mini_cross):
    out, _ = _export(tmp_path, mini_playbook, cross=mini_cross)
    alpha = (out / "Themes" / "alpha.md").read_text()
    assert "## Cross-asset corroborators" in alpha
    assert "| [[BCO_USD]] | up | 2 |" in alpha
    assert "| [[USD_CAD]] | down | 0.5 |" in alpha
    # beta has no cross-asset entry => placeholder
    beta = (out / "Themes" / "beta.md").read_text()
    assert "No cross-asset corroborators" in beta


# ------------------------------------------------------------------------- #
# Manifest.
# ------------------------------------------------------------------------- #


def test_manifest_written_with_hashes_and_counts(tmp_path, mini_playbook):
    _export(tmp_path, mini_playbook)
    manifest = json.loads((tmp_path / ".manifest.json").read_text())
    assert manifest["bead"] == "CL-uuy0"
    assert manifest["counts"]["themes"] == 2
    assert manifest["counts"]["instruments"] == 8
    # sha256 of the source is a 64-hex digest
    digest = manifest["sources"]["event_playbooks.yaml"]
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


# ------------------------------------------------------------------------- #
# Seed concept validation.
# ------------------------------------------------------------------------- #


def test_seed_notes_copied_and_valid(tmp_path, mini_playbook, seed_dir):
    out, result = _export(tmp_path, mini_playbook, seed=seed_dir)
    assert (out / "Concepts" / "Crude.md").exists()
    assert (out / "Concepts" / "Freight.md").exists()
    assert result.concept_count == 2


def test_seed_note_with_bad_link_fails_loud(tmp_path, mini_playbook):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "Broken.md").write_text(
        "# Broken\nLinks to [[does_not_exist]].\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        _export(tmp_path, mini_playbook, seed=seed)


# ------------------------------------------------------------------------- #
# Fail-loud paths.
# ------------------------------------------------------------------------- #


def test_unknown_kind_fails_loud(tmp_path):
    data = {
        "themes": {
            "bad": {
                "name": "Bad", "description": "d", "watch_terms": ["x"],
                "instruments": [
                    {"instrument": "BCO_USD", "kind": "crypto", "direction": "long",
                     "rationale": "r"},
                ],
            }
        }
    }
    pb = _write_yaml(tmp_path / "pb.yaml", data)
    with pytest.raises(ValueError, match="unknown kind"):
        _export(tmp_path, pb)


def test_theme_missing_instruments_fails_loud(tmp_path):
    data = {
        "themes": {
            "bad": {
                "name": "Bad", "description": "d", "watch_terms": ["x"],
                # no 'instruments'
            }
        }
    }
    pb = _write_yaml(tmp_path / "pb.yaml", data)
    with pytest.raises(ValueError, match="instruments"):
        _export(tmp_path, pb)


def test_theme_missing_name_fails_loud(tmp_path):
    data = {
        "themes": {
            "bad": {
                "description": "d", "watch_terms": ["x"],
                "instruments": [
                    {"instrument": "BCO_USD", "kind": "oanda", "direction": "long",
                     "rationale": "r"},
                ],
            }
        }
    }
    pb = _write_yaml(tmp_path / "pb.yaml", data)
    with pytest.raises(ValueError, match="name"):
        _export(tmp_path, pb)


def test_conflicting_instrument_kind_fails_loud(tmp_path):
    # BCO_USD is oanda in one theme and equity_watch in another => config drift
    data = {
        "themes": {
            "a": {
                "name": "A", "description": "d", "watch_terms": ["x"],
                "instruments": [
                    {"instrument": "BCO_USD", "kind": "oanda", "direction": "long",
                     "rationale": "r"},
                ],
            },
            "b": {
                "name": "B", "description": "d", "watch_terms": ["y"],
                "instruments": [
                    {"instrument": "BCO_USD", "kind": "equity_watch",
                     "direction": "watch", "rationale": "r"},
                ],
            },
        }
    }
    pb = _write_yaml(tmp_path / "pb.yaml", data)
    with pytest.raises(ValueError, match="conflicting kinds"):
        _export(tmp_path, pb)


def test_no_duplicate_output_filenames(tmp_path, mini_playbook):
    # A clean run must not collide; assert the guard recorded exactly the
    # expected file count (2 themes + 8 instruments + coverage + readme).
    out, result = _export(tmp_path, mini_playbook)
    assert result.files_written == 2 + 8 + 1 + 1
    # and every written path is unique on disk
    written = list(out.rglob("*.md"))
    assert len(written) == len({p.resolve() for p in written})
