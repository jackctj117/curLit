"""Unit tests for the Obsidian knowledge-graph exporter (CL-uuy0, CL-w0ox).

Pure, offline, tmp_path only — no network, no live DB. Exercises the mini-fixture
graph (two overlapping themes incl. a watch-only theme) and the fail-loud paths.
The opt-in `--discovered` layer (CL-w0ox) is tested with an INJECTED in-memory
sqlite engine, so those tests need no Postgres either.
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
                {
                    "instrument": "BCO_USD",
                    "kind": "oanda",
                    "direction": "long",
                    "rationale": "Brent | benchmark",
                },
                {
                    "instrument": "WTICO_USD",
                    "kind": "oanda",
                    "direction": "long",
                    "rationale": "WTI follows",
                },
                {
                    "instrument": "XAU_USD",
                    "kind": "oanda",
                    "direction": "long",
                    "rationale": "safe haven",
                },
                {
                    "instrument": "USD_CAD",
                    "kind": "fx",
                    "direction": "short",
                    "rationale": "petro fx",
                },
                {
                    "instrument": "FRO",
                    "kind": "equity_watch",
                    "direction": "watch",
                    "rationale": "tanker (shared 1)",
                },
                {
                    "instrument": "STNG",
                    "kind": "equity_watch",
                    "direction": "watch",
                    "rationale": "product tanker (shared 2)",
                },
                {
                    "instrument": "XOM",
                    "kind": "equity_watch",
                    "direction": "watch",
                    "rationale": "oil major (shared 3)",
                },
            ],
        },
        "beta": {
            "name": "Beta watch-only theme",
            "description": "An advisory-only equity theme.",
            "watch_terms": ["equities"],
            "instruments": [
                {
                    "instrument": "FRO",
                    "kind": "equity_watch",
                    "direction": "watch",
                    "rationale": "shared 1",
                },
                {
                    "instrument": "STNG",
                    "kind": "equity_watch",
                    "direction": "watch",
                    "rationale": "shared 2",
                },
                {
                    "instrument": "XOM",
                    "kind": "equity_watch",
                    "direction": "watch",
                    "rationale": "shared 3",
                },
                {
                    "instrument": "OXY",
                    "kind": "equity_watch",
                    "direction": "watch",
                    "rationale": "oil major",
                },
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
        # Isolate from the repo's committed templates — the templates-copy path
        # has its own dedicated test with a tmp templates dir.
        templates_dir=None,
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
                "name": "One",
                "description": "d",
                "watch_terms": ["x"],
                "instruments": [
                    {
                        "instrument": "BCO_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    },
                    {
                        "instrument": "WTICO_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    },
                ],
            },
            "two": {
                "name": "Two",
                "description": "d",
                "watch_terms": ["y"],
                "instruments": [
                    {
                        "instrument": "BCO_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    },
                    {
                        "instrument": "XAU_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    },
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
                "name": "Small",
                "description": "d",
                "watch_terms": ["x"],
                "instruments": [
                    {
                        "instrument": "BCO_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    },
                    {"instrument": "USD_CAD", "kind": "fx", "direction": "short", "rationale": "r"},
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
    (seed / "Broken.md").write_text("# Broken\nLinks to [[does_not_exist]].\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does_not_exist"):
        _export(tmp_path, mini_playbook, seed=seed)


# ------------------------------------------------------------------------- #
# Fail-loud paths.
# ------------------------------------------------------------------------- #


def test_unknown_kind_fails_loud(tmp_path):
    data = {
        "themes": {
            "bad": {
                "name": "Bad",
                "description": "d",
                "watch_terms": ["x"],
                "instruments": [
                    {
                        "instrument": "BCO_USD",
                        "kind": "crypto",
                        "direction": "long",
                        "rationale": "r",
                    },
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
                "name": "Bad",
                "description": "d",
                "watch_terms": ["x"],
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
                "description": "d",
                "watch_terms": ["x"],
                "instruments": [
                    {
                        "instrument": "BCO_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    },
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
                "name": "A",
                "description": "d",
                "watch_terms": ["x"],
                "instruments": [
                    {
                        "instrument": "BCO_USD",
                        "kind": "oanda",
                        "direction": "long",
                        "rationale": "r",
                    },
                ],
            },
            "b": {
                "name": "B",
                "description": "d",
                "watch_terms": ["y"],
                "instruments": [
                    {
                        "instrument": "BCO_USD",
                        "kind": "equity_watch",
                        "direction": "watch",
                        "rationale": "r",
                    },
                ],
            },
        }
    }
    pb = _write_yaml(tmp_path / "pb.yaml", data)
    with pytest.raises(ValueError, match="conflicting kinds"):
        _export(tmp_path, pb)


def test_no_duplicate_output_filenames(tmp_path, mini_playbook):
    # A clean run must not collide; assert the guard recorded exactly the
    # expected file count: 2 themes + 8 instruments + Coverage + Dashboard +
    # README + graph.json (no seed => 0 concepts; no templates dir => 0
    # templates in this fixture).
    out, result = _export(tmp_path, mini_playbook)
    assert result.files_written == 2 + 8 + 1 + 1 + 1 + 1
    # and every written .md path is unique on disk
    written = list(out.rglob("*.md"))
    assert len(written) == len({p.resolve() for p in written})


# ------------------------------------------------------------------------- #
# Enriched frontmatter (Dataview-queryable, 100% derivable).
# ------------------------------------------------------------------------- #


def _frontmatter_block(text: str) -> str:
    """Return the YAML frontmatter (the first `---`-delimited block) of a note."""
    assert text.startswith("---\n")
    end = text.index("\n---", 4)
    return text[4:end]


def test_theme_frontmatter_has_overlaps_and_coverage_flags(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    beta_fm = _frontmatter_block((out / "Themes" / "beta.md").read_text())
    # beta is all equity_watch => WATCH-ONLY + SINGLE-KIND + THIN (4 < 8), and
    # overlaps alpha (shares FRO/STNG/XOM = 3).
    assert "coverage_flags: [WATCH-ONLY, THIN, SINGLE-KIND]" in beta_fm
    assert "overlaps: [alpha]" in beta_fm
    assert "watch_term_count: 1" in beta_fm  # beta has one watch term
    alpha_fm = _frontmatter_block((out / "Themes" / "alpha.md").read_text())
    # alpha has a tradable leg + 7 instruments + mixed kinds => no WATCH-ONLY /
    # SINGLE-KIND, but THIN (7 < 8). Overlaps beta.
    assert "coverage_flags: [THIN]" in alpha_fm
    assert "WATCH-ONLY" not in alpha_fm
    assert "overlaps: [beta]" in alpha_fm
    assert "watch_term_count: 2" in alpha_fm


def test_theme_frontmatter_flags_match_coverage(tmp_path, mini_playbook):
    """The flag logic in the frontmatter must equal what Coverage.md renders."""
    out, _ = _export(tmp_path, mini_playbook)
    cov = (out / "Coverage.md").read_text()
    for key in ("alpha", "beta"):
        fm = _frontmatter_block((out / "Themes" / f"{key}.md").read_text())
        flags_line = next(ln for ln in fm.splitlines() if ln.startswith("coverage_flags:"))
        # parse "coverage_flags: [A, B]" -> {"A", "B"}
        inside = flags_line.split("[", 1)[1].rsplit("]", 1)[0]
        fm_flags = {f.strip() for f in inside.split(",") if f.strip()}
        cov_line = next(ln for ln in cov.splitlines() if f"[[{key}]]" in ln)
        cov_flags = {f for f in ("WATCH-ONLY", "THIN", "SINGLE-KIND") if f"`{f}`" in cov_line}
        assert fm_flags == cov_flags, (key, fm_flags, cov_flags)


def test_theme_frontmatter_keeps_existing_keys(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    fm = _frontmatter_block((out / "Themes" / "alpha.md").read_text())
    for key in ("type: theme", "source:", "instrument_count:", "tradable_count:"):
        assert key in fm


def test_instrument_frontmatter_has_themes_and_count(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    # FRO is watched by both alpha and beta => theme_count 2, themes sorted.
    fro_fm = _frontmatter_block((out / "Instruments" / "FRO.md").read_text())
    assert "themes: [alpha, beta]" in fro_fm
    assert "theme_count: 2" in fro_fm
    # OXY only in beta => theme_count 1.
    oxy_fm = _frontmatter_block((out / "Instruments" / "OXY.md").read_text())
    assert "themes: [beta]" in oxy_fm
    assert "theme_count: 1" in oxy_fm
    # existing keys preserved
    assert "type: instrument" in fro_fm
    assert "kind: equity_watch" in fro_fm
    assert "tradable: false" in fro_fm


# ------------------------------------------------------------------------- #
# Dashboard (Dataview MOC).
# ------------------------------------------------------------------------- #


def test_dashboard_written_with_dataview_blocks(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    dash = out / "Dashboard.md"
    assert dash.exists()
    text = dash.read_text()
    assert "```dataview" in text
    # the four required query surfaces
    assert 'FROM "Themes"' in text
    assert "WHERE coverage_flags" in text
    assert "SORT theme_count DESC" in text
    assert "WHERE tradable = false" in text
    # degrade-gracefully note + cross-links to the static snapshot
    assert "[[Coverage]]" in text
    assert "[[README]]" in text
    assert "Dataview" in text


# ------------------------------------------------------------------------- #
# .obsidian/graph.json — colour-grouped graph.
# ------------------------------------------------------------------------- #


def test_graph_json_written_with_three_color_groups(tmp_path, mini_playbook):
    out, _ = _export(tmp_path, mini_playbook)
    graph_path = out / ".obsidian" / "graph.json"
    assert graph_path.exists()
    graph = json.loads(graph_path.read_text())  # valid JSON
    groups = graph["colorGroups"]
    assert isinstance(groups, list) and len(groups) == 3
    queries = {g["query"] for g in groups}
    assert queries == {"tag:#theme", "tag:#instrument", "tag:#concept"}
    rgbs = set()
    for g in groups:
        assert g["color"]["a"] == 1
        rgb = g["color"]["rgb"]
        assert isinstance(rgb, int)
        rgbs.add(rgb)
    assert len(rgbs) == 3  # three visually distinct colours


# ------------------------------------------------------------------------- #
# Templates copied into the vault.
# ------------------------------------------------------------------------- #


def test_templates_copied_into_vault(tmp_path, mini_playbook):
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "Theme.md").write_text(
        "---\ntype: theme\ntags: [theme]\n---\n# <% tp.file.title %>\n",
        encoding="utf-8",
    )
    (templates / "Event.md").write_text(
        "---\ntype: event\n---\n# <% tp.file.title %>\n", encoding="utf-8"
    )
    out = tmp_path / "vault"
    ekg.export_vault(
        out_dir=out,
        seed_dir=tmp_path / "no_seed",
        playbooks_path=mini_playbook,
        cross_asset_path=None,
        retail_path=None,
        manifest_path=tmp_path / ".manifest.json",
        templates_dir=templates,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    assert (out / "Templates" / "Theme.md").exists()
    assert (out / "Templates" / "Event.md").exists()
    # copied verbatim (Templater syntax preserved, inert here)
    assert "<% tp.file.title %>" in (out / "Templates" / "Theme.md").read_text()


def test_committed_templates_match_vault_conventions():
    """The repo's parked templates carry `type` frontmatter matching the vault."""
    tpl_dir = ekg.DEFAULT_TEMPLATES
    names = {p.name for p in tpl_dir.glob("*.md")}
    assert {"Theme.md", "Company.md", "Ticker.md", "Event.md", "Dashboard.md"} <= names
    for path in tpl_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")  # has YAML frontmatter
        assert "type:" in text  # aligns to the vault's `type` convention


# ------------------------------------------------------------------------- #
# Discovered layer (CL-w0ox) — the OPT-IN live-DB layer. A fake sqlite engine
# is injected so these run with no Postgres. The pure path above must stay
# byte-identical; these tests only touch the --discovered branch.
#
# Seeded so the mini playbook (`alpha` watches XOM etc.) makes XOM an
# already-a-playbook instrument that MUST be skipped, VG a net-new discovery
# (with a sec_name), TANK a net-new discovery whose name falls back to a trimmed
# security_name, and USD_JPY an underscore-FX symbol that MUST be skipped.
# ------------------------------------------------------------------------- #


def _make_fake_engine():
    """An in-memory sqlite engine mirroring the queried columns of the three
    real tables, seeded with a handful of niche + non-niche trade ideas."""
    from sqlalchemy import create_engine, text

    engine = create_engine("sqlite://")  # shared in-memory for this connection
    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE geo_events (id INTEGER PRIMARY KEY, theme TEXT, headline TEXT)")
        )
        conn.execute(
            text(
                "CREATE TABLE trade_ideas (id INTEGER PRIMARY KEY, "
                "geo_event_id INTEGER, ticker TEXT, confidence REAL, "
                "rationale TEXT, notes TEXT, created_at TEXT)"
            )
        )
        conn.execute(text("CREATE TABLE symbols (symbol TEXT, sec_name TEXT, security_name TEXT)"))
        conn.execute(
            text(
                "INSERT INTO geo_events (id, theme, headline) VALUES "
                "(1, 'alpha', 'Hormuz transit risk spikes'), "
                "(2, 'beta', 'Equity supply squeeze'), "
                "(3, 'alpha', 'Second alpha event')"
            )
        )
        # VG: two niche ideas across two events => idea_count 2, themes {alpha,
        # beta}; the higher-confidence row (0.5, event 2 'beta') is the rep.
        # TANK: one niche idea, name falls back to trimmed security_name.
        # XOM: a niche idea but XOM is already a playbook instrument => skipped.
        # USD_JPY: an underscore-FX niche idea => skipped defensively.
        # ZZZZ: a NON-niche idea (notes lack 'niche') => not selected at all.
        conn.execute(
            text(
                "INSERT INTO trade_ideas "
                "(id, geo_event_id, ticker, confidence, rationale, notes, "
                "created_at) VALUES "
                "(1, 1, 'VG', 0.28, 'low-conf VG chain', "
                "'[niche 3hop] first', '2026-01-01T00:00:00+00:00'), "
                "(2, 2, 'VG', 0.5, 'REP VG multi-hop rationale', "
                "'[niche 2hop] second', '2026-01-02T00:00:00+00:00'), "
                "(3, 3, 'TANK', 0.4, 'tanker play', "
                "'[niche] tanker', '2026-01-03T00:00:00+00:00'), "
                "(4, 1, 'XOM', 0.6, 'oil major', "
                "'[niche] major', '2026-01-01T00:00:00+00:00'), "
                "(5, 1, 'USD_JPY', 0.6, 'fx pair', "
                "'[niche] fx', '2026-01-01T00:00:00+00:00'), "
                "(6, 2, 'ZZZZ', 0.9, 'not niche', "
                "'ordinary event idea', '2026-01-01T00:00:00+00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO symbols (symbol, sec_name, security_name) VALUES "
                "('VG', 'Venture Global, Inc.', "
                "'Venture Global, Inc. Class A common stock'), "
                "('TANK', NULL, 'Tanker Co. Ltd. - Class A Common Stock'), "
                "('XOM', 'Exxon Mobil Corporation', 'Exxon Mobil Corp')"
            )
        )
    return engine


def _export_discovered(tmp_path, playbook, engine):
    out = tmp_path / "vault"
    result = ekg.export_vault(
        out_dir=out,
        seed_dir=tmp_path / "no_seed",
        playbooks_path=playbook,
        cross_asset_path=None,
        retail_path=None,
        manifest_path=tmp_path / ".manifest.json",
        templates_dir=None,
        generated_at="2026-03-15T00:00:00+00:00",
        discovered=True,
        discovered_engine=engine,
    )
    return out, result


def test_clean_company_name_prefers_sec_name_and_trims_boilerplate():
    # sec_name wins verbatim
    assert ekg._clean_company_name("Venture Global, Inc.", "anything") == "Venture Global, Inc."
    # falls back to security_name with the class boilerplate trimmed, keeping
    # the corporate designator
    assert (
        ekg._clean_company_name(None, "Tanker Co. Ltd. - Class A Common Stock") == "Tanker Co. Ltd."
    )
    assert ekg._clean_company_name(None, "Foo Corp Common Stock") == "Foo Corp"
    assert ekg._clean_company_name(None, "Bar Ordinary Shares") == "Bar"
    assert ekg._clean_company_name(None, None) == ""


def test_discovered_note_created_for_net_new_ticker(tmp_path, mini_playbook):
    out, result = _export_discovered(tmp_path, mini_playbook, _make_fake_engine())
    vg = out / "Discovered" / "VG.md"
    assert vg.exists()
    text = vg.read_text()
    fm = _frontmatter_block(text)
    assert "type: discovered" in fm
    assert "ticker: VG" in fm
    assert "company_name: Venture Global, Inc." in fm
    assert "tags: [discovered]" in fm
    assert "idea_count: 2" in fm  # two niche VG ideas
    assert "max_confidence: 0.5" in fm
    assert result.discovered_included is True
    assert result.discovered_count == 2  # VG + TANK (XOM/USD_JPY/ZZZZ excluded)


def test_discovered_note_links_to_each_theme(tmp_path, mini_playbook):
    out, _ = _export_discovered(tmp_path, mini_playbook, _make_fake_engine())
    text = (out / "Discovered" / "VG.md").read_text()
    # VG fired under both alpha and beta events => a wikilink to each theme so
    # Obsidian draws the theme<->discovered edge.
    assert "[[alpha]]" in text
    assert "[[beta]]" in text
    fm = _frontmatter_block(text)
    assert "themes: [alpha, beta]" in fm
    # the representative (highest-confidence) rationale + its headline
    assert "REP VG multi-hop rationale" in text
    assert "Equity supply squeeze" in text


def test_discovered_banner_and_snapshot_date(tmp_path, mini_playbook):
    out, _ = _export_discovered(tmp_path, mini_playbook, _make_fake_engine())
    text = (out / "Discovered" / "VG.md").read_text()
    assert "Discovered by the niche agent from LIVE events" in text
    assert "snapshot as of 2026-03-15" in text  # generated_at date
    assert "make knowledge-vault-live" in text
    assert "not a position" in text.lower()


def test_discovered_company_name_falls_back_to_trimmed_security_name(tmp_path, mini_playbook):
    out, _ = _export_discovered(tmp_path, mini_playbook, _make_fake_engine())
    tank = out / "Discovered" / "TANK.md"
    assert tank.exists()
    # sec_name is NULL => trimmed security_name, corporate designator kept
    assert "company_name: Tanker Co. Ltd." in _frontmatter_block(tank.read_text())


def test_already_playbook_instrument_not_duplicated(tmp_path, mini_playbook):
    out, _ = _export_discovered(tmp_path, mini_playbook, _make_fake_engine())
    # XOM is a playbook instrument (alpha/beta watch it) => teal node already,
    # must NOT be re-emitted as a discovered node.
    assert (out / "Instruments" / "XOM.md").exists()
    assert not (out / "Discovered" / "XOM.md").exists()


def test_underscore_fx_and_non_niche_skipped(tmp_path, mini_playbook):
    out, result = _export_discovered(tmp_path, mini_playbook, _make_fake_engine())
    # USD_JPY is an underscore-FX symbol => skipped; ZZZZ's notes lack 'niche'.
    assert not (out / "Discovered" / "USD_JPY.md").exists()
    assert not (out / "Discovered" / "ZZZZ.md").exists()
    discovered = {p.stem for p in (out / "Discovered").glob("*.md")}
    assert discovered == {"VG", "TANK"}


def test_discovered_graph_has_four_color_groups(tmp_path, mini_playbook):
    out, _ = _export_discovered(tmp_path, mini_playbook, _make_fake_engine())
    graph = json.loads((out / ".obsidian" / "graph.json").read_text())
    groups = graph["colorGroups"]
    assert len(groups) == 4
    queries = {g["query"] for g in groups}
    assert queries == {
        "tag:#theme",
        "tag:#instrument",
        "tag:#concept",
        "tag:#discovered",
    }
    green = next(g for g in groups if g["query"] == "tag:#discovered")
    assert green["color"]["rgb"] == 0x27AE60  # distinct green


def test_discovered_dashboard_section_and_manifest(tmp_path, mini_playbook):
    out, result = _export_discovered(tmp_path, mini_playbook, _make_fake_engine())
    dash = (out / "Dashboard.md").read_text()
    assert "Discovered tickers" in dash
    assert 'FROM "Discovered"' in dash
    assert "TABLE company_name, themes, idea_count, max_confidence" in dash
    assert "SORT idea_count DESC" in dash
    manifest = json.loads((tmp_path / ".manifest.json").read_text())
    assert manifest["discovered"]["included"] is True
    assert manifest["discovered"]["count"] == 2
    assert manifest["discovered"]["snapshot"] == "2026-03-15"


def test_db_unreachable_degrades_to_pure_vault_with_warning(tmp_path, mini_playbook, caplog):
    import logging

    class _BoomEngine:
        def connect(self):  # noqa: ANN001, ANN201
            raise RuntimeError("connection refused")

    with caplog.at_level(logging.WARNING):
        out, result = _export_discovered(tmp_path, mini_playbook, _BoomEngine())
    # DB blip must not nuke the vault: the pure layer is still there ...
    assert (out / "Themes" / "alpha.md").exists()
    assert (out / "Instruments" / "XOM.md").exists()
    # ... but the Discovered layer is absent and the run degraded to 3 groups.
    assert not (out / "Discovered").exists()
    graph = json.loads((out / ".obsidian" / "graph.json").read_text())
    assert len(graph["colorGroups"]) == 3
    assert result.discovered_included is False
    assert result.discovered_count == 0
    # warned loudly + recorded in the manifest
    assert any("unreachable" in r.message.lower() for r in caplog.records)
    manifest = json.loads((tmp_path / ".manifest.json").read_text())
    assert manifest["discovered"]["included"] is False


def test_pure_path_omits_discovered_layer(tmp_path, mini_playbook):
    """Without --discovered the output has no Discovered/, 3 color groups, and no
    dashboard section — the pure contract is preserved."""
    out, result = _export(tmp_path, mini_playbook)
    assert not (out / "Discovered").exists()
    graph = json.loads((out / ".obsidian" / "graph.json").read_text())
    assert len(graph["colorGroups"]) == 3
    assert "Discovered tickers" not in (out / "Dashboard.md").read_text()
    assert result.discovered_included is False
    manifest = json.loads((tmp_path / ".manifest.json").read_text())
    assert manifest["discovered"] == {"included": False, "count": 0, "snapshot": ""}
