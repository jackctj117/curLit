"""Obsidian knowledge-graph exporter (CL-uuy0) — RESEARCH ONLY, OFFLINE.

A pure YAML->Markdown generator that renders the geopolitical event playbooks
into an Obsidian vault so a human can explore the theme <-> instrument <-> region
graph. This is a one-way, dev/operator tool:

  * It is NOT under src/, is NOT a daemon, is NOT imported by the trading engine,
    and its output is NEVER read back into the trading system.
  * It is PURE: no network, no database. It reads the YAML configs and writes
    Markdown files. It runs offline in CI.
  * The YAML configs are the single source of truth. Obsidian never wins — the
    vault is a regeneratable build artifact. Regenerate with `make knowledge-vault`.

It FAILS LOUD on: an unknown instrument kind, a duplicate output filename/id, a
theme missing a required field, or a hand-authored seed note whose [[wikilinks]]
do not resolve to a generated theme/instrument/concept.

Usage:
    .venv/bin/python scripts/export_knowledge_graph.py
    .venv/bin/python scripts/export_knowledge_graph.py --out DIR --seed DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------- #
# Constants — the verified source data contract.
# ------------------------------------------------------------------------- #

VALID_KINDS = frozenset({"oanda", "fx", "equity_watch", "polymarket"})
TRADABLE_KINDS = frozenset({"oanda", "fx"})
REQUIRED_THEME_FIELDS = ("name", "description", "watch_terms", "instruments")
OVERLAP_THRESHOLD = 3  # themes sharing >= this many instruments are "overlapping"
THIN_THRESHOLD = 8     # a theme with fewer than this many instruments is THIN

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAYBOOKS = _REPO_ROOT / "configs" / "event_playbooks.yaml"
DEFAULT_CROSS_ASSET = _REPO_ROOT / "configs" / "cross_asset_checks.yaml"
DEFAULT_RETAIL = _REPO_ROOT / "configs" / "retail_proxies.yaml"
DEFAULT_OUT = _REPO_ROOT / "knowledge" / "obsidian" / "vault"
DEFAULT_SEED = _REPO_ROOT / "knowledge" / "obsidian" / "seed"
DEFAULT_MANIFEST = _REPO_ROOT / "knowledge" / "obsidian" / ".manifest.json"

BANNER = (
    "> [!info] Generated from configs/event_playbooks.yaml — NOT live state. "
    "Regenerate with `make knowledge-vault`; do not hand-edit."
)

# Domain tags — a light curated classification so the graph gets colour and the
# theme frontmatter carries a searchable domain. Purely cosmetic metadata.
DOMAIN_TAGS: dict[str, str] = {
    "energy_chokepoint": "energy",
    "red_sea_shipping": "shipping",
    "oil_supply_shock": "energy",
    "russia_ukraine": "conflict",
    "africa_power_shift": "africa",
    "drc_copper_cobalt": "metals",
    "sahel_gold_uranium": "metals",
    "guinea_iron_bauxite": "metals",
    "south_africa_pgm_gold": "metals",
    "taiwan_semiconductor": "tech",
    "black_sea_grain": "agriculture",
    "war_escalation": "conflict",
    "cb_surprise": "macro",
    "natural_disaster": "macro",
    "sanctions_trade": "macro",
    "pharma_api_supply": "supply-chain",
}

# The read-only SQL an operator can eyeball to see which wired themes have
# actually fired a live urgency-7 event. NEVER executed here (stays pure).
URGENCY_SQL = (
    "SELECT ge.theme, COUNT(*) FILTER (WHERE (assessment->>'urgency')::float >= 7) "
    "AS urgent FROM geo_events ge WHERE assessment ? 'urgency' "
    "GROUP BY ge.theme ORDER BY urgent DESC;"
)

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")


# ------------------------------------------------------------------------- #
# Dataclass configs — no dict-groping in the rendering logic.
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Instrument:
    """One instrument row inside a theme's playbook entry."""

    symbol: str
    kind: str
    direction: str
    rationale: str


@dataclass(frozen=True)
class CrossAsset:
    """A cross-asset corroborator for a theme (from cross_asset_checks.yaml)."""

    symbol: str
    expected_direction: str
    weight: float


@dataclass
class Theme:
    """A geopolitical theme and everything the vault renders about it."""

    key: str
    name: str
    description: str
    watch_terms: list[str]
    instruments: list[Instrument]
    cross_asset: list[CrossAsset] = field(default_factory=list)

    @property
    def instrument_symbols(self) -> set[str]:
        return {i.symbol for i in self.instruments}

    @property
    def tradable_count(self) -> int:
        return sum(1 for i in self.instruments if i.kind in TRADABLE_KINDS)

    @property
    def kind_breakdown(self) -> Counter[str]:
        return Counter(i.kind for i in self.instruments)


@dataclass(frozen=True)
class ExportResult:
    """Summary of a run — used by the manifest and by tests."""

    theme_count: int
    instrument_count: int
    concept_count: int
    node_count: int
    link_count: int
    files_written: int


# ------------------------------------------------------------------------- #
# Loading + validation (fail loud).
# ------------------------------------------------------------------------- #


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return data


def load_themes(
    playbooks_path: Path,
    cross_asset_path: Path | None = None,
) -> dict[str, Theme]:
    """Parse the playbooks (+ optional cross-asset) into validated Theme objects.

    Fails loud on a theme missing a required field, on an unknown instrument
    kind, and on a malformed instrument row.
    """
    raw = _load_yaml(playbooks_path)
    themes_raw = raw.get("themes")
    if not isinstance(themes_raw, dict) or not themes_raw:
        raise ValueError(f"{playbooks_path} has no non-empty 'themes' mapping")

    cross_by_theme = _load_cross_asset(cross_asset_path) if cross_asset_path else {}

    themes: dict[str, Theme] = {}
    for key, body in themes_raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"theme {key!r} is not a mapping")
        missing = [f for f in REQUIRED_THEME_FIELDS if f not in body]
        if missing:
            raise ValueError(
                f"theme {key!r} missing required field(s): {', '.join(missing)}"
            )
        instruments = _parse_instruments(key, body["instruments"])
        watch_terms = body["watch_terms"]
        if not isinstance(watch_terms, list):
            raise ValueError(f"theme {key!r} watch_terms is not a list")
        themes[key] = Theme(
            key=key,
            name=str(body["name"]),
            description=str(body["description"]).strip(),
            watch_terms=[str(w) for w in watch_terms],
            instruments=instruments,
            cross_asset=cross_by_theme.get(key, []),
        )
    return themes


def _parse_instruments(theme_key: str, rows: object) -> list[Instrument]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"theme {theme_key!r} has no 'instruments' list")
    out: list[Instrument] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"theme {theme_key!r} has a non-mapping instrument row")
        for f in ("instrument", "kind", "direction"):
            if f not in row:
                raise ValueError(
                    f"theme {theme_key!r} instrument row missing {f!r}: {row}"
                )
        kind = str(row["kind"])
        if kind not in VALID_KINDS:
            raise ValueError(
                f"theme {theme_key!r} instrument {row['instrument']!r} has unknown "
                f"kind {kind!r} (valid: {', '.join(sorted(VALID_KINDS))})"
            )
        out.append(
            Instrument(
                symbol=str(row["instrument"]),
                kind=kind,
                direction=str(row["direction"]),
                rationale=str(row.get("rationale", "")).strip(),
            )
        )
    return out


def _load_cross_asset(path: Path) -> dict[str, list[CrossAsset]]:
    raw = _load_yaml(path)
    themes_raw = raw.get("themes", {})
    if not isinstance(themes_raw, dict):
        raise ValueError(f"{path} has no 'themes' mapping")
    out: dict[str, list[CrossAsset]] = {}
    for key, body in themes_raw.items():
        rows = body.get("instruments", []) if isinstance(body, dict) else []
        entries: list[CrossAsset] = []
        for row in rows:
            entries.append(
                CrossAsset(
                    symbol=str(row["instrument"]),
                    expected_direction=str(row["expected_direction"]),
                    weight=float(row.get("weight", 1.0)),
                )
            )
        if entries:
            out[key] = entries
    return out


def load_retail_proxies(path: Path | None) -> dict[str, str]:
    """Return {symbol -> one-line proxy summary} for the optional enrichment.

    Skips silently if the file is absent — it is enrichment, not required.
    """
    if path is None or not path.exists():
        return {}
    raw = _load_yaml(path)
    instruments = raw.get("instruments", {})
    if not isinstance(instruments, dict):
        return {}
    out: dict[str, str] = {}
    for sym, body in instruments.items():
        if not isinstance(body, dict):
            continue
        proxies = body.get("proxies") or []
        if not proxies:
            continue
        parts = [f"proxies: {', '.join(str(p) for p in proxies)}"]
        inverse = body.get("inverse") or []
        if inverse:
            parts.append(f"inverse: {', '.join(str(i) for i in inverse)}")
        note = body.get("note")
        if note:
            parts.append(str(note))
        out[str(sym)] = " — ".join(parts)
    return out


# ------------------------------------------------------------------------- #
# Graph analysis.
# ------------------------------------------------------------------------- #


def compute_overlaps(themes: dict[str, Theme]) -> dict[str, list[str]]:
    """For each theme, the other theme keys sharing >= OVERLAP_THRESHOLD instruments."""
    keys = list(themes.keys())
    overlaps: dict[str, list[str]] = {k: [] for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            shared = themes[a].instrument_symbols & themes[b].instrument_symbols
            if len(shared) >= OVERLAP_THRESHOLD:
                overlaps[a].append(b)
                overlaps[b].append(a)
    for k in overlaps:
        overlaps[k].sort()
    return overlaps


def instrument_watchers(themes: dict[str, Theme]) -> dict[str, list[tuple[str, str]]]:
    """Return {symbol -> [(theme_key, direction), ...]} sorted by theme key."""
    out: dict[str, list[tuple[str, str]]] = {}
    for key, theme in themes.items():
        for inst in theme.instruments:
            out.setdefault(inst.symbol, []).append((key, inst.direction))
    for sym in out:
        out[sym].sort()
    return out


def compute_flags(theme: Theme) -> list[str]:
    """Coverage flags for a theme, most-important first."""
    flags: list[str] = []
    if theme.tradable_count == 0:
        flags.append("WATCH-ONLY")
    if len(theme.instruments) < THIN_THRESHOLD:
        flags.append("THIN")
    if len(theme.kind_breakdown) == 1:
        flags.append("SINGLE-KIND")
    return flags


# ------------------------------------------------------------------------- #
# Rendering.
# ------------------------------------------------------------------------- #


def _frontmatter(fields: dict[str, object]) -> str:
    lines = ["---"]
    for k, v in fields.items():
        if isinstance(v, list):
            rendered = "[" + ", ".join(str(x) for x in v) + "]"
            lines.append(f"{k}: {rendered}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def render_theme_note(
    theme: Theme,
    overlaps: list[str],
    themes: dict[str, Theme],
) -> str:
    domain = DOMAIN_TAGS.get(theme.key)
    tags = ["theme"] + ([domain] if domain else [])
    fm = _frontmatter(
        {
            "type": "theme",
            "tags": tags,
            "source": f"configs/event_playbooks.yaml#{theme.key}",
            "instrument_count": len(theme.instruments),
            "tradable_count": theme.tradable_count,
        }
    )
    lines = [fm, "", BANNER, "", f"# {theme.name}", "", theme.description, ""]

    lines.append("## Watch terms")
    lines.append("")
    for term in theme.watch_terms:
        lines.append(f"- {term}")
    lines.append("")

    lines.append("## Instruments")
    lines.append("")
    lines.append("| Instrument | Kind | Direction | Rationale |")
    lines.append("| --- | --- | --- | --- |")
    for inst in theme.instruments:
        rationale = inst.rationale.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| [[{inst.symbol}]] | {inst.kind} | {inst.direction} | {rationale} |"
        )
    lines.append("")

    lines.append("## Cross-asset corroborators")
    lines.append("")
    if theme.cross_asset:
        lines.append("| Instrument | Expected direction | Weight |")
        lines.append("| --- | --- | --- |")
        for ca in theme.cross_asset:
            lines.append(f"| [[{ca.symbol}]] | {ca.expected_direction} | {ca.weight:g} |")
    else:
        lines.append("_No cross-asset corroborators configured for this theme._")
    lines.append("")

    lines.append("## Overlapping themes")
    lines.append("")
    lines.append(
        f"_Themes sharing at least {OVERLAP_THRESHOLD} instruments with this one._"
    )
    lines.append("")
    if overlaps:
        for other in overlaps:
            shared = theme.instrument_symbols & themes[other].instrument_symbols
            lines.append(f"- [[{other}]] ({len(shared)} shared)")
    else:
        lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


def render_instrument_note(
    symbol: str,
    kind: str,
    watchers: list[tuple[str, str]],
    retail: str | None,
) -> str:
    tradable = kind in TRADABLE_KINDS
    tags = ["instrument", kind] + (["tradable"] if tradable else [])
    fm = _frontmatter(
        {
            "type": "instrument",
            "kind": kind,
            "tradable": tradable,
            "tags": tags,
        }
    )
    lines = [fm, "", BANNER, "", f"# {symbol}", ""]
    lines.append(f"- **Kind:** {kind}")
    lines.append(f"- **Tradable (machine-executable FX/CFD leg):** {'yes' if tradable else 'no'}")
    lines.append("")
    lines.append(
        "> Themes that watch this instrument appear automatically as Obsidian "
        "**backlinks** (see the linked-mentions pane) — the multi-hop feature. "
        "The table below adds what each watching theme is *not* visible from a "
        "backlink alone: the direction it wants this instrument to move."
    )
    lines.append("")
    lines.append("## Watched by")
    lines.append("")
    lines.append("| Theme | Direction |")
    lines.append("| --- | --- |")
    for theme_key, direction in watchers:
        lines.append(f"| [[{theme_key}]] | {direction} |")
    lines.append("")
    if retail:
        lines.append("## Retail proxy")
        lines.append("")
        lines.append(f"- {retail}")
        lines.append("")
    return "\n".join(lines)


def _sort_key_coverage(theme: Theme, flags: list[str]) -> tuple[int, int, int, str]:
    # Most under-prepared at the top: WATCH-ONLY first, then THIN, then
    # ascending tradable_count.
    return (
        0 if "WATCH-ONLY" in flags else 1,
        0 if "THIN" in flags else 1,
        theme.tradable_count,
        theme.key,
    )


def render_coverage_note(
    themes: dict[str, Theme],
    overlaps: dict[str, list[str]],
) -> str:
    rows: list[tuple[Theme, list[str]]] = []
    for theme in themes.values():
        flags = compute_flags(theme)
        rows.append((theme, flags))
    rows.sort(key=lambda r: _sort_key_coverage(r[0], r[1]))

    lines = [
        _frontmatter({"type": "dashboard", "tags": ["coverage", "dashboard"]}),
        "",
        BANNER,
        "",
        "# Coverage dashboard",
        "",
        "Per-theme instrument coverage. Rows are sorted so the most "
        "under-prepared themes sit at the top: **WATCH-ONLY** first (no "
        "machine-tradable FX/CFD leg — the desk can only advise), then "
        "**THIN** (< 8 instruments), then by ascending tradable count.",
        "",
        "**Flags:** `WATCH-ONLY` = tradable_count == 0 · "
        "`THIN` = < 8 instruments · `SINGLE-KIND` = all instruments one kind.",
        "",
        "| Theme | Total | Kind breakdown | Tradable | Overlaps | Flags |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for theme, flags in rows:
        kb = theme.kind_breakdown
        breakdown = (
            f"{kb.get('oanda', 0)} oanda / {kb.get('fx', 0)} fx / "
            f"{kb.get('equity_watch', 0)} equity_watch / {kb.get('polymarket', 0)} pm"
        )
        flag_str = " ".join(f"`{f}`" for f in flags) if flags else ""
        lines.append(
            f"| [[{theme.key}]] | {len(theme.instruments)} | {breakdown} | "
            f"{theme.tradable_count} | {len(overlaps[theme.key])} | {flag_str} |"
        )
    lines.append("")
    lines.append("## Cross-reference: which themes have actually fired?")
    lines.append("")
    lines.append(
        "This vault is generated from static config — it shows what curLit is "
        "*wired* to react to, not what has actually happened. To see which wired "
        "themes have fired a live urgency-7 event, run this **read-only** query "
        "against the trading DB yourself (the exporter never touches the DB):"
    )
    lines.append("")
    lines.append("```sql")
    lines.append(URGENCY_SQL)
    lines.append("```")
    lines.append("")
    lines.append(
        "Themes wired here but absent (or near-zero) in that result are "
        "*wired-but-never-fired* — candidates for review."
    )
    lines.append("")
    return "\n".join(lines)


def render_vault_readme() -> str:
    lines = [
        "# curLit knowledge-graph vault (CL-uuy0)",
        "",
        BANNER,
        "",
        "An **offline, research-only** Obsidian vault of the geopolitical "
        "event playbooks: the theme <-> instrument <-> region graph for human "
        "exploration.",
        "",
        "## What this is (and is not)",
        "",
        "- **Source of truth is the YAML** (`configs/event_playbooks.yaml`, "
        "`configs/cross_asset_checks.yaml`). Obsidian **never wins** — this "
        "vault is a regeneratable build artifact.",
        "- It is **NOT part of the live fleet**: not a daemon, not imported by "
        "the engine, never read back into the trading system.",
        "- It is **pure**: generated with no network and no database access.",
        "",
        "## How to open",
        "",
        "1. In Obsidian: **Open folder as vault** and point it at this directory, "
        "or copy this folder into an existing vault.",
        "2. Open `Coverage.md` first — it surfaces under-wired / watch-only "
        "themes.",
        "3. Use the graph view and the **backlinks** pane on any "
        "`Instruments/<SYMBOL>` note to see every theme that watches it "
        "(multi-hop).",
        "",
        "## Regenerate",
        "",
        "```",
        "make knowledge-vault",
        "```",
        "",
        "Do not hand-edit the generated notes under `Themes/` and "
        "`Instruments/`; edit the YAML and regenerate. Hand-authored bridge "
        "notes live in `Concepts/` (authored in `knowledge/obsidian/seed/`).",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------- #
# Seed (concept) validation.
# ------------------------------------------------------------------------- #


def extract_wikilinks(text: str) -> list[str]:
    """Return the link targets of every [[wikilink]] in text (stripping #anchors/|alias)."""
    return [m.group(1).strip() for m in _WIKILINK_RE.finditer(text)]


def validate_seed_notes(
    seed_dir: Path,
    known_targets: set[str],
) -> dict[str, str]:
    """Read every concept note in seed_dir, validate its wikilinks, return {name -> body}.

    A concept note may link to a generated theme/instrument OR to another
    concept note. Fails loud on any unresolved link.
    """
    if not seed_dir.exists():
        return {}
    concept_names = {p.stem for p in sorted(seed_dir.glob("*.md"))}
    resolvable = known_targets | concept_names
    out: dict[str, str] = {}
    for path in sorted(seed_dir.glob("*.md")):
        body = path.read_text(encoding="utf-8")
        for target in extract_wikilinks(body):
            if target not in resolvable:
                raise ValueError(
                    f"seed concept {path.name!r} links to [[{target}]] which "
                    "resolves to no generated theme/instrument/concept"
                )
        out[path.stem] = body
    return out


# ------------------------------------------------------------------------- #
# Atomic write + collision detection.
# ------------------------------------------------------------------------- #


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


class _CollisionGuard:
    """Fails loud if two generated files would land at the same path."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def write(self, path: Path, content: str) -> None:
        key = str(path.resolve())
        if key in self._seen:
            raise ValueError(f"duplicate output filename/id collision: {path}")
        self._seen.add(key)
        _atomic_write(path, content)

    @property
    def count(self) -> int:
        return len(self._seen)


# ------------------------------------------------------------------------- #
# Manifest.
# ------------------------------------------------------------------------- #


def _sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    manifest_path: Path,
    sources: dict[str, Path | None],
    result: ExportResult,
    generated_at: str,
) -> None:
    manifest = {
        "generated_at": generated_at,
        "bead": "CL-uuy0",
        "sources": {name: _sha256(p) for name, p in sources.items()},
        "counts": {
            "themes": result.theme_count,
            "instruments": result.instrument_count,
            "concepts": result.concept_count,
            "nodes": result.node_count,
            "links": result.link_count,
            "files": result.files_written,
        },
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# ------------------------------------------------------------------------- #
# Orchestration.
# ------------------------------------------------------------------------- #


def export_vault(
    out_dir: Path,
    seed_dir: Path,
    playbooks_path: Path = DEFAULT_PLAYBOOKS,
    cross_asset_path: Path | None = DEFAULT_CROSS_ASSET,
    retail_path: Path | None = DEFAULT_RETAIL,
    manifest_path: Path | None = DEFAULT_MANIFEST,
    generated_at: str | None = None,
) -> ExportResult:
    """Generate the full Obsidian vault. Pure: reads YAML, writes Markdown."""
    themes = load_themes(playbooks_path, cross_asset_path)
    retail = load_retail_proxies(retail_path)
    overlaps = compute_overlaps(themes)
    watchers = instrument_watchers(themes)

    guard = _CollisionGuard()
    link_count = 0

    # Themes/
    for key, theme in themes.items():
        note = render_theme_note(theme, overlaps[key], themes)
        guard.write(out_dir / "Themes" / f"{key}.md", note)
        link_count += len(extract_wikilinks(note))

    # Instruments/  (one per unique symbol; symbol IS the note title)
    all_symbols = sorted(watchers.keys())
    symbol_kind: dict[str, str] = {}
    for theme in themes.values():
        for inst in theme.instruments:
            # A symbol's kind is stable across themes; first seen wins, and we
            # assert consistency to catch config drift.
            prior = symbol_kind.get(inst.symbol)
            if prior is not None and prior != inst.kind:
                raise ValueError(
                    f"instrument {inst.symbol!r} has conflicting kinds "
                    f"{prior!r} and {inst.kind!r} across themes"
                )
            symbol_kind.setdefault(inst.symbol, inst.kind)
    for symbol in all_symbols:
        note = render_instrument_note(
            symbol, symbol_kind[symbol], watchers[symbol], retail.get(symbol)
        )
        guard.write(out_dir / "Instruments" / f"{symbol}.md", note)
        link_count += len(extract_wikilinks(note))

    # Concepts/  (validate seed wikilinks against generated targets, then copy)
    known_targets = set(themes.keys()) | set(all_symbols)
    concepts = validate_seed_notes(seed_dir, known_targets)
    for name, body in concepts.items():
        guard.write(out_dir / "Concepts" / f"{name}.md", body)
        link_count += len(extract_wikilinks(body))

    # Coverage.md
    coverage = render_coverage_note(themes, overlaps)
    guard.write(out_dir / "Coverage.md", coverage)
    link_count += len(extract_wikilinks(coverage))

    # README.md
    guard.write(out_dir / "README.md", render_vault_readme())

    node_count = len(themes) + len(all_symbols) + len(concepts)
    result = ExportResult(
        theme_count=len(themes),
        instrument_count=len(all_symbols),
        concept_count=len(concepts),
        node_count=node_count,
        link_count=link_count,
        files_written=guard.count,
    )

    if manifest_path is not None:
        stamp = generated_at or datetime.now(UTC).isoformat()
        write_manifest(
            manifest_path,
            {
                "event_playbooks.yaml": playbooks_path,
                "cross_asset_checks.yaml": cross_asset_path,
                "retail_proxies.yaml": retail_path,
            },
            result,
            stamp,
        )

    logger.info(
        "vault generated: %d themes, %d instruments, %d concepts, %d links -> %s",
        result.theme_count,
        result.instrument_count,
        result.concept_count,
        result.link_count,
        out_dir,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export an offline Obsidian knowledge-graph vault (CL-uuy0)."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output vault dir")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED, help="seed concepts dir")
    parser.add_argument(
        "--playbooks", type=Path, default=DEFAULT_PLAYBOOKS, help="event playbooks YAML"
    )
    parser.add_argument(
        "--cross-asset", type=Path, default=DEFAULT_CROSS_ASSET, help="cross-asset YAML"
    )
    parser.add_argument(
        "--retail", type=Path, default=DEFAULT_RETAIL, help="retail proxies YAML"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="staleness manifest path"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    result = export_vault(
        out_dir=args.out,
        seed_dir=args.seed,
        playbooks_path=args.playbooks,
        cross_asset_path=args.cross_asset,
        retail_path=args.retail,
        manifest_path=args.manifest,
    )
    logger.info(
        "wrote %d files (%d nodes, %d links) to %s",
        result.files_written,
        result.node_count,
        result.link_count,
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
