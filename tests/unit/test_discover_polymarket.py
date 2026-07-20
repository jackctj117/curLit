"""Tests for the Polymarket discovery script (CL-3t4j v2 follow-up).

Exercises the pure functions (filter, rank, extract token_id, merge)
without hitting the live Gamma API. main() is exercised through
its argv interface with monkeypatched fetch_active_markets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.discover_polymarket_markets import (
    extract_yes_token_id,
    filter_and_rank,
    get_volume,
    is_fx_macro_relevant,
    merge_into_config,
)

# ---------------------------------------------------------------------- #
# Filtering
# ---------------------------------------------------------------------- #


class TestIsFXMacroRelevant:
    def test_fed_match(self) -> None:
        assert is_fx_macro_relevant({
            "question": "Will the Fed cut rates by 25bps in June?",
        })

    def test_dollar_match(self) -> None:
        assert is_fx_macro_relevant({
            "question": "Will EUR break parity with the dollar by Q3?",
        })

    def test_sports_market_excluded(self) -> None:
        assert not is_fx_macro_relevant({
            "question": "Will LeBron win MVP this season?",
            "description": "Basketball award prediction market",
        })

    def test_keyword_in_description(self) -> None:
        assert is_fx_macro_relevant({
            "question": "Generic title",
            "description": "Resolves on the next CPI release.",
        })


class TestExtractYesTokenId:
    def test_native_list(self) -> None:
        m = {"clobTokenIds": ["yes-tok", "no-tok"]}
        assert extract_yes_token_id(m) == "yes-tok"

    def test_json_string(self) -> None:
        # Real Gamma API often returns this field as a JSON-encoded string
        m = {"clobTokenIds": json.dumps(["yes-tok", "no-tok"])}
        assert extract_yes_token_id(m) == "yes-tok"

    def test_missing_returns_none(self) -> None:
        assert extract_yes_token_id({}) is None

    def test_empty_list_returns_none(self) -> None:
        assert extract_yes_token_id({"clobTokenIds": []}) is None

    def test_unparseable_returns_none(self) -> None:
        assert extract_yes_token_id({"clobTokenIds": "not json"}) is None


class TestGetVolume:
    def test_volumeNum_preferred(self) -> None:
        assert get_volume({"volumeNum": 5000.0, "volume": 1000}) == 5000.0

    def test_volume_fallback(self) -> None:
        assert get_volume({"volume": 250}) == 250.0

    def test_missing_returns_zero(self) -> None:
        assert get_volume({}) == 0.0

    def test_unparseable_returns_zero(self) -> None:
        assert get_volume({"volumeNum": "junk"}) == 0.0


class TestFilterAndRank:
    def test_filters_and_sorts_by_volume(self) -> None:
        markets = [
            {"slug": "low-vol-fed", "question": "Fed cut?",
             "clobTokenIds": ["a", "b"], "volume": 5000},
            {"slug": "high-vol-fed", "question": "Fed cut probability",
             "clobTokenIds": ["c", "d"], "volume": 100000},
            {"slug": "mid-vol-cpi", "question": "CPI > 3%?",
             "clobTokenIds": ["e", "f"], "volume": 50000},
            {"slug": "sports", "question": "MVP race?",
             "clobTokenIds": ["g", "h"], "volume": 99999},
            {"slug": "no-tokens", "question": "Will Fed surprise?",
             "volume": 200000},
        ]
        out = filter_and_rank(markets, min_volume_usd=10_000, limit=10)
        slugs = [m["slug"] for m in out]
        # Sports filtered out (no fx/macro match)
        # no-tokens filtered out (no clobTokenIds)
        # low-vol-fed filtered out (5k < 10k threshold)
        assert slugs == ["high-vol-fed", "mid-vol-cpi"]

    def test_limit_caps_results(self) -> None:
        markets = [
            {"slug": f"fed-{i}", "question": "Fed move?",
             "clobTokenIds": ["a", "b"], "volume": float(100000 - i * 1000)}
            for i in range(20)
        ]
        out = filter_and_rank(markets, min_volume_usd=0, limit=5)
        assert len(out) == 5
        # Should be top 5 by volume (highest first)
        assert [m["slug"] for m in out] == [f"fed-{i}" for i in range(5)]


# ---------------------------------------------------------------------- #
# YAML merge
# ---------------------------------------------------------------------- #


class TestMergeIntoConfig:
    def test_replaces_placeholder_with_real_market(
        self, tmp_path: Path,
    ) -> None:
        existing = tmp_path / "polymarket_markets.yaml"
        existing.write_text(yaml.safe_dump({
            "markets": [
                {"slug": "fed-cut-jun-2026",
                 "token_id": "PLACEHOLDER_FED_JUN_YES",
                 "fx_relevance": "high"},
            ],
        }))
        discovered = [
            {"slug": "fed-cut-jun-2026",
             "question": "Will the Fed cut by 25bps in June 2026?",
             "clobTokenIds": ["real-yes-token", "real-no-token"],
             "volumeNum": 250000,
             "endDate": "2026-06-15"},
        ]
        new_doc, stats = merge_into_config(existing, discovered)
        assert stats["replaced"] == 1
        assert stats["added"] == 0
        assert stats["placeholders_remaining"] == 0
        # Real entry has the new token_id but kept the operator's
        # fx_relevance field
        entry = new_doc["markets"][0]
        assert entry["token_id"] == "real-yes-token"
        assert entry["fx_relevance"] == "high"
        assert entry["slug"] == "fed-cut-jun-2026"

    def test_preserves_real_entries_unchanged(
        self, tmp_path: Path,
    ) -> None:
        existing = tmp_path / "polymarket_markets.yaml"
        existing.write_text(yaml.safe_dump({
            "markets": [
                {"slug": "fed-cut-jun-2026",
                 "token_id": "operator-curated-token",
                 "fx_relevance": "high"},
            ],
        }))
        # Discovery returns same slug with a different token_id;
        # operator's manual choice should win
        discovered = [
            {"slug": "fed-cut-jun-2026",
             "question": "Fed cut?",
             "clobTokenIds": ["different-token", "no"],
             "volumeNum": 100000},
        ]
        new_doc, stats = merge_into_config(existing, discovered)
        assert stats["preserved"] == 1
        assert stats["replaced"] == 0
        assert new_doc["markets"][0]["token_id"] == "operator-curated-token"

    def test_appends_new_discoveries(self, tmp_path: Path) -> None:
        existing = tmp_path / "polymarket_markets.yaml"
        existing.write_text(yaml.safe_dump({
            "markets": [
                {"slug": "old", "token_id": "old-token"},
            ],
        }))
        discovered = [
            {"slug": "new-fed-cut",
             "question": "New fed market",
             "clobTokenIds": ["nt", "nf"],
             "volumeNum": 50000},
        ]
        new_doc, stats = merge_into_config(existing, discovered)
        assert stats["preserved"] == 1
        assert stats["added"] == 1
        slugs = [m["slug"] for m in new_doc["markets"]]
        assert slugs == ["old", "new-fed-cut"]

    def test_keeps_orphaned_placeholders(self, tmp_path: Path) -> None:
        # An operator-listed placeholder slug that discovery didn't
        # find. Operator may still want to fill it manually; don't
        # delete.
        existing = tmp_path / "polymarket_markets.yaml"
        existing.write_text(yaml.safe_dump({
            "markets": [
                {"slug": "operator-only", "token_id": "PLACEHOLDER_X"},
            ],
        }))
        new_doc, stats = merge_into_config(existing, [])
        assert stats["placeholders_remaining"] == 1
        assert new_doc["markets"][0]["slug"] == "operator-only"

    def test_no_existing_yaml_creates_fresh(
        self, tmp_path: Path,
    ) -> None:
        ghost = tmp_path / "no-such.yaml"
        discovered = [
            {"slug": "first-market",
             "question": "Fed move?",
             "clobTokenIds": ["t", "n"],
             "volumeNum": 100000},
        ]
        new_doc, stats = merge_into_config(ghost, discovered)
        assert stats["added"] == 1
        assert stats["preserved"] == 0
        assert new_doc["markets"][0]["slug"] == "first-market"


# ---------------------------------------------------------------------- #
# main() via monkeypatched API + argv
# ---------------------------------------------------------------------- #


class TestMainEntry:
    def test_dry_run_doesnt_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text(yaml.safe_dump({"markets": []}))

        def fake_fetch(api_url: str = "", limit: int = 200, timeout_sec: float = 30) -> list[dict[str, Any]]:
            return [
                {"slug": "fed-test", "question": "Fed cut?",
                 "clobTokenIds": ["yes", "no"], "volumeNum": 50000},
            ]

        monkeypatch.setattr(
            "scripts.discover_polymarket_markets.fetch_active_markets",
            fake_fetch,
        )
        from scripts.discover_polymarket_markets import main
        rc = main([
            "--mode", "fx",
            "--config", str(cfg),
            "--limit", "5",
            "--min-volume", "1000",
            "--dry-run",
        ])
        assert rc == 0
        # YAML unchanged on dry-run
        loaded = yaml.safe_load(cfg.read_text())
        assert loaded == {"markets": []}

    def test_real_run_writes_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text(yaml.safe_dump({"markets": []}))

        def fake_fetch(api_url: str = "", limit: int = 200, timeout_sec: float = 30) -> list[dict[str, Any]]:
            return [
                {"slug": "fed-cut-jun-2026",
                 "question": "Will the Fed cut?",
                 "clobTokenIds": ["yes-tok-123", "no-tok-456"],
                 "volumeNum": 75000,
                 "endDate": "2026-06-15"},
            ]

        monkeypatch.setattr(
            "scripts.discover_polymarket_markets.fetch_active_markets",
            fake_fetch,
        )
        from scripts.discover_polymarket_markets import main
        rc = main([
            "--mode", "fx",
            "--config", str(cfg),
            "--limit", "5",
            "--min-volume", "1000",
        ])
        assert rc == 0
        loaded = yaml.safe_load(cfg.read_text())
        assert len(loaded["markets"]) == 1
        entry = loaded["markets"][0]
        assert entry["slug"] == "fed-cut-jun-2026"
        assert entry["token_id"] == "yes-tok-123"
        assert "discovered_volume_usd" in entry
        assert entry["end_date"] == "2026-06-15"

    def test_api_failure_returns_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = tmp_path / "p.yaml"
        cfg.write_text("markets: []")

        def boom(api_url: str = "", limit: int = 200, timeout_sec: float = 30) -> list[dict[str, Any]]:
            raise ConnectionError("offline")

        monkeypatch.setattr(
            "scripts.discover_polymarket_markets.fetch_active_markets",
            boom,
        )
        from scripts.discover_polymarket_markets import main
        rc = main(["--mode", "fx", "--config", str(cfg)])
        assert rc == 1


# ---------------------------------------------------------------------- #
# Geopolitical discovery (CL-r1ep)
# ---------------------------------------------------------------------- #

from scripts.discover_polymarket_markets import (  # noqa: E402
    discover_geo_markets,
    extract_yes_prob,
    match_theme,
    merge_geo_config,
)

_THEME_TERMS = {
    "energy_chokepoint": ("strait of hormuz", "hormuz closure", "tanker attacked"),
    "taiwan_semiconductor": ("taiwan strait", "taiwan blockade", "taiwan invasion"),
    "russia_ukraine": ("kerch bridge", "ceasefire talks", "crimea"),
    "africa_power_shift": ("military coup", "coup attempt", "junta"),
}


class TestMatchTheme:
    def test_matches_hormuz(self) -> None:
        m = {"question": "Will the Strait of Hormuz be closed in 2026?",
             "slug": "hormuz-closed-2026"}
        matched = match_theme(m, _THEME_TERMS)
        assert matched is not None
        assert matched[0] == "energy_chokepoint"
        assert matched[1] == "strait of hormuz"

    def test_matches_slug_when_question_generic(self) -> None:
        m = {"question": "Big geopolitical event?", "slug": "taiwan-blockade-q4"}
        matched = match_theme(m, _THEME_TERMS)
        assert matched is not None
        assert matched[0] == "taiwan_semiconductor"

    def test_no_match_returns_none(self) -> None:
        m = {"question": "Will Bitcoin hit 200k?", "slug": "btc-200k"}
        assert match_theme(m, _THEME_TERMS) is None

    def test_best_score_wins_on_multiple_hits(self) -> None:
        # Two terms from russia_ukraine, one from africa_power_shift.
        m = {"question": "Crimea ceasefire talks and a coup?", "slug": "x"}
        matched = match_theme(m, _THEME_TERMS)
        assert matched is not None
        assert matched[0] == "russia_ukraine"

    def test_empty_haystack(self) -> None:
        assert match_theme({"question": "", "slug": ""}, _THEME_TERMS) is None


class TestExtractYesProb:
    def test_json_string(self) -> None:
        assert extract_yes_prob({"outcomePrices": json.dumps(["0.63", "0.37"])}) == pytest.approx(0.63)

    def test_native_list(self) -> None:
        assert extract_yes_prob({"outcomePrices": [0.2, 0.8]}) == pytest.approx(0.2)

    def test_missing(self) -> None:
        assert extract_yes_prob({}) is None

    def test_unparseable(self) -> None:
        assert extract_yes_prob({"outcomePrices": "nope"}) is None


class TestDiscoverGeoMarkets:
    def test_filters_and_tags_theme(self) -> None:
        markets = [
            {"slug": "hormuz-closed", "question": "Will the Strait of Hormuz close?",
             "clobTokenIds": json.dumps(["y1", "n1"]),
             "outcomePrices": json.dumps(["0.18", "0.82"]),
             "volumeNum": 80000, "endDate": "2026-12-31"},
            {"slug": "sports", "question": "Will the Lakers win?",
             "clobTokenIds": json.dumps(["y2", "n2"]), "volumeNum": 500000},
            {"slug": "low-vol", "question": "Taiwan blockade?",
             "clobTokenIds": json.dumps(["y3", "n3"]), "volumeNum": 100},
            {"slug": "no-token", "question": "Taiwan invasion?", "volumeNum": 99999},
        ]
        out = discover_geo_markets(markets, _THEME_TERMS, min_volume_usd=5000, limit=10)
        # Only the hormuz market: sports has no theme, low-vol below floor,
        # no-token lacks a clobTokenId.
        assert len(out) == 1
        e = out[0]
        assert e["slug"] == "hormuz-closed"
        assert e["theme"] == "energy_chokepoint"
        assert e["yes_token_id"] == "y1"
        assert e["yes_prob"] == pytest.approx(0.18)
        assert e["end_date"] == "2026-12-31"
        assert "discovered_at" in e

    def test_sorts_by_volume(self) -> None:
        markets = [
            {"slug": "a", "question": "Hormuz closure?",
             "clobTokenIds": ["ya", "na"], "volumeNum": 10000},
            {"slug": "b", "question": "Taiwan blockade?",
             "clobTokenIds": ["yb", "nb"], "volumeNum": 90000},
        ]
        out = discover_geo_markets(markets, _THEME_TERMS, min_volume_usd=0, limit=10)
        assert [e["slug"] for e in out] == ["b", "a"]

    def test_limit_caps(self) -> None:
        markets = [
            {"slug": f"coup-{i}", "question": "military coup?",
             "clobTokenIds": ["y", "n"], "volumeNum": float(100000 - i)}
            for i in range(10)
        ]
        out = discover_geo_markets(markets, _THEME_TERMS, min_volume_usd=0, limit=3)
        assert len(out) == 3


class TestMergeGeoConfig:
    def test_appends_new_dedup(self, tmp_path: Path) -> None:
        cfg = tmp_path / "geo.yaml"
        cfg.write_text(yaml.safe_dump({"markets": [
            {"slug": "existing", "theme": "russia_ukraine",
             "yes_token_id": "e"},
        ]}))
        discovered = [
            {"slug": "existing", "theme": "russia_ukraine", "yes_token_id": "x"},
            {"slug": "new", "theme": "energy_chokepoint", "yes_token_id": "n"},
        ]
        new_doc, stats = merge_geo_config(cfg, discovered)
        assert stats["existing"] == 1
        assert stats["added"] == 1
        assert stats["skipped_duplicate"] == 1
        slugs = [m["slug"] for m in new_doc["markets"]]
        assert slugs == ["existing", "new"]
        # The pre-existing entry is preserved verbatim (operator may have
        # hand-tuned it) — discovery's competing token_id does NOT win.
        assert new_doc["markets"][0]["yes_token_id"] == "e"

    def test_fresh_when_no_file(self, tmp_path: Path) -> None:
        ghost = tmp_path / "nope.yaml"
        new_doc, stats = merge_geo_config(ghost, [
            {"slug": "first", "theme": "war_escalation", "yes_token_id": "t"},
        ])
        assert stats["added"] == 1
        assert new_doc["markets"][0]["slug"] == "first"


class TestGeoMainEntry:
    def test_geo_run_writes_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = tmp_path / "geo.yaml"

        def fake_fetch(api_url: str = "", limit: int = 200,
                       timeout_sec: float = 30,
                       order_by_volume24hr: bool = False) -> list[dict[str, Any]]:
            return [
                {"slug": "hormuz-closed-2026",
                 "question": "Will the Strait of Hormuz be closed in 2026?",
                 "clobTokenIds": json.dumps(["yes-tok", "no-tok"]),
                 "outcomePrices": json.dumps(["0.18", "0.82"]),
                 "volumeNum": 120000, "endDate": "2026-12-31"},
                {"slug": "lakers", "question": "Lakers win?",
                 "clobTokenIds": json.dumps(["a", "b"]), "volumeNum": 500000},
            ]

        monkeypatch.setattr(
            "scripts.discover_polymarket_markets.fetch_active_markets",
            fake_fetch,
        )
        from scripts.discover_polymarket_markets import main
        rc = main([
            "--mode", "geo", "--out", str(out),
            "--min-volume", "1000", "--limit", "50",
        ])
        assert rc == 0
        loaded = yaml.safe_load(out.read_text())
        assert len(loaded["markets"]) == 1
        e = loaded["markets"][0]
        assert e["slug"] == "hormuz-closed-2026"
        assert e["theme"] == "energy_chokepoint"
        assert e["yes_token_id"] == "yes-tok"

    def test_geo_dry_run_no_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        out = tmp_path / "geo.yaml"

        def fake_fetch(api_url: str = "", limit: int = 200,
                       timeout_sec: float = 30,
                       order_by_volume24hr: bool = False) -> list[dict[str, Any]]:
            return [
                {"slug": "taiwan-blockade-2026", "question": "Taiwan blockade in 2026?",
                 "clobTokenIds": json.dumps(["y", "n"]), "volumeNum": 60000},
            ]

        monkeypatch.setattr(
            "scripts.discover_polymarket_markets.fetch_active_markets",
            fake_fetch,
        )
        from scripts.discover_polymarket_markets import main
        rc = main(["--mode", "geo", "--out", str(out), "--dry-run",
                   "--min-volume", "1000"])
        assert rc == 0
        assert not out.exists()
