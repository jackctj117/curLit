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
        rc = main(["--config", str(cfg)])
        assert rc == 1
