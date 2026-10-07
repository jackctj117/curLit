"""Explicit strategy registry (CL-nix8).

Oracles:
  * the substring dispatcher as it existed at 5d44d14 (copied verbatim
    below) — the live config must resolve IDENTICALLY under the registry;
  * the ``class:`` dotted path each live_portfolio.yaml entry declares,
    imported independently with importlib.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
import yaml

import src.runtime.run_engine as run_engine
from src.runtime.run_engine import (
    STRATEGY_CLASS_REGISTRY,
    StrategyConfigError,
    resolve_strategy_entries,
)
from src.strategies.carry_vol_filter import CarryVolFilterStrategy
from src.strategies.cb_sentiment_shift import CBSentimentShiftStrategy
from src.strategies.event_driven import EventDrivenStrategy
from src.strategies.rate_diff_mean_reversion import RateDiffMRStrategy

pytestmark = pytest.mark.unit

LIVE_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "live_portfolio.yaml"


def _legacy_substring_route(sid: str) -> type | None:
    """run_engine.build_strategies dispatch at 5d44d14 (the reference path)."""
    if "rate_diff" in sid:
        return RateDiffMRStrategy
    elif "sentiment" in sid or "cb" in sid:
        return CBSentimentShiftStrategy
    elif "carry" in sid or "vol_filter" in sid:
        return CarryVolFilterStrategy
    elif "event" in sid:
        return EventDrivenStrategy
    return None  # silently dropped


def _import_class(path: str) -> type:
    module, _, name = path.rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    assert isinstance(cls, type)
    return cls


def _live_entries() -> list[dict[str, Any]]:
    raw = yaml.safe_load(LIVE_CONFIG.read_text())
    entries = raw["strategies"]
    assert isinstance(entries, list) and entries
    return entries


class TestLiveConfigDifferential:
    def test_live_config_resolves_identically_to_legacy_dispatch(self) -> None:
        entries = _live_entries()
        resolved = resolve_strategy_entries(entries)
        enabled = [e for e in entries if e.get("enabled", True) is not False]
        assert [sid for sid, _, _ in resolved] == [e["id"] for e in enabled]
        for (sid, cls, scfg), entry in zip(resolved, enabled, strict=True):
            legacy = _legacy_substring_route(sid)
            assert legacy is not None, f"legacy dispatch dropped live id {sid!r}"
            assert cls is legacy, sid
            # Second, independent oracle: the class the YAML itself names.
            assert cls is _import_class(entry["class"]), sid
            assert scfg == (entry.get("config") or {})

    def test_live_config_has_the_four_live_strategies(self) -> None:
        classes = [cls for _, cls, _ in resolve_strategy_entries(_live_entries())]
        assert sorted(c.__name__ for c in classes) == sorted(
            [
                "RateDiffMRStrategy",
                "CBSentimentShiftStrategy",
                "CarryVolFilterStrategy",
                "EventDrivenStrategy",
            ]
        )

    def test_build_strategies_builds_live_config_in_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Relative state paths (data/...) land under tmp, never the repo.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CURLIT_LIQUIDITY_PROFILE", str(tmp_path / "absent.json"))
        monkeypatch.setattr(run_engine, "_build_db_engine", lambda: sa.create_engine("sqlite://"))
        monkeypatch.setattr(run_engine, "log_startup_health", lambda engine: None)
        config = yaml.safe_load(LIVE_CONFIG.read_text())
        built = run_engine.build_strategies(config, broker=None, oms=None)
        expected = [
            _legacy_substring_route(e["id"])
            for e in config["strategies"]
            if e.get("enabled", True) is not False
        ]
        assert [type(s) for s in built] == expected

    def test_registry_class_paths_match_live_yaml(self) -> None:
        for entry in _live_entries():
            assert entry["class"] in STRATEGY_CLASS_REGISTRY


class TestFailLoud:
    @pytest.mark.parametrize(
        ("entries", "fragment"),
        [
            ([{"id": "momentum_breakout"}], "unknown id"),
            ([{"id": "x", "class": "src.strategies.nope.Nope"}], "not a registered"),
            ([{"id": "cb_sentiment_shift"}, {"id": "cb_sentiment_shift"}], "duplicate"),
            (
                [
                    {
                        "id": "event_driven",
                        "class": "src.strategies.rate_diff_mean_reversion.RateDiffMRStrategy",
                    }
                ],
                "registered to",
            ),
            ([{"id": ""}], "missing or empty"),
            ([{"config": {}}], "missing or empty"),
            (["event_driven"], "expected a mapping"),
            ({"id": "event_driven"}, "expected a list"),
            (None, "present but empty"),  # bare `strategies:` in YAML
            ([{"id": "event_driven", "config": [1, 2]}], "must be a mapping"),
        ],
    )
    def test_invalid_entries_raise(self, entries: Any, fragment: str) -> None:
        with pytest.raises(StrategyConfigError, match=fragment):
            resolve_strategy_entries(entries)

    def test_duplicate_detected_even_when_one_is_disabled(self) -> None:
        with pytest.raises(StrategyConfigError, match="duplicate"):
            resolve_strategy_entries(
                [{"id": "event_driven", "enabled": False}, {"id": "event_driven"}]
            )

    def test_disabled_unknown_entry_is_parked_not_fatal(self) -> None:
        resolved = resolve_strategy_entries(
            [{"id": "shadow_thing", "class": "src.strategies.x.Y", "enabled": False}]
        )
        assert resolved == []

    def test_bare_strategies_key_does_not_fall_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = MagicMock(side_effect=AssertionError("DB touched before validation"))
        monkeypatch.setattr(run_engine, "_build_db_engine", db)
        config = yaml.safe_load("strategies:\n")
        assert config == {"strategies": None}
        with pytest.raises(StrategyConfigError, match="present but empty"):
            run_engine.build_strategies(config, broker=None, oms=None)
        db.assert_not_called()

    def test_build_fails_before_touching_the_db(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db = MagicMock(side_effect=AssertionError("DB touched before validation"))
        monkeypatch.setattr(run_engine, "_build_db_engine", db)
        with pytest.raises(StrategyConfigError, match="unknown id"):
            run_engine.build_strategies(
                {"strategies": [{"id": "momentum_breakout"}]}, broker=None, oms=None
            )
        db.assert_not_called()


class TestNoSubstringMisrouting:
    """Ids that the legacy substring dispatch would have mis-built."""

    def test_class_field_wins_over_misleading_id(self) -> None:
        sid = "carry_hedge_vs_rate_diff"
        assert _legacy_substring_route(sid) is RateDiffMRStrategy  # the old bug
        resolved = resolve_strategy_entries(
            [{"id": sid, "class": "src.strategies.carry_vol_filter.CarryVolFilterStrategy"}]
        )
        assert resolved[0][1] is CarryVolFilterStrategy

    def test_substring_lookalike_without_class_fails(self) -> None:
        # Legacy: "eventual_cb_rate_diff" contains "rate_diff" → built as
        # rate-diff MR. Now: not a canonical id and no class → refuse.
        assert _legacy_substring_route("eventual_cb_rate_diff") is RateDiffMRStrategy
        with pytest.raises(StrategyConfigError):
            resolve_strategy_entries([{"id": "eventual_cb_rate_diff"}])
