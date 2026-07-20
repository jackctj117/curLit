"""Tests for the event playbook loader (CL-6iu7).

Covers: the real configs/event_playbooks.yaml loads and passes schema
sanity (tradables match the OANDA symbol shape, equities are watch-only,
required themes present), plus fail-loud validation on bad configs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.events.playbooks import (
    INSTRUMENT_RE,
    TRADABLE_KINDS,
    VALID_DIRECTIONS,
    VALID_KINDS,
    all_tradable_instruments,
    load_playbooks,
)

REQUIRED_THEMES = {
    "energy_chokepoint",
    "oil_supply_shock",
    "war_escalation",
    "cb_surprise",
    "natural_disaster",
    "sanctions_trade",
}


class TestRealConfig:
    def test_loads_and_has_required_themes(self) -> None:
        pbs = load_playbooks("configs/event_playbooks.yaml")
        assert set(pbs) >= REQUIRED_THEMES

    def test_every_theme_is_complete(self) -> None:
        for key, pb in load_playbooks("configs/event_playbooks.yaml").items():
            assert pb.name, key
            assert pb.description, key
            assert len(pb.watch_terms) >= 3, key
            assert len(pb.instruments) >= 3, key

    def test_schema_sanity(self) -> None:
        pbs = load_playbooks("configs/event_playbooks.yaml")
        for pb in pbs.values():
            for inst in pb.instruments:
                assert inst.kind in VALID_KINDS
                assert inst.direction in VALID_DIRECTIONS
                assert inst.rationale, f"{pb.key}/{inst.instrument} missing rationale"
                if inst.kind in TRADABLE_KINDS:
                    assert INSTRUMENT_RE.match(inst.instrument), (
                        f"{pb.key}/{inst.instrument} not OANDA-shaped"
                    )
                if inst.kind == "equity_watch":
                    assert inst.direction == "watch", (
                        f"{pb.key}/{inst.instrument}: equities are alert-only"
                    )

    def test_every_theme_has_a_tradable(self) -> None:
        # A theme with only watches can never produce a trade signal.
        for key, pb in load_playbooks("configs/event_playbooks.yaml").items():
            assert pb.tradable_instruments, key

    def test_tradable_union_helper(self) -> None:
        pbs = load_playbooks("configs/event_playbooks.yaml")
        tradables = all_tradable_instruments(pbs)
        assert "BCO_USD" in tradables
        assert "XAU_USD" in tradables
        # Equity tickers must never leak into the tradable whitelist.
        assert "FRO" not in tradables
        assert "NVDA" not in tradables


class TestValidation:
    def _write(self, tmp_path: Path, doc: dict) -> Path:
        p = tmp_path / "pb.yaml"
        p.write_text(yaml.safe_dump(doc))
        return p

    def _theme(self, **overrides: object) -> dict:
        base: dict = {
            "name": "T",
            "description": "d",
            "watch_terms": ["a phrase"],
            "instruments": [{
                "instrument": "EUR_USD", "kind": "fx",
                "direction": "long", "rationale": "r",
            }],
        }
        base.update(overrides)
        return base

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_playbooks(tmp_path / "nope.yaml")

    def test_missing_themes_raises(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, {"not_themes": {}})
        with pytest.raises(ValueError, match="themes"):
            load_playbooks(p)

    def test_empty_watch_terms_raises(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, {"themes": {"t": self._theme(watch_terms=[])}})
        with pytest.raises(ValueError, match="watch_terms"):
            load_playbooks(p)

    def test_bad_kind_raises(self, tmp_path: Path) -> None:
        theme = self._theme(instruments=[{
            "instrument": "EUR_USD", "kind": "cfd",
            "direction": "long", "rationale": "r",
        }])
        p = self._write(tmp_path, {"themes": {"t": theme}})
        with pytest.raises(ValueError, match="kind"):
            load_playbooks(p)

    def test_lowercase_tradable_raises(self, tmp_path: Path) -> None:
        theme = self._theme(instruments=[{
            "instrument": "eur_usd", "kind": "fx",
            "direction": "long", "rationale": "r",
        }])
        p = self._write(tmp_path, {"themes": {"t": theme}})
        with pytest.raises(ValueError, match="must match"):
            load_playbooks(p)

    def test_directional_equity_raises(self, tmp_path: Path) -> None:
        theme = self._theme(instruments=[{
            "instrument": "NVDA", "kind": "equity_watch",
            "direction": "short", "rationale": "r",
        }])
        p = self._write(tmp_path, {"themes": {"t": theme}})
        with pytest.raises(ValueError, match="alert-only"):
            load_playbooks(p)

    def test_polymarket_slug_allowed_lowercase(self, tmp_path: Path) -> None:
        theme = self._theme(instruments=[
            {"instrument": "EUR_USD", "kind": "fx",
             "direction": "long", "rationale": "r"},
            {"instrument": "some-event-slug-2026", "kind": "polymarket",
             "direction": "watch", "rationale": "r"},
        ])
        p = self._write(tmp_path, {"themes": {"t": theme}})
        pbs = load_playbooks(p)
        kinds = {i.kind for i in pbs["t"].instruments}
        assert kinds == {"fx", "polymarket"}
