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
    # CL-01zt expansion
    "russia_ukraine",
    "africa_power_shift",
    "drc_copper_cobalt",
    "sahel_gold_uranium",
    "guinea_iron_bauxite",
    "south_africa_pgm_gold",
    "red_sea_shipping",
    "taiwan_semiconductor",
    "black_sea_grain",
}

#: Themes whose equity watches exist BECAUSE of territorial exposure —
#: every equity rationale must name the country/site (CL-01zt: "what
#: companies have that territory").
AFRICA_THEMES = {
    "africa_power_shift",
    "drc_copper_cobalt",
    "sahel_gold_uranium",
    "guinea_iron_bauxite",
    "south_africa_pgm_gold",
}

#: Loose country/site token list for the territorial-exposure check.
TERRITORY_TOKENS = (
    "mali", "ghana", "drc", "congo", "zambia", "niger", "guinea",
    "south africa", "sa ", "zimbabwe", "burkina", "senegal", "tanzania",
    "sahel", "katanga", "kolwezi", "kamoa", "kipushi", "kisanfu",
    "tenke", "mutanda", "fekola", "loulo", "obuasi", "ahafo", "akyem",
    "kibali", "kansanshi", "sentinel", "dasa", "rustenburg", "marikana",
    "mogalakwena", "south deep", "mponeng", "siguiri", "simandou",
    "cbg", "pilbara", "geita", "iduapriem",
)


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
        # CL-01zt additions must be vetted tradables.
        assert "XCU_USD" in tradables
        assert "XPT_USD" in tradables
        assert "USD_ZAR" in tradables
        assert "WHEAT_USD" in tradables
        # Equity tickers must never leak into the tradable whitelist.
        assert "FRO" not in tradables
        assert "NVDA" not in tradables
        assert "GOLD" not in tradables  # Barrick's ticker, not XAU
        assert "CORN" not in tradables  # Teucrium ETF; CORN_USD is the CFD


class TestTerritorialExposure:
    """CL-01zt: the operator's ask is 'which companies have that
    territory' — Africa-theme equity watches must carry it in the
    rationale, loosely checked against a country/site token list."""

    def test_africa_equity_rationales_name_territory(self) -> None:
        pbs = load_playbooks("configs/event_playbooks.yaml")
        for key in AFRICA_THEMES:
            pb = pbs[key]
            equities = [i for i in pb.instruments if i.kind == "equity_watch"]
            assert equities, f"{key}: no equity watch entries"
            for inst in equities:
                low = inst.rationale.lower()
                assert any(tok in low for tok in TERRITORY_TOKENS), (
                    f"{key}/{inst.instrument}: rationale must name the "
                    f"territory/asset, got: {inst.rationale!r}"
                )

    def test_africa_themes_have_commodity_tradable(self) -> None:
        # Every Africa theme needs at least one reachable tradable leg
        # (metal CFD or ZAR) — alerts alone can't act.
        pbs = load_playbooks("configs/event_playbooks.yaml")
        for key in AFRICA_THEMES:
            names = {i.instrument for i in pbs[key].tradable_instruments}
            assert names & {
                "XAU_USD", "XCU_USD", "XPT_USD", "XPD_USD", "USD_ZAR",
            }, key

    def test_exchange_suffixed_equities_load(self) -> None:
        # Non-US listings (IVN.TO, GLEN.L, ...) are valid equity_watch
        # ids — only tradables are held to the OANDA symbol shape.
        pbs = load_playbooks("configs/event_playbooks.yaml")
        drc = {i.instrument for i in pbs["drc_copper_cobalt"].instruments}
        assert "IVN.TO" in drc
        assert "GLEN.L" in drc


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
