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
    # CL-lu80
    "pharma_api_supply",
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
    "mali",
    "ghana",
    "drc",
    "congo",
    "zambia",
    "niger",
    "guinea",
    "south africa",
    "sa ",
    "zimbabwe",
    "burkina",
    "senegal",
    "tanzania",
    "sahel",
    "katanga",
    "kolwezi",
    "kamoa",
    "kipushi",
    "kisanfu",
    "tenke",
    "mutanda",
    "fekola",
    "loulo",
    "obuasi",
    "ahafo",
    "akyem",
    "kibali",
    "kansanshi",
    "sentinel",
    "dasa",
    "rustenburg",
    "marikana",
    "mogalakwena",
    "south deep",
    "mponeng",
    "siguiri",
    "simandou",
    "cbg",
    "pilbara",
    "geita",
    "iduapriem",
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
                "XAU_USD",
                "XCU_USD",
                "XPT_USD",
                "XPD_USD",
                "USD_ZAR",
            }, key

    def test_exchange_suffixed_equities_load(self) -> None:
        # Non-US listings (IVN.TO, GLEN.L, ...) are valid equity_watch
        # ids — only tradables are held to the OANDA symbol shape.
        pbs = load_playbooks("configs/event_playbooks.yaml")
        drc = {i.instrument for i in pbs["drc_copper_cobalt"].instruments}
        assert "IVN.TO" in drc
        assert "GLEN.L" in drc


class TestPharmaApiSupply:
    """CL-lu80: the pharma theme has NO pharma OANDA leg — it is almost
    entirely equity_watch (alert-only). Reasons must distinguish
    China/India-input-exposed generics from diversified/resilient names.
    """

    @pytest.fixture(scope="class")
    def pharma(self):
        return load_playbooks("configs/event_playbooks.yaml")["pharma_api_supply"]

    def test_loads_with_watch_terms(self, pharma) -> None:
        assert pharma.watch_terms
        assert len(pharma.instruments) >= 3

    def test_equities_are_watch_only(self, pharma) -> None:
        # Every equity is alert-only (kind equity_watch, forced watch).
        equities = [i for i in pharma.instruments if i.kind == "equity_watch"]
        assert equities, "pharma theme has no equity watches"
        for inst in equities:
            assert inst.direction == "watch", inst.instrument
        # Exposure names present.
        names = {i.instrument for i in equities}
        assert {"TEVA", "VTRS", "RDY"} <= names  # exposed generics
        assert {"TMO", "PFE", "JNJ"} <= names  # diversified

    def test_no_pharma_oanda_symbol_invented(self, pharma) -> None:
        # The only tradable allowed is the broad SPX500_USD risk-off leg
        # (optional weak proxy) — never an invented pharma CFD.
        tradables = {i.instrument for i in pharma.tradable_instruments}
        assert tradables <= {"SPX500_USD"}, tradables

    def test_reasons_distinguish_exposed_vs_diversified(self, pharma) -> None:
        reasons = [i.rationale.lower() for i in pharma.instruments]
        # At least one reason names the China/India/generics input exposure.
        assert any(any(tok in r for tok in ("china", "india", "generics")) for r in reasons)
        # At least one reason marks a diversified/resilient beneficiary.
        assert any(
            any(tok in r for tok in ("diversified", "resilient", "beneficiary")) for r in reasons
        )

    def test_gdelt_query_under_length_ceiling(self, pharma) -> None:
        from src.data.gdelt import build_theme_query

        q = build_theme_query(pharma)
        assert len(q) <= 200, f"pharma GDELT query is {len(q)} chars"


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
            "instruments": [
                {
                    "instrument": "EUR_USD",
                    "kind": "fx",
                    "direction": "long",
                    "rationale": "r",
                }
            ],
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
        theme = self._theme(
            instruments=[
                {
                    "instrument": "EUR_USD",
                    "kind": "cfd",
                    "direction": "long",
                    "rationale": "r",
                }
            ]
        )
        p = self._write(tmp_path, {"themes": {"t": theme}})
        with pytest.raises(ValueError, match="kind"):
            load_playbooks(p)

    def test_lowercase_tradable_raises(self, tmp_path: Path) -> None:
        theme = self._theme(
            instruments=[
                {
                    "instrument": "eur_usd",
                    "kind": "fx",
                    "direction": "long",
                    "rationale": "r",
                }
            ]
        )
        p = self._write(tmp_path, {"themes": {"t": theme}})
        with pytest.raises(ValueError, match="must match"):
            load_playbooks(p)

    def test_directional_equity_raises(self, tmp_path: Path) -> None:
        theme = self._theme(
            instruments=[
                {
                    "instrument": "NVDA",
                    "kind": "equity_watch",
                    "direction": "short",
                    "rationale": "r",
                }
            ]
        )
        p = self._write(tmp_path, {"themes": {"t": theme}})
        with pytest.raises(ValueError, match="alert-only"):
            load_playbooks(p)

    def test_polymarket_slug_allowed_lowercase(self, tmp_path: Path) -> None:
        theme = self._theme(
            instruments=[
                {"instrument": "EUR_USD", "kind": "fx", "direction": "long", "rationale": "r"},
                {
                    "instrument": "some-event-slug-2026",
                    "kind": "polymarket",
                    "direction": "watch",
                    "rationale": "r",
                },
            ]
        )
        p = self._write(tmp_path, {"themes": {"t": theme}})
        pbs = load_playbooks(p)
        kinds = {i.kind for i in pbs["t"].instruments}
        assert kinds == {"fx", "polymarket"}

    # ---- last_reviewed (CL-ylak) -------------------------------------

    def test_last_reviewed_parses_yaml_date(self, tmp_path: Path) -> None:
        from datetime import date

        p = self._write(tmp_path, {"themes": {"t": self._theme(last_reviewed=date(2026, 7, 20))}})
        assert load_playbooks(p)["t"].last_reviewed == date(2026, 7, 20)

    def test_last_reviewed_parses_iso_string(self, tmp_path: Path) -> None:
        from datetime import date

        p = self._write(tmp_path, {"themes": {"t": self._theme(last_reviewed="2026-07-20")}})
        assert load_playbooks(p)["t"].last_reviewed == date(2026, 7, 20)

    def test_last_reviewed_absent_is_none(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, {"themes": {"t": self._theme()}})
        assert load_playbooks(p)["t"].last_reviewed is None

    def test_last_reviewed_malformed_raises(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, {"themes": {"t": self._theme(last_reviewed="not-a-date")}})
        with pytest.raises(ValueError, match="last_reviewed"):
            load_playbooks(p)

    def test_real_yaml_dates_every_theme(self) -> None:
        # The shipped config MUST date every theme, or the stale-fact ceiling
        # silently no-ops for the undated ones.
        pbs = load_playbooks("configs/event_playbooks.yaml")
        undated = [k for k, v in pbs.items() if v.last_reviewed is None]
        assert undated == []

    # ---- tier (CL-gn6k) ----------------------------------------------

    def test_tier_defaults_to_specific(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, {"themes": {"t": self._theme()}})
        assert load_playbooks(p)["t"].tier == "specific"

    def test_tier_generic_parses(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, {"themes": {"t": self._theme(tier="generic")}})
        assert load_playbooks(p)["t"].tier == "generic"

    def test_tier_invalid_raises(self, tmp_path: Path) -> None:
        p = self._write(tmp_path, {"themes": {"t": self._theme(tier="middling")}})
        with pytest.raises(ValueError, match="tier"):
            load_playbooks(p)

    def test_real_yaml_generic_tier_is_the_catch_alls(self) -> None:
        pbs = load_playbooks("configs/event_playbooks.yaml")
        generic = {k for k, v in pbs.items() if v.tier == "generic"}
        assert generic == {"war_escalation", "natural_disaster", "africa_power_shift"}
