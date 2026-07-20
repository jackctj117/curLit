"""Unit tests for the cross-asset confirmation layer (CL-6mzn).

Covers:
  * cross_asset_confirmation vote math — all agree / mixed / disagree,
    weighting, missing-data exclusion, zero-data → unknown, and the
    risk-off SIGN conventions (a Taiwan event with the yen strengthening
    is USD_JPY DOWN = agrees);
  * config load + validation of the REAL configs/cross_asset_checks.yaml
    (OANDA-shaped instruments, valid directions) plus fail-loud on bad
    configs;
  * the render helpers (digest HTML + strategy plain-text): line format,
    ✓/✗ marks, HTML escaping, the low-score fade warning, and
    unknown → omitted.

No DB / network — a fake DataProvider supplies canned prices.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.events.cross_asset import (
    CrossAssetConfig,
    CrossAssetResult,
    InstrumentMove,
    cross_asset_confirmation,
    load_cross_asset_config,
)
from src.events.digest import build_cross_asset_line
from src.strategies.event_driven import EventDrivenStrategy

# --------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------- #

SINCE = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
NOW = SINCE + timedelta(hours=2)


class FakeProvider:
    """DataProvider stand-in keyed by (instrument, is_now).

    ``prices`` maps instrument → (p0, p1): the value at ``since`` and the
    value at ``now``. A ``None`` for either leg (or an instrument absent
    from the map) means "no data" for that leg — the confirmation layer
    must EXCLUDE it from the vote, not fail.
    """

    def __init__(self, prices: dict[str, tuple[float | None, float | None]]) -> None:
        self.prices = prices

    def get_latest_value(self, instrument: str, as_of: datetime) -> float | None:
        leg = self.prices.get(instrument)
        if leg is None:
            return None
        p0, p1 = leg
        return p1 if as_of >= NOW else p0


def _cfg(entries: dict[str, list[dict[str, Any]]]) -> CrossAssetConfig:
    """Build a CrossAssetConfig in-memory (bypasses YAML) for vote tests."""
    from src.events.cross_asset import CrossAssetCheck

    themes = {
        theme: tuple(
            CrossAssetCheck(
                instrument=e["instrument"],
                expected_direction=e["expected_direction"],
                weight=e.get("weight", 1.0),
            )
            for e in items
        )
        for theme, items in entries.items()
    }
    return CrossAssetConfig(themes=themes)


def _run(provider: Any, theme: str, cfg: CrossAssetConfig, **kw: Any) -> CrossAssetResult:
    return cross_asset_confirmation(provider, theme, SINCE, cfg, now=NOW, **kw)


# ===================================================================== #
# Vote math
# ===================================================================== #


class TestVoteMath:
    def test_all_agree_high_score_confirmed(self) -> None:
        cfg = _cfg({"t": [
            {"instrument": "BCO_USD", "expected_direction": "up"},
            {"instrument": "WTICO_USD", "expected_direction": "up"},
        ]})
        # Both rose → both agree.
        prov = FakeProvider({"BCO_USD": (100.0, 102.0), "WTICO_USD": (80.0, 81.0)})
        res = _run(prov, "t", cfg)
        assert res.confirmed is True
        assert res.score == pytest.approx(1.0)
        assert res.n_voting == 2
        assert res.n_agree == 2

    def test_mixed_score_reflects_fraction(self) -> None:
        cfg = _cfg({"t": [
            {"instrument": "A_USD", "expected_direction": "up"},
            {"instrument": "B_USD", "expected_direction": "up"},
            {"instrument": "C_USD", "expected_direction": "up"},
        ]})
        # A up (agree), B down (disagree), C up (agree) → 2/3.
        prov = FakeProvider({
            "A_USD": (100.0, 101.0),
            "B_USD": (100.0, 99.0),
            "C_USD": (100.0, 105.0),
        })
        res = _run(prov, "t", cfg)
        assert res.score == pytest.approx(2.0 / 3.0)
        assert res.confirmed is True  # 0.667 >= 0.5
        assert res.n_agree == 2
        assert res.n_voting == 3

    def test_disagree_not_confirmed(self) -> None:
        cfg = _cfg({"t": [
            {"instrument": "A_USD", "expected_direction": "up"},
            {"instrument": "B_USD", "expected_direction": "up"},
        ]})
        # Both fell → both disagree with "up".
        prov = FakeProvider({"A_USD": (100.0, 98.0), "B_USD": (100.0, 97.0)})
        res = _run(prov, "t", cfg)
        assert res.confirmed is False
        assert res.score == pytest.approx(0.0)

    def test_weight_leans_on_primary(self) -> None:
        cfg = _cfg({"t": [
            {"instrument": "BIG_USD", "expected_direction": "up", "weight": 3.0},
            {"instrument": "SMALL_USD", "expected_direction": "up", "weight": 1.0},
        ]})
        # BIG agrees, SMALL disagrees → weighted 3/4 = 0.75.
        prov = FakeProvider({"BIG_USD": (100.0, 105.0), "SMALL_USD": (100.0, 99.0)})
        res = _run(prov, "t", cfg)
        assert res.score == pytest.approx(0.75)
        assert res.confirmed is True

    def test_missing_data_excluded_from_vote(self) -> None:
        cfg = _cfg({"t": [
            {"instrument": "HAS_USD", "expected_direction": "up"},
            {"instrument": "GONE_USD", "expected_direction": "up"},
        ]})
        # GONE has no data — must be excluded, not counted as a fail.
        prov = FakeProvider({"HAS_USD": (100.0, 102.0)})
        res = _run(prov, "t", cfg)
        assert res.confirmed is True
        assert res.score == pytest.approx(1.0)  # only HAS voted, and it agreed
        assert res.n_voting == 1
        gone = next(d for d in res.details if d.instrument == "GONE_USD")
        assert gone.agrees is None
        assert gone.actual_move_pct is None

    def test_partial_leg_missing_excluded(self) -> None:
        cfg = _cfg({"t": [{"instrument": "X_USD", "expected_direction": "up"}]})
        # p1 leg is None (no current price) → excluded → unknown.
        prov = FakeProvider({"X_USD": (100.0, None)})
        res = _run(prov, "t", cfg)
        assert res.confirmed is None
        assert res.n_voting == 0

    def test_zero_data_is_unknown_never_blocks(self) -> None:
        cfg = _cfg({"t": [
            {"instrument": "A_USD", "expected_direction": "up"},
            {"instrument": "B_USD", "expected_direction": "up"},
        ]})
        prov = FakeProvider({})  # nothing resolves
        res = _run(prov, "t", cfg)
        assert res.confirmed is None  # UNKNOWN — never blocks
        assert res.score == pytest.approx(0.0)
        assert res.n_voting == 0

    def test_unconfigured_theme_is_unknown(self) -> None:
        cfg = _cfg({"t": [{"instrument": "A_USD", "expected_direction": "up"}]})
        prov = FakeProvider({"A_USD": (100.0, 101.0)})
        res = _run(prov, "no_such_theme", cfg)
        assert res.confirmed is None
        assert res.details == []

    def test_flat_move_does_not_corroborate(self) -> None:
        cfg = _cfg({"t": [{"instrument": "A_USD", "expected_direction": "up"}]})
        # Dead flat: p0 == p1. The related asset ISN'T moving = fade signal.
        prov = FakeProvider({"A_USD": (100.0, 100.0)})
        res = _run(prov, "t", cfg)
        assert res.confirmed is False
        move = res.details[0]
        assert move.agrees is False
        assert move.actual_move_pct == pytest.approx(0.0)

    def test_custom_threshold(self) -> None:
        cfg = _cfg({"t": [
            {"instrument": "A_USD", "expected_direction": "up"},
            {"instrument": "B_USD", "expected_direction": "up"},
            {"instrument": "C_USD", "expected_direction": "up"},
        ]})
        # 2/3 agree. threshold 0.7 → NOT confirmed; 0.6 → confirmed.
        prov = FakeProvider({
            "A_USD": (100.0, 101.0),
            "B_USD": (100.0, 101.0),
            "C_USD": (100.0, 99.0),
        })
        assert _run(prov, "t", cfg, threshold=0.7).confirmed is False
        assert _run(prov, "t", cfg, threshold=0.6).confirmed is True


# ===================================================================== #
# Risk-off sign conventions — the easy-to-get-wrong part
# ===================================================================== #


class TestRiskOffSigns:
    def test_taiwan_yen_strengthening_is_usdjpy_down_agrees(self) -> None:
        """A Taiwan risk-off event: the yen STRENGTHENS, which is USD_JPY
        FALLING. The config encodes USD_JPY expected_direction=down, so a
        falling USD_JPY must AGREE."""
        cfg = load_cross_asset_config()
        # USD_JPY 150 → 148 (yen stronger), XAU up (haven), SPX down (risk-off).
        prov = FakeProvider({
            "USD_JPY": (150.0, 148.0),   # yen strengthened → pair DOWN
            "XAU_USD": (2000.0, 2020.0),  # haven bid UP
            "SPX500_USD": (5000.0, 4950.0),  # risk-off DOWN
        })
        res = _run(prov, "taiwan_semiconductor", cfg)
        jpy = next(d for d in res.details if d.instrument == "USD_JPY")
        assert jpy.expected == "down"
        assert jpy.actual_move_pct < 0  # pair fell
        assert jpy.agrees is True       # falling pair matches "down"
        assert res.confirmed is True
        assert res.score == pytest.approx(1.0)

    def test_taiwan_yen_weakening_is_usdjpy_up_disagrees(self) -> None:
        """Inverse: if USD_JPY RISES (yen weakens), that is NOT the
        risk-off read and must DISAGREE."""
        cfg = load_cross_asset_config()
        prov = FakeProvider({
            "USD_JPY": (150.0, 152.0),  # yen weakened → pair UP → disagrees "down"
            "XAU_USD": (2000.0, 1990.0),  # gold fell → disagrees "up"
            "SPX500_USD": (5000.0, 5050.0),  # equities rose → disagrees "down"
        })
        res = _run(prov, "taiwan_semiconductor", cfg)
        jpy = next(d for d in res.details if d.instrument == "USD_JPY")
        assert jpy.agrees is False
        assert res.confirmed is False
        assert res.score == pytest.approx(0.0)

    def test_energy_petro_fx_usdcad_down_agrees(self) -> None:
        """Energy chokepoint: crude spikes, CAD strengthens → USD_CAD
        DOWN, which the config expects and must AGREE."""
        cfg = load_cross_asset_config()
        prov = FakeProvider({
            "BCO_USD": (80.0, 84.0),     # brent up
            "WTICO_USD": (76.0, 79.0),   # wti up
            "USD_CAD": (1.36, 1.35),     # CAD stronger → pair DOWN
            "USD_NOK": (10.5, 10.3),     # NOK stronger → pair DOWN
        })
        res = _run(prov, "energy_chokepoint", cfg)
        cad = next(d for d in res.details if d.instrument == "USD_CAD")
        assert cad.expected == "down"
        assert cad.agrees is True
        assert res.confirmed is True


# ===================================================================== #
# Config load + validation
# ===================================================================== #


class TestConfigLoad:
    def test_real_config_loads_and_is_shaped(self) -> None:
        cfg = load_cross_asset_config()
        assert cfg.themes  # non-empty
        # Every listed instrument is OANDA-shaped and every direction valid.
        import re

        oanda = re.compile(r"^[A-Z0-9_]+$")
        for theme, checks in cfg.themes.items():
            assert checks, f"{theme} has no checks"
            for c in checks:
                assert oanda.match(c.instrument), (theme, c.instrument)
                assert c.expected_direction in ("up", "down")
                assert c.weight > 0

    def test_real_config_taiwan_yen_is_down(self) -> None:
        """Lock the risk-off sign in the shipped config: taiwan USD_JPY
        MUST be down (yen-strengthening), never up."""
        cfg = load_cross_asset_config()
        jpy = next(
            c for c in cfg.checks_for("taiwan_semiconductor")
            if c.instrument == "USD_JPY"
        )
        assert jpy.expected_direction == "down"

    def test_real_config_covers_required_themes(self) -> None:
        cfg = load_cross_asset_config()
        for theme in (
            "energy_chokepoint", "oil_supply_shock", "red_sea_shipping",
            "taiwan_semiconductor", "russia_ukraine", "drc_copper_cobalt",
            "black_sea_grain", "guinea_iron_bauxite",
        ):
            assert cfg.checks_for(theme), f"missing theme {theme}"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cross_asset_config(tmp_path / "nope.yaml")

    def test_bad_direction_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "themes:\n  t:\n    instruments:\n"
            "      - instrument: BCO_USD\n        expected_direction: sideways\n"
        )
        with pytest.raises(ValueError, match="expected_direction"):
            load_cross_asset_config(p)

    def test_non_oanda_instrument_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "themes:\n  t:\n    instruments:\n"
            "      - instrument: brent.crude\n        expected_direction: up\n"
        )
        with pytest.raises(ValueError, match="OANDA-shaped"):
            load_cross_asset_config(p)

    def test_empty_instruments_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("themes:\n  t:\n    instruments: []\n")
        with pytest.raises(ValueError, match="non-empty"):
            load_cross_asset_config(p)

    def test_negative_weight_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "themes:\n  t:\n    instruments:\n"
            "      - instrument: BCO_USD\n        expected_direction: up\n"
            "        weight: -1\n"
        )
        with pytest.raises(ValueError, match="weight"):
            load_cross_asset_config(p)


# ===================================================================== #
# Render helpers — digest HTML + strategy plain-text
# ===================================================================== #


def _result(
    confirmed: bool | None,
    moves: list[tuple[str, str, float | None, bool | None]],
) -> CrossAssetResult:
    details = [
        InstrumentMove(instrument=i, expected=e, actual_move_pct=m, agrees=a)
        for i, e, m, a in moves
    ]
    voting = [d for d in details if d.agrees is not None]
    n = len(voting)
    score = (sum(1 for d in voting if d.agrees) / n) if n else 0.0
    return CrossAssetResult(confirmed=confirmed, score=score, details=details)


class TestDigestLine:
    def test_confirms_format(self) -> None:
        res = _result(True, [
            ("BCO_USD", "up", 1.8, True),
            ("USD_CAD", "down", -0.2, True),
            ("WTICO_USD", "up", -0.1, False),
        ])
        line = build_cross_asset_line(res)
        assert line is not None
        assert "BCO_USD +1.8% ✓" in line
        assert "USD_CAD -0.2% ✓" in line
        assert "WTICO_USD -0.1% ✗" in line
        assert "confirms (2/3)" in line
        assert "·" in line
        assert line.startswith("<b>Cross-asset:</b>")

    def test_low_score_fade_warning(self) -> None:
        res = _result(False, [
            ("BCO_USD", "up", -1.0, False),
            ("USD_CAD", "down", 0.5, False),
        ])
        line = build_cross_asset_line(res)
        assert line is not None
        assert "NOT confirming — fade risk" in line
        assert "(0/2)" in line

    def test_unknown_omitted(self) -> None:
        res = _result(None, [("BCO_USD", "up", None, None)])
        assert build_cross_asset_line(res) is None

    def test_none_result_omitted(self) -> None:
        assert build_cross_asset_line(None) is None

    def test_html_escaped(self) -> None:
        # A hostile instrument name (defensive — names come from config,
        # but the line is embedded in a Telegram-HTML body).
        res = _result(True, [("A<b>&_USD", "up", 1.0, True)])
        line = build_cross_asset_line(res)
        assert "<b>&" not in line.replace("<b>Cross-asset:</b>", "")
        assert "&lt;b&gt;&amp;_USD" in line


class TestStrategyPlainLine:
    def test_plain_confirms(self) -> None:
        res = _result(True, [
            ("BCO_USD", "up", 1.8, True),
            ("USD_CAD", "down", -0.2, True),
        ])
        line = EventDrivenStrategy._cross_asset_line(res)
        assert line == "Cross-asset: BCO_USD +1.8% ✓ · USD_CAD -0.2% ✓ · confirms (2/2)"
        assert "<b>" not in line  # plain text, no markup

    def test_plain_fade(self) -> None:
        res = _result(False, [("BCO_USD", "up", -1.0, False)])
        line = EventDrivenStrategy._cross_asset_line(res)
        assert "NOT confirming — fade risk (0/1)" in line

    def test_plain_unknown_omitted(self) -> None:
        assert EventDrivenStrategy._cross_asset_line(
            _result(None, [("BCO_USD", "up", None, None)]),
        ) is None
        assert EventDrivenStrategy._cross_asset_line(None) is None

    def test_notes_summary(self) -> None:
        res = _result(True, [
            ("BCO_USD", "up", 1.8, True),
            ("USD_CAD", "down", -0.2, True),
        ])
        note = EventDrivenStrategy._cross_asset_summary(res)
        assert note == "cross-asset: confirms 2/2 (BCO_USD +1.8%, USD_CAD -0.2%)"

    def test_notes_summary_fade(self) -> None:
        res = _result(False, [("BCO_USD", "up", -1.0, False)])
        note = EventDrivenStrategy._cross_asset_summary(res)
        assert "NOT confirming (fade risk) 0/1" in note

    def test_notes_summary_unknown(self) -> None:
        assert EventDrivenStrategy._cross_asset_summary(
            _result(None, [("BCO_USD", "up", None, None)]),
        ) is None
