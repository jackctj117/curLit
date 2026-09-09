"""Trading startup rejects missing/malformed safety values, without echoing them."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.risk.env_config import env_bool, env_float, env_int
from src.risk.risk_profile import _build_profile, load_active_profile


@pytest.mark.parametrize(
    "section", ["sizing", "kill_switches", "strategy_gates", "holding", "bias"]
)
@pytest.mark.parametrize("invalid", [None, [], False, "wrong", {"typo": 1}])
def test_unknown_or_malformed_sections_reject(section, invalid):
    with pytest.raises(ValueError):
        _build_profile("fixture", {section: invalid})


@pytest.mark.parametrize(
    "field,value",
    [
        ("kelly_fraction", True),
        ("kelly_fraction", "0.25"),
        ("kelly_fraction", float("nan")),
        ("max_position_pct", -0.1),
        ("max_position_pct", 1.01),
    ],
)
def test_fraction_types_and_ranges(field, value):
    with pytest.raises(ValueError):
        _build_profile("fixture", {"sizing": {field: value}})


@given(st.floats(allow_nan=True, allow_infinity=True))
def test_fraction_acceptance_matches_mathematical_unit_interval(value):
    if math.isfinite(value) and 0 <= value <= 1:
        assert (
            _build_profile("fixture", {"sizing": {"kelly_fraction": value}}).sizing.kelly_fraction
            == value
        )
    else:
        with pytest.raises(ValueError):
            _build_profile("fixture", {"sizing": {"kelly_fraction": value}})


@pytest.mark.parametrize(
    "body",
    [
        "active: a\nprofiles: {a: {inherits: a}}",
        "active: a\nprofiles: {a: {inherits: b}, b: {inherits: a}}",
        "active: a\nprofiles: {a: {inherits: null}}",
        "active: a\nprofiles: {a: {}}\nunknown: 1",
        "active: a\nprofiles: {a: {sizing: {kelly_fraction: .2, kelly_fraction: .3}}}",
        "active: a\nprofiles: {a: {}, b: {sizing: {typo: .2}}}",
        "active: a\nprofiles: {a: {holding: {max_holding_days: 1.5}}}",
        "active: a\nprofiles: {a: {kill_switches: {daily_loss_limit_pct: .03}}}",
        "active: a\nprofiles: {a: {bias: {prefer_polymarket_no: 'false'}}}",
    ],
)
def test_profile_structure_inheritance_and_duplicate_keys(tmp_path, body):
    path = tmp_path / "risk.yaml"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_active_profile(path)


@pytest.mark.parametrize(
    "parser,bad",
    [
        (env_float, "nan"),
        (env_float, "inf"),
        (env_float, "secret-example"),
        (env_int, "1.5"),
        (env_bool, "ture"),
        (env_bool, ""),
    ],
)
def test_invalid_env_refuses_without_echoing_value(monkeypatch, parser, bad):
    monkeypatch.setenv("FIXTURE_SETTING", bad)
    with pytest.raises(ValueError) as err:
        parser("FIXTURE_SETTING", 1)
    assert "FIXTURE_SETTING" in str(err.value)
    if bad:
        assert bad not in str(err.value)


@pytest.mark.parametrize(
    "raw,expected", [("true", True), ("1", True), ("no", False), ("off", False)]
)
def test_documented_boolean_literals(monkeypatch, raw, expected):
    monkeypatch.setenv("FIXTURE_SETTING", raw)
    assert env_bool("FIXTURE_SETTING", not expected) is expected


def test_explicit_unknown_environment_never_uses_dev_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("CURLIT_RISK_PROFILE", "missing")
    with pytest.raises(ValueError):
        load_active_profile(tmp_path / "absent", allow_development_defaults=True)


@pytest.mark.parametrize("book", ["options", "equities"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("min_confidence", 1.1),
        ("entry_delay_min", -1),
        ("min_alignment", float("nan")),
        ("require_red_team", "false"),
    ],
)
def test_execution_configs_reject_invalid_direct_inputs(book, field, value):
    from src.execution.alpaca_equity_executor import EquityExecConfig
    from src.execution.alpaca_options_executor import OptionsExecConfig

    cls = OptionsExecConfig if book == "options" else EquityExecConfig
    with pytest.raises(ValueError):
        cls(**{field: value})


def test_documented_execution_sentinels_and_contract_ranges():
    from src.execution.alpaca_options import ContractSelectionConfig
    from src.execution.alpaca_options_executor import OptionsExecConfig

    cfg = OptionsExecConfig(
        min_alignment=-1.01,
        entry_delay_override_conf=1.01,
        max_entry_spread_pct=1.01,
        entry_delay_min=0,
    )
    assert cfg.min_alignment == -1.01
    for kwargs in (
        {"default_dte_days": 0},
        {"strike_window": float("inf")},
        {"default_moneyness": True},
    ):
        with pytest.raises(ValueError):
            ContractSelectionConfig(**kwargs)


@pytest.mark.parametrize("book", ["options", "equities"])
def test_invalid_daemon_env_never_reaches_a_broker_client(monkeypatch, book):
    import importlib

    daemon = importlib.import_module("scripts.execute_" + book)
    prefix = "OPT" if book == "options" else "EQ"
    monkeypatch.setenv(f"ALPACA_{prefix}_MIN_CONFIDENCE", "nan")
    with pytest.raises(ValueError, match="startup refused"):
        daemon._config_from_env()
