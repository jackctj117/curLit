"""CL-7ys8: unknown/proxy evidence cannot satisfy a research gate."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.research.agents.reviewer import Position
from src.research.backtest_runner import (
    _compute_decay_severity,
    _compute_edge_concentration,
    _compute_regime_diversified,
    _read_execution_symbol,
    _read_symbols,
    make_backtest_runner,
)
from src.research.verdict import ParsedRule, Verdict, compute_verdict, evaluate_threshold


@pytest.mark.parametrize(
    "values", [[], [1.0], [0.0] * 9, [1.0, float("nan")] * 4, [float("inf")] * 8]
)
def test_unavailable_fold_diagnostics_remain_unknown(values: list[float]) -> None:
    folds = pd.DataFrame({"test_sharpe": values})
    assert _compute_edge_concentration(folds) is None
    assert _compute_regime_diversified(folds) is None
    assert _compute_decay_severity(folds) is None


def test_missing_fold_columns_are_unknown() -> None:
    for calculate in (
        _compute_edge_concentration,
        _compute_regime_diversified,
        _compute_decay_severity,
    ):
        assert calculate(pd.DataFrame({"unrelated": [1] * 9})) is None


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf"), True, "1.0"])
def test_non_numeric_or_nonfinite_evidence_cannot_pass_inequality(value: Any) -> None:
    passed, missing, _ = evaluate_threshold("sharpe != 0", {"sharpe": value})
    assert not passed
    assert missing


@pytest.mark.parametrize("value", [None, "unknown", "invalid_output", float("nan")])
def test_unknown_decay_cannot_pass_negative_membership(value: Any) -> None:
    passed, missing, _ = evaluate_threshold(
        "decay_severity not in ['STRONG', 'MODERATE']",
        {"decay_severity": value},
    )
    assert not passed
    assert missing


def test_proxy_provenance_overrides_favorable_value() -> None:
    report = {
        "edge_concentration": 0.25,
        "_metrics_provenance": {
            "edge_concentration": "PROXY: temporal folds, not feature attribution"
        },
    }
    result = compute_verdict(
        report,
        [ParsedRule("B.1", "edge", "edge_concentration <= 0.50")],
        Position.PROMOTE,
        Position.PROMOTE,
    )
    assert result.verdict == Verdict.ESCALATE
    assert result.rule_evaluations[0].missing
    assert not result.rule_evaluations[0].passed


@pytest.mark.parametrize("status", ["proxy", "insufficient"])
def test_structured_evidence_survives_serialization(status: str) -> None:
    report = {
        "edge_concentration": 0.25,
        "_metric_evidence": {
            "edge_concentration": {
                "status": status,
                "observation_count": 6,
                "method": "fold_proxy",
                "reason": "not measured feature attribution",
            }
        },
    }
    restored = json.loads(json.dumps(report, allow_nan=False))
    passed, missing, value = evaluate_threshold("edge_concentration <= 0.50", restored)
    assert (passed, missing, value) == (False, True, 0.25)


def test_measured_evidence_can_pass_or_fail_without_becoming_unknown() -> None:
    for value, expected in [(0.25, True), (0.75, False)]:
        report = {
            "edge_concentration": value,
            "_metric_evidence": {
                "edge_concentration": {
                    "status": "measured",
                    "observation_count": 100,
                    "method": "fixture_feature_attribution",
                    "reason": "",
                }
            },
        }
        passed, missing, _ = evaluate_threshold("edge_concentration <= 0.50", report)
        assert passed is expected
        assert not missing


@given(st.lists(st.integers(min_value=1, max_value=1000), min_size=2, max_size=20))
def test_edge_proxy_matches_independent_integer_ratio(values: list[int]) -> None:
    measured = _compute_edge_concentration(pd.DataFrame({"test_sharpe": values}))
    assert measured == pytest.approx(max(values) / sum(values))
    assert measured is not None and 0 < measured <= 1


@pytest.mark.parametrize("symbols", [
    None, [], "EURUSD", {"EURUSD"}, [None], [True], [""], ["eurusd"],
    ["EURUSD "], ["EURUSD", "EURUSD"], ["EUR USD"],
])
def test_malformed_symbols_never_become_a_default_pair(symbols: Any) -> None:
    candidate = type("Candidate", (), {"symbols": symbols})
    with pytest.raises(ValueError):
        _read_symbols(candidate)


def test_failed_constructor_cannot_fall_back_to_eurusd() -> None:
    class Candidate:
        def __init__(self) -> None:
            raise RuntimeError("fixture constructor failure")

    with pytest.raises(ValueError, match="construction failed") as error:
        _read_symbols(Candidate)
    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize("symbol", ["DXY", "US_2Y", "IMAGINARY", None, ""])
def test_unsupported_execution_is_not_data_availability(symbol: Any) -> None:
    candidate = type("Candidate", (), {"execution_symbol": symbol})
    with pytest.raises(ValueError):
        _read_execution_symbol(candidate, [symbol])


def test_single_execution_and_cross_symbol_inputs_are_explicit() -> None:
    candidate = type("Candidate", (), {"symbols": ["USDJPY"]})
    assert _read_execution_symbol(candidate, _read_symbols(candidate)) == "USDJPY"
    candidate = type("Candidate", (), {"symbols": ["DXY", "USDJPY"], "execution_symbol": "USDJPY"})
    assert _read_execution_symbol(candidate, _read_symbols(candidate)) == "USDJPY"


def test_unknown_execution_is_rejected_before_data_fetch(tmp_path: Path) -> None:
    path = tmp_path / "candidate.py"
    path.write_text(
        "class Candidate:\n    symbols = ['IMAGINARY']\n"
        "    def fit(self, data): pass\n    def generate_signals(self, data): pass\n"
    )
    provider = Mock()
    with pytest.raises(ValueError, match="supported trading instrument"):
        make_backtest_runner(provider)(path)
    provider.get_aligned_series.assert_not_called()
