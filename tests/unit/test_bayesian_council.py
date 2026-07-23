"""Tests for the Bayesian modeling council orchestration (CL-l0x).

Mocks the LLM agents (modeler + fit_evaluator) so we don't burn API
budget. The fit-runner tool runs for real on a tiny synthetic panel.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")

from src.research.agents.bayesian_council import (  # noqa: E402
    run_bayesian_council,
)


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    for p, beta in [("EURUSD", 0.5), ("GBPUSD", 0.6)]:
        spread = rng.uniform(-1, 1, 100)
        ret = beta * spread * 0.001 + rng.normal(0, 0.005, 100)
        for s, r in zip(spread, ret, strict=False):
            rows.append({"pair": p, "rate_diff": s, "log_return": r})
    return pd.DataFrame(rows)


_PANEL_SCHEMA = {"pair": "str", "rate_diff": "float", "log_return": "float"}

_VALID_SPEC = {
    "model_family": "hierarchical_linear",
    "rationale": "FX-rate diff partial pooling",
    "outcome": {"variable": "log_return", "type": "continuous"},
    "predictors": [
        {
            "variable": "rate_diff",
            "role": "fixed_effect",
            "prior": {"family": "Normal", "mu": 0.0, "sigma": 1.0},
        }
    ],
    "hierarchy": {
        "grouping": "pair",
        "varying_intercept": True,
        "varying_slopes": ["rate_diff"],
        "pooling": "partial",
    },
    "noise": {"family": "Normal", "scale_prior": {"family": "HalfNormal", "sigma": 0.05}},
    "fit_settings": {"draws": 200, "tune": 200, "chains": 2, "target_accept": 0.95},
    "diagnostic_thresholds": {
        "rhat_max": 1.05,
        "ess_min_per_chain": 50,
        "divergence_rate_max": 0.05,
    },
    "expected_runtime_sec": 30,
    "expected_post_outputs": ["per_pair_beta_posterior"],
}


def _stub_agent(text: str) -> MagicMock:
    """Build a stub agent that returns ``text`` from .run()."""
    mock = MagicMock()
    resp = MagicMock()
    resp.agent_name = "stub"
    resp.role = "stub"
    resp.text = text
    resp.model = "fake-model"
    resp.provider = "fake"
    resp.input_tokens = 100
    resp.output_tokens = 100
    resp.usd_cost = 0.001
    resp.elapsed_sec = 0.1
    mock.run.return_value = resp
    return mock


class TestShipPath:
    def test_ship_verdict_short_circuits_loop(self) -> None:
        modeler = _stub_agent(json.dumps(_VALID_SPEC))
        evaluator = _stub_agent(
            json.dumps(
                {
                    "verdict": "ship",
                    "verdict_reason": "All diagnostics pass",
                    "concerns": [],
                    "calibration": {
                        "posterior_predictive_coverage_95": 0.92,
                        "log_likelihood_oos": -100.0,
                    },
                    "follow_up_actions": [],
                }
            )
        )
        outcome = run_bayesian_council(
            hypothesis="Rate diff predicts FX returns hierarchically across G10",
            panel=_panel(),
            panel_schema=_PANEL_SCHEMA,
            modeler=modeler,
            fit_evaluator=evaluator,
        )
        assert outcome.verdict == "ship"
        assert outcome.iterations == 1
        # Modeler called once (no iteration), evaluator called once.
        assert modeler.run.call_count == 1
        assert evaluator.run.call_count == 1


class TestIteratePath:
    def test_iterate_re_runs_modeler_with_concerns(self) -> None:
        modeler = _stub_agent(json.dumps(_VALID_SPEC))
        # First call returns "iterate", second returns "ship"
        eval_responses = [
            json.dumps(
                {
                    "verdict": "iterate",
                    "verdict_reason": "R-hat borderline",
                    "concerns": [
                        {
                            "severity": "warn",
                            "category": "convergence",
                            "detail": "R-hat 1.02",
                            "suggested_fix": "Raise tune to 1000",
                        }
                    ],
                    "calibration": {
                        "posterior_predictive_coverage_95": 0.85,
                        "log_likelihood_oos": -110.0,
                    },
                    "follow_up_actions": ["raise tune"],
                }
            ),
            json.dumps(
                {
                    "verdict": "ship",
                    "verdict_reason": "Now passes after tune raise",
                    "concerns": [],
                    "calibration": {
                        "posterior_predictive_coverage_95": 0.93,
                        "log_likelihood_oos": -100.0,
                    },
                    "follow_up_actions": [],
                }
            ),
        ]
        eval_mock = MagicMock()
        eval_mock.run.side_effect = lambda *_a, **_k: _wrap(eval_responses.pop(0))

        outcome = run_bayesian_council(
            hypothesis="x",
            panel=_panel(),
            panel_schema=_PANEL_SCHEMA,
            modeler=modeler,
            fit_evaluator=eval_mock,
            max_iterations=3,
        )
        assert outcome.verdict == "ship"
        assert outcome.iterations == 2
        # Second modeler call should have received the concerns.
        second_call_input = modeler.run.call_args_list[1].args[0]
        assert "previous_evaluator_concerns" in second_call_input


def _wrap(text: str) -> MagicMock:
    """Wrap a string in a fake AgentResponse for side_effect lambdas."""
    resp = MagicMock()
    resp.agent_name = "stub"
    resp.role = "stub"
    resp.text = text
    resp.model = "fake"
    resp.provider = "fake"
    resp.input_tokens = 100
    resp.output_tokens = 100
    resp.usd_cost = 0.001
    resp.elapsed_sec = 0.1
    return resp


class TestRejectPath:
    def test_modeler_returns_unparseable_aborts(self) -> None:
        modeler = _stub_agent("not even json {{{")
        evaluator = _stub_agent(json.dumps({"verdict": "ship"}))
        outcome = run_bayesian_council(
            hypothesis="x",
            panel=_panel(),
            panel_schema=_PANEL_SCHEMA,
            modeler=modeler,
            fit_evaluator=evaluator,
        )
        assert outcome.verdict == "reject"
        # Evaluator should never have been called — we bailed at the
        # modeler stage.
        assert evaluator.run.call_count == 0

    def test_max_iterations_with_iterate_becomes_reject(self) -> None:
        modeler = _stub_agent(json.dumps(_VALID_SPEC))
        # Always iterate — never converges.
        evaluator = _stub_agent(
            json.dumps(
                {
                    "verdict": "iterate",
                    "verdict_reason": "Still concerning",
                    "concerns": [
                        {
                            "severity": "warn",
                            "category": "convergence",
                            "detail": "R-hat",
                            "suggested_fix": "raise tune",
                        }
                    ],
                    "calibration": {
                        "posterior_predictive_coverage_95": 0.8,
                        "log_likelihood_oos": -110.0,
                    },
                    "follow_up_actions": [],
                }
            )
        )
        outcome = run_bayesian_council(
            hypothesis="x",
            panel=_panel(),
            panel_schema=_PANEL_SCHEMA,
            modeler=modeler,
            fit_evaluator=evaluator,
            max_iterations=2,
        )
        assert outcome.verdict == "reject"
        assert outcome.iterations == 2
        assert "Exceeded max_iterations" in outcome.final_evaluator_response["verdict_reason"]


class TestTranscript:
    def test_transcript_records_all_steps(self) -> None:
        modeler = _stub_agent(json.dumps(_VALID_SPEC))
        evaluator = _stub_agent(json.dumps({"verdict": "ship", "verdict_reason": "ok"}))
        outcome = run_bayesian_council(
            hypothesis="x",
            panel=_panel(),
            panel_schema=_PANEL_SCHEMA,
            modeler=modeler,
            fit_evaluator=evaluator,
        )
        # 3 entries per iteration (modeler, tool, evaluator); 1 iteration → 3.
        assert len(outcome.transcript) == 3
        assert outcome.transcript[0]["agent"] == "modeler"
        assert outcome.transcript[1]["tool"] == "bayesian_rate_diff_fit"
        assert outcome.transcript[2]["agent"] == "fit_evaluator"
