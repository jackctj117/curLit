"""Bayesian modeling council (CL-l0x).

Two-agent + tool orchestration:

  Modeler        → JSON spec (from research_prompts/modeler.md)
  bayesian_rate_diff_fit (tool, not an agent) → posterior + diagnostics
  Fit Evaluator  → JSON verdict (from research_prompts/fit_evaluator.md)

Iterates up to ``max_iterations`` on an "iterate" verdict. Returns the
final ship/iterate/reject outcome plus the full transcript so the
orchestrator can persist it for audit.

This council does NOT live inside the bull/bear debate. It sits next
to it as a parallel pipeline that strategies opt into when they need
calibrated uncertainty (rate_diff hierarchical, GP fair value, regime-
state inference). The promotion review (bull/bear) remains the gate
on whether *any* strategy ships; this council is the gate on whether
its *modeling is rigorous enough* to ship.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.research.agents.base import Agent, AgentResponse

logger = logging.getLogger(__name__)


# Hard cap on iterations. After 3 the council should escalate the
# model_family choice rather than keep tweaking priors. The
# fit_evaluator prompt internalizes this rule but we enforce it in
# code as a safety net.
_DEFAULT_MAX_ITERATIONS: int = 3


@dataclass
class BayesianCouncilOutcome:
    verdict: str  # "ship" | "iterate" | "reject"
    final_spec: dict[str, Any]  # the spec that produced the final fit
    final_diagnostics: dict[str, Any]  # bayesian_rate_diff.diagnose() output
    final_evaluator_response: dict[str, Any]  # fit_evaluator's parsed JSON
    iterations: int
    transcript: list[dict[str, Any]] = field(default_factory=list)


def _try_parse_json(text: str) -> dict[str, Any]:
    """Lenient JSON parse — strips a leading/trailing ```json fence if
    the LLM wrapped its output despite the prompt asking it not to."""
    body = text.strip()
    if body.startswith("```"):
        # Strip ```json or ``` opening + closing fence
        body = body.split("\n", 1)[1] if "\n" in body else body
        if body.endswith("```"):
            body = body.rsplit("```", 1)[0]
    parsed: dict[str, Any] = json.loads(body)
    return parsed


def run_bayesian_council(
    hypothesis: str,
    panel: pd.DataFrame,
    panel_schema: dict[str, str],
    modeler: Agent,
    fit_evaluator: Agent,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> BayesianCouncilOutcome:
    """Run the modeler → fit → evaluator loop.

    ``panel_schema`` is ``{column_name: dtype_str}``. Lets the Modeler
    cite real columns instead of hallucinating.

    The fit tool (``bayesian_rate_diff.fit``) is fixed in v1 — the
    Modeler's spec is read mostly for the diagnostic_thresholds and
    fit_settings fields; the model structure itself is the
    hierarchical-linear shape from CL-l0x. v2 (filed) generalizes the
    fit tool to dispatch on ``model_family``.
    """
    from src.models.bayesian_rate_diff import BayesianRateDiffConfig, diagnose, fit

    transcript: list[dict[str, Any]] = []
    last_spec: dict[str, Any] = {}
    last_evaluator_response: dict[str, Any] = {}
    last_diagnostics: dict[str, Any] = {}
    last_verdict = "reject"

    iteration = 0
    spec_iteration_input = (
        f"<hypothesis>\n{hypothesis}\n</hypothesis>\n\n"
        f"<panel_schema>\n{json.dumps(panel_schema, indent=2)}\n</panel_schema>"
    )

    while iteration < max_iterations:
        iteration += 1

        # --- Modeler produces a spec --------------------------------
        modeler_resp = modeler.run(spec_iteration_input)
        transcript.append(
            {
                "iteration": iteration,
                "agent": "modeler",
                "response": _agent_resp_dict(modeler_resp),
            }
        )
        try:
            spec = _try_parse_json(modeler_resp.text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Modeler iteration %d returned non-JSON; aborting council",
                iteration,
            )
            transcript.append(
                {
                    "iteration": iteration,
                    "agent": "modeler",
                    "error": f"json parse: {exc}",
                }
            )
            return BayesianCouncilOutcome(
                verdict="reject",
                final_spec={},
                final_diagnostics={},
                final_evaluator_response={
                    "verdict": "reject",
                    "verdict_reason": "Modeler did not produce parseable JSON",
                },
                iterations=iteration,
                transcript=transcript,
            )
        last_spec = spec

        # --- Fit-runner tool ----------------------------------------
        fit_settings = spec.get("fit_settings") or {}
        cfg = BayesianRateDiffConfig(
            draws=int(fit_settings.get("draws", 2000)),
            tune=int(fit_settings.get("tune", 1000)),
            chains=int(fit_settings.get("chains", 4)),
            target_accept=float(fit_settings.get("target_accept", 0.95)),
        )
        try:
            fit_result = fit(panel, cfg)
        except Exception as exc:
            logger.warning(
                "Fit-runner failed on iteration %d: %s: %s",
                iteration,
                type(exc).__name__,
                exc,
            )
            transcript.append(
                {
                    "iteration": iteration,
                    "tool": "bayesian_rate_diff_fit",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return BayesianCouncilOutcome(
                verdict="reject",
                final_spec=spec,
                final_diagnostics={"error": str(exc)},
                final_evaluator_response={
                    "verdict": "reject",
                    "verdict_reason": f"Fit failed: {exc}",
                },
                iterations=iteration,
                transcript=transcript,
            )
        diagnostics = diagnose(fit_result)
        last_diagnostics = diagnostics

        transcript.append(
            {
                "iteration": iteration,
                "tool": "bayesian_rate_diff_fit",
                "diagnostics": diagnostics,
                "elapsed_sec": fit_result.elapsed_sec,
                "n_draws": fit_result.n_draws,
                "n_chains": fit_result.n_chains,
            }
        )

        # --- Fit evaluator ------------------------------------------
        eval_input = (
            f"<spec>\n{json.dumps(spec, indent=2)}\n</spec>\n\n"
            f"<diagnostics>\n{json.dumps(diagnostics, indent=2)}\n</diagnostics>\n\n"
            f"<posterior_summary>\n"
            f"beta:\n{fit_result.beta_summary.to_string()}\n\n"
            f"alpha:\n{fit_result.alpha_summary.to_string()}\n"
            f"</posterior_summary>"
        )
        evaluator_resp = fit_evaluator.run(eval_input)
        transcript.append(
            {
                "iteration": iteration,
                "agent": "fit_evaluator",
                "response": _agent_resp_dict(evaluator_resp),
            }
        )
        try:
            ev = _try_parse_json(evaluator_resp.text)
        except json.JSONDecodeError:
            ev = {
                "verdict": "reject",
                "verdict_reason": "Evaluator did not produce parseable JSON",
            }
        last_evaluator_response = ev

        verdict = str(ev.get("verdict", "reject")).lower()
        last_verdict = verdict

        if verdict == "ship":
            break
        if verdict == "reject":
            break
        # iterate: feed the evaluator's concerns back into the modeler
        spec_iteration_input = (
            f"<hypothesis>\n{hypothesis}\n</hypothesis>\n\n"
            f"<panel_schema>\n{json.dumps(panel_schema, indent=2)}\n</panel_schema>\n\n"
            f"<previous_spec>\n{json.dumps(spec, indent=2)}\n</previous_spec>\n\n"
            f"<previous_evaluator_concerns>\n{json.dumps(ev.get('concerns', []), indent=2)}\n"
            f"</previous_evaluator_concerns>\n\n"
            "Revise the spec to address the concerns above."
        )

    # Loop exited — either due to ship/reject or max_iterations.
    if iteration >= max_iterations and last_verdict == "iterate":
        # Hit the iteration cap — escalate to reject so we don't ship a
        # model that the evaluator never blessed.
        last_evaluator_response = {
            **last_evaluator_response,
            "verdict": "reject",
            "verdict_reason": (
                f"Exceeded max_iterations={max_iterations} without converging on a shippable spec"
            ),
        }
        last_verdict = "reject"

    return BayesianCouncilOutcome(
        verdict=last_verdict,
        final_spec=last_spec,
        final_diagnostics=last_diagnostics,
        final_evaluator_response=last_evaluator_response,
        iterations=iteration,
        transcript=transcript,
    )


def _agent_resp_dict(resp: AgentResponse) -> dict[str, Any]:
    """Trimmed AgentResponse → JSON-serializable dict for the transcript."""
    return {
        "agent_name": resp.agent_name,
        "role": resp.role,
        "text": resp.text,
        "model": resp.model,
        "provider": resp.provider,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "usd_cost": resp.usd_cost,
        "elapsed_sec": resp.elapsed_sec,
    }
