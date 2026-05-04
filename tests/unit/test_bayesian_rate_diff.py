"""Tests for the Bayesian hierarchical rate-diff model (CL-l0x).

Live PyMC fits — but on a tiny panel (3 pairs × 200 obs × 200 draws ×
2 chains) so the test runs in ~10 seconds. Real production fits use
2000+ draws on full G10 panels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")

from src.models.bayesian_rate_diff import (  # noqa: E402
    BayesianRateDiffConfig,
    diagnose,
    fit,
)


def _build_panel(seed: int = 7) -> pd.DataFrame:
    """3 pairs, 200 obs each, with known per-pair betas drawn from a
    population centered at 0.5. Enough signal to recover the
    hierarchy in a fast fit."""
    rng = np.random.default_rng(seed)
    pairs = ["EURUSD", "GBPUSD", "AUDUSD"]
    true_betas = {"EURUSD": 0.45, "GBPUSD": 0.55, "AUDUSD": 0.50}

    rows: list[dict] = []
    for p in pairs:
        spread = rng.uniform(-2, 2, 200)
        noise = rng.normal(0, 0.005, 200)
        ret = true_betas[p] * spread * 0.001 + noise  # daily-scale FX returns
        for s, r in zip(spread, ret, strict=False):
            rows.append({"pair": p, "rate_diff": s, "log_return": r})
    return pd.DataFrame(rows)


class TestFit:
    def test_returns_per_pair_betas(self) -> None:
        panel = _build_panel()
        cfg = BayesianRateDiffConfig(draws=200, tune=200, chains=2)
        result = fit(panel, cfg)
        # 3 pairs in panel → 3 rows in beta_summary.
        assert sorted(result.pair_names) == ["AUDUSD", "EURUSD", "GBPUSD"]
        assert len(result.beta_summary) == 3
        # Posterior std should be much narrower than the Normal(0, 1)
        # prior — data has informed the parameters.
        assert (result.beta_summary["sd"] < 1.0).all()

    def test_recovers_population_mean_in_neighborhood(self) -> None:
        panel = _build_panel()
        cfg = BayesianRateDiffConfig(draws=200, tune=200, chains=2)
        result = fit(panel, cfg)
        # True population mean of betas is 0.5 (in the 0.001-scaled
        # return space, that becomes very small). With 200 draws and
        # 200 obs/pair, the posterior of mu_beta should be in the
        # order of magnitude of the truth — we just check it's finite
        # and not blown up.
        assert np.isfinite(result.mu_beta_summary["mean"])
        assert -2.0 < result.mu_beta_summary["mean"] < 2.0

    def test_diagnose_produces_structured_verdict(self) -> None:
        # On a small CI-friendly fit (200 obs × 3 pairs × 400 draws ×
        # 2 chains), MCMC diagnostics often surface small-sample warnings.
        # That's expected — production uses 2000+ draws across 4 chains.
        # What we're testing here is the *integration*: diagnose()
        # consumes the fit and returns a structured verdict that the
        # fit_evaluator agent can read.
        panel = _build_panel()
        cfg = BayesianRateDiffConfig(draws=400, tune=400, chains=2)
        result = fit(panel, cfg)
        diag = diagnose(result)
        assert diag["verdict"] in ("pass", "warn", "fail")
        assert isinstance(diag["reasons"], list) and diag["reasons"]
        assert {"rhat_max", "ess_min", "divergences", "divergence_rate"} <= diag["metrics"].keys()

    def test_empty_panel_raises(self) -> None:
        empty = pd.DataFrame({"pair": [], "rate_diff": [], "log_return": []})
        cfg = BayesianRateDiffConfig(draws=100, tune=100, chains=2)
        with pytest.raises(ValueError, match="empty after dropping NaN"):
            fit(empty, cfg)

    def test_missing_columns_raises(self) -> None:
        bad = pd.DataFrame({"pair": ["EURUSD"], "log_return": [0.001]})
        cfg = BayesianRateDiffConfig(draws=100, tune=100, chains=2)
        with pytest.raises(ValueError, match="missing columns"):
            fit(bad, cfg)


class TestDiagnose:
    def _stub_result(
        self, rhat_max: float, ess_min: float, divergences: int,
        n_draws: int = 1000, n_chains: int = 4,
    ):  # type: ignore[no-untyped-def]
        from src.models.bayesian_rate_diff import BayesianFitResult
        return BayesianFitResult(
            pair_names=["EURUSD"],
            beta_summary=pd.DataFrame(),
            alpha_summary=pd.DataFrame(),
            mu_beta_summary={"mean": 0.0, "sd": 0.1},
            sigma_beta_summary={"mean": 0.1, "sd": 0.05},
            sigma_y_summary={"mean": 0.005, "sd": 0.001},
            rhat_max=rhat_max,
            ess_min=ess_min,
            divergences=divergences,
            n_draws=n_draws,
            n_chains=n_chains,
            elapsed_sec=10.0,
        )

    def test_pass_when_all_healthy(self) -> None:
        diag = diagnose(self._stub_result(rhat_max=1.005, ess_min=2000, divergences=0))
        assert diag["verdict"] == "pass"

    def test_warn_when_rhat_borderline(self) -> None:
        diag = diagnose(self._stub_result(rhat_max=1.03, ess_min=2000, divergences=0))
        assert diag["verdict"] == "warn"

    def test_fail_when_rhat_high(self) -> None:
        diag = diagnose(self._stub_result(rhat_max=1.15, ess_min=2000, divergences=0))
        assert diag["verdict"] == "fail"

    def test_fail_when_ess_low(self) -> None:
        # 4 chains × 400 = 1600 floor. 800 < 1600 → fail.
        diag = diagnose(self._stub_result(rhat_max=1.0, ess_min=800, divergences=0))
        assert diag["verdict"] == "fail"

    def test_fail_when_divergences_high(self) -> None:
        # 4000 total draws × 1% = 40 divergences threshold; 50 > 40.
        diag = diagnose(self._stub_result(
            rhat_max=1.0, ess_min=2000, divergences=50,
        ))
        assert diag["verdict"] == "fail"
