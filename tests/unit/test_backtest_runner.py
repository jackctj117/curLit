"""Unit tests for the research backtest_runner (CL-p9ix).

Doesn't hit Postgres or real data — uses a fake DataProvider that
returns a synthetic OHLCV DataFrame. Tests cover:

  * import of a generated strategy file (success + failure modes)
  * symbol extraction from a class-level attribute
  * end-to-end run produces a metrics dict whose top-level keys match
    every threshold path in REVIEW_RULES.md
  * crude B-rule proxies (edge_concentration, regime_diversified,
    decay_severity) compute correctly from synthetic fold metrics
  * raises on data-fetch failure (turned into REJECTED by Implementer)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.backtest_runner import (
    _compute_decay_severity,
    _compute_edge_concentration,
    _compute_regime_diversified,
    _find_strategy_class,
    _import_strategy_module,
    _read_symbols,
    make_backtest_runner,
)

# ---------------------------------------------------------------------- #
# Fake DataProvider
# ---------------------------------------------------------------------- #


@dataclass
class _FakeProvider:
    """Stand-in for DataProvider.get_aligned_series. Returns whatever
    DataFrame the fixture sets in ``data``."""

    data: pd.DataFrame

    def get_aligned_series(
        self,
        symbols: list[str],  # noqa: ARG002
        start: object,  # noqa: ARG002
        end: object,  # noqa: ARG002
    ) -> pd.DataFrame:
        return self.data


def _synth_ohlcv(
    n: int = 1500,
    seed: int = 42,
    symbol: str = "EURUSD",
) -> pd.DataFrame:
    """Random-walk close prices with a daily index, returned as a wide
    DataFrame with one column named ``symbol`` (matches what
    DataProvider.get_aligned_series produces). Walk-forward needs at
    least cfg.min_history=756 rows + an OOS window."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame({symbol: close}, index=idx)


def _write_strategy(tmp_path: Path, body: str, name: str = "stub") -> Path:
    p = tmp_path / f"{name}.py"
    p.write_text(body)
    return p


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


_VALID_STRATEGY = """
import pandas as pd

class StubStrategy:
    symbols = ['EURUSD']
    id = 'stub'

    def fit(self, train_data):
        # Trivial mean-reversion: buy when close is below recent mean
        self._mean = float(train_data['close'].rolling(20).mean().iloc[-1])

    def generate_signals(self, test_data):
        return (test_data['close'] < self._mean).astype(float) - 0.5
"""


_BROKEN_IMPORT = """
import nonexistent_module
class Broken:
    pass
"""


_NO_STRATEGY_PROTOCOL = """
class JustData:
    foo = 1
"""


# ---------------------------------------------------------------------- #
# Module imports
# ---------------------------------------------------------------------- #


class TestImportStrategy:
    def test_imports_valid_file(self, tmp_path: Path) -> None:
        path = _write_strategy(tmp_path, _VALID_STRATEGY)
        module = _import_strategy_module(path)
        assert hasattr(module, "StubStrategy")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _import_strategy_module(tmp_path / "ghost.py")

    def test_broken_import_propagates(self, tmp_path: Path) -> None:
        path = _write_strategy(tmp_path, _BROKEN_IMPORT)
        # ImportError or ModuleNotFoundError both fine
        with pytest.raises((ImportError, ModuleNotFoundError)):
            _import_strategy_module(path)


class TestFindStrategyClass:
    def test_picks_protocol_compliant_class(self, tmp_path: Path) -> None:
        # Multiple top-level classes; should pick the one with
        # fit + generate_signals
        body = """
class Helper:
    pass

class MyStrategy:
    def fit(self, t): pass
    def generate_signals(self, t): pass
"""
        module = _import_strategy_module(_write_strategy(tmp_path, body))
        cls = _find_strategy_class(module)
        assert cls.__name__ == "MyStrategy"

    def test_no_class_raises(self, tmp_path: Path) -> None:
        body = "x = 1\n"
        module = _import_strategy_module(_write_strategy(tmp_path, body))
        with pytest.raises(ValueError, match="no top-level class"):
            _find_strategy_class(module)

    def test_falls_back_to_first_class_when_no_protocol_match(
        self,
        tmp_path: Path,
    ) -> None:
        # No class has fit + generate_signals — pick the first one
        module = _import_strategy_module(
            _write_strategy(tmp_path, _NO_STRATEGY_PROTOCOL),
        )
        cls = _find_strategy_class(module)
        assert cls.__name__ == "JustData"


class TestReadSymbols:
    def test_class_level_list(self) -> None:
        class A:
            symbols = ["EURUSD", "USDJPY"]

        assert _read_symbols(A) == ["EURUSD", "USDJPY"]

    def test_default_when_missing(self) -> None:
        class B:
            pass

        assert _read_symbols(B) == ["EURUSD"]


class TestReadExecutionSymbol:
    def test_explicit_execution_symbol(self) -> None:
        from src.research.backtest_runner import _read_execution_symbol

        class A:
            symbols = ["DXY", "EURUSD"]
            execution_symbol = "EURUSD"

        assert _read_execution_symbol(A, A.symbols) == "EURUSD"

    def test_defaults_to_first_symbol(self) -> None:
        from src.research.backtest_runner import _read_execution_symbol

        class A:
            symbols = ["EURUSD", "USDJPY"]

        assert _read_execution_symbol(A, A.symbols) == "EURUSD"

    def test_invalid_execution_symbol_falls_back(self) -> None:
        from src.research.backtest_runner import _read_execution_symbol

        class A:
            symbols = ["EURUSD"]
            execution_symbol = "GBPUSD"  # not in symbols

        assert _read_execution_symbol(A, A.symbols) == "EURUSD"


# ---------------------------------------------------------------------- #
# B-rule proxies
# ---------------------------------------------------------------------- #


class TestEdgeConcentration:
    def test_all_in_one_fold_returns_one(self) -> None:
        fm = pd.DataFrame(
            {
                "test_sharpe": [2.0, 0.0, 0.0, 0.0],
            }
        )
        assert _compute_edge_concentration(fm) == pytest.approx(1.0)

    def test_evenly_split_returns_one_over_n(self) -> None:
        fm = pd.DataFrame({"test_sharpe": [0.5, 0.5, 0.5, 0.5]})
        assert _compute_edge_concentration(fm) == pytest.approx(0.25)

    def test_too_few_folds_returns_default(self) -> None:
        fm = pd.DataFrame({"test_sharpe": [1.0]})
        assert _compute_edge_concentration(fm) == 0.50


class TestRegimeDiversified:
    def test_two_positive_folds_diverse(self) -> None:
        fm = pd.DataFrame({"test_sharpe": [0.5, -0.2, 0.7, -0.1]})
        assert _compute_regime_diversified(fm) is True

    def test_one_positive_fold_concentrated(self) -> None:
        fm = pd.DataFrame({"test_sharpe": [0.5, -0.2, -0.7, -0.1]})
        assert _compute_regime_diversified(fm) is False

    def test_too_few_folds_defaults_true(self) -> None:
        fm = pd.DataFrame({"test_sharpe": [0.5]})
        assert _compute_regime_diversified(fm) is True


class TestDecaySeverity:
    def test_strong_decay(self) -> None:
        fm = pd.DataFrame(
            {
                "test_sharpe": [
                    1.0,
                    1.0,
                    1.0,  # oldest third (positive)
                    0.5,
                    0.0,
                    0.0,  # middle
                    -0.8,
                    -0.8,
                    -0.8,  # newest third (negative, > 50% of oldest)
                ]
            }
        )
        assert _compute_decay_severity(fm) == "STRONG"

    def test_moderate_decay(self) -> None:
        fm = pd.DataFrame(
            {
                "test_sharpe": [
                    1.0,
                    1.0,
                    1.0,  # oldest
                    0.5,
                    0.5,
                    0.0,
                    -0.1,
                    -0.1,
                    -0.1,  # newest sign-flipped but mild
                ]
            }
        )
        assert _compute_decay_severity(fm) == "MODERATE"

    def test_no_decay(self) -> None:
        fm = pd.DataFrame(
            {
                "test_sharpe": [
                    0.5,
                    0.5,
                    0.5,
                    0.6,
                    0.7,
                    0.6,
                    0.7,
                    0.8,
                    0.7,
                ]
            }
        )
        assert _compute_decay_severity(fm) == "NONE"

    def test_too_few_folds(self) -> None:
        fm = pd.DataFrame({"test_sharpe": [1.0, 0.5]})
        assert _compute_decay_severity(fm) == "NONE"


# ---------------------------------------------------------------------- #
# End-to-end runner
# ---------------------------------------------------------------------- #


class TestRunnerEndToEnd:
    def test_returns_metrics_dict_with_threshold_paths(
        self,
        tmp_path: Path,
    ) -> None:
        strat_path = _write_strategy(tmp_path, _VALID_STRATEGY, name="stub_e2e")
        provider = _FakeProvider(data=_synth_ohlcv(n=1500))
        runner = make_backtest_runner(
            data_provider=provider,  # type: ignore[arg-type]
            bootstrap_n=200,  # speed up the test
        )
        metrics = runner(strat_path)
        # Every threshold path in REVIEW_RULES.md must be top-level
        # resolvable.
        assert "oos_metrics" in metrics
        for k in (
            "sharpe",
            "n_trades",
            "hit_rate",
            "max_drawdown",
            "profit_factor",
        ):
            assert k in metrics["oos_metrics"], f"missing oos_metrics.{k}"
        assert "sharpe_ci_95" in metrics
        for k in ("low", "high"):
            assert k in metrics["sharpe_ci_95"]
        for top in (
            "is_oos_sharpe_ratio",
            "edge_concentration",
            "regime_diversified",
            "decay_severity",
        ):
            assert top in metrics, f"missing top-level {top}"
        # Provenance documents what's a proxy
        assert "_metrics_provenance" in metrics
        assert "PROXY" in metrics["_metrics_provenance"]["edge_concentration"]

    def test_raises_on_no_data(self, tmp_path: Path) -> None:
        strat_path = _write_strategy(tmp_path, _VALID_STRATEGY, name="stub_nodata")
        provider = _FakeProvider(data=pd.DataFrame())
        runner = make_backtest_runner(
            data_provider=provider,  # type: ignore[arg-type]
            bootstrap_n=100,
        )
        with pytest.raises(ValueError, match="no data"):
            runner(strat_path)

    def test_raises_when_execution_symbol_column_missing(
        self,
        tmp_path: Path,
    ) -> None:
        strat_path = _write_strategy(tmp_path, _VALID_STRATEGY, name="stub_nocol")
        # DataFrame with neither 'close' nor a column matching the
        # strategy's execution_symbol (defaults to symbols[0] = EURUSD)
        provider = _FakeProvider(
            data=pd.DataFrame(
                {"foo": [1.0, 2.0]},
                index=pd.date_range("2018-01-01", periods=2),
            )
        )
        runner = make_backtest_runner(
            data_provider=provider,  # type: ignore[arg-type]
            bootstrap_n=100,
        )
        with pytest.raises(
            ValueError,
            match="missing execution_symbol column",
        ):
            runner(strat_path)

    def test_raises_on_strategy_import_error(self, tmp_path: Path) -> None:
        strat_path = _write_strategy(tmp_path, _BROKEN_IMPORT, name="broken")
        provider = _FakeProvider(data=_synth_ohlcv())
        runner = make_backtest_runner(
            data_provider=provider,  # type: ignore[arg-type]
            bootstrap_n=100,
        )
        with pytest.raises((ImportError, ModuleNotFoundError)):
            runner(strat_path)


# ---------------------------------------------------------------------- #
# Loop integration — runner + Implementer + verdict engine
# ---------------------------------------------------------------------- #


class TestLoopWiring:
    """Verify the runner output flows into the Implementer's report
    such that compute_verdict can resolve every threshold path."""

    def test_implementer_spreads_top_level(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Implementer's _build_report should spread the runner's
        keys at top level so verdict.compute_verdict reads them
        without nested-path traversal."""
        from src.research.agents.implementer import Implementer
        from src.research.config import (
            AgentConfig,
            ProviderConfig,
            ResearchConfig,
        )
        from src.research.llm.client import (
            Driver,
            LLMResponse,
            register_driver,
        )

        class _RunnerCanned(Driver):
            name = "runner-canned"

            def __init__(self, api_key: str = "x") -> None:
                super().__init__(api_key)
                self.canned_text = (
                    "## Strategy code\n```python\nclass S:\n    pass\n```\n"
                    "## Prediction\n- `predicted_sharpe_range`: [0.5, 0.9]\n"
                    "**FINAL_POSITION**: IMPLEMENTED\n"
                )

            def complete(
                self,
                messages: object,  # noqa: ARG002
                model: str,
                max_tokens: int = 4096,  # noqa: ARG002
                temperature: float = 0.0,  # noqa: ARG002
                **kwargs: object,
            ) -> LLMResponse:
                return LLMResponse(
                    text=self.canned_text,
                    model=model,
                    provider=self.name,
                    input_tokens=10,
                    output_tokens=20,
                    usd_cost=0.001,
                    elapsed_sec=0.01,
                )

        register_driver("runner-canned", _RunnerCanned)
        monkeypatch.setenv("RUNNER_KEY", "fake")
        prompt = tmp_path / "p.md"
        prompt.write_text("stub")
        cfg = ResearchConfig(
            providers={
                "runner-canned": ProviderConfig(
                    api_key_env="RUNNER_KEY",
                    default_model="m",
                )
            },
            agents={
                "implementer": AgentConfig(
                    provider="runner-canned",
                    role="implementer",
                    prompt_path=str(prompt),
                    model=None,
                )
            },
            debates={},  # type: ignore[arg-type]
        )
        impl = Implementer.from_config(name="implementer", research_config=cfg)
        hyp = tmp_path / "hyp.md"
        hyp.write_text("# stub\n")

        canned_metrics = {
            "oos_metrics": {
                "sharpe": 0.7,
                "n_trades": 50,
                "hit_rate": 0.6,
                "max_drawdown": -0.1,
                "profit_factor": 1.5,
            },
            "sharpe_ci_95": {"low": 0.2, "high": 1.2},
            "is_oos_sharpe_ratio": 1.4,
            "edge_concentration": 0.3,
            "regime_diversified": True,
            "decay_severity": "NONE",
        }

        def fake_runner(_p: Path) -> dict[str, object]:
            return canned_metrics

        result = impl.implement(
            hypothesis_path=hyp,
            strategy_slug="t",
            backtest_runner=fake_runner,
            code_dir=tmp_path / "exp",
            report_dir=tmp_path / "rep",
        )
        assert result.report_path is not None
        import json

        report = json.loads(result.report_path.read_text())
        # Top-level paths the verdict engine reads
        assert report["oos_metrics"]["sharpe"] == 0.7
        assert report["sharpe_ci_95"]["low"] == 0.2
        assert report["edge_concentration"] == 0.3
        assert report["regime_diversified"] is True
        assert report["decay_severity"] == "NONE"
        # Reserved keys not clobbered
        assert report["strategy_slug"] == "t"
        assert report["gates"]["backtest"] == "ran"
        # Backwards-compatible nested copy still present
        assert report["backtest_metrics"]["oos_metrics"]["sharpe"] == 0.7


# Suppress unused-import warning for Callable
_ = Callable
