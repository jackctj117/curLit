"""manage_strategies CLI wiring + removal-flow regressions (CL-9dhg).

Two verified bugs:

1. ``_build_coordinator`` imported ``_build_coordinator_for_admin`` from
   run_engine, which never existed — every mutating subcommand died on
   ImportError. It must instead reuse run_engine's real builders
   (build_broker / build_strategies / build_coordinator).

2. Even with a coordinator, a fresh CLI process has EMPTY cross-tick
   target memory (``_strategy_targets`` seeds only inside
   ``process_intents``, which never runs in the CLI). ``remove_strategy``
   computes co-holder shares from that memory, so an unseeded removal
   submits ZERO-target liquidations that flatten every co-holding
   strategy while printing success. The CLI must seed
   (``_seed_targets_from_books`` + ``_targets_seeded = True``) BEFORE
   calling ``remove_strategy``.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest
from scripts import manage_strategies


class _FakeCoordinator:
    """Records the ORDER of seeding vs. removal calls."""

    def __init__(self, strategy_ids: list[str]) -> None:
        self.strategies = dict.fromkeys(strategy_ids, object())
        self.calls: list[str] = []
        self._strategy_targets: dict[str, dict[str, float]] = {}
        self._targets_seeded = False

    def seed_targets_from_books(self) -> None:
        self.calls.append("seed")
        self._targets_seeded = True

    def _seed_targets_from_books(self) -> None:
        self.calls.append("seed")

    def remove_strategy(self, strategy_id: str) -> None:
        # The whole point: memory must be seeded by the time removal
        # computes co-holder shares.
        assert self._targets_seeded, (
            "remove_strategy called with unseeded cross-tick memory — "
            "co-holder shares would compute as 0 and the removal would "
            "flatten every strategy on shared symbols"
        )
        self.calls.append(f"remove:{strategy_id}")


def _remove_args(strategy: str, *, confirm: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        strategy=strategy, confirm=confirm, broker="paper",
    )


def test_remove_seeds_memory_before_remove_strategy(monkeypatch):
    coord = _FakeCoordinator(["rate_diff_mr", "event_driven"])
    monkeypatch.setattr(
        manage_strategies, "_build_coordinator", lambda mode: coord,
    )

    rc = manage_strategies.cmd_remove(_remove_args("rate_diff_mr"))

    assert rc == 0
    assert coord.calls == ["seed", "remove:rate_diff_mr"]
    assert coord._targets_seeded is True


def test_remove_without_confirm_refuses_and_does_not_touch_state(monkeypatch):
    coord = _FakeCoordinator(["rate_diff_mr"])
    monkeypatch.setattr(
        manage_strategies, "_build_coordinator", lambda mode: coord,
    )

    rc = manage_strategies.cmd_remove(
        _remove_args("rate_diff_mr", confirm=False),
    )

    assert rc == 2
    assert coord.calls == []
    assert coord._targets_seeded is False


def test_remove_unknown_strategy_refuses(monkeypatch):
    coord = _FakeCoordinator(["rate_diff_mr"])
    monkeypatch.setattr(
        manage_strategies, "_build_coordinator", lambda mode: coord,
    )

    rc = manage_strategies.cmd_remove(_remove_args("nope"))

    assert rc == 2
    assert coord.calls == []


def test_main_remove_end_to_end_seeds_before_removal(monkeypatch):
    """Through argparse: main(['remove', ...]) reaches the seeded path."""
    coord = _FakeCoordinator(["carry_vol"])
    monkeypatch.setattr(
        manage_strategies, "_build_coordinator", lambda mode: coord,
    )
    # Keep the unit test hermetic — don't load the operator's .env.
    monkeypatch.setattr(manage_strategies, "load_project_env", lambda: None)

    rc = manage_strategies.main(
        ["remove", "--strategy", "carry_vol", "--confirm"],
    )

    assert rc == 0
    assert coord.calls == ["seed", "remove:carry_vol"]


def test_build_coordinator_reuses_run_engine_builders(monkeypatch):
    """_build_coordinator must wire the REAL run_engine builders (the old
    import target ``_build_coordinator_for_admin`` never existed)."""
    from src.runtime import run_engine

    calls: list[str] = []
    sentinel_coord = object()
    fake_broker = object()
    fake_strategies = [object()]
    fake_config = {"strategies": []}

    monkeypatch.setattr(
        run_engine, "load_config", lambda path: calls.append("config") or fake_config,
    )
    monkeypatch.setattr(
        run_engine, "build_broker",
        lambda mode: calls.append(f"broker:{mode}") or fake_broker,
    )
    monkeypatch.setattr(
        run_engine, "build_trade_journal", lambda: calls.append("journal"),
    )
    monkeypatch.setattr(
        run_engine, "build_blackout_evaluator",
        lambda config: calls.append("blackout"),
    )

    def fake_build_strategies(config: Any, broker: Any, oms: Any) -> list[Any]:
        assert config is fake_config
        assert broker is fake_broker
        assert oms is not None
        calls.append("strategies")
        return fake_strategies

    monkeypatch.setattr(run_engine, "build_strategies", fake_build_strategies)

    def fake_build_coordinator(
        config: Any, strategies: Any, oms: Any, broker: Any,
        blackout_evaluator: Any = None,
    ) -> Any:
        assert strategies is fake_strategies
        assert broker is fake_broker
        calls.append("coordinator")
        return sentinel_coord

    monkeypatch.setattr(run_engine, "build_coordinator", fake_build_coordinator)

    coord = manage_strategies._build_coordinator("paper")

    assert coord is sentinel_coord
    assert calls == [
        "config", "broker:paper", "journal", "strategies", "blackout",
        "coordinator",
    ]


def test_build_coordinator_fails_loud_when_state_unreachable(monkeypatch):
    """run_engine.build_coordinator returns None when Postgres is down —
    the engine may degrade to legacy mode, but an admin CLI that mutates
    allocations must refuse to run instead."""
    from src.runtime import run_engine

    monkeypatch.setattr(run_engine, "load_config", lambda path: {})
    monkeypatch.setattr(run_engine, "build_broker", lambda mode: object())
    monkeypatch.setattr(run_engine, "build_trade_journal", lambda: None)
    monkeypatch.setattr(
        run_engine, "build_blackout_evaluator", lambda config: None,
    )
    monkeypatch.setattr(
        run_engine, "build_strategies", lambda config, broker, oms: [],
    )
    monkeypatch.setattr(
        run_engine,
        "build_coordinator",
        lambda config, strategies, oms, broker, blackout_evaluator=None: None,
    )

    with pytest.raises(RuntimeError, match="PortfolioCoordinator"):
        manage_strategies._build_coordinator("paper")
