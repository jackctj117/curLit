"""Shared-DB-engine wiring in the boot path (CL-7vn9 / CL-8s2a).

Before this, every DB-backed builder in ``run_engine`` called
``create_engine`` for itself — 5 separate SQLAlchemy engines, so 5
independent connection pools sat idle (~25 conns) against a single-process
engine that only ever needs one. These tests pin the fix: exactly ONE engine
is built and the SAME object is threaded to every builder, while each builder
still fails open (returns None) when the engine is absent/unusable.

No network, no database — ``create_engine`` and the leaf store constructors
are all monkeypatched. A SQLAlchemy ``Engine`` is threadsafe and pools
per-operation, so a single shared engine is the correct primitive here.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.runtime import run_engine as re


class _FakeEngine:
    """Stand-in for a SQLAlchemy Engine — identity is all the tests check."""

    def __init__(self, tag: str = "engine") -> None:
        self.tag = tag


def test_build_shared_db_engine_calls_create_engine_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_build() -> Any:
        calls.append("build")
        return _FakeEngine()

    monkeypatch.setattr(re, "_build_db_engine", fake_build)
    eng = re.build_shared_db_engine()
    assert isinstance(eng, _FakeEngine)
    assert calls == ["build"]  # exactly one engine minted


def test_build_shared_db_engine_fails_open_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> Any:
        raise RuntimeError("no DB url")

    monkeypatch.setattr(re, "_build_db_engine", boom)
    # Never raises — a DB-less boot still starts; builders degrade on None.
    assert re.build_shared_db_engine() is None


def test_run_engine_threads_one_engine_to_every_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression guard: run_engine builds the shared engine ONCE and
    hands the SAME object to journal / snapshot / strategies / coordinator /
    kill-switch builders — no builder mints its own."""
    import asyncio

    shared = _FakeEngine("shared")
    seen_engines: dict[str, Any] = {}
    create_engine_calls: list[str] = []

    # Count every low-level engine construction — must be exactly one.
    def fake_low_level() -> Any:
        create_engine_calls.append("create")
        return shared

    monkeypatch.setattr(re, "_build_db_engine", fake_low_level)

    # Capture the engine each builder receives; return harmless stand-ins.
    monkeypatch.setattr(
        re, "build_trade_journal",
        lambda engine=None: seen_engines.setdefault("journal", engine),
    )
    monkeypatch.setattr(
        re, "build_feature_snapshot_store",
        lambda engine=None: seen_engines.setdefault("snapshot", engine),
    )

    def fake_strategies(config, broker, oms, snapshot_store=None, engine=None):  # noqa: ANN001, ANN202
        seen_engines["strategies"] = engine
        return []

    monkeypatch.setattr(re, "build_strategies", fake_strategies)

    def fake_coordinator(config, strategies, oms, broker,  # noqa: ANN001, ANN202
                         blackout_evaluator=None, engine=None):
        seen_engines["coordinator"] = engine
        return None

    monkeypatch.setattr(re, "build_coordinator", fake_coordinator)

    def fake_kill(broker, oms, engine=None):  # noqa: ANN001, ANN202
        seen_engines["killswitch"] = engine
        return object()

    monkeypatch.setattr(re, "build_kill_switch_manager", fake_kill)

    # Neutralize everything else run_engine touches so it returns after wiring.
    monkeypatch.setattr(re, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(re, "load_config", lambda path: {})
    monkeypatch.setattr(re, "build_broker", lambda mode: object())
    monkeypatch.setattr(re, "effective_broker_mode", lambda mode, broker: mode)
    monkeypatch.setattr(re, "build_blackout_evaluator", lambda config: None)
    monkeypatch.setattr(re, "build_cold_start_reconciler",
                        lambda *a, **k: None)

    class _StopEngine:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def run(self) -> None:
            return

    monkeypatch.setattr(re, "LiveEngine", _StopEngine)
    # set_runtime import inside run_engine — stub the web api wiring.
    import sys
    import types
    fake_web = types.ModuleType("src.web.api")
    fake_web.set_runtime = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.web.api", fake_web)
    # Skip the source-drift watcher import path cheaply.
    fake_drift = types.ModuleType("src.runtime.source_drift")

    class _NoWatcher:
        async def run_forever(self) -> None:
            return

    fake_drift.SourceDriftWatcher = _NoWatcher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.runtime.source_drift", fake_drift)

    asyncio.run(re.run_engine("paper"))

    # Exactly one engine minted, and the identical object reached each builder.
    assert create_engine_calls == ["create"]
    assert seen_engines["journal"] is shared
    assert seen_engines["snapshot"] is shared
    assert seen_engines["strategies"] is shared
    assert seen_engines["coordinator"] is shared
    assert seen_engines["killswitch"] is shared


def test_builders_fall_back_to_local_engine_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compat for out-of-lane zero-arg callers (manage_strategies,
    scripts): a builder called with no engine still builds one locally rather
    than degrading — the original contract."""
    local = _FakeEngine("local")
    monkeypatch.setattr(re, "_build_db_engine", lambda: local)
    monkeypatch.setattr(re, "TradeJournal", lambda engine: ("journal", engine))
    monkeypatch.setattr(
        re, "FeatureSnapshotStore", lambda engine: ("snap", engine),
    )

    assert re.build_trade_journal() == ("journal", local)
    assert re.build_feature_snapshot_store() == ("snap", local)


def test_builders_fail_open_on_construction_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store that can't construct (DB unreachable) still yields None so the
    engine boots without audit / snapshots rather than refusing to trade."""
    eng = _FakeEngine()

    def boom(engine: Any) -> Any:
        raise RuntimeError("DB unreachable")

    monkeypatch.setattr(re, "TradeJournal", boom)
    monkeypatch.setattr(re, "FeatureSnapshotStore", boom)

    assert re.build_trade_journal(eng) is None
    assert re.build_feature_snapshot_store(eng) is None
