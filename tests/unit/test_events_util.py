"""Tests for the events shared kernel (CL-ikz2 — review §6.2.1).

Covers the utilities the events modules used to re-implement per file:
``env_flag`` truthiness semantics (identical to the three deleted
``_env_flag`` copies), both clamp calling conventions (the raising
impact-agent form and the default-returning niche form), and atomic JSON
persistence (valid content, in-place replace, no tmp litter, tmp cleanup
on failure).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.events._util import atomic_write_json, clamp_float, clamp_int, env_flag

# --------------------------------------------------------------------- #
# env_flag
# --------------------------------------------------------------------- #


class TestEnvFlag:
    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", " yes ", "On", "  ON "])
    def test_truthy_values(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURLIT_TEST_FLAG", raw)
        assert env_flag("CURLIT_TEST_FLAG", default=False) is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "garbage", "2"])
    def test_falsy_values(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURLIT_TEST_FLAG", raw)
        assert env_flag("CURLIT_TEST_FLAG", default=True) is False

    def test_unset_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CURLIT_TEST_FLAG", raising=False)
        assert env_flag("CURLIT_TEST_FLAG", default=True) is True
        assert env_flag("CURLIT_TEST_FLAG", default=False) is False

    def test_shared_by_events_modules(self) -> None:
        """The per-module ``_env_flag`` copies are gone — triage, the niche
        agent and the critic all use the ONE kernel function."""
        from src.events import adversarial_critic, niche_agent, triage

        assert not hasattr(triage, "_env_flag")
        assert not hasattr(adversarial_critic, "_env_flag")
        assert niche_agent.env_flag is env_flag
        assert triage.env_flag is env_flag
        assert adversarial_critic.env_flag is env_flag


# --------------------------------------------------------------------- #
# clamps — both calling conventions
# --------------------------------------------------------------------- #


class TestClampInt:
    def test_bounds_and_rounding(self) -> None:
        assert clamp_int("7.6", 1, 10) == 8
        assert clamp_int(99, 1, 10) == 10
        assert clamp_int(-5, 1, 10) == 1
        assert clamp_int(5, 1, 10) == 5

    def test_raises_without_default(self) -> None:
        # The impact-agent convention: the caller owns the except.
        with pytest.raises(TypeError):
            clamp_int(None, 1, 10)
        with pytest.raises(ValueError):
            clamp_int("junk", 1, 10)

    def test_default_on_bad_input(self) -> None:
        # The niche-agent convention: fail-soft to the supplied default.
        assert clamp_int(None, 1, 10, 4) == 4
        assert clamp_int("junk", 1, 10, 4) == 4
        assert clamp_int(7, 1, 10, 4) == 7


class TestClampFloat:
    def test_bounds(self) -> None:
        assert clamp_float("0.75", 0.0, 1.0) == 0.75
        assert clamp_float(3.2, 0.0, 1.0) == 1.0
        assert clamp_float(-0.5, 0.0, 1.0) == 0.0

    def test_raises_without_default(self) -> None:
        with pytest.raises(TypeError):
            clamp_float(None, 0.0, 1.0)
        with pytest.raises(ValueError):
            clamp_float("junk", 0.0, 1.0)

    def test_default_on_bad_input(self) -> None:
        assert clamp_float(None, 0.0, 1.0, 0.4) == 0.4
        assert clamp_float("junk", 0.0, 1.0, 0.4) == 0.4
        assert clamp_float(0.9, 0.0, 1.0, 0.4) == 0.9


# --------------------------------------------------------------------- #
# atomic_write_json
# --------------------------------------------------------------------- #


class TestAtomicWriteJson:
    def test_writes_valid_json_and_creates_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "state.json"
        atomic_write_json(target, {"a": 1, "items": [1, 2, 3]})
        assert json.loads(target.read_text()) == {"a": 1, "items": [1, 2, 3]}

    def test_replaces_existing_and_leaves_no_tmp_litter(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})
        assert json.loads(target.read_text()) == {"v": 2}
        assert [f.name for f in tmp_path.iterdir()] == ["state.json"]

    def test_non_json_values_fall_back_to_str(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        atomic_write_json(target, {"path": Path("x/y")})  # default=str
        assert json.loads(target.read_text()) == {"path": "x/y"}

    def test_failure_leaves_target_intact_and_no_tmp(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})

        class _Unserialisable:
            def __str__(self) -> str:  # even default=str blows up
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            atomic_write_json(target, {"bad": _Unserialisable()})
        # Old state survives untouched; the tmp file was cleaned up.
        assert json.loads(target.read_text()) == {"v": 1}
        assert [f.name for f in tmp_path.iterdir()] == ["state.json"]


# --------------------------------------------------------------------- #
# ThreadLocalClient (CL-4pyb / CL-818b) — per-thread LLM clients
# --------------------------------------------------------------------- #


class TestThreadLocalClient:
    """The claude-code driver shells out from a per-instance temp cwd, so a
    client shared across fan-out workers means concurrent `claude` subprocesses
    in ONE workdir. Each thread must get its own; an injected client must be
    reused verbatim so unit tests never spawn the real CLI."""

    def test_injected_client_reused_and_never_duplicated(self) -> None:
        from src.events._util import ThreadLocalClient

        stub = object()
        holder = ThreadLocalClient(stub)
        assert holder.is_default is False
        assert holder.get() is stub

        seen = []
        threads = [threading.Thread(target=lambda: seen.append(holder.get())) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == [stub] * 4  # same object in every thread

    def test_default_client_is_distinct_per_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.events import _util

        made: list[object] = []

        def _fake_get_client(provider: str) -> object:
            c = object()
            made.append(c)
            return c

        # get_client is imported inside .get(); patch at its source module.
        import src.research.llm as llm_mod

        monkeypatch.setattr(llm_mod, "get_client", _fake_get_client)

        holder = _util.ThreadLocalClient(None)
        assert holder.is_default is True

        # CL-6sdh: keep four workers alive together. Fast sequential Thread
        # lifetimes can reuse OS identifiers and overwrite the old test's dict.
        barrier = threading.Barrier(4, timeout=5)  # Bound a broken fixture, not production work.

        def _grab() -> tuple[int, object, object]:
            barrier.wait()
            return threading.get_ident(), holder.get(), holder.get()

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_grab) for _ in range(4)]
            results = [future.result(timeout=10) for future in futures]

        assert len({ident for ident, _, _ in results}) == 4
        assert all(first is second for _, first, second in results)
        clients = [first for _, first, _ in results]
        assert len(clients) == 4
        assert len({id(c) for c in clients}) == 4  # one distinct client per thread
        assert len(made) == 4

    def test_thread_oracle_rejects_shared_client_mutation(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.events._util import ThreadLocalClient

        shared = object()
        monkeypatch.setattr(ThreadLocalClient, "get", lambda _self: shared)
        with pytest.raises(AssertionError):
            self.test_default_client_is_distinct_per_thread(monkeypatch)

    def test_same_thread_reuses_its_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.research.llm as llm_mod
        from src.events import _util

        monkeypatch.setattr(llm_mod, "get_client", lambda provider: object())
        holder = _util.ThreadLocalClient(None)
        assert holder.get() is holder.get()  # cached per thread, not per call
