"""Tests for SourceDriftWatcher (CL-9eli post-mortem hook).

Covers:
  - Baseline-vs-current matching → no drift
  - Modified file → drift detected
  - Deleted file → drift detected
  - Glob pattern resolution
  - run_forever sets the Prometheus gauge once on first detection
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.monitoring.metrics import engine_source_drift
from src.runtime.source_drift import SourceDriftWatcher


@pytest.fixture
def fake_repo(tmp_path):  # type: ignore[no-untyped-def]
    """A temp repo with a couple source files for the watcher to track."""
    (tmp_path / "src" / "data").mkdir(parents=True)
    (tmp_path / "src" / "strategies").mkdir(parents=True)
    (tmp_path / "src" / "data" / "provider.py").write_text(
        "# provider v1\n",
    )
    (tmp_path / "src" / "strategies" / "alpha.py").write_text(
        "# alpha v1\n",
    )
    (tmp_path / "src" / "strategies" / "beta.py").write_text(
        "# beta v1\n",
    )
    return tmp_path


def _patterns() -> tuple[str, ...]:
    return ("src/data/provider.py", "src/strategies/*.py")


class TestSnapshot:
    def test_baseline_resolves_globs(self, fake_repo: Path) -> None:
        w = SourceDriftWatcher(_patterns(), root=fake_repo)
        # provider.py + 2 strategy files
        assert len(w.watched) == 3
        assert all(p.exists() for p in w.watched)


class TestCheck:
    def test_no_drift_initially(self, fake_repo: Path) -> None:
        w = SourceDriftWatcher(_patterns(), root=fake_repo)
        assert w.check() == []

    def test_modified_file_detected(self, fake_repo: Path) -> None:
        w = SourceDriftWatcher(_patterns(), root=fake_repo)
        (fake_repo / "src" / "data" / "provider.py").write_text("# provider v2\n")
        drift = w.check()
        assert len(drift) == 1
        assert drift[0].name == "provider.py"

    def test_multiple_files_detected(self, fake_repo: Path) -> None:
        w = SourceDriftWatcher(_patterns(), root=fake_repo)
        (fake_repo / "src" / "strategies" / "alpha.py").write_text("# alpha v2\n")
        (fake_repo / "src" / "strategies" / "beta.py").write_text("# beta v2\n")
        drift = w.check()
        assert len(drift) == 2

    def test_deleted_file_detected(self, fake_repo: Path) -> None:
        w = SourceDriftWatcher(_patterns(), root=fake_repo)
        (fake_repo / "src" / "data" / "provider.py").unlink()
        drift = w.check()
        assert len(drift) == 1
        assert drift[0].name == "provider.py"

    def test_unrelated_file_change_ignored(self, fake_repo: Path) -> None:
        w = SourceDriftWatcher(_patterns(), root=fake_repo)
        # Touch a file not in the watch patterns.
        (fake_repo / "src" / "unwatched.py").write_text("# whatever\n")
        assert w.check() == []


class TestRunForever:
    @pytest.mark.asyncio
    async def test_sets_gauge_on_drift(self, fake_repo: Path) -> None:
        # Reset gauge state before the test (it's a process-global
        # singleton).
        engine_source_drift.set(0)

        w = SourceDriftWatcher(_patterns(), root=fake_repo)
        # Run the watcher with a tiny interval so we don't wait 5 minutes.
        task = asyncio.create_task(w.run_forever(interval_sec=0))
        try:
            # Let the loop run a tick on the clean baseline.
            await asyncio.sleep(0.05)
            # Mutate a watched file.
            (fake_repo / "src" / "data" / "provider.py").write_text("# v2\n")
            # Wait for the watcher to notice (next tick).
            await asyncio.sleep(0.05)
            assert engine_source_drift._value.get() == 1.0  # type: ignore[attr-defined]
        finally:
            task.cancel()
            with contextlib_suppress():
                await task


class contextlib_suppress:  # tiny ctx manager for async cancel cleanup
    def __enter__(self) -> "contextlib_suppress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return exc_type is asyncio.CancelledError
