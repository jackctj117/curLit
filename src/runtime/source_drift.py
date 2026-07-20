"""Source-drift watcher — detect when on-disk source diverges from the
running process (the CL-9eli / 2026-05-04 incident).

The failure mode: an engine boots on Wednesday, a fix lands on Friday,
nobody restarts. The engine keeps logging the same AttributeError every
signal cycle — alive, heartbeat fresh, but operating on dead code.

This watcher hashes a curated list of source files at startup and re-
hashes them periodically. When the on-disk hash diverges from the
in-memory hash, it logs CRITICAL and sets ``fx_engine_source_drift = 1``
so Grafana / Telegram can alert. We deliberately do *not* auto-halt:
the operator has the context to decide whether the change is a real
fix that needs a restart vs an in-progress edit they didn't intend the
engine to see.

Files watched are the ones whose stale-vs-fresh divergence has produced
a real incident or has high blast radius:

  - src/data/provider.py            — the CL-9eli incident
  - src/strategies/*.py             — all strategy code paths
  - src/portfolio/coordinator.py    — allocation routing
  - src/execution/oms.py            — order placement
  - src/runtime/live_engine.py      — the loop itself
  - src/runtime/run_engine.py       — startup wiring

Cadence: 5 minutes. Drift comes from `git pull` or text edits, both
infrequent — sub-minute polling adds noise, multi-hour misses the
incident. 5 min is one daily-cycle's worth of resolution and a tiny
fraction of the typical strategy interval (1h–24h).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)


# 5 minutes — see module docstring rationale.
_DEFAULT_CHECK_INTERVAL_SEC: int = 300

# The watch list. Glob patterns are resolved at startup so adding new
# strategy files later is automatic. Paths are relative to the repo root
# (same cwd as run_engine).
_DEFAULT_WATCH_PATTERNS: tuple[str, ...] = (
    "src/data/provider.py",
    "src/strategies/*.py",
    "src/portfolio/coordinator.py",
    "src/portfolio/attribution.py",
    "src/execution/oms.py",
    "src/runtime/live_engine.py",
    "src/runtime/run_engine.py",
)


def _hash_file(path: Path) -> str:
    """SHA256 of file contents. Raises FileNotFoundError if missing —
    the caller treats that as drift (file was deleted under us)."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _resolve_paths(
    patterns: Iterable[str], root: Path,
) -> list[Path]:
    """Expand glob patterns relative to ``root``."""
    out: list[Path] = []
    for pat in patterns:
        if "*" in pat:
            out.extend(sorted(root.glob(pat)))
        else:
            p = root / pat
            if p.exists():
                out.append(p)
    return sorted(set(out))


class SourceDriftWatcher:
    """Snapshots source-file hashes at construction; .check() returns the
    list of paths whose on-disk content has changed since."""

    def __init__(
        self,
        watch_patterns: Iterable[str] | None = None,
        root: Path | None = None,
    ) -> None:
        self.root = root or Path.cwd()
        patterns = tuple(watch_patterns or _DEFAULT_WATCH_PATTERNS)
        self.watched = _resolve_paths(patterns, self.root)
        self.baseline: dict[Path, str] = {
            p: _hash_file(p) for p in self.watched
        }
        logger.info(
            "SourceDriftWatcher monitoring %d files; baseline captured",
            len(self.watched),
        )

    def check(self) -> list[Path]:
        """Return the list of paths whose hash has changed (or which have
        gone missing). Empty list = clean."""
        drifted: list[Path] = []
        for p, baseline_hash in self.baseline.items():
            try:
                current = _hash_file(p)
            except FileNotFoundError:
                drifted.append(p)
                continue
            if current != baseline_hash:
                drifted.append(p)
        return drifted

    async def run_forever(
        self, interval_sec: int = _DEFAULT_CHECK_INTERVAL_SEC,
    ) -> None:
        """Background task: periodic check, log + emit gauge on drift.

        Sticks the alert "on" once drift is detected — the operator must
        restart (which captures a fresh baseline) to clear it. We don't
        auto-clear because a same-source-edited-back-to-original would
        spuriously clear the alert during partial work.
        """
        from src.monitoring.metrics import engine_source_drift

        engine_source_drift.set(0)
        drift_logged = False
        while True:
            await asyncio.sleep(interval_sec)
            try:
                drifted = self.check()
            except Exception:
                logger.exception("SourceDriftWatcher.check raised")
                continue
            if drifted and not drift_logged:
                paths = ", ".join(str(p.relative_to(self.root)) for p in drifted)
                logger.critical(
                    "SOURCE DRIFT DETECTED — engine is running stale code "
                    "vs working tree. Restart to load latest. Files: %s",
                    paths,
                )
                engine_source_drift.set(1)
                drift_logged = True
            elif drifted and drift_logged:
                # Already alerted; emit a heartbeat WARNING so the operator
                # knows the alert is still live, not stale itself.
                logger.warning(
                    "Source drift still present (%d files diverged)",
                    len(drifted),
                )
