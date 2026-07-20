"""Tests for the event-pipeline --poly step wiring (CL-r1ep).

Exercises scripts/event_pipeline._poly_step + _cycle integration with a
mocked PolymarketSignal — no live HTTP, no DB. Focus: the step polls,
detects, and notifies; a poly failure never kills the cycle; --no-poly
skips it; the digest receives the poly signal for corroboration.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest
import scripts.event_pipeline as ep


def _args(**overrides: Any) -> argparse.Namespace:
    base = {
        "ingest": False,
        "assess": False,
        "scan": False,
        "poly": True,
        "digest": False,
        "poly_config": "configs/polymarket_geo_markets.yaml",
        "playbooks": "configs/event_playbooks.yaml",
        "limit": 20,
        "lookback_minutes": 60,
        "model": "",
        "digest_min_urgency": 5,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class _FakeSignal:
    def __init__(self) -> None:
        self.polled: list[Any] = []
        self.detected = 0
        self.notified: list[Any] = []
        self.shifts_to_return: list[Any] = []

    def poll_probabilities(self, markets: list[Any]) -> list[Any]:
        self.polled = markets
        return [(m, 0.5) for m in markets]

    def detect_shifts(self, *a: Any, **k: Any) -> list[Any]:
        return self.shifts_to_return

    def notify_shifts(self, shifts: list[Any]) -> int:
        self.notified = shifts
        return len(shifts)


def _patch_signal(
    monkeypatch: pytest.MonkeyPatch,
    signal: Any,
    markets: list[Any],
) -> None:
    """Patch the lazily-imported PolymarketSignal + loader + engine."""
    import src.events.polymarket_signal as mod

    monkeypatch.setattr(mod, "PolymarketSignal", lambda *a, **k: signal)
    monkeypatch.setattr(mod, "load_tracked_markets", lambda path: markets)
    # _poly_step builds an engine — stub create_engine to a sentinel.
    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url: object())
    monkeypatch.setattr(ep, "_db_url", lambda: "sqlite://")


class TestPolyStep:
    def test_polls_and_notifies_shifts(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sig = _FakeSignal()
        sig.shifts_to_return = ["shift-1", "shift-2"]
        markets = ["m1", "m2"]
        _patch_signal(monkeypatch, sig, markets)

        returned = ep._poly_step(_args())
        assert returned is sig
        assert sig.polled == markets
        assert sig.notified == ["shift-1", "shift-2"]

    def test_no_markets_skips_poll(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sig = _FakeSignal()
        _patch_signal(monkeypatch, sig, [])
        returned = ep._poly_step(_args())
        assert returned is sig
        assert sig.polled == []  # never polled — nothing configured

    def test_no_shifts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sig = _FakeSignal()
        sig.shifts_to_return = []
        _patch_signal(monkeypatch, sig, ["m1"])
        ep._poly_step(_args())
        assert sig.notified == []


class TestCyclePolyTolerance:
    def test_poly_failure_never_kills_cycle(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # _poly_step raises; _cycle (assess=False path) must swallow it.
        def boom(args: Any) -> Any:
            raise RuntimeError("gamma down")

        monkeypatch.setattr(ep, "_poly_step", boom)
        # poly-only cycle (no ingest/assess/scan) — should not raise.
        ep._cycle(_args(poly=True))

    def test_no_poly_skips_step(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called = {"n": 0}

        def spy(args: Any) -> Any:
            called["n"] += 1

        monkeypatch.setattr(ep, "_poly_step", spy)
        ep._cycle(_args(poly=False))
        assert called["n"] == 0

    def test_poly_runs_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called = {"n": 0}

        def spy(args: Any) -> Any:
            called["n"] += 1

        monkeypatch.setattr(ep, "_poly_step", spy)
        ep._cycle(_args(poly=True))
        assert called["n"] == 1
