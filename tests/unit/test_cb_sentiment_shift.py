"""CB sentiment-shift strategy — entry cap (CL-34by).

The max_concurrent_positions cap was checked only once per tick (against the
pre-tick count), so a single tick with several fresh signals could open past
the cap. It must be enforced per-entry within the tick.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from src.strategies.cb_sentiment_shift import (
    CBSentimentConfig,
    CBSentimentShiftStrategy,
    OpenPosition,
)


class TestStatePersistence:
    """CL-885p (P0 part 2): CB persists open_positions to a durable file and
    reloads on __init__ (before the cold-start reconcile), so a restart's live
    legs are seen as MATCHED instead of flattened as orphans."""

    def test_save_and_reload_roundtrip(self, tmp_path) -> None:
        path = str(tmp_path / "cb.json")
        s1 = CBSentimentShiftStrategy(CBSentimentConfig(state_path=path))
        s1.open_positions["EURUSD"] = OpenPosition(
            symbol="EURUSD", entry_ts=datetime.now(UTC), entry_price=1.10,
            quantity=-1000.0, direction=-1, stop_loss=1.11, source_cb="FED",
        )
        s1._save_state()
        # A fresh instance reloads the book on construction.
        s2 = CBSentimentShiftStrategy(CBSentimentConfig(state_path=path))
        assert "EURUSD" in s2.open_positions
        p = s2.open_positions["EURUSD"]
        assert p.quantity == -1000.0
        assert p.symbol == "EURUSD"
        assert p.direction == -1

    def test_missing_file_is_cold_start(self, tmp_path) -> None:
        s = CBSentimentShiftStrategy(
            CBSentimentConfig(state_path=str(tmp_path / "none.json")),
        )
        assert s.open_positions == {}

    def test_corrupt_file_fails_loud(self, tmp_path) -> None:
        import json
        path = tmp_path / "cb.json"
        path.write_text("{ not valid json")
        with pytest.raises(json.JSONDecodeError):
            CBSentimentShiftStrategy(CBSentimentConfig(state_path=str(path)))

    def test_no_path_is_in_memory_only(self, tmp_path) -> None:
        s = CBSentimentShiftStrategy(CBSentimentConfig(state_path=None))
        s.open_positions["EURUSD"] = OpenPosition(
            symbol="EURUSD", entry_ts=datetime.now(UTC), entry_price=1.10,
            quantity=1000.0, direction=1, stop_loss=1.09,
        )
        s._save_state()  # no path → no-op, nothing written
        assert not list(tmp_path.glob("*.json"))


class _FlatBroker:
    def get_positions(self) -> list[Any]:
        return []

    def get_account(self) -> Any:
        return SimpleNamespace(equity=100_000.0)


class _HoldingBroker:
    def __init__(self, positions: list[Any]) -> None:
        self._positions = positions

    def get_positions(self) -> list[Any]:
        return self._positions

    def get_account(self) -> Any:
        return SimpleNamespace(equity=100_000.0)


def _open_pos(pair: str, qty: float) -> OpenPosition:
    return OpenPosition(
        symbol=pair, entry_ts=datetime.now(UTC), entry_price=1.10,
        quantity=qty, direction=1 if qty > 0 else -1, stop_loss=1.09,
        source_cb="FED",
    )


class TestPhantomDrop:
    """CL-0h30 (P0): a leg the broker never opened (rejected entry) is dropped
    from open_positions at the start of the tick, not managed as a phantom."""

    def test_phantom_dropped_when_broker_flat(self, monkeypatch) -> None:
        strat = CBSentimentShiftStrategy(CBSentimentConfig(max_concurrent_positions=3))
        strat.open_positions["EURUSD"] = _open_pos("EURUSD", 1000.0)
        monkeypatch.setattr(strat, "_check_new_events", lambda: [])
        monkeypatch.setattr(strat, "_refresh_thresholds", lambda: None)
        monkeypatch.setattr(strat, "_update_trailing_stops", lambda prices: [])
        asyncio.run(strat.generate_intents({}, _FlatBroker()))
        assert "EURUSD" not in strat.open_positions

    def test_real_position_kept(self, monkeypatch) -> None:
        from src.execution.broker import Position
        strat = CBSentimentShiftStrategy(CBSentimentConfig(max_concurrent_positions=3))
        strat.open_positions["EURUSD"] = _open_pos("EURUSD", -1000.0)
        broker = _HoldingBroker([Position("EURUSD", -1000.0, 1.10)])
        monkeypatch.setattr(strat, "_check_new_events", lambda: [])
        monkeypatch.setattr(strat, "_refresh_thresholds", lambda: None)
        monkeypatch.setattr(strat, "_update_trailing_stops", lambda prices: [])
        asyncio.run(strat.generate_intents({}, broker))
        assert "EURUSD" in strat.open_positions  # backed by the broker → kept

    def test_phantom_drop_skipped_on_broker_read_failure(self, monkeypatch) -> None:
        class _NoPos:
            def get_account(self) -> Any:
                return SimpleNamespace(equity=100_000.0)
        strat = CBSentimentShiftStrategy(CBSentimentConfig(max_concurrent_positions=3))
        strat.open_positions["EURUSD"] = _open_pos("EURUSD", 1000.0)
        monkeypatch.setattr(strat, "_check_new_events", lambda: [])
        monkeypatch.setattr(strat, "_refresh_thresholds", lambda: None)
        monkeypatch.setattr(strat, "_update_trailing_stops", lambda prices: [])
        asyncio.run(strat.generate_intents({}, _NoPos()))
        assert "EURUSD" in strat.open_positions  # best-effort: belief kept


class _Broker:
    def get_account(self) -> Any:
        return SimpleNamespace(equity=100_000.0)


def _prices(pairs: list[str]) -> dict[str, dict[str, Any]]:
    return {p: {"bid": 1.1000, "ask": 1.1002} for p in pairs}


def _signals(pairs: list[str]) -> list[dict[str, Any]]:
    return [
        {"pair": p, "direction": 1, "cb": "FED", "shift": 0.5, "doc_id": f"d{i}"}
        for i, p in enumerate(pairs)
    ]


def test_cap_enforced_within_single_tick(monkeypatch) -> None:
    cfg = CBSentimentConfig(max_concurrent_positions=3)
    strat = CBSentimentShiftStrategy(cfg)
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]  # 5 fresh signals
    monkeypatch.setattr(strat, "_check_new_events", lambda: _signals(pairs))
    monkeypatch.setattr(strat, "_refresh_thresholds", lambda: None)
    intents = asyncio.run(strat.generate_intents(_prices(pairs), _Broker()))
    entries = [i for i in intents if i.target_position != 0]
    assert len(entries) == 3          # capped in-tick, not 5
    assert len(strat.open_positions) == 3


def test_under_cap_opens_all(monkeypatch) -> None:
    cfg = CBSentimentConfig(max_concurrent_positions=5)
    strat = CBSentimentShiftStrategy(cfg)
    pairs = ["EURUSD", "GBPUSD"]
    monkeypatch.setattr(strat, "_check_new_events", lambda: _signals(pairs))
    monkeypatch.setattr(strat, "_refresh_thresholds", lambda: None)
    intents = asyncio.run(strat.generate_intents(_prices(pairs), _Broker()))
    assert len([i for i in intents if i.target_position != 0]) == 2
