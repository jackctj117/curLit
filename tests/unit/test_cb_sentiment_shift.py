"""CB sentiment-shift strategy — entry cap (CL-34by).

The max_concurrent_positions cap was checked only once per tick (against the
pre-tick count), so a single tick with several fresh signals could open past
the cap. It must be enforced per-entry within the tick.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from src.strategies.cb_sentiment_shift import (
    CBSentimentConfig,
    CBSentimentShiftStrategy,
)


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
