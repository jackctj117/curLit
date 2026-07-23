"""PaperBroker.stream_prices fail-closed ticks (CL-8lv6 P0).

The old stream fabricated bid=1.1000/ask=1.1002 for EVERY requested symbol,
ignoring set_price — the live engine's paper mode fills _last_prices from
this stream, so all instruments were marked ~1.10 and paper-soak
sizing/stops/PnL were meaningless. Now, like get_price: only symbols with a
real set_price value are yielded (deterministic ±0.5 bp synthetic spread
around the current mid); unpriced symbols are ABSENT, never fabricated, and
set_price updates are reflected on the next pass.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.execution.paper_broker import STREAM_HALF_SPREAD_BPS, PaperBroker


def _collect(
    broker: PaperBroker,
    symbols: list[str],
    n: int,
) -> list[dict[str, Any]]:
    """Pull n ticks from the (infinite) stream, with a hang guard."""

    async def run() -> list[dict[str, Any]]:
        agen = broker.stream_prices(symbols)
        ticks: list[dict[str, Any]] = []
        try:
            for _ in range(n):
                ticks.append(await asyncio.wait_for(agen.__anext__(), 5.0))
        finally:
            await agen.aclose()
        return ticks

    return asyncio.run(run())


def _fast_broker() -> PaperBroker:
    return PaperBroker(stream_interval_sec=0.001)


class TestStreamPrices:
    def test_priced_symbol_ticks_around_set_price(self) -> None:
        b = _fast_broker()
        b.set_price("EURUSD", 1.1000, 1.1002)
        (tick,) = _collect(b, ["EURUSD"], 1)
        assert tick["symbol"] == "EURUSD"
        assert set(tick) == {"symbol", "bid", "ask", "ts"}  # engine shape
        mid = 1.1001
        assert tick["bid"] < mid < tick["ask"]
        assert (tick["bid"] + tick["ask"]) / 2 == pytest.approx(mid)
        # Deterministic ±0.5 bp synthetic spread.
        half = mid * (STREAM_HALF_SPREAD_BPS / 10_000.0)
        assert tick["bid"] == pytest.approx(mid - half)
        assert tick["ask"] == pytest.approx(mid + half)

    def test_unpriced_symbol_never_appears(self) -> None:
        b = _fast_broker()
        b.set_price("EURUSD", 1.1000, 1.1002)
        # 3 ticks span >=3 stream passes: GBPUSD was skipped every pass,
        # not merely reordered.
        ticks = _collect(b, ["EURUSD", "GBPUSD"], 3)
        assert [t["symbol"] for t in ticks] == ["EURUSD"] * 3

    def test_no_priced_symbols_yields_nothing(self) -> None:
        # Fail closed all the way: nothing priced -> nothing fabricated.
        b = _fast_broker()

        async def run() -> None:
            agen = b.stream_prices(["XAUUSD"])
            try:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(agen.__anext__(), 0.05)
            finally:
                await agen.aclose()

        asyncio.run(run())

    def test_set_price_update_is_reflected(self) -> None:
        b = _fast_broker()
        b.set_price("EURUSD", 1.1000, 1.1002)

        async def run() -> tuple[dict[str, Any], dict[str, Any]]:
            agen = b.stream_prices(["EURUSD"])
            try:
                first = await asyncio.wait_for(agen.__anext__(), 5.0)
                b.set_price("EURUSD", 1.2000, 1.2002)
                second = await asyncio.wait_for(agen.__anext__(), 5.0)
            finally:
                await agen.aclose()
            return first, second

        first, second = asyncio.run(run())
        assert (first["bid"] + first["ask"]) / 2 == pytest.approx(1.1001)
        assert (second["bid"] + second["ask"]) / 2 == pytest.approx(1.2001)

    def test_symbol_priced_mid_stream_starts_ticking(self) -> None:
        # The engine's paper mode may wire set_price after the stream task
        # starts — the symbol must appear as soon as it gains a price.
        b = _fast_broker()
        b.set_price("EURUSD", 1.1000, 1.1002)

        async def run() -> list[str]:
            agen = b.stream_prices(["EURUSD", "GBPUSD"])
            seen: list[str] = []
            try:
                seen.append((await asyncio.wait_for(agen.__anext__(), 5.0))["symbol"])
                b.set_price("GBPUSD", 1.3000, 1.3002)
                for _ in range(2):
                    seen.append(
                        (await asyncio.wait_for(agen.__anext__(), 5.0))["symbol"],
                    )
            finally:
                await agen.aclose()
            return seen

        seen = asyncio.run(run())
        assert seen[0] == "EURUSD"
        assert "GBPUSD" in seen[1:]
