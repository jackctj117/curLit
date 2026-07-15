"""Tests for the on-chain reconciler (CL-poly-3 scaffold).

Mocks web3.eth + the OrderFilled event emitter so we don't need a live
Polygon RPC. Verifies dedup, summary counts, and matched/orphan logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.execution.polymarket_reconciler import (
    OnchainFill,
    ReconcileSummary,
    _dedup,
    _normalize,
    reconcile,
)


# Valid 20-byte hex addresses — eth_utils (present when the [polymarket]
# extra is installed) checksum-validates these; short strings like
# "0xfunder" raise ValueError.
_FUNDER = "0x" + "fa" * 20
_OTHER = "0x" + "0b" * 20


def _fake_event(
    order_hash: str = "0xdead",
    block_number: int = 100,
    tx_hash: str = "0xbeef",
    maker: str = _FUNDER,
    taker: str = _OTHER,
):  # type: ignore[no-untyped-def]
    """Test factory. Both order_hash and tx_hash must use hex-only chars
    after the 0x prefix (we pad to 64 nybbles = 32 bytes)."""
    return {
        "args": {
            "orderHash": bytes.fromhex(order_hash[2:].zfill(64)),
            "maker": maker,
            "taker": taker,
            "makerAssetId": 1,
            "takerAssetId": 2,
            "makerAmountFilled": 500_000,
            "takerAmountFilled": 1_000,
            "fee": 0,
        },
        "blockNumber": block_number,
        "transactionHash": bytes.fromhex(tx_hash[2:].zfill(64)),
    }


class TestNormalize:
    def test_event_to_struct(self) -> None:
        ev = _fake_event(order_hash="0xab")
        f = _normalize(ev)
        assert isinstance(f, OnchainFill)
        assert f.maker_amount_filled == 500_000
        assert f.taker_amount_filled == 1_000


class TestDedup:
    def test_drops_repeats_on_tx_orderhash(self) -> None:
        a = _normalize(_fake_event(order_hash="0xab", tx_hash="0xfeed"))
        # Same (tx_hash, order_hash) — must dedup.
        b = _normalize(_fake_event(order_hash="0xab", tx_hash="0xfeed"))
        c = _normalize(_fake_event(order_hash="0xcd", tx_hash="0xfeed"))
        out = _dedup([a, b, c])
        assert len(out) == 2  # a (or b — either) + c


class TestReconcile:
    def _w3_with(self, fills) -> object:  # type: ignore[no-untyped-def]
        """Fake web3 that returns the given fills via OrderFilled.get_logs."""
        contract = MagicMock()
        # Maker filter pass returns all fills; taker pass returns none.
        contract.events.OrderFilled.get_logs.side_effect = [fills, []]
        eth = MagicMock()
        eth.contract.return_value = contract
        eth.block_number = 200
        w3 = MagicMock()
        w3.eth = eth
        return w3

    def test_all_matched_yields_zero_diff(self) -> None:
        ev = _fake_event(order_hash="0xab")
        w3 = self._w3_with([ev])
        journal = [{"order_hash": ev["args"]["orderHash"].hex()}]
        summary = reconcile(
            w3=w3,
            funder_address=_FUNDER,
            journal_fills=journal,
            since_block=100,
        )
        assert isinstance(summary, ReconcileSummary)
        assert summary.matched == 1
        assert summary.onchain_only == 0
        assert summary.journal_only == 0

    def test_onchain_only_flagged(self) -> None:
        ev = _fake_event(order_hash="0xab")
        w3 = self._w3_with([ev])
        # Empty journal — all on-chain fills are unaccounted-for.
        summary = reconcile(
            w3=w3,
            funder_address=_FUNDER,
            journal_fills=[],
            since_block=100,
        )
        assert summary.onchain_count == 1
        assert summary.onchain_only == 1
        assert summary.journal_only == 0

    def test_journal_only_flagged(self) -> None:
        w3 = self._w3_with([])  # no on-chain logs
        summary = reconcile(
            w3=w3,
            funder_address=_FUNDER,
            journal_fills=[{"order_hash": "0xabsentfromchain"}],
            since_block=100,
        )
        assert summary.onchain_count == 0
        assert summary.journal_only == 1
        assert summary.onchain_only == 0
