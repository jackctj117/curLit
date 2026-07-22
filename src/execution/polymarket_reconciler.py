"""On-chain reconciliation for Polymarket fills (CL-poly-3 scaffold).

CL-unlt diffs OANDA REST fills vs the internal trade journal. The same
shape applies to Polymarket — different transport.

Sources of truth, in increasing canonicality:
  1. Internal journal (we already write to ``trade_journal_events``).
  2. CLOB REST ``/data/trades?user=<funder>``. Includes orderID;
     occasionally lossy on edge cases.
  3. **On-chain ``OrderFilled`` events on CTFExchange.** Canonical —
     anything matched and settled emits.

Reference: Plan §6.

This module is a SCAFFOLD. The diff/persist tooling shares with CL-unlt
(``src/execution/broker_reconciliation.py``) and lands at
CL-poly-3 acceptance.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# CTF Exchange contract on Polygon mainnet. The matching contract — when
# our limit orders cross the book the SDK posts to this contract for
# settlement. Verify per-bringup; Polymarket has rotated this once
# (V1 → V2) historically.
_CTF_EXCHANGE_MAINNET: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Maximum block range per RPC query. Public providers cap at 2-10k;
# we use 2k to stay safe across the spread of providers an operator
# might choose. See plan §6 op note.
_DEFAULT_LOG_CHUNK_SIZE: int = 2_000


# OrderFilled event ABI — minimal, just what reconciliation reads.
_ORDER_FILLED_ABI: dict[str, Any] = {
    "anonymous": False,
    "name": "OrderFilled",
    "type": "event",
    "inputs": [
        {"indexed": True,  "name": "orderHash",          "type": "bytes32"},
        {"indexed": True,  "name": "maker",              "type": "address"},
        {"indexed": True,  "name": "taker",              "type": "address"},
        {"indexed": False, "name": "makerAssetId",       "type": "uint256"},
        {"indexed": False, "name": "takerAssetId",       "type": "uint256"},
        {"indexed": False, "name": "makerAmountFilled",  "type": "uint256"},
        {"indexed": False, "name": "takerAmountFilled",  "type": "uint256"},
        {"indexed": False, "name": "fee",                "type": "uint256"},
    ],
}


@dataclass(frozen=True)
class OnchainFill:
    """Normalized OrderFilled event for reconciler diff."""

    order_hash: str
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    maker_amount_filled: int  # raw uint (token decimals applied by caller)
    taker_amount_filled: int
    fee: int
    block_number: int
    tx_hash: str


def fetch_onchain_fills(
    w3: Any,
    funder_address: str,
    from_block: int,
    to_block: int | str = "latest",
    *,
    contract_address: str = _CTF_EXCHANGE_MAINNET,
    chunk_size: int = _DEFAULT_LOG_CHUNK_SIZE,
) -> list[OnchainFill]:
    """Pull OrderFilled events where the funder is maker or taker.

    Two passes (maker-side + taker-side) because the event indexes both
    fields and `argument_filters` only takes one OR'able set.

    Returns events in chronological order.
    """
    # eth_utils is optional. Production runs have it via web3; tests
    # don't pull the EVM stack just for an EIP-55 case-fold step.
    # `str` (not ChecksumAddress) since the ImportError fallback keeps the
    # caller-supplied plain string; ChecksumAddress is a str NewType so the
    # checksummed branch assigns cleanly.
    funder_cs: str
    try:
        # Import from the canonical submodule — the eth_utils package
        # re-exports to_checksum_address without declaring it for mypy.
        from eth_utils.address import to_checksum_address as _checksum
        funder_cs = _checksum(funder_address)
    except ImportError:
        funder_cs = funder_address

    contract = w3.eth.contract(
        address=contract_address, abi=[_ORDER_FILLED_ABI],
    )

    to_block_int = (
        w3.eth.block_number if to_block == "latest" else int(to_block)
    )

    out: list[OnchainFill] = []
    for arg_filter in [{"maker": funder_cs}, {"taker": funder_cs}]:
        for start in range(from_block, to_block_int + 1, chunk_size):
            end = min(start + chunk_size - 1, to_block_int)
            try:
                logs = contract.events.OrderFilled.get_logs(
                    from_block=start, to_block=end,
                    argument_filters=arg_filter,
                )
            except Exception:
                logger.warning(
                    "polymarket reconciler: log fetch failed blocks "
                    "%d-%d filter=%s", start, end, arg_filter,
                    exc_info=True,
                )
                continue
            for ev in logs:
                out.append(_normalize(ev))

    out.sort(key=lambda f: (f.block_number, f.order_hash))
    return _dedup(out)


def _normalize(ev: Any) -> OnchainFill:
    args = ev["args"]
    return OnchainFill(
        order_hash=args["orderHash"].hex(),
        maker=args["maker"],
        taker=args["taker"],
        maker_asset_id=int(args["makerAssetId"]),
        taker_asset_id=int(args["takerAssetId"]),
        maker_amount_filled=int(args["makerAmountFilled"]),
        taker_amount_filled=int(args["takerAmountFilled"]),
        fee=int(args["fee"]),
        block_number=int(ev["blockNumber"]),
        tx_hash=ev["transactionHash"].hex(),
    )


def _dedup(fills: Iterable[OnchainFill]) -> list[OnchainFill]:
    """A fill where the funder is BOTH maker and taker (rare — self-match)
    appears twice in our two-pass pull. Dedup on (tx_hash, order_hash).
    """
    seen: set[tuple[str, str]] = set()
    out: list[OnchainFill] = []
    for f in fills:
        key = (f.tx_hash, f.order_hash)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


@dataclass(frozen=True)
class ReconcileSummary:
    """Returned by ``reconcile()``. Mirrors CL-unlt's report shape so a
    single Grafana dashboard panel handles both venues."""

    onchain_count: int
    journal_count: int
    matched: int
    onchain_only: int
    journal_only: int
    last_reconciled_block: int


def reconcile(
    *,
    w3: Any,
    funder_address: str,
    journal_fills: list[dict[str, Any]],
    since_block: int,
    contract_address: str = _CTF_EXCHANGE_MAINNET,
) -> ReconcileSummary:
    """Compare on-chain fills against the internal trade journal slice.

    ``journal_fills`` is whatever ``read_fills(broker='polymarket',
    since_block=...)`` returns from the existing journal module. Each
    entry must have a ``order_hash`` field that matches the on-chain
    log's ``orderHash``.

    Returns a summary the operator dashboard renders. Real diff
    (which entries are missing) is logged at WARNING level for
    triage; the summary just carries the counts.
    """
    onchain = fetch_onchain_fills(
        w3, funder_address, since_block,
        contract_address=contract_address,
    )
    onchain_by_hash = {f.order_hash: f for f in onchain}
    journal_by_hash = {
        j["order_hash"]: j for j in journal_fills if "order_hash" in j
    }

    matched_keys = set(onchain_by_hash) & set(journal_by_hash)
    onchain_only_keys = set(onchain_by_hash) - matched_keys
    journal_only_keys = set(journal_by_hash) - matched_keys

    if onchain_only_keys:
        logger.warning(
            "polymarket reconcile: %d onchain fills NOT in journal: %s",
            len(onchain_only_keys), sorted(onchain_only_keys)[:20],
        )
    if journal_only_keys:
        logger.warning(
            "polymarket reconcile: %d journal fills NOT onchain: %s",
            len(journal_only_keys), sorted(journal_only_keys)[:20],
        )

    last_block = max(
        (f.block_number for f in onchain), default=since_block,
    )
    return ReconcileSummary(
        onchain_count=len(onchain),
        journal_count=len(journal_fills),
        matched=len(matched_keys),
        onchain_only=len(onchain_only_keys),
        journal_only=len(journal_only_keys),
        last_reconciled_block=last_block,
    )
