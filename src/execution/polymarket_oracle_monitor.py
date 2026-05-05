"""UMA optimistic-oracle resolution monitor (CL-poly-3 scaffold).

Polymarket markets resolve via the UMA Optimistic Oracle. The
``UmaCtfAdapter`` contract on Polygon emits ``QuestionInitialized``
when a market is proposed for resolution and ``QuestionResolved`` when
the proposed answer is finalized. Disputes can pause resolution for
days; the monitor surfaces those events to the operator (and,
eventually, the kill switch).

Reference: Plan §4.

Failure modes the monitor watches for:
  * Disputed resolution → market frozen → capital locked.
  * Wrong-resolution risk → DVM returns 0.5 (un-resolvable) → both
    YES and NO redeem at $0.50.
  * Pre-close freeze → trading halts before tokens redeem.

This module is a SCAFFOLD. Wiring to Pushover/Telegram alerts and the
production kill switch happens in CL-poly-3 acceptance.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# UMA CTF Adapter on Polygon mainnet. The adapter contract is the
# bridge between Polymarket's CTF Exchange and UMA's optimistic oracle.
# Verify before each new mainnet bringup — UMA has rotated adapters.
_UMA_CTF_ADAPTER_MAINNET: str = "0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74"

# Polygon block time ~2s. We poll every 30s — plenty of resolution to
# catch disputes in time, light enough on the RPC quota that we don't
# hit Alchemy/Infura rate limits.
_DEFAULT_POLL_INTERVAL_SEC: int = 30

# Block-range chunk size for log queries. Public RPCs cap range at
# ~2000-10000 blocks per request; 2000 is the safe default that works
# across all major providers without retries.
_DEFAULT_LOG_CHUNK_SIZE: int = 2_000


# Minimal ABI for the events we care about. Full adapter ABI is large;
# we only need the resolution events.
_QUESTION_RESOLVED_ABI: dict[str, Any] = {
    "anonymous": False,
    "name": "QuestionResolved",
    "type": "event",
    "inputs": [
        {"indexed": True,  "name": "questionID", "type": "bytes32"},
        {"indexed": False, "name": "outcome",    "type": "uint256"},
        {"indexed": False, "name": "settlementPrice", "type": "int256"},
    ],
}

_QUESTION_INITIALIZED_ABI: dict[str, Any] = {
    "anonymous": False,
    "name": "QuestionInitialized",
    "type": "event",
    "inputs": [
        {"indexed": True,  "name": "questionID", "type": "bytes32"},
        {"indexed": False, "name": "requestTimestamp", "type": "uint256"},
        {"indexed": False, "name": "ancillaryData", "type": "bytes"},
    ],
}


def watch_resolutions(
    w3: Any,
    our_question_ids: set[str],
    on_event: Callable[[dict[str, Any]], None],
    *,
    adapter_address: str = _UMA_CTF_ADAPTER_MAINNET,
    poll_interval_sec: int = _DEFAULT_POLL_INTERVAL_SEC,
    chunk_size: int = _DEFAULT_LOG_CHUNK_SIZE,
    stop_after: int | None = None,
) -> int:
    """Poll for QuestionResolved events on questions in ``our_question_ids``.

    Args:
      w3: connected Web3 client (from ``polymarket_chain.make_w3``).
      our_question_ids: hex-string question IDs we hold positions on.
        Strip 0x prefix; comparison uses .hex() output.
      on_event: callback invoked once per matching event with a
        normalized dict ``{question_id, outcome, settlement_price,
        block_number, tx_hash}``.
      adapter_address: UMA CTF Adapter contract address. Override
        when the adapter rotates.
      poll_interval_sec: gap between polls.
      chunk_size: block-range size per RPC call.
      stop_after: stop after N polls (used in tests). None = run forever.

    Returns:
      Number of matching events delivered.
    """
    contract = w3.eth.contract(
        address=adapter_address, abi=[_QUESTION_RESOLVED_ABI],
    )

    last_block = w3.eth.block_number
    delivered = 0
    polls = 0

    logger.info(
        "polymarket oracle monitor: watching %d questions starting at block %d",
        len(our_question_ids), last_block,
    )

    while True:
        latest = w3.eth.block_number
        if latest > last_block:
            for start in range(last_block + 1, latest + 1, chunk_size):
                end = min(start + chunk_size - 1, latest)
                try:
                    logs = contract.events.QuestionResolved.get_logs(
                        from_block=start, to_block=end,
                    )
                except Exception:
                    logger.warning(
                        "oracle monitor: log fetch failed for blocks "
                        "%d-%d", start, end, exc_info=True,
                    )
                    continue

                for ev in logs:
                    qid = ev["args"]["questionID"].hex()
                    if qid not in our_question_ids:
                        continue
                    payload = {
                        "question_id": qid,
                        "outcome": int(ev["args"]["outcome"]),
                        "settlement_price": int(ev["args"]["settlementPrice"]),
                        "block_number": int(ev["blockNumber"]),
                        "tx_hash": ev["transactionHash"].hex(),
                    }
                    try:
                        on_event(payload)
                    except Exception:
                        logger.exception(
                            "oracle monitor on_event handler raised",
                        )
                    delivered += 1
            last_block = latest

        polls += 1
        if stop_after is not None and polls >= stop_after:
            return delivered
        time.sleep(poll_interval_sec)
