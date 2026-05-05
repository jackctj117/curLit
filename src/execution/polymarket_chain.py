"""Web3 client + signer wiring for Polymarket (CL-poly-3 scaffold).

Single entrypoint ``make_w3(env)`` returns a connected ``Web3`` and a
``LocalAccount`` (signer key in memory only). Both web3 and eth-account
are optional deps — if not installed the function raises ImportError
with an actionable message rather than crashing somewhere deep.

Reference: Plan §1.

This module deliberately doesn't expose the private key after
construction. The Account object's signer_pk is derived once and
attached to the LocalAccount; everything else uses signed messages.
"""

from __future__ import annotations

import logging
from typing import Any

from src.execution.polymarket_secrets import load_polymarket_creds

logger = logging.getLogger(__name__)


# 10s default RPC timeout. Polygon RPC providers (Alchemy, Infura) can
# spike to ~3s on bad days; 10s catches a stuck connection without
# wedging the broker indefinitely.
_DEFAULT_RPC_TIMEOUT_SEC: int = 10

# Wall-clock vs latest block tolerance for the preflight check.
# Polygon block time is ~2s; if the latest block is more than 30s old
# the RPC is either lagging badly or returning a stale tip and we
# should not place orders against it.
_RPC_FRESHNESS_TOLERANCE_SEC: int = 30


def make_w3(env: str) -> tuple[Any, Any]:
    """Return ``(web3.Web3, eth_account.LocalAccount)`` for ``env``.

    Args:
      env: "mainnet" | "amoy".

    Raises:
      ImportError: web3 or eth-account missing. Install via the
        ``polymarket`` extras: ``pip install '.[polymarket]'``.
      AssertionError: RPC unreachable, wrong chain id, or stale block.
      RuntimeError: vault creds missing.
    """
    try:
        from eth_account import Account
        from web3 import Web3
    except ImportError as exc:
        msg = (
            "web3 + eth-account required for Polymarket live broker. "
            "Install via `pip install web3 eth-account` or use the "
            "polymarket extras."
        )
        raise ImportError(msg) from exc

    creds = load_polymarket_creds(env)

    w3 = Web3(Web3.HTTPProvider(
        creds.rpc_url,
        request_kwargs={"timeout": _DEFAULT_RPC_TIMEOUT_SEC},
    ))
    if not w3.is_connected():
        msg = f"polymarket RPC unreachable at {creds.rpc_url}"
        raise AssertionError(msg)

    chain_id = w3.eth.chain_id
    if chain_id != creds.chain_id:
        msg = (
            f"polymarket RPC returned chain_id={chain_id}, "
            f"expected {creds.chain_id} for env={env}"
        )
        raise AssertionError(msg)

    acct = Account.from_key(creds.signer_pk)
    logger.info(
        "polymarket chain ready: env=%s chain_id=%d signer=%s funder=%s",
        env, chain_id, acct.address, creds.funder_address,
    )
    return w3, acct


def latest_block_age_sec(w3: Any) -> float:
    """Wall-clock age of the latest block. Used by preflight."""
    import time as _time

    block = w3.eth.get_block("latest")
    return float(_time.time() - block["timestamp"])


def assert_rpc_fresh(w3: Any) -> None:
    """Raise AssertionError if the latest block is suspiciously old."""
    age = latest_block_age_sec(w3)
    if age > _RPC_FRESHNESS_TOLERANCE_SEC:
        msg = (
            f"polymarket RPC stale: latest block is {age:.1f}s old, "
            f"tolerance is {_RPC_FRESHNESS_TOLERANCE_SEC}s. The RPC "
            f"provider may be lagging — switch endpoint or wait."
        )
        raise AssertionError(msg)
