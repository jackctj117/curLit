"""Polymarket secrets loader (CL-poly-3 scaffold).

Mirrors the OANDA secrets pattern (vault-isolated, never on disk
plaintext). Loads:

  signer_pk          # 0x-prefixed hex private key for EIP-712 signing
  api_key            # uuid returned by /auth/api-key
  api_secret         # base64 hmac secret
  api_passphrase     # passphrase string
  funder_address     # 0x... checksummed; the wallet holding USDC
  rpc_url            # e.g. Alchemy/Infura Polygon endpoint

Vault layout per environment:

  secret/trading/polymarket/mainnet/{signer_pk, api_key, api_secret,
                                     api_passphrase, funder_address, rpc_url}
  secret/trading/polymarket/amoy/{...}        # testnet mirror

Two key roles, separable:
  - **Funder key** (cold/manual): deposits USDC.e, approves
    CTFExchange allowance. Used rarely; stays cold.
  - **Signer/API key** (hot, vault-resident): signs EIP-712 limit orders,
    submits to CLOB. This is what the broker uses at runtime.

Reference: Plan §1.

This module is a SCAFFOLD. Wiring to the real vault agent (CL-r48)
plus per-load audit logging happens in the live broker bringup
(CL-poly-3 acceptance).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Polygon mainnet chain id — fixed by the network.
_POLYGON_MAINNET_CHAIN_ID: int = 137
# Amoy testnet replaces the deprecated Mumbai testnet (2024).
_AMOY_TESTNET_CHAIN_ID: int = 80002


@dataclass(frozen=True)
class PolymarketCreds:
    """All credentials needed to sign + submit Polymarket orders.

    Frozen so the structure can't be accidentally mutated mid-run.
    The signer_pk lives in memory only — never written to disk by
    this module."""

    signer_pk: str
    api_key: str
    api_secret: str
    api_passphrase: str
    funder_address: str
    chain_id: int
    rpc_url: str


#: Loaded creds per env. Was an ``lru_cache`` (CL-co66): equivalent
#: retention, but a bare lru_cache gives the operator NO way to reason about
#: — or purge — hot signer-key material sitting in the process. This explicit
#: cache is the same memoization with a documented lifetime and a real
#: :func:`clear_polymarket_creds` kill path.
_CREDS_CACHE: dict[str, PolymarketCreds] = {}
#: env → did the VAULT path produce these creds (vs the env-var fallback)?
#: Recorded at load so :func:`loaded_via_vault` never has to re-read the
#: signer key out of the vault just to answer the question.
_VAULT_SOURCED: dict[str, bool] = {}


def clear_polymarket_creds(env: str | None = None) -> None:
    """Drop cached creds (all envs, or just ``env``) — CL-co66.

    Purges this module's reference to the hot signer key. Note this cannot
    scrub copies already handed to callers (``ClobClient``, ``LocalAccount``),
    and Python strings are immutable so the bytes may persist until GC; it
    bounds THIS module's retention, which is what it can honestly promise."""
    if env is None:
        _CREDS_CACHE.clear()
        _VAULT_SOURCED.clear()
    else:
        _CREDS_CACHE.pop(env, None)
        _VAULT_SOURCED.pop(env, None)


def load_polymarket_creds(env: str) -> PolymarketCreds:
    """Return PolymarketCreds for ``env`` ∈ {"mainnet", "amoy"}.

    Reads from the vault agent (CL-r48). On **amoy/dev** an env-var fallback
    is allowed for workstation convenience.

    On **mainnet the env-var fallback is REFUSED** (CL-co66): a real-money
    signing key must never come from the process environment, where it lands
    in ``.env`` files, shell history, crash dumps and child processes. The
    preflight already asserted this after the fact; refusing at the LOADER
    makes it structural rather than a check someone can forget to run.

    Results are cached per env — see :data:`_CREDS_CACHE` and
    :func:`clear_polymarket_creds`.
    """
    if env not in {"mainnet", "amoy"}:
        msg = f"unknown polymarket env: {env}"
        raise ValueError(msg)
    cached = _CREDS_CACHE.get(env)
    if cached is not None:
        return cached

    chain_id = _POLYGON_MAINNET_CHAIN_ID if env == "mainnet" else _AMOY_TESTNET_CHAIN_ID

    # Try vault first. Lazy import so a development install without
    # the vault agent module still loads this file.
    creds_dict = _try_vault(env)
    from_vault = creds_dict is not None
    if creds_dict is None:
        if env == "mainnet":
            msg = (
                "polymarket mainnet creds must come from the VAULT — refusing the "
                "env-var fallback for a real-money signer key (CL-co66). Start the "
                "vault agent and store secret/trading/polymarket/mainnet/*."
            )
            raise RuntimeError(msg)
        creds_dict = _fallback_env(env)

    creds = PolymarketCreds(
        signer_pk=creds_dict["signer_pk"],
        api_key=creds_dict["api_key"],
        api_secret=creds_dict["api_secret"],
        api_passphrase=creds_dict["api_passphrase"],
        funder_address=creds_dict["funder_address"],
        chain_id=chain_id,
        rpc_url=creds_dict["rpc_url"],
    )
    _CREDS_CACHE[env] = creds
    _VAULT_SOURCED[env] = from_vault
    return creds


#: Back-compat shim: the loader used to be an ``lru_cache``, so callers and
#: tests reset it via ``load_polymarket_creds.cache_clear()``.
load_polymarket_creds.cache_clear = clear_polymarket_creds  # type: ignore[attr-defined]


def _try_vault(env: str) -> dict[str, str] | None:
    """Return creds dict from the vault agent, or None if unavailable.

    We swallow ImportError + connection errors so the dev path (env vars)
    still works on a workstation without the vault running. Production
    preflight asserts the vault path WAS used."""
    try:
        from src.security.vault_client import VaultClient
    except ImportError:
        logger.debug(
            "polymarket secrets: vault client not importable — trying env-var fallback",
        )
        return None

    try:
        v = VaultClient()
        base = f"trading/polymarket/{env}"
        return {
            "signer_pk": v.get(f"{base}/signer_pk"),
            "api_key": v.get(f"{base}/api_key"),
            "api_secret": v.get(f"{base}/api_secret"),
            "api_passphrase": v.get(f"{base}/api_passphrase"),
            "funder_address": v.get(f"{base}/funder_address"),
            "rpc_url": v.get(f"{base}/rpc_url"),
        }
    except Exception:
        logger.warning(
            "polymarket secrets: vault read failed — trying env-var fallback",
            exc_info=True,
        )
        return None


def _fallback_env(env: str) -> dict[str, str]:
    """Read from environment variables. Used in development. The
    PRODUCTION preflight (polymarket_preflight) asserts this path was
    NOT used on mainnet."""
    prefix = f"POLYMARKET_{env.upper()}_"
    keys = [
        "SIGNER_PK",
        "API_KEY",
        "API_SECRET",
        "API_PASSPHRASE",
        "FUNDER_ADDRESS",
        "RPC_URL",
    ]
    out: dict[str, str] = {}
    missing: list[str] = []
    for k in keys:
        full = prefix + k
        v = os.environ.get(full)
        if not v:
            missing.append(full)
        else:
            out[k.lower()] = v
    if missing:
        msg = f"polymarket {env} creds missing from vault AND env vars: {missing}"
        raise RuntimeError(msg)
    logger.warning(
        "polymarket %s creds loaded from env vars (DEV path) — production must use vault",
        env,
    )
    return out


def loaded_via_vault(env: str) -> bool:
    """Used by the preflight check: did the vault path produce creds?

    Returns True iff a real vault read produced the creds now in use. False
    means the env-var fallback fired. Mainnet preflight asserts True (and
    since CL-co66 mainnet cannot reach the fallback at all).

    Reads the flag RECORDED AT LOAD rather than re-running ``_try_vault``:
    the old version pulled the signer key out of the vault a second time —
    an extra round-trip and an extra copy of hot key material in memory —
    purely to answer a yes/no question. Loads once if nothing is cached yet.
    """
    if env not in _VAULT_SOURCED:
        load_polymarket_creds(env)
    return _VAULT_SOURCED.get(env, False)
