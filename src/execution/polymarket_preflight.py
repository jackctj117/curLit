"""Polymarket pre-flight checks (CL-poly-3).

The mainnet gate. Run AT BROKER CONSTRUCTION TIME (run_engine wires
this in front of PolymarketBroker for the live env). Returns a list
of failure reasons; an empty list = green light.

Reference: Plan §7.

Each check exists because a real failure mode in the plan can produce
silent loss otherwise. Keep them ordered fastest-first so the operator
sees the first failure quickly.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from decimal import Decimal

logger = logging.getLogger(__name__)


# Minimum MATIC balance for gas float. ~0.5 MATIC ≈ 200 typical fills'
# worth of gas at normal Polygon prices, with headroom for spikes.
_MIN_MATIC_GAS_FLOAT: Decimal = Decimal("0.5")

# Minimum USDC balance to permit live trading. Below this, an operator
# error (trying to trade an empty wallet) is more likely than
# a real strategy. Tunable via env var.
_DEFAULT_MIN_USDC_BALANCE: Decimal = Decimal("10")


def run(
    env: str,
    *,
    require_vault: bool = True,
    min_usdc_balance: Decimal = _DEFAULT_MIN_USDC_BALANCE,
    extra_checks: list[Callable[[], str | None]] | None = None,
) -> list[str]:
    """Run all preflight checks. Return a list of failure reason strings;
    empty = pass.

    The mainnet broker rejects construction unless this returns ``[]``.
    Amoy (testnet) skips ``require_vault`` so a developer with env-var
    creds can still smoke-test the pipeline.

    Reference: Plan §7. Operator can add more checks via
    ``extra_checks`` — each callable returns either a failure string or
    None.
    """
    failures: list[str] = []

    failures.extend(_check_creds_present(env))

    if require_vault:
        failures.extend(_check_loaded_via_vault(env))

    failures.extend(_check_chain_connectivity(env))
    failures.extend(_check_balances(env, min_usdc_balance))
    failures.extend(_check_allowance(env))

    if extra_checks:
        for fn in extra_checks:
            try:
                msg = fn()
            except Exception as exc:
                msg = f"extra preflight check raised: {type(exc).__name__}: {exc}"
            if msg:
                failures.append(msg)

    if failures:
        logger.error(
            "polymarket preflight FAILED for env=%s: %d issue(s)",
            env,
            len(failures),
        )
    else:
        logger.info("polymarket preflight PASSED for env=%s", env)
    return failures


# --- individual checks ------------------------------------------------


def _check_creds_present(env: str) -> list[str]:
    try:
        from src.execution.polymarket_secrets import load_polymarket_creds

        load_polymarket_creds(env)
        return []
    except Exception as exc:
        return [f"creds load: {type(exc).__name__}: {exc}"]


def _check_loaded_via_vault(env: str) -> list[str]:
    """Mainnet must NOT use the env-var fallback. Asserts the vault
    path produced creds.

    The override env-var ``POLYMARKET_ALLOW_ENV_FALLBACK=1`` exists so a
    developer doing local mainnet smoke (against a deliberately tiny
    cap) can opt out — but it's loud (printed at WARNING) and would be
    spotted in any audit log.
    """
    if os.environ.get("POLYMARKET_ALLOW_ENV_FALLBACK") == "1":
        logger.warning(
            "POLYMARKET_ALLOW_ENV_FALLBACK=1 set — vault check skipped",
        )
        return []
    try:
        from src.execution.polymarket_secrets import loaded_via_vault

        if not loaded_via_vault(env):
            return [
                "secrets came from env vars, not the vault. "
                "Set vault entries under secret/trading/polymarket/"
                f"{env}/ or set POLYMARKET_ALLOW_ENV_FALLBACK=1 if "
                "you've thought about it.",
            ]
    except Exception as exc:
        return [f"vault check raised: {type(exc).__name__}: {exc}"]
    return []


def _check_chain_connectivity(env: str) -> list[str]:
    try:
        from src.execution.polymarket_chain import assert_rpc_fresh, make_w3

        w3, _ = make_w3(env)
        assert_rpc_fresh(w3)
        return []
    except Exception as exc:
        return [f"chain connectivity: {type(exc).__name__}: {exc}"]


def _check_balances(env: str, min_usdc: Decimal) -> list[str]:
    """USDC.e balance >= min_usdc, MATIC float >= floor.

    The USDC and MATIC checks are read-only RPC calls; they don't sign
    anything. If web3 is missing entirely the check skips with an
    informative failure.
    """
    failures: list[str] = []
    try:
        from src.execution.polymarket_chain import make_w3
        from src.execution.polymarket_secrets import load_polymarket_creds
    except ImportError as exc:
        return [f"balance check: web3/secrets imports failed: {exc}"]

    try:
        w3, _ = make_w3(env)
        creds = load_polymarket_creds(env)
        funder = creds.funder_address
        # MATIC balance — native, in wei (10^18). To human float.
        wei_balance = w3.eth.get_balance(funder)
        matic = Decimal(wei_balance) / Decimal(10**18)
        if matic < _MIN_MATIC_GAS_FLOAT:
            failures.append(
                f"funder MATIC balance {matic} below floor {_MIN_MATIC_GAS_FLOAT}",
            )
    except Exception as exc:
        failures.append(f"matic balance check: {type(exc).__name__}: {exc}")

    # USDC balance check is best-effort — needs the USDC contract
    # ABI which we don't bundle. The CLOB itself reports balance via
    # /balance-allowance; py-clob-client wraps it. Defer to broker
    # construction-time fetch when we have the SDK; for now, just
    # warn that this preflight is a best-effort.
    if min_usdc > 0:
        logger.info(
            "polymarket preflight: USDC balance check delegated to broker "
            "construction (CLOB /balance-allowance call)",
        )

    return failures


def _check_allowance(env: str) -> list[str]:
    """CTFExchange allowance > 0 — read-only ERC-20 allowance check.

    Without an allowance the broker can place but not settle orders.
    The check is delegated to the SDK's balance_allowance call once
    py-clob-client is installed; this scaffold returns a no-op so
    Amoy preflight doesn't block on it before the SDK is wired.
    """
    logger.debug(
        "polymarket preflight: allowance check delegated to broker bringup",
    )
    return []
