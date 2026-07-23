"""Polymarket emergency kill switch — cancel-all + revoke-allowance (CL-983f).

The "pull the plug" tool for the live Polymarket broker. Two steps, in
order:

  1. **Cancel ALL open orders** on the CLOB for the configured account
     (``py-clob-client``'s ``cancel_all`` — Level-2 auth).
  2. **Revoke the CTFExchange USDC.e allowance**: ERC-20
     ``approve(CTFExchange, 0)`` signed with the hot signer key and sent
     via web3. With the allowance at zero, nothing can settle against
     the wallet even if an order slipped through step 1.

Both steps are attempted even if the first fails — an emergency tool
must not stop halfway because one leg errored. Every action is logged;
any failure lands in ``KillSwitchReport.errors`` and the CLI exits
non-zero.

Deliberately dependency-light: py-clob-client + web3 (both already
required by the live broker), stdlib otherwise. Chain plumbing reuses
``polymarket_chain.make_w3`` and creds come from
``polymarket_secrets.load_polymarket_creds``.

Scope note: this revokes the *default* CTFExchange allowance only —
the venue the live broker (``polymarket_broker.py``, signature_type=0,
neg_risk=False) trades on. If an operator ever enables neg-risk
markets, the NegRiskCtfExchange allowance needs its own revoke.

Usage (via the thin CLI shim in ``scripts/polymarket_kill_switch.py``):

  .venv/bin/python -m scripts.polymarket_kill_switch --env amoy --dry-run
  .venv/bin/python -m scripts.polymarket_kill_switch --env amoy
  .venv/bin/python -m scripts.polymarket_kill_switch --env mainnet --yes-i-mean-it
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvContracts:
    """On-chain addresses the kill switch touches, per environment."""

    ctf_exchange: str
    usdc_collateral: str


# Checksummed contract addresses. Mirrors py-clob-client's
# ``config.get_contract_config`` (neg_risk=False) and the reconciler's
# ``_CTF_EXCHANGE_MAINNET`` — kept as local literals so this tool works
# even if the SDK's config module moves. Verify per-bringup; Polymarket
# has rotated the exchange contract once (V1 -> V2) historically.
CONTRACTS: dict[str, EnvContracts] = {
    "mainnet": EnvContracts(
        ctf_exchange="0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
        usdc_collateral="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    ),
    "amoy": EnvContracts(
        ctf_exchange="0xdFE02Eb6733538f8Ea35D585af8DE5958AD99E40",
        usdc_collateral="0x9c4E1703476E875070EE25b56A58B008CFb8FA78",
    ),
}

# CLOB REST hosts — same table as polymarket_broker.py.
_HOSTS: dict[str, str] = {
    "mainnet": "https://clob.polymarket.com",
    "amoy": "https://clob-amoy.polymarket.com",
}

# Minimal ERC-20 ABI — just the two functions the revoke path needs.
_ERC20_ABI: list[dict[str, Any]] = [
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
]

# How long to wait for the revoke tx to be mined. Polygon block time is
# ~2s; 120s covers severe gas-price congestion before we declare the
# revoke unconfirmed and exit non-zero for the operator to chase.
_RECEIPT_TIMEOUT_SEC: int = 120


@dataclass
class KillSwitchReport:
    """Everything the kill switch did (or would do, in dry-run)."""

    env: str
    dry_run: bool
    open_order_ids: list[str] | None = None
    cancel_response: dict[str, Any] | None = None
    allowance_before: int | None = None
    revoke_tx_hash: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def run_kill_switch(
    env: str,
    *,
    dry_run: bool,
    client: Any | None = None,
    w3: Any | None = None,
    acct: Any | None = None,
    funder_address: str | None = None,
) -> KillSwitchReport:
    """Execute (or dry-run) the full kill-switch sequence for ``env``.

    Args:
      env: "mainnet" | "amoy".
      dry_run: True = read-only. Lists open orders + current allowance,
        prints what WOULD be cancelled/revoked, sends nothing.
      client / w3 / acct / funder_address: injectable for tests. When
        None, built from vault/env creds via the standard plumbing
        (``ClobClient`` + ``polymarket_chain.make_w3``).

    Returns:
      KillSwitchReport. ``report.ok`` is False if ANY step failed —
      the CLI maps that to a non-zero exit.

    Raises:
      ValueError: unknown env. Everything else is captured per-step in
      ``report.errors`` so the second step still runs.
    """
    if env not in CONTRACTS:
        msg = f"unknown polymarket env: {env}"
        raise ValueError(msg)

    report = KillSwitchReport(env=env, dry_run=dry_run)
    logger.warning(
        "polymarket KILL SWITCH engaged: env=%s dry_run=%s",
        env,
        dry_run,
    )

    _step_cancel_all(report, client=client)
    _step_revoke_allowance(
        report,
        w3=w3,
        acct=acct,
        funder_address=funder_address,
    )

    if report.ok:
        logger.warning(
            "polymarket kill switch COMPLETE: env=%s dry_run=%s cancelled=%s revoke_tx=%s",
            env,
            dry_run,
            report.open_order_ids,
            report.revoke_tx_hash,
        )
    else:
        logger.error(
            "polymarket kill switch FINISHED WITH ERRORS: env=%s %s",
            env,
            report.errors,
        )
    return report


# --- step 1: CLOB cancel-all -------------------------------------------


def _step_cancel_all(report: KillSwitchReport, *, client: Any | None) -> None:
    """Cancel every open order for the account. Dry-run only lists them."""
    try:
        if client is None:
            client = _make_clob_client(report.env)

        # Informational listing first — the operator needs to see what
        # is (or would be) cancelled. In dry-run this IS the step, so a
        # listing failure is a hard error there; in live mode we still
        # fire cancel_all even if the listing failed.
        try:
            open_orders = client.get_orders()
            report.open_order_ids = [
                str(o.get("id") or o.get("orderID") or "?") for o in open_orders
            ]
            for oid in report.open_order_ids:
                logger.info(
                    "polymarket kill switch: open order %s%s",
                    oid,
                    " (would cancel)" if report.dry_run else "",
                )
            logger.warning(
                "polymarket kill switch: %d open order(s) on %s",
                len(report.open_order_ids),
                report.env,
            )
        except Exception as exc:
            if report.dry_run:
                raise
            logger.warning(
                "polymarket kill switch: open-order listing failed (%s) — "
                "proceeding to cancel_all anyway",
                exc,
                exc_info=True,
            )

        if report.dry_run:
            logger.warning(
                "polymarket kill switch DRY RUN: would call cancel_all() on %s",
                report.env,
            )
            return

        resp = client.cancel_all()
        report.cancel_response = dict(resp) if resp else {}
        not_cancelled = report.cancel_response.get("not_canceled") or {}
        if not_cancelled:
            msg = f"cancel_all left orders standing: {not_cancelled}"
            raise RuntimeError(msg)
        logger.warning(
            "polymarket kill switch: cancel_all OK: %s",
            report.cancel_response,
        )
    except Exception as exc:
        detail = f"cancel-all step FAILED: {type(exc).__name__}: {exc}"
        report.errors.append(detail)
        logger.exception("polymarket kill switch: %s", detail)


def _make_clob_client(env: str) -> Any:
    """Build a Level-2-authed ClobClient — same recipe as the broker."""
    from py_clob_client.client import ClobClient
    from py_clob_client.constants import AMOY, POLYGON

    from src.execution.polymarket_secrets import load_polymarket_creds

    creds = load_polymarket_creds(env)
    client = ClobClient(
        host=_HOSTS[env],
        key=creds.signer_pk,
        chain_id=POLYGON if env == "mainnet" else AMOY,
        funder=creds.funder_address,
        signature_type=0,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    return client


# --- step 2: revoke CTFExchange allowance ------------------------------


def _step_revoke_allowance(
    report: KillSwitchReport,
    *,
    w3: Any | None,
    acct: Any | None,
    funder_address: str | None,
) -> None:
    """approve(CTFExchange, 0) on the USDC.e contract from the signer key."""
    contracts = CONTRACTS[report.env]
    try:
        if w3 is None or acct is None:
            from src.execution.polymarket_chain import make_w3
            from src.execution.polymarket_secrets import load_polymarket_creds

            w3, acct = make_w3(report.env)
            if funder_address is None:
                funder_address = load_polymarket_creds(report.env).funder_address

        usdc = w3.eth.contract(
            address=contracts.usdc_collateral,
            abi=_ERC20_ABI,
        )
        current = int(
            usdc.functions.allowance(
                acct.address,
                contracts.ctf_exchange,
            ).call(),
        )
        report.allowance_before = current
        logger.warning(
            "polymarket kill switch: allowance(owner=%s, spender=%s) = %d",
            acct.address,
            contracts.ctf_exchange,
            current,
        )

        # approve() only zeroes the SENDER's allowance. If the funder is
        # a different wallet (proxy custody), its allowance needs the
        # cold funder key — flag it loudly but don't fail the hot-path
        # revoke over it.
        if funder_address is not None and funder_address.lower() != str(acct.address).lower():
            try:
                funder_allowance = int(
                    usdc.functions.allowance(
                        funder_address,
                        contracts.ctf_exchange,
                    ).call(),
                )
                if funder_allowance > 0:
                    logger.error(
                        "polymarket kill switch: funder %s still has "
                        "allowance %d for CTFExchange — the hot signer "
                        "key CANNOT revoke it. Revoke manually with the "
                        "cold funder key.",
                        funder_address,
                        funder_allowance,
                    )
            except Exception:
                logger.warning(
                    "polymarket kill switch: funder allowance read failed",
                    exc_info=True,
                )

        if report.dry_run:
            logger.warning(
                "polymarket kill switch DRY RUN: would send approve(%s, 0) on USDC %s from %s",
                contracts.ctf_exchange,
                contracts.usdc_collateral,
                acct.address,
            )
            return

        tx = usdc.functions.approve(contracts.ctf_exchange, 0).build_transaction(
            {
                "from": acct.address,
                "nonce": w3.eth.get_transaction_count(acct.address),
            }
        )
        signed = acct.sign_transaction(tx)
        # web3 v7 / eth-account >= 0.11 use snake_case; keep the camelCase
        # fallback so an older pinned eth-account doesn't break the tool.
        raw = getattr(signed, "raw_transaction", None)
        if raw is None:
            raw = signed.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        tx_hash_hex = tx_hash.hex() if hasattr(tx_hash, "hex") else str(tx_hash)
        logger.warning(
            "polymarket kill switch: revoke tx sent: %s — waiting for receipt (timeout %ds)",
            tx_hash_hex,
            _RECEIPT_TIMEOUT_SEC,
        )

        receipt = w3.eth.wait_for_transaction_receipt(
            tx_hash,
            timeout=_RECEIPT_TIMEOUT_SEC,
        )
        if int(receipt["status"]) != 1:
            msg = f"revoke tx {tx_hash_hex} REVERTED (status=0)"
            raise RuntimeError(msg)

        report.revoke_tx_hash = tx_hash_hex
        logger.warning(
            "polymarket kill switch: allowance revoked: tx=%s",
            tx_hash_hex,
        )
    except Exception as exc:
        detail = f"revoke-allowance step FAILED: {type(exc).__name__}: {exc}"
        report.errors.append(detail)
        logger.exception("polymarket kill switch: %s", detail)


# --- CLI ----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Exit codes: 0 = all steps OK, 1 = a step failed,
    2 = refused (mainnet without the explicit confirmation flag)."""
    p = argparse.ArgumentParser(
        description=(
            "EMERGENCY: cancel ALL open Polymarket CLOB orders and revoke "
            "the CTFExchange USDC.e allowance (approve 0). Use --dry-run "
            "first to see what would happen."
        ),
    )
    p.add_argument(
        "--env",
        required=True,
        choices=sorted(CONTRACTS),
        help="Target environment (explicit — no default).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Read-only: list open orders + current allowance, send nothing.",
    )
    p.add_argument(
        "--yes-i-mean-it",
        action="store_true",
        help="Required to run against mainnet (not needed for --dry-run).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.env == "mainnet" and not args.dry_run and not args.yes_i_mean_it:
        print(
            "REFUSED: --env mainnet cancels real orders and revokes a real "
            "allowance. Re-run with --yes-i-mean-it (or --dry-run).",
        )
        return 2

    report = run_kill_switch(args.env, dry_run=args.dry_run)

    prefix = "[DRY RUN] " if report.dry_run else ""
    print(f"{prefix}env={report.env}")
    print(f"{prefix}open orders: {report.open_order_ids}")
    print(f"{prefix}allowance before: {report.allowance_before}")
    if not report.dry_run:
        print(f"cancel response: {report.cancel_response}")
        print(f"revoke tx: {report.revoke_tx_hash}")
    for err in report.errors:
        print(f"ERROR: {err}")

    return 0 if report.ok else 1
