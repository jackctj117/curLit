"""Tests for the Polymarket kill switch (CL-983f).

Everything is mocked — no CLOB, no RPC, no signing key. Covers:
  * dry-run: lists orders + allowance, sends NOTHING
  * action path: cancel_all fired, approve(CTFExchange, 0) tx built
    with the right spender and actually broadcast
  * fail-loud: step failures land in report.errors (and both steps
    still run), CLI exits non-zero
  * mainnet requires --yes-i-mean-it
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.execution.polymarket_kill_switch import (
    CONTRACTS,
    KillSwitchReport,
    main,
    run_kill_switch,
)

_TX_HASH_BYTES = bytes.fromhex("ab" * 32)


def _mock_client(
    orders: list[dict] | None = None,
    cancel_response: dict | None = None,
) -> MagicMock:
    client = MagicMock(name="ClobClient")
    client.get_orders.return_value = (
        orders if orders is not None else [{"id": "ord-1"}, {"id": "ord-2"}]
    )
    client.cancel_all.return_value = (
        cancel_response
        if cancel_response is not None
        else {"canceled": ["ord-1", "ord-2"], "not_canceled": {}}
    )
    return client


def _mock_chain(allowance: int = 123_000_000) -> tuple[MagicMock, MagicMock]:
    """(w3, acct) mocks wired for the allowance-read + approve(0) path."""
    w3 = MagicMock(name="Web3")
    contract = w3.eth.contract.return_value
    contract.functions.allowance.return_value.call.return_value = allowance
    contract.functions.approve.return_value.build_transaction.return_value = {
        "from": "0xSigner", "nonce": 7,
    }
    w3.eth.get_transaction_count.return_value = 7
    w3.eth.send_raw_transaction.return_value = _TX_HASH_BYTES
    w3.eth.wait_for_transaction_receipt.return_value = {"status": 1}

    acct = MagicMock(name="LocalAccount")
    acct.address = "0xSigner"
    acct.sign_transaction.return_value = SimpleNamespace(
        raw_transaction=b"signed-raw-tx",
    )
    return w3, acct


class TestDryRun:
    def test_dry_run_sends_nothing(self) -> None:
        client = _mock_client()
        w3, acct = _mock_chain()

        report = run_kill_switch(
            "amoy", dry_run=True, client=client, w3=w3, acct=acct,
        )

        assert report.ok
        assert report.dry_run is True
        client.cancel_all.assert_not_called()
        w3.eth.send_raw_transaction.assert_not_called()
        acct.sign_transaction.assert_not_called()

    def test_dry_run_reports_orders_and_allowance(self) -> None:
        client = _mock_client(orders=[{"id": "abc"}])
        w3, acct = _mock_chain(allowance=42_000_000)

        report = run_kill_switch(
            "amoy", dry_run=True, client=client, w3=w3, acct=acct,
        )

        assert report.open_order_ids == ["abc"]
        assert report.allowance_before == 42_000_000
        assert report.revoke_tx_hash is None
        assert report.cancel_response is None

    def test_dry_run_listing_failure_is_an_error(self) -> None:
        # In dry-run, the listing IS the cancel step — a failure there
        # means the dry run couldn't verify anything.
        client = _mock_client()
        client.get_orders.side_effect = ConnectionError("CLOB down")
        w3, acct = _mock_chain()

        report = run_kill_switch(
            "amoy", dry_run=True, client=client, w3=w3, acct=acct,
        )

        assert not report.ok
        assert any("cancel-all" in e for e in report.errors)


class TestActionPath:
    def test_cancel_all_called(self) -> None:
        client = _mock_client()
        w3, acct = _mock_chain()

        report = run_kill_switch(
            "amoy", dry_run=False, client=client, w3=w3, acct=acct,
        )

        assert report.ok, report.errors
        client.cancel_all.assert_called_once_with()
        assert report.cancel_response == {
            "canceled": ["ord-1", "ord-2"], "not_canceled": {},
        }

    @pytest.mark.parametrize("env", ["amoy", "mainnet"])
    def test_approve_zero_built_for_right_spender(self, env: str) -> None:
        client = _mock_client()
        w3, acct = _mock_chain()

        report = run_kill_switch(
            env, dry_run=False, client=client, w3=w3, acct=acct,
        )

        assert report.ok, report.errors
        # The ERC-20 contract is the env's USDC.e collateral...
        assert w3.eth.contract.call_count == 1
        assert (
            w3.eth.contract.call_args.kwargs["address"]
            == CONTRACTS[env].usdc_collateral
        )
        # ...and the approval zeroed is for the env's CTFExchange.
        contract = w3.eth.contract.return_value
        contract.functions.approve.assert_called_once_with(
            CONTRACTS[env].ctf_exchange, 0,
        )
        contract.functions.allowance.assert_any_call(
            acct.address, CONTRACTS[env].ctf_exchange,
        )

    def test_revoke_tx_signed_and_broadcast(self) -> None:
        client = _mock_client()
        w3, acct = _mock_chain()

        report = run_kill_switch(
            "amoy", dry_run=False, client=client, w3=w3, acct=acct,
        )

        acct.sign_transaction.assert_called_once()
        w3.eth.send_raw_transaction.assert_called_once_with(b"signed-raw-tx")
        assert report.revoke_tx_hash == _TX_HASH_BYTES.hex()

    def test_cancel_failure_still_revokes(self) -> None:
        # Emergency semantics: a dead CLOB must not stop the on-chain
        # revoke. Both steps run; the report carries the failure.
        client = _mock_client()
        client.cancel_all.side_effect = ConnectionError("CLOB down")
        w3, acct = _mock_chain()

        report = run_kill_switch(
            "amoy", dry_run=False, client=client, w3=w3, acct=acct,
        )

        assert not report.ok
        assert any("cancel-all" in e for e in report.errors)
        w3.eth.send_raw_transaction.assert_called_once()
        assert report.revoke_tx_hash == _TX_HASH_BYTES.hex()

    def test_reverted_revoke_tx_is_an_error(self) -> None:
        client = _mock_client()
        w3, acct = _mock_chain()
        w3.eth.wait_for_transaction_receipt.return_value = {"status": 0}

        report = run_kill_switch(
            "amoy", dry_run=False, client=client, w3=w3, acct=acct,
        )

        assert not report.ok
        assert any("REVERTED" in e for e in report.errors)
        assert report.revoke_tx_hash is None

    def test_cancel_all_leaving_orders_is_an_error(self) -> None:
        client = _mock_client(
            cancel_response={
                "canceled": ["ord-1"],
                "not_canceled": {"ord-2": "stuck"},
            },
        )
        w3, acct = _mock_chain()

        report = run_kill_switch(
            "amoy", dry_run=False, client=client, w3=w3, acct=acct,
        )

        assert not report.ok
        assert any("standing" in e for e in report.errors)

    def test_funder_mismatch_does_not_fail_hot_revoke(self) -> None:
        # A distinct cold funder with its own allowance is flagged in
        # logs but must not break the signer-side revoke.
        client = _mock_client()
        w3, acct = _mock_chain()

        report = run_kill_switch(
            "amoy", dry_run=False, client=client, w3=w3, acct=acct,
            funder_address="0xCOLDFUNDER",
        )

        assert report.ok, report.errors


class TestUnknownEnv:
    def test_unknown_env_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown polymarket env"):
            run_kill_switch("mumbai", dry_run=True)


class TestCLI:
    def test_mainnet_without_confirmation_refused(self) -> None:
        with patch(
            "src.execution.polymarket_kill_switch.run_kill_switch",
        ) as run_mock:
            rc = main(["--env", "mainnet"])
        assert rc == 2
        run_mock.assert_not_called()

    def test_mainnet_dry_run_needs_no_confirmation(self) -> None:
        report = KillSwitchReport(env="mainnet", dry_run=True)
        with patch(
            "src.execution.polymarket_kill_switch.run_kill_switch",
            return_value=report,
        ) as run_mock:
            rc = main(["--env", "mainnet", "--dry-run"])
        assert rc == 0
        run_mock.assert_called_once_with("mainnet", dry_run=True)

    def test_mainnet_with_confirmation_runs(self) -> None:
        report = KillSwitchReport(env="mainnet", dry_run=False)
        with patch(
            "src.execution.polymarket_kill_switch.run_kill_switch",
            return_value=report,
        ) as run_mock:
            rc = main(["--env", "mainnet", "--yes-i-mean-it"])
        assert rc == 0
        run_mock.assert_called_once_with("mainnet", dry_run=False)

    def test_step_failure_exits_nonzero(self) -> None:
        report = KillSwitchReport(env="amoy", dry_run=False)
        report.errors.append("revoke-allowance step FAILED: boom")
        with patch(
            "src.execution.polymarket_kill_switch.run_kill_switch",
            return_value=report,
        ):
            rc = main(["--env", "amoy"])
        assert rc == 1

    def test_env_is_required(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_script_shim_importable(self) -> None:
        # The operator-facing entrypoint must resolve to the module CLI.
        from scripts.polymarket_kill_switch import main as script_main
        assert script_main is main
