"""Tests for the Polymarket secrets loader (CL-poly-3 scaffold)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.execution.polymarket_secrets import (
    PolymarketCreds,
    clear_polymarket_creds,
    load_polymarket_creds,
    loaded_via_vault,
)

_REQUIRED_ENV: tuple[tuple[str, str], ...] = (
    ("SIGNER_PK", "0xabc"),
    ("API_KEY", "uuid-1234"),
    ("API_SECRET", "secret-1234"),
    ("API_PASSPHRASE", "pass-1234"),
    ("FUNDER_ADDRESS", "0xfunder"),
    ("RPC_URL", "https://example/rpc"),
)


def _env_for(env: str) -> dict[str, str]:
    prefix = f"POLYMARKET_{env.upper()}_"
    return {prefix + k: v for k, v in _REQUIRED_ENV}


class TestEnvFallback:
    def setup_method(self) -> None:
        # lru_cache means the loader memoizes per-env. Reset before
        # each test so env-var changes are observed.
        load_polymarket_creds.cache_clear()

    def test_amoy_loads_from_env(self) -> None:
        env_vars = _env_for("amoy")
        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch(
                "src.execution.polymarket_secrets._try_vault",
                return_value=None,
            ),
        ):
            creds = load_polymarket_creds("amoy")
        assert isinstance(creds, PolymarketCreds)
        assert creds.signer_pk == "0xabc"
        assert creds.api_key == "uuid-1234"
        assert creds.chain_id == 80002  # Amoy

    def test_mainnet_chain_id(self) -> None:
        # Mainnet must come from the vault (CL-co66), so exercise chain_id
        # through the vault path rather than the refused env fallback.
        vault = {k.lower(): v for k, v in _REQUIRED_ENV}
        with patch(
            "src.execution.polymarket_secrets._try_vault",
            return_value=vault,
        ):
            creds = load_polymarket_creds("mainnet")
        assert creds.chain_id == 137

    def test_mainnet_refuses_env_fallback(self) -> None:
        """CL-co66: a real-money signing key must never come from the process
        environment (.env files, shell history, crash dumps, child procs).
        The preflight asserted this after the fact; the loader now refuses."""
        env_vars = _env_for("mainnet")
        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch(
                "src.execution.polymarket_secrets._try_vault",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="must come from the VAULT"),
        ):
            load_polymarket_creds("mainnet")

    def test_amoy_still_allows_env_fallback(self) -> None:
        # Dev/testnet convenience is unchanged — only mainnet is hardened.
        with (
            patch.dict(os.environ, _env_for("amoy"), clear=False),
            patch(
                "src.execution.polymarket_secrets._try_vault",
                return_value=None,
            ),
        ):
            assert load_polymarket_creds("amoy").signer_pk == "0xabc"

    def test_missing_env_var_raises(self) -> None:
        # Set everything except RPC_URL.
        env_vars = _env_for("amoy")
        env_vars.pop("POLYMARKET_AMOY_RPC_URL")
        # Clean up any residual mainnet vars too.
        with (
            patch.dict(
                os.environ,
                {**env_vars},
                clear=True,
            ),
            patch(
                "src.execution.polymarket_secrets._try_vault",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="missing from vault AND env"),
        ):
            load_polymarket_creds("amoy")


class TestVaultPath:
    def setup_method(self) -> None:
        load_polymarket_creds.cache_clear()

    def test_vault_returns_dict_short_circuits_env(self) -> None:
        canned = {
            "signer_pk": "vault-key",
            "api_key": "vault-uuid",
            "api_secret": "vault-secret",
            "api_passphrase": "vault-pass",
            "funder_address": "0xvault",
            "rpc_url": "https://vault/rpc",
        }
        # Even with bad env vars, vault wins.
        with (
            patch(
                "src.execution.polymarket_secrets._try_vault",
                return_value=canned,
            ),
        ):
            creds = load_polymarket_creds("mainnet")
        assert creds.signer_pk == "vault-key"
        assert creds.rpc_url == "https://vault/rpc"


class TestEnvValidation:
    def test_unknown_env_raises(self) -> None:
        load_polymarket_creds.cache_clear()
        with pytest.raises(ValueError, match="unknown polymarket env"):
            load_polymarket_creds("kovan")


class TestCredRetention:
    """CL-co66: the loader memoized creds in a bare lru_cache, so hot signer-key
    material sat in the process with no way to reason about or purge it."""

    def setup_method(self) -> None:
        clear_polymarket_creds()

    def teardown_method(self) -> None:
        clear_polymarket_creds()

    def _vault(self) -> dict[str, str]:
        return {k.lower(): v for k, v in _REQUIRED_ENV}

    def test_creds_are_cached_then_purgeable(self) -> None:
        with patch(
            "src.execution.polymarket_secrets._try_vault",
            return_value=self._vault(),
        ) as vault:
            first = load_polymarket_creds("mainnet")
            second = load_polymarket_creds("mainnet")
            assert first is second  # memoized — one vault read
            assert vault.call_count == 1

            clear_polymarket_creds("mainnet")
            third = load_polymarket_creds("mainnet")
            assert third is not first  # purged → re-read
            assert vault.call_count == 2

    def test_clear_all_envs(self) -> None:
        from src.execution import polymarket_secrets as mod

        with patch(
            "src.execution.polymarket_secrets._try_vault",
            return_value=self._vault(),
        ):
            load_polymarket_creds("mainnet")
            load_polymarket_creds("amoy")
        assert mod._CREDS_CACHE  # populated
        clear_polymarket_creds()
        assert mod._CREDS_CACHE == {}  # no signer key retained here
        assert mod._VAULT_SOURCED == {}

    def test_loaded_via_vault_does_not_reread_the_signer_key(self) -> None:
        """The old loaded_via_vault() called _try_vault() again — pulling the
        signer key out of the vault a second time (extra round-trip, extra
        copy of hot key material) just to answer a yes/no question."""
        with patch(
            "src.execution.polymarket_secrets._try_vault",
            return_value=self._vault(),
        ) as vault:
            load_polymarket_creds("mainnet")
            assert vault.call_count == 1
            assert loaded_via_vault("mainnet") is True
            assert vault.call_count == 1  # answered from the recorded flag

    def test_loaded_via_vault_false_on_env_fallback(self) -> None:
        with (
            patch.dict(os.environ, _env_for("amoy"), clear=False),
            patch(
                "src.execution.polymarket_secrets._try_vault",
                return_value=None,
            ),
        ):
            load_polymarket_creds("amoy")
            assert loaded_via_vault("amoy") is False
