"""Tests for the Polymarket secrets loader (CL-poly-3 scaffold)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.execution.polymarket_secrets import (
    PolymarketCreds,
    load_polymarket_creds,
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
        env_vars = _env_for("mainnet")
        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch(
                "src.execution.polymarket_secrets._try_vault",
                return_value=None,
            ),
        ):
            creds = load_polymarket_creds("mainnet")
        assert creds.chain_id == 137

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
