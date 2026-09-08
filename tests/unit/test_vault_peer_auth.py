"""Vault agent Unix-socket peer authentication (CL-xdnh P1).

Hermetic: sockets live in a short tempdir (AF_UNIX path limit ~104 chars on
macOS makes pytest's tmp_path too long), vault files in tmp_path; the real
vault / .env are never touched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import stat
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.security import vault_agent, vault_wire

pytestmark = pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"),
    reason="AF_UNIX peer credentials need Linux or macOS",
)


@pytest.fixture
def sock_dir():
    d = tempfile.mkdtemp(prefix="vsock-")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def _reset_warn_flag(monkeypatch):
    monkeypatch.setattr(vault_agent, "_warned_unsupported", False)


# --- peer_uid --------------------------------------------------------------


def test_peer_uid_socketpair_reports_own_uid():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert vault_agent.peer_uid(a) == os.getuid()
        assert vault_agent.peer_uid(b) == os.getuid()
    finally:
        a.close()
        b.close()


def test_peer_uid_none_on_query_failure(monkeypatch):
    """A socket that can't answer the credential query -> None (fail closed)."""

    class Broken:
        def getsockopt(self, *args):
            raise OSError("no creds")

    assert vault_agent.peer_uid(Broken()) is None


# --- peer_authorized branches ---------------------------------------------


def test_peer_authorized_same_uid():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert vault_agent.peer_authorized(a) is True
    finally:
        a.close()
        b.close()


def test_peer_authorized_rejects_foreign_uid(monkeypatch, caplog):
    monkeypatch.setattr(vault_agent, "peer_uid", lambda conn: os.getuid() + 1)
    with caplog.at_level(logging.WARNING, logger="src.security.vault_agent"):
        assert vault_agent.peer_authorized(object()) is False
    assert "rejected connection from uid" in caplog.text


def test_peer_authorized_fails_closed_on_unknown_platform(monkeypatch, caplog):
    monkeypatch.setattr(vault_agent, "peer_uid", lambda conn: None)
    with caplog.at_level(logging.WARNING, logger="src.security.vault_agent"):
        assert vault_agent.peer_authorized(object()) is False
        assert vault_agent.peer_authorized(object()) is False
    # warning about missing platform support is logged exactly once
    assert caplog.text.count("no peer-credential support") == 1


# --- end-to-end over a real AF_UNIX socket --------------------------------


async def _request(path: str, payload: dict) -> bytes:
    """One framed request/response round-trip (CL-1ho7 wire protocol).

    Authorized requests must return a complete frame; truncation is a failure.
    Rejection uses a separate raw-byte oracle below."""
    reader, writer = await asyncio.open_unix_connection(path)
    try:
        await vault_wire.send_framed_async(writer, json.dumps(payload).encode())
        return await vault_wire.recv_framed_async(reader)
    finally:
        writer.close()


async def _assert_rejected(path: str, payload: dict[str, str]) -> None:
    """CL-6sdh: EOF/reset is allowed, but even ONE response byte is a failure.

    Use a raw socket: a framing decoder can consume a partial header/body then
    raise IncompleteReadError, hiding bytes from the former rejection oracle.
    Kernel reads preserve queued bytes; StreamReader can instead raise a stored
    reset before exposing its buffer. No credentials or operational socket used.
    """
    loop = asyncio.get_running_loop()
    body = json.dumps(payload).encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.setblocking(False)
        await loop.sock_connect(sock, path)
        # Auth may reject before the request write. Still inspect replies.
        with suppress(BrokenPipeError, ConnectionResetError):
            # Independent wire vector, not the production frame encoder.
            await loop.sock_sendall(sock, len(body).to_bytes(4, "big") + body)
        try:
            received = await asyncio.wait_for(loop.sock_recv(sock, 1), timeout=5)
        except ConnectionResetError:
            received = b""
        assert received == b"", "unauthorized peer received a response byte"


def test_agent_serves_same_uid_client(sock_dir):
    async def run() -> None:
        path = str(sock_dir / "agent.sock")
        server = await vault_agent.serve(path, {"OANDA_API_KEY": "sekrit"})
        try:
            raw = await _request(path, {"action": "get", "name": "OANDA_API_KEY"})
            assert json.loads(raw) == {"ok": True, "value": "sekrit"}
            raw = await _request(path, {"action": "list"})
            assert json.loads(raw) == {"ok": True, "names": ["OANDA_API_KEY"]}
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


def test_agent_drops_foreign_uid_without_response(sock_dir, monkeypatch):
    monkeypatch.setattr(vault_agent, "peer_uid", lambda conn: os.getuid() + 1)
    decode = AsyncMock(wraps=vault_agent.recv_framed_async)
    monkeypatch.setattr(vault_agent, "recv_framed_async", decode)

    async def run() -> None:
        path = str(sock_dir / "agent.sock")
        server = await vault_agent.serve(path, {"OANDA_API_KEY": "sekrit"})
        try:
            await _assert_rejected(path, {"action": "get", "name": "OANDA_API_KEY"})
            decode.assert_not_awaited()  # Not even the protocol decoder may run.
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


def test_agent_drops_client_when_platform_unsupported(sock_dir, monkeypatch):
    monkeypatch.setattr(vault_agent, "peer_uid", lambda conn: None)
    decode = AsyncMock(wraps=vault_agent.recv_framed_async)
    monkeypatch.setattr(vault_agent, "recv_framed_async", decode)

    async def run() -> None:
        path = str(sock_dir / "agent.sock")
        server = await vault_agent.serve(path, {"K": "v"})
        try:
            await _assert_rejected(path, {"action": "list"})
            decode.assert_not_awaited()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize("leaked", [b"\x00", b"secret", b'\x00\x00\x00\x02{}'])
@pytest.mark.parametrize("abort", [False, True])
def test_rejection_oracle_detects_leaking_server_mutation(
    sock_dir: Path, leaked: bytes, abort: bool,
) -> None:
    async def run() -> None:
        path = str(sock_dir / "leaking.sock")

        async def leak(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            # Send partial headers, unframed bytes, or a complete response.
            # All must fail the oracle even when the connection then resets.
            writer.write(leaked)
            await writer.drain()
            if abort:
                writer.transport.abort()
            else:
                writer.close()

        server = await asyncio.start_unix_server(leak, path)
        try:
            with pytest.raises(AssertionError, match="response byte"):
                await _assert_rejected(path, {"action": "list"})
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


# --- socket file / directory hardening ------------------------------------


def test_socket_file_mode_is_0600(sock_dir):
    async def run() -> None:
        path = str(sock_dir / "agent.sock")
        server = await vault_agent.serve(path, {})
        try:
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


def test_harden_tightens_owned_loose_parent_dir(sock_dir):
    loose = sock_dir / "loose"
    loose.mkdir()
    os.chmod(loose, 0o755)
    target = loose / "agent.sock"
    target.touch()
    vault_agent._harden_socket_path(str(target))
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(loose).st_mode) == 0o700
