"""Length-prefixed vault wire protocol (CL-1ho7, E13).

Covers the shared framing helpers (size guards, short-read recv loop,
EOF mid-frame) plus end-to-end round-trips through the real agent with
payloads larger than the old single-shot ``recv(4096)`` could carry —
via ``VaultClient`` and ``scripts.rotate_secrets._read_vault_via_agent``.

Hermetic: sockets live in a short ``mkdtemp`` dir (macOS ~104-char
AF_UNIX ``sun_path`` limit makes pytest's tmp_path too long); the real
vault / .env / live agent socket are never touched.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import struct
import sys
import tempfile
from pathlib import Path

import pytest

from src.security import vault_agent, vault_wire
from src.security.vault_client import VaultClient
from src.security.vault_wire import (
    MAX_FRAME_BYTES,
    FrameError,
    encode_frame,
    recv_framed,
    send_framed,
)


@pytest.fixture
def sock_dir():
    d = tempfile.mkdtemp(prefix="vwire-")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- encode_frame ----------------------------------------------------------


def test_encode_frame_prefixes_length():
    assert encode_frame(b"abc") == b"\x00\x00\x00\x03abc"


def test_encode_frame_rejects_empty_payload():
    with pytest.raises(FrameError, match="zero-length"):
        encode_frame(b"")


def test_encode_frame_rejects_oversize_payload():
    with pytest.raises(FrameError, match="exceeds cap"):
        encode_frame(b"x" * 11, max_size=10)


# --- sync send/recv over a socketpair --------------------------------------


def test_round_trip_over_socketpair():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        payload = json.dumps({"action": "get", "name": "K"}).encode()
        send_framed(a, payload)
        assert recv_framed(b) == payload
    finally:
        a.close()
        b.close()


def test_recv_framed_handles_short_reads():
    """A peer that dribbles one byte per recv still yields a whole frame."""

    class DribbleSocket:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._pos = 0

        def recv(self, n: int) -> bytes:
            if self._pos >= len(self._data):
                return b""
            chunk = self._data[self._pos : self._pos + 1]  # ignore n: 1 byte
            self._pos += 1
            return chunk

    payload = b"p" * 300
    assert recv_framed(DribbleSocket(encode_frame(payload))) == payload


def test_recv_framed_rejects_zero_length_header():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        a.sendall(struct.pack(">I", 0))
        with pytest.raises(FrameError, match="zero-length"):
            recv_framed(b)
    finally:
        a.close()
        b.close()


def test_recv_framed_rejects_oversize_header_before_reading_body():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        a.sendall(struct.pack(">I", MAX_FRAME_BYTES + 1))
        with pytest.raises(FrameError, match="exceeds cap"):
            recv_framed(b)
    finally:
        a.close()
        b.close()


def test_recv_framed_raises_on_eof_mid_frame():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        a.sendall(struct.pack(">I", 100) + b"only-part")
        a.close()
        with pytest.raises(FrameError, match="closed mid-frame"):
            recv_framed(b)
    finally:
        b.close()


# --- async helpers ---------------------------------------------------------


def test_async_recv_rejects_oversize_header():
    async def run() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", MAX_FRAME_BYTES + 1))
        with pytest.raises(FrameError, match="exceeds cap"):
            await vault_wire.recv_framed_async(reader)

    asyncio.run(run())


def test_async_recv_raises_incomplete_on_eof_mid_frame():
    async def run() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", 50) + b"short")
        reader.feed_eof()
        with pytest.raises(asyncio.IncompleteReadError):
            await vault_wire.recv_framed_async(reader)

    asyncio.run(run())


# --- end-to-end through the real agent (payloads > 4096 bytes) -------------

pytestmark_e2e = pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"),
    reason="AF_UNIX peer credentials need Linux or macOS",
)


@pytestmark_e2e
def test_vault_client_round_trips_payload_larger_than_4096(sock_dir):
    """The old recv(4096) truncated this; framing must not."""
    big_value = "v" * 10_000

    async def run() -> None:
        path = str(sock_dir / "agent.sock")
        server = await vault_agent.serve(path, {"BIG": big_value})
        try:
            loop = asyncio.get_running_loop()
            client = VaultClient(socket_path=path)
            value = await loop.run_in_executor(None, client.get, "BIG")
            assert value == big_value
            names = await loop.run_in_executor(None, client.list)
            assert names == ["BIG"]
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


@pytestmark_e2e
def test_rotate_secrets_reads_large_vault_via_agent(sock_dir, monkeypatch):
    """_read_vault_via_agent survives a list response far beyond one
    recv buffer (200 names) and pulls every credential intact."""
    from scripts import rotate_secrets

    creds = {f"CREDENTIAL_{i:04d}": f"value-{i}-" + "x" * 40 for i in range(200)}
    path = str(sock_dir / "agent.sock")
    monkeypatch.setattr(rotate_secrets, "_VAULT_AGENT_SOCK", path)

    async def run() -> None:
        server = await vault_agent.serve(path, dict(creds))
        try:
            loop = asyncio.get_running_loop()
            out = await loop.run_in_executor(None, rotate_secrets._read_vault_via_agent)
            assert out == creds
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())


@pytestmark_e2e
def test_agent_drops_malformed_frame_without_response(sock_dir):
    """An authorized peer sending garbage framing gets dropped silently."""

    async def run() -> None:
        path = str(sock_dir / "agent.sock")
        server = await vault_agent.serve(path, {"K": "v"})
        try:
            reader, writer = await asyncio.open_unix_connection(path)
            try:
                writer.write(struct.pack(">I", MAX_FRAME_BYTES + 1) + b"junk")
                await writer.drain()
                assert await reader.read() == b""  # closed, nothing leaked
            finally:
                writer.close()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())
