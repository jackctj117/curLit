"""Length-prefixed framing for the vault agent socket (CL-1ho7, E13).

The agent and its clients used to exchange bare JSON blobs with a single
``recv(4096)`` on each side — no message boundary, silent truncation the
moment a vault outgrew one datagram-sized read. Every message now travels
as::

    [4-byte big-endian unsigned length][payload]

Both sync (``socket``) and asyncio (``StreamReader``/``StreamWriter``)
helpers live here so the agent, ``VaultClient``, and
``scripts/rotate_secrets._read_vault_via_agent`` can never drift apart.

Guards, enforced on BOTH send and receive (fail loud):

* zero-length frames are rejected — an empty message is always a bug;
* frames above ``MAX_FRAME_BYTES`` (4 MiB) are rejected before any
  allocation, so a hostile/byzantine peer can't balloon memory;
* short reads are handled with a recv loop; EOF mid-frame raises.
"""

from __future__ import annotations

import asyncio
import socket
import struct

#: Hard ceiling on a single framed message. The whole vault serialized as
#: JSON is far below this; anything bigger is a bug or an attack.
MAX_FRAME_BYTES = 4 * 1024 * 1024

_LEN_STRUCT = struct.Struct(">I")


class FrameError(ValueError):
    """A frame violated the wire protocol (empty, oversize, or truncated)."""


def _check_size(length: int, max_size: int) -> None:
    if length == 0:
        msg = "zero-length frame rejected"
        raise FrameError(msg)
    if length > max_size:
        msg = f"frame of {length} bytes exceeds cap of {max_size}"
        raise FrameError(msg)


def encode_frame(payload: bytes, max_size: int = MAX_FRAME_BYTES) -> bytes:
    """``payload`` -> ``len-prefix + payload`` (size-checked)."""
    _check_size(len(payload), max_size)
    return _LEN_STRUCT.pack(len(payload)) + payload


# --- sync (socket) side ----------------------------------------------------


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes, looping over short reads. Raises
    ``FrameError`` if the peer closes mid-frame."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            msg = f"connection closed mid-frame ({n - remaining}/{n} bytes read)"
            raise FrameError(msg)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_framed(sock: socket.socket, payload: bytes, max_size: int = MAX_FRAME_BYTES) -> None:
    """Write one length-prefixed message to a connected socket."""
    sock.sendall(encode_frame(payload, max_size))


def recv_framed(sock: socket.socket, max_size: int = MAX_FRAME_BYTES) -> bytes:
    """Read one length-prefixed message from a connected socket."""
    (length,) = _LEN_STRUCT.unpack(_recv_exact(sock, _LEN_STRUCT.size))
    _check_size(length, max_size)
    return _recv_exact(sock, length)


# --- asyncio side ----------------------------------------------------------


async def send_framed_async(
    writer: asyncio.StreamWriter,
    payload: bytes,
    max_size: int = MAX_FRAME_BYTES,
) -> None:
    """Write one length-prefixed message to an asyncio stream."""
    writer.write(encode_frame(payload, max_size))
    await writer.drain()


async def recv_framed_async(
    reader: asyncio.StreamReader,
    max_size: int = MAX_FRAME_BYTES,
) -> bytes:
    """Read one length-prefixed message from an asyncio stream.

    Raises ``FrameError`` on a zero-length/oversize header and
    ``asyncio.IncompleteReadError`` if the peer closes mid-frame.
    """
    header = await reader.readexactly(_LEN_STRUCT.size)
    (length,) = _LEN_STRUCT.unpack(header)
    _check_size(length, max_size)
    return await reader.readexactly(length)
