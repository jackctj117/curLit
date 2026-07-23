"""Vault agent — in-memory credential server via Unix socket.

Peer authentication (CL-xdnh P1): every connection's peer UID is read from
the kernel (SO_PEERCRED on Linux, LOCAL_PEERCRED on macOS) and must equal
this process's UID. Unknown/unsupported platforms fail CLOSED — the
connection is rejected and a warning is logged once. The socket file is
created under a 0o177 umask and pinned to mode 0o600; a loose parent
directory we own is tightened, one we don't own is warned about.

Wire protocol (CL-1ho7): every request/response is a 4-byte big-endian
length prefix + JSON payload (src.security.vault_wire) — framing runs
strictly AFTER peer auth; malformed frames drop the connection.
"""

import asyncio
import getpass
import json
import logging
import os
import socket
import stat
import struct
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOCKET_PATH = "/run/fx-vault-agent.sock"


from src.security.vault_codec import (  # noqa: E402
    derive_key,
    passphrase_weakness,
    unseal,
)
from src.security.vault_wire import (  # noqa: E402
    FrameError,
    recv_framed_async,
    send_framed_async,
)

# --- Peer credential authentication (CL-xdnh) ------------------------------

# macOS: getsockopt(SOL_LOCAL, LOCAL_PEERCRED) -> struct xucred
#   { u_int cr_version; uid_t cr_uid; short cr_ngroups; gid_t cr_groups[16]; }
_SOL_LOCAL = 0
_LOCAL_PEERCRED = 1
_XUCRED_FMT = "IIh16I"  # native alignment: 4+4+2+(2 pad)+64 = 76 bytes

# Linux: getsockopt(SOL_SOCKET, SO_PEERCRED) -> struct ucred {pid, uid, gid}
_UCRED_FMT = "3i"

_warned_unsupported = False


def peer_uid(conn: socket.socket) -> int | None:
    """Kernel-reported UID of the peer on an AF_UNIX socket.

    Returns None when the platform is unsupported or the credential query
    fails — callers must treat None as REJECT (fail closed).
    """
    try:
        if sys.platform.startswith("linux"):
            so_peercred = getattr(socket, "SO_PEERCRED", 17)
            data = conn.getsockopt(
                socket.SOL_SOCKET,
                so_peercred,
                struct.calcsize(_UCRED_FMT),
            )
            _pid, uid, _gid = struct.unpack(_UCRED_FMT, data)
            return int(uid)
        if sys.platform == "darwin":
            data = conn.getsockopt(
                _SOL_LOCAL,
                _LOCAL_PEERCRED,
                struct.calcsize(_XUCRED_FMT),
            )
            _version, uid, _ngroups = struct.unpack_from("IIh", data)
            return int(uid)
    except (OSError, struct.error) as exc:
        logger.warning("vault agent: peer credential query failed: %s", exc)
        return None
    return None


def peer_authorized(conn: socket.socket) -> bool:
    """True iff the connecting peer runs as the same UID as this agent."""
    global _warned_unsupported
    uid = peer_uid(conn)
    if uid is None:
        if not _warned_unsupported:
            _warned_unsupported = True
            logger.warning(
                "vault agent: no peer-credential support on platform %r — "
                "rejecting ALL connections (fail closed)",
                sys.platform,
            )
        return False
    if uid != os.getuid():
        logger.warning(
            "vault agent: rejected connection from uid %d (expected %d)",
            uid,
            os.getuid(),
        )
        return False
    return True


def _harden_socket_path(path: str) -> None:
    """Pin the socket file to 0o600 and tighten/flag a loose parent dir."""
    os.chmod(path, 0o600)
    parent = Path(path).resolve().parent
    st = parent.stat()
    mode = stat.S_IMODE(st.st_mode)
    if st.st_uid == os.getuid():
        if mode & 0o077:
            os.chmod(parent, mode & ~0o077)
            logger.info(
                "vault agent: tightened socket dir %s from %o to %o",
                parent,
                mode,
                mode & ~0o077,
            )
    elif mode & 0o022:
        logger.warning(
            "vault agent: socket dir %s (uid %d, mode %o) is writable by "
            "group/other — the socket can be replaced by another user",
            parent,
            st.st_uid,
            mode,
        )


def decrypt_vault(path: Path, key: bytes) -> dict[str, Any]:
    """Open a vault file of ANY historical schema/backend via the shared
    codec (CL-ujm6) — v1 "ct" and v2 "ciphertext", split or appended tag."""
    data = json.loads(path.read_text())
    return json.loads(unseal(data, key))  # type: ignore[no-any-return]


async def handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, creds: dict[str, Any]
) -> None:
    try:
        sock = writer.get_extra_info("socket")
        if sock is None or not peer_authorized(sock):
            return  # drop unauthenticated peers without a response
        # Length-prefixed framing (CL-1ho7) — ONLY after peer auth, so an
        # unauthorized peer never even gets a protocol read. Malformed
        # frames (empty/oversize/truncated) drop the connection without a
        # response, same as an unauthenticated peer.
        try:
            data = await recv_framed_async(reader)
        except (FrameError, asyncio.IncompleteReadError) as exc:
            logger.warning("vault agent: dropping client on bad frame: %s", exc)
            return
        req = json.loads(data)
        if req.get("action") == "get":
            resp = (
                {"ok": True, "value": creds.get(req.get("name", ""))}
                if req.get("name") in creds
                else {"ok": False, "error": "not_found"}
            )
        elif req.get("action") == "list":
            resp = {"ok": True, "names": list(creds.keys())}
        else:
            resp = {"ok": False, "error": "unknown_action"}
        await send_framed_async(writer, json.dumps(resp).encode())
    finally:
        writer.close()


async def serve(socket_path: str, credentials: dict[str, Any]) -> asyncio.Server:
    """Bind the agent socket with a race-free restrictive mode and return
    the serving asyncio.Server (caller owns its lifecycle)."""

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await handle_client(reader, writer, credentials)

    if os.path.exists(socket_path):
        os.unlink(socket_path)

    # umask closes the bind->chmod race: the socket is never group/other
    # accessible, not even for an instant.
    old_umask = os.umask(0o177)
    try:
        server = await asyncio.start_unix_server(handler, socket_path)
    finally:
        os.umask(old_umask)
    _harden_socket_path(socket_path)
    return server


async def main_async() -> None:
    vault_path = Path(os.environ.get("VAULT_PATH", "vault.enc"))
    salt_path = Path(os.environ.get("VAULT_SALT", "vault.salt"))

    passphrase = getpass.getpass("Vault passphrase: ")
    salt = salt_path.read_bytes()
    key = derive_key(passphrase, salt)
    credentials = decrypt_vault(vault_path, key)

    # Existing vault opened fine — never fail on weakness here, but tell the
    # operator loudly to rotate (CL-qyav P2).
    weakness = passphrase_weakness(passphrase)
    if weakness is not None:
        logger.warning(
            "Vault passphrase is WEAK (%s). Rotate it: re-seal the vault "
            "with a stronger passphrase (see scripts/rotate_secrets.py); "
            "old sealed data stays readable.",
            weakness,
        )

    # The passphrase CANNOT be wiped from memory: Python str is immutable, so
    # any "zeroing" loop (a prior version here had one) only builds new string
    # objects while the original bytes stay on the heap until GC — and copies
    # (getpass buffers, PBKDF2 input) survive anyway. Its lifetime is
    # process-scoped by design; the real mitigations are the kernel peer-UID
    # auth on the socket (same-UID only, fail-closed) and the 0600 file/socket
    # permissions — not in-memory hygiene. (CL-8lv6 P1)

    server = await serve(SOCKET_PATH, credentials)
    print(f"Vault agent listening on {SOCKET_PATH}")

    async with server:
        await server.serve_forever()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
