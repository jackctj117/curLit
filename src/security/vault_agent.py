"""Vault agent — in-memory credential server via Unix socket."""

import asyncio
import getpass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SOCKET_PATH = "/run/fx-vault-agent.sock"


def derive_key(passphrase: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 600_000, 32)


def decrypt_vault(path: Path, key: bytes) -> dict[str, Any]:
    data = json.loads(path.read_text())
    nonce = bytes.fromhex(data["nonce"])
    ct = bytes.fromhex(data["ciphertext"])
    try:
        from wolfcrypt.ciphers import MODE_GCM, Aes
        aes = Aes(key, MODE_GCM, nonce)
        return json.loads(aes.decrypt(ct, bytes.fromhex(data.get("tag", ""))))  # type: ignore[no-any-return]
    except ImportError:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return json.loads(AESGCM(key).decrypt(nonce, ct, None))  # type: ignore[no-any-return]


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, creds: dict[str, Any]) -> None:
    try:
        data = await reader.read(4096)
        req = json.loads(data)
        if req.get("action") == "get":
            resp = {"ok": True, "value": creds.get(req.get("name", ""))} if req.get("name") in creds else {"ok": False, "error": "not_found"}
        elif req.get("action") == "list":
            resp = {"ok": True, "names": list(creds.keys())}
        else:
            resp = {"ok": False, "error": "unknown_action"}
        writer.write(json.dumps(resp).encode())
        await writer.drain()
    finally:
        writer.close()


async def main_async() -> None:
    vault_path = Path(os.environ.get("VAULT_PATH", "vault.enc"))
    salt_path = Path(os.environ.get("VAULT_SALT", "vault.salt"))

    passphrase = getpass.getpass("Vault passphrase: ")
    salt = salt_path.read_bytes()
    key = derive_key(passphrase, salt)
    credentials = decrypt_vault(vault_path, key)

    # Zero passphrase
    for i in range(len(passphrase)):
        passphrase = passphrase[:i] + "\x00" + passphrase[i+1:] if i < len(passphrase) else passphrase

    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await handle_client(reader, writer, credentials)

    server = await asyncio.start_unix_server(handler, SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    print(f"Vault agent listening on {SOCKET_PATH}")

    async with server:
        await server.serve_forever()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
