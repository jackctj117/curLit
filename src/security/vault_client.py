"""Vault client — Unix socket client for credential retrieval."""

import json
import logging
import socket

logger = logging.getLogger(__name__)

SOCKET_PATH = "/run/fx-vault-agent.sock"


class VaultClient:
    def __init__(self, socket_path: str = SOCKET_PATH, timeout: float = 5.0) -> None:
        self.socket_path = socket_path
        self.timeout = timeout

    def get(self, name: str) -> str:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
            sock.sendall(json.dumps({"action": "get", "name": name}).encode())
            data = sock.recv(4096)
            resp = json.loads(data)
            if not resp.get("ok"):
                raise KeyError(f"Credential '{name}': {resp.get('error', 'unknown')}")
            return str(resp["value"])
        except FileNotFoundError as exc:
            raise KeyError(
                f"Vault agent not running (socket {self.socket_path} not found)",
            ) from exc
        finally:
            sock.close()

    def list(self) -> list[str]:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self.socket_path)
            sock.sendall(json.dumps({"action": "list"}).encode())
            data = sock.recv(4096)
            resp = json.loads(data)
            return [str(n) for n in resp.get("names", [])]
        finally:
            sock.close()
