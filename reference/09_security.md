# 09 — Security

Credential vault using wolfCrypt, vault agent daemon, paper-backed recovery, SSH hardening.

## Credential Vault

**Location:** `src/security/vault.py`
**Purpose:** Encrypted credential store. Uses wolfCrypt AES-256-GCM with PBKDF2 key derivation.

```python
import os
import json
import secrets
import logging
from pathlib import Path
from dataclasses import dataclass

from wolfcrypt.ciphers import Aes
from wolfcrypt.hashes import HmacSha256


logger = logging.getLogger(__name__)


@dataclass
class VaultConfig:
    vault_path: Path = Path('/opt/fx-system/vault.enc')
    salt_path: Path = Path('/opt/fx-system/vault.salt')
    pbkdf2_iterations: int = 600000
    aes_key_size: int = 32  # AES-256


class CredentialVault:
    """
    Encrypted credential storage using wolfCrypt.
    
    File format:
    - Salt file (16 bytes, plaintext): random per-vault salt
    - Vault file (encrypted): nonce (12 bytes) + ciphertext + tag (16 bytes)
    
    Key derivation: PBKDF2-HMAC-SHA256(passphrase, salt, iterations) → 32 bytes
    Encryption: AES-256-GCM with random nonce per write
    """
    
    def __init__(self, config: VaultConfig = None):
        self.config = config or VaultConfig()
        self._unlocked_credentials: dict[str, str] | None = None
        self._key: bytes | None = None
    
    def initialize(self, passphrase: str):
        """Create new vault with random salt."""
        if self.config.vault_path.exists():
            raise RuntimeError("Vault already exists")
        
        salt = secrets.token_bytes(16)
        self.config.salt_path.write_bytes(salt)
        os.chmod(self.config.salt_path, 0o400)
        
        self._key = self._derive_key(passphrase, salt)
        self._unlocked_credentials = {}
        self._save()
        
        logger.info(f"Vault initialized at {self.config.vault_path}")
    
    def unlock(self, passphrase: str) -> bool:
        if not self.config.vault_path.exists():
            raise RuntimeError("Vault doesn't exist; run initialize first")
        
        salt = self.config.salt_path.read_bytes()
        self._key = self._derive_key(passphrase, salt)
        
        try:
            encrypted = self.config.vault_path.read_bytes()
            nonce = encrypted[:12]
            tag = encrypted[-16:]
            ciphertext = encrypted[12:-16]
            
            aes = Aes(self._key, Aes.MODE_GCM, IV=nonce)
            plaintext = aes.decrypt(ciphertext, tag)
            
            self._unlocked_credentials = json.loads(plaintext.decode('utf-8'))
            return True
        except Exception as e:
            logger.error(f"Vault unlock failed: {e}")
            self._key = None
            return False
    
    def get(self, key: str) -> str | None:
        if self._unlocked_credentials is None:
            raise RuntimeError("Vault is locked")
        return self._unlocked_credentials.get(key)
    
    def set(self, key: str, value: str):
        if self._unlocked_credentials is None:
            raise RuntimeError("Vault is locked")
        self._unlocked_credentials[key] = value
        self._save()
    
    def delete(self, key: str):
        if self._unlocked_credentials is None:
            raise RuntimeError("Vault is locked")
        self._unlocked_credentials.pop(key, None)
        self._save()
    
    def list_keys(self) -> list[str]:
        if self._unlocked_credentials is None:
            raise RuntimeError("Vault is locked")
        return list(self._unlocked_credentials.keys())
    
    def lock(self):
        self._unlocked_credentials = None
        self._key = None
    
    def rotate_passphrase(self, new_passphrase: str):
        if self._unlocked_credentials is None:
            raise RuntimeError("Vault must be unlocked to rotate")
        
        new_salt = secrets.token_bytes(16)
        self.config.salt_path.write_bytes(new_salt)
        self._key = self._derive_key(new_passphrase, new_salt)
        self._save()
    
    def _derive_key(self, passphrase: str, salt: bytes) -> bytes:
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.aes_key_size,
            salt=salt,
            iterations=self.config.pbkdf2_iterations,
        )
        return kdf.derive(passphrase.encode('utf-8'))
    
    def _save(self):
        plaintext = json.dumps(self._unlocked_credentials).encode('utf-8')
        nonce = secrets.token_bytes(12)
        
        aes = Aes(self._key, Aes.MODE_GCM, IV=nonce)
        ciphertext, tag = aes.encrypt(plaintext)
        
        encrypted = nonce + ciphertext + tag
        
        # Atomic write
        tmp = self.config.vault_path.with_suffix('.tmp')
        tmp.write_bytes(encrypted)
        os.chmod(tmp, 0o400)
        tmp.rename(self.config.vault_path)
```

## Vault Agent Daemon

**Location:** `src/security/vault_agent.py`
**Purpose:** Long-running daemon that holds vault unlocked in memory. Other processes connect via Unix socket.

```python
import os
import socket
import json
import logging
import struct
import threading
from pathlib import Path
from getpass import getpass

from src.security.vault import CredentialVault


logger = logging.getLogger(__name__)


SOCKET_PATH = '/run/fx-vault.sock'
ALLOWED_UID = None  # Set at startup


class VaultAgent:
    def __init__(self, vault: CredentialVault):
        self.vault = vault
        self._server_socket = None
        self._running = False
    
    def serve(self):
        if Path(SOCKET_PATH).exists():
            os.unlink(SOCKET_PATH)
        
        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o600)  # Owner-only access
        self._server_socket.listen(5)
        self._running = True
        
        logger.info(f"Vault agent listening on {SOCKET_PATH}")
        
        while self._running:
            try:
                client, _ = self._server_socket.accept()
                # Verify peer credentials
                creds = client.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, 
                    struct.calcsize('3i')
                )
                pid, uid, gid = struct.unpack('3i', creds)
                
                if uid != os.getuid():
                    logger.warning(f"Rejected connection from UID {uid}")
                    client.close()
                    continue
                
                threading.Thread(
                    target=self._handle_client, args=(client,)
                ).start()
            except Exception as e:
                logger.exception(f"Accept error: {e}")
    
    def _handle_client(self, client: socket.socket):
        try:
            data = self._recv_msg(client)
            if data is None:
                return
            
            request = json.loads(data)
            command = request.get('command')
            
            if command == 'get':
                value = self.vault.get(request['key'])
                response = {'value': value}
            elif command == 'list':
                response = {'keys': self.vault.list_keys()}
            elif command == 'health':
                response = {'status': 'ok'}
            else:
                response = {'error': f'Unknown command: {command}'}
            
            self._send_msg(client, json.dumps(response).encode())
        except Exception as e:
            logger.exception(f"Client handler error: {e}")
            try:
                self._send_msg(client, json.dumps({'error': str(e)}).encode())
            except:
                pass
        finally:
            client.close()
    
    def _recv_msg(self, sock: socket.socket) -> bytes | None:
        len_bytes = sock.recv(4)
        if len(len_bytes) < 4:
            return None
        msg_len = struct.unpack('!I', len_bytes)[0]
        if msg_len > 1024 * 1024:
            return None
        return sock.recv(msg_len)
    
    def _send_msg(self, sock: socket.socket, msg: bytes):
        sock.sendall(struct.pack('!I', len(msg)) + msg)


def main():
    logging.basicConfig(level=logging.INFO)
    
    vault = CredentialVault()
    
    # Prompt for passphrase (or read from systemd-ask-password)
    passphrase = getpass("Vault passphrase: ")
    if not vault.unlock(passphrase):
        logger.error("Failed to unlock vault")
        return 1
    
    logger.info("Vault unlocked")
    
    agent = VaultAgent(vault)
    try:
        agent.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        vault.lock()
        if Path(SOCKET_PATH).exists():
            os.unlink(SOCKET_PATH)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
```

## Vault Client

**Location:** `src/security/vault_client.py`
**Purpose:** Lightweight client used by application code to query the vault agent.

```python
import socket
import json
import struct
from pathlib import Path

SOCKET_PATH = '/run/fx-vault.sock'


class VaultClient:
    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
    
    def get(self, key: str) -> str | None:
        return self._request({'command': 'get', 'key': key}).get('value')
    
    def list_keys(self) -> list[str]:
        return self._request({'command': 'list'}).get('keys', [])
    
    def health_check(self) -> bool:
        try:
            return self._request({'command': 'health'}).get('status') == 'ok'
        except Exception:
            return False
    
    def _request(self, payload: dict) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        try:
            data = json.dumps(payload).encode()
            sock.sendall(struct.pack('!I', len(data)) + data)
            
            len_bytes = sock.recv(4)
            if len(len_bytes) < 4:
                raise RuntimeError("Short response")
            msg_len = struct.unpack('!I', len_bytes)[0]
            response = sock.recv(msg_len)
            
            return json.loads(response)
        finally:
            sock.close()
```

## Vault Initialization Script

**Location:** `scripts/initialize_vault.py`
**Purpose:** First-time vault setup. Generates passphrase, creates BIP39 recovery seed, prints paper backup.

```python
import secrets
import sys
from getpass import getpass
from pathlib import Path

from src.security.vault import CredentialVault, VaultConfig


# Diceware-style word list for human-friendly passphrases
DICEWARE_WORDS = [
    # Subset shown — full list at https://www.eff.org/dice
    'abacus', 'abdomen', 'abdominal', 'abide', 'abiding', 'ability',
    # ... 7,776 words total
]


def generate_diceware_passphrase(n_words: int = 6) -> str:
    return '-'.join(
        secrets.choice(DICEWARE_WORDS) for _ in range(n_words)
    )


def generate_bip39_mnemonic() -> str:
    """Generate 24-word BIP39 mnemonic for recovery seed."""
    from mnemonic import Mnemonic
    mnemo = Mnemonic('english')
    return mnemo.generate(strength=256)  # 24 words


def main():
    print("=" * 70)
    print("FX System Vault Initialization")
    print("=" * 70)
    print()
    
    config = VaultConfig()
    if config.vault_path.exists():
        print(f"ERROR: Vault already exists at {config.vault_path}")
        return 1
    
    print("Generating passphrase...")
    passphrase = generate_diceware_passphrase()
    
    print("Generating recovery mnemonic (BIP39)...")
    mnemonic = generate_bip39_mnemonic()
    
    print()
    print("=" * 70)
    print("WRITE THESE DOWN ON PAPER NOW:")
    print("=" * 70)
    print()
    print(f"DAILY PASSPHRASE: {passphrase}")
    print()
    print(f"RECOVERY MNEMONIC: {mnemonic}")
    print()
    print("Recommended: Print this page, store the printout in a fireproof safe.")
    print("Do NOT save this to disk anywhere on this machine.")
    print()
    input("Press Enter when you have written it down safely... ")
    print()
    
    confirm = input("Type the passphrase to confirm you have it: ")
    if confirm.strip() != passphrase:
        print("ERROR: Passphrase mismatch. Restart and try again.")
        return 1
    
    confirm_mn = input("Type the first 4 words of the mnemonic: ")
    expected = ' '.join(mnemonic.split()[:4])
    if confirm_mn.strip().lower() != expected.lower():
        print("ERROR: Mnemonic mismatch. Restart and try again.")
        return 1
    
    print("Confirmed. Initializing vault...")
    vault = CredentialVault(config)
    vault.initialize(passphrase)
    
    # Encrypt mnemonic with passphrase for digital backup
    recovery_path = Path('/opt/fx-system/recovery.enc')
    save_recovery_encrypted(mnemonic, passphrase, recovery_path)
    
    print()
    print(f"Vault created: {config.vault_path}")
    print(f"Recovery seed (encrypted): {recovery_path}")
    print()
    print("Add credentials with: python scripts/vault_add.py KEY_NAME")
    return 0


def save_recovery_encrypted(mnemonic: str, passphrase: str, path: Path):
    """Save mnemonic encrypted with the passphrase (defense in depth)."""
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    
    salt = secrets.token_bytes(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600000
    )
    key = kdf.derive(passphrase.encode())
    nonce = secrets.token_bytes(12)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, mnemonic.encode(), None)
    
    path.write_bytes(salt + nonce + ciphertext)
    import os
    os.chmod(path, 0o400)


if __name__ == '__main__':
    sys.exit(main())
```

## Add Credential Script

**Location:** `scripts/vault_add.py`
**Purpose:** Add or update a credential in the vault.

```python
import sys
from getpass import getpass

from src.security.vault import CredentialVault


def main():
    if len(sys.argv) < 2:
        print("Usage: vault_add.py KEY_NAME")
        return 1
    
    key = sys.argv[1]
    
    vault = CredentialVault()
    passphrase = getpass("Vault passphrase: ")
    if not vault.unlock(passphrase):
        print("Failed to unlock vault")
        return 1
    
    value = getpass(f"Value for {key}: ")
    confirm = getpass(f"Confirm: ")
    
    if value != confirm:
        print("Values don't match")
        return 1
    
    vault.set(key, value)
    print(f"Stored {key}")
    
    vault.lock()
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

## SSH Key Derivation

**Location:** `src/security/ssh_keys.py`
**Purpose:** Deterministically derive SSH keys from BIP39 mnemonic for recovery.

```python
def derive_ssh_key_from_mnemonic(mnemonic: str, key_index: int = 0) -> bytes:
    """
    Derive ed25519 SSH private key from BIP39 mnemonic.
    Different key_index values give different keys from same mnemonic.
    """
    from mnemonic import Mnemonic
    import hashlib
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    
    mnemo = Mnemonic('english')
    seed = mnemo.to_seed(mnemonic, passphrase=f'ssh-{key_index}')
    
    # Take first 32 bytes for Ed25519 seed
    key_bytes = hashlib.sha256(seed).digest()
    
    private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
    
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
```

## TPM Sealing (Optional)

**Location:** `src/security/tpm_seal.py`
**Purpose:** Seal vault passphrase to TPM2 PCRs for auto-unlock when machine state matches.

```python
import subprocess
from pathlib import Path


SEAL_DIR = Path('/var/lib/fx-vault')


def seal_passphrase_to_tpm(passphrase: str):
    """
    Seal the passphrase using TPM2.
    Will only unseal if PCR values match (boot integrity preserved).
    """
    SEAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create primary key
    subprocess.run([
        'tpm2_createprimary', '-C', 'o', 
        '-c', str(SEAL_DIR / 'primary.ctx')
    ], check=True)
    
    # Create sealing key (sealed against PCRs 0,2,4,7)
    subprocess.run([
        'tpm2_create',
        '-C', str(SEAL_DIR / 'primary.ctx'),
        '-i', '-',  # Read from stdin
        '-u', str(SEAL_DIR / 'sealed.pub'),
        '-r', str(SEAL_DIR / 'sealed.priv'),
        '-L', 'sha256:0,2,4,7',  # PCR policy
    ], input=passphrase.encode(), check=True)


def unseal_passphrase_from_tpm() -> str:
    """Try to unseal. Will fail if boot state has changed."""
    primary = SEAL_DIR / 'primary.ctx'
    if not primary.exists():
        # Recreate primary key (deterministic from owner hierarchy)
        subprocess.run([
            'tpm2_createprimary', '-C', 'o', '-c', str(primary)
        ], check=True)
    
    subprocess.run([
        'tpm2_load',
        '-C', str(primary),
        '-u', str(SEAL_DIR / 'sealed.pub'),
        '-r', str(SEAL_DIR / 'sealed.priv'),
        '-c', str(SEAL_DIR / 'sealed.ctx'),
    ], check=True)
    
    result = subprocess.run([
        'tpm2_unseal', '-c', str(SEAL_DIR / 'sealed.ctx'),
        '-p', 'pcr:sha256:0,2,4,7',
    ], capture_output=True, check=True)
    
    return result.stdout.decode()
```
