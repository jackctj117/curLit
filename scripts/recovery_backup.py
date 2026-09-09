"""Explicit operator backup and isolated full restore proof (CL-0deu.14).

Source is SELECT/pg_dump only. The disposable destination has no host mounts,
published ports, or network. Plaintext archives remain in memory, never on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from scripts.alpaca_recovery_report import write_private
from sqlalchemy import create_engine, text

from src.data.db_env import build_db_url
from src.dotenv_bootstrap import load_project_env

logger = logging.getLogger(__name__)


def run(args: list[str], data: bytes | None = None, *, timeout: int = 300) -> bytes:
    """Checked subprocess; never print raw DB data or credential-bearing errors."""
    result = subprocess.run(args, input=data, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} operation failed: exit {result.returncode}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-container", required=True)
    parser.add_argument("--restore-image", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.restore_image.startswith("sha256:"):
        parser.error("restore image must be an already-inspected immutable local image ID")
    if not args.key_file.is_file() or args.key_file.stat().st_mode & 0o077:
        parser.error("existing private backup key required")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    load_project_env(args.env_file)
    engine = create_engine(build_db_url(), connect_args={"connect_timeout": 10})
    destination = "curlit-restore-proof-" + uuid.uuid4().hex
    created = False
    try:
        logger.info("Capturing a consistent read-only source snapshot and encrypted archive")
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:  # noqa: SIM117
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text("SET LOCAL statement_timeout = '60s'"))
                snapshot_id = str(conn.execute(text("SELECT pg_export_snapshot()")).scalar_one())
                tables = list(
                    conn.execute(
                        text(
                            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
                        )
                    ).scalars()
                )
                # SQL identifiers originate in pg_catalog, quoted by the dialect.
                quote = engine.dialect.identifier_preparer.quote
                counts = {
                    name: conn.execute(
                        text(f"SELECT count(*) FROM public.{quote(name)}")
                    ).scalar_one()
                    for name in tables
                }
                archive = run(
                    [
                        "docker",
                        "exec",
                        args.source_container,
                        "pg_dump",
                        "-U",
                        "fx",
                        "-d",
                        "fx",
                        "-Fc",
                        "--no-owner",
                        "--no-acl",
                        "--snapshot",
                        snapshot_id,
                    ]
                )
        archive_hash = hashlib.sha256(archive).hexdigest()
        encrypted = run(
            [
                "gpg",
                "--batch",
                "--pinentry-mode",
                "loopback",
                "--symmetric",
                "--cipher-algo",
                "AES256",
                "--passphrase-file",
                str(args.key_file),
            ],
            archive,
        )
        path = args.output_dir / "database.pgdump.gpg"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(encrypted)
            output.flush()
            os.fsync(output.fileno())
        decrypted = run(
            [
                "gpg",
                "--batch",
                "--pinentry-mode",
                "loopback",
                "--decrypt",
                "--passphrase-file",
                str(args.key_file),
                str(path),
            ]
        )
        if hashlib.sha256(decrypted).hexdigest() != archive_hash:
            raise RuntimeError("encrypted backup round-trip mismatch")
        logger.info("Starting isolated disposable Timescale restore destination")
        run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--network",
                "none",
                "--name",
                destination,
                "-e",
                "POSTGRES_USER=fx",
                "-e",
                "POSTGRES_DB=curlit_test_restore",
                "-e",
                "POSTGRES_HOST_AUTH_METHOD=trust",
                args.restore_image,
            ]
        )
        created = True
        # Bounded readiness loop in a child, not an unbounded daemon wait.
        run(
            [
                "docker",
                "exec",
                destination,
                "sh",
                "-c",
                "for n in $(seq 1 30); do pg_isready -U fx -d curlit_test_restore && exit 0; "
                "sleep 1; done; exit 1",
            ],
            timeout=40,
        )
        prefix = [
            "docker",
            "exec",
            "-i",
            destination,
            "psql",
            "-U",
            "fx",
            "-d",
            "curlit_test_restore",
            "-v",
            "ON_ERROR_STOP=1",
            "-At",
        ]
        run(
            prefix, b"CREATE EXTENSION IF NOT EXISTS timescaledb; SELECT timescaledb_pre_restore();"
        )
        logger.info("Restoring full decrypted archive; errors are fatal")
        run(
            [
                "docker",
                "exec",
                "-i",
                destination,
                "pg_restore",
                "-U",
                "fx",
                "-d",
                "curlit_test_restore",
                "--exit-on-error",
                "--no-owner",
                "--no-acl",
            ],
            decrypted,
            timeout=600,
        )
        run(prefix, b"SELECT timescaledb_post_restore();")
        restored = {}
        for name in tables:
            restored[name] = int(
                run(prefix, (f"SELECT count(*) FROM public.{quote(name)};").encode()).strip()
            )
        if restored != counts:
            raise RuntimeError("restored public table counts differ from exported source snapshot")
        write_private(
            args.output_dir / "manifest.json",
            {
                "captured_at": datetime.now(UTC).isoformat(),
                "source_snapshot": snapshot_id,
                "archive_sha256": archive_hash,
                "encrypted_sha256": hashlib.sha256(encrypted).hexdigest(),
                "source_table_counts": counts,
                "restored_table_counts": restored,
                "full_restore_verified": True,
                "restore_image": args.restore_image,
                "limitations": ["row counts are not row-by-row content equivalence"],
            },
        )
        logger.info(
            "Full restore passed: %d public tables match consistent source counts", len(counts)
        )
        return 0
    finally:
        engine.dispose()
        if created:
            logger.info("Removing only the disposable restore-proof container")
            run(["docker", "stop", destination], timeout=40)


if __name__ == "__main__":
    raise SystemExit(main())
