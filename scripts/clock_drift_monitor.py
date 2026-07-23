"""NTP / system clock drift monitor (CL-gnhu).

Polls the system clock against a reference NTP server every minute.
Emits ``fx_clock_drift_seconds`` for Prometheus and logs WARNING when
drift exceeds the threshold.

Why 100ms? FX strategies don't trade microseconds, but Reconciliation
matching uses a 30-second window (CL-unlt). Drift above ~5 seconds
already hits that window's tolerance; 100ms is the alert level so we
catch developing drift well before it produces silent matching errors.

Runtime contract: meant to run as a long-lived process (systemd or
docker), not a cron job. We need the trend, not just snapshots.

Usage:
  .venv/bin/python -m scripts.clock_drift_monitor [--interval 60]
                                                  [--threshold 0.1]
                                                  [--ntp-server pool.ntp.org]
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import time

logger = logging.getLogger(__name__)


# 100 ms — drift level that triggers a WARNING. See module docstring.
_DEFAULT_THRESHOLD_SEC: float = 0.1

# 60 seconds — poll cadence. Sub-minute polling adds load on public NTP
# pool without buying useful resolution; multi-minute misses fast drift
# events (e.g. VM time-stops on host migration).
_DEFAULT_INTERVAL_SEC: int = 60

# Public NTP pool. Pin to .pool addresses so the OS resolves a fresh IP
# per query (load-spreads, fault-tolerant). Override on operator-provided
# private NTP infra.
_DEFAULT_NTP_SERVER: str = "pool.ntp.org"

# NTP epoch is 1900-01-01; Unix is 1970-01-01. Difference in seconds
# (70 years × 365.25 days × 86400 s + leap days).
_NTP_EPOCH_OFFSET: int = 2_208_988_800

# 48 bytes — fixed NTPv3 packet size. mode=3 (client) in the first byte.
_NTP_PACKET_FORMAT = "!12I"
_NTP_REQUEST_FIRST_BYTE: int = 0x1B


def query_ntp_offset(server: str, timeout_sec: float = 5.0) -> float:
    """Return clock offset (server_time - local_time) in seconds.

    Positive offset means the local clock is BEHIND the NTP server.
    Negative means the local clock is AHEAD. Either direction beyond
    the threshold is alert-worthy.

    Implements a minimal NTPv3 client — avoids the ntplib dependency
    that can disappear from PyPI. ~30 lines for what we need.
    """
    addr = (server, 123)
    packet = bytearray(48)
    packet[0] = _NTP_REQUEST_FIRST_BYTE

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_sec)
    try:
        t1 = time.time()
        sock.sendto(packet, addr)
        data, _ = sock.recvfrom(48)
        t4 = time.time()
    finally:
        sock.close()

    fields = struct.unpack(_NTP_PACKET_FORMAT, data)
    # Server's transmit timestamp = words 10-11 (seconds + fraction).
    server_secs = fields[10] - _NTP_EPOCH_OFFSET
    server_frac = fields[11] / 2**32
    server_time = server_secs + server_frac

    # Use the round-trip midpoint as our best estimate of when the
    # server stamped the response (RFC 5905 §8). Subtracting from the
    # server's stamp gives the offset.
    midpoint = (t1 + t4) / 2
    return server_time - midpoint


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Monitor system clock drift vs NTP server.",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=_DEFAULT_INTERVAL_SEC,
        help="Seconds between checks (default 60)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD_SEC,
        help="Drift threshold in seconds for WARNING (default 0.1)",
    )
    p.add_argument(
        "--ntp-server",
        default=_DEFAULT_NTP_SERVER,
        help="NTP server hostname (default pool.ntp.org)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Importing here so the script can be invoked --help without the
    # prometheus_client dependency in lightweight contexts.
    from src.monitoring.metrics import clock_drift_seconds

    logger.info(
        "Clock drift monitor: server=%s interval=%ds threshold=%.3fs",
        args.ntp_server,
        args.interval,
        args.threshold,
    )

    while True:
        try:
            offset = query_ntp_offset(args.ntp_server)
            clock_drift_seconds.set(offset)
            if abs(offset) >= args.threshold:
                logger.warning(
                    "Clock drift %.3fs exceeds threshold %.3fs (server=%s)",
                    offset,
                    args.threshold,
                    args.ntp_server,
                )
            else:
                logger.debug("Clock drift %.4fs (within threshold)", offset)
        except Exception as exc:
            logger.warning(
                "NTP query failed: %s: %s — will retry next interval",
                type(exc).__name__,
                exc,
            )
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
