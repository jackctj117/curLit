"""Tests for the NTPv3 packet handling in scripts.clock_drift_monitor (CL-gnhu).

The actual NTP query needs network access; tests here cover the parsing
correctness via a synthesized server response. End-to-end is covered
by the morning_health_check.sh integration on real boxes.
"""

from __future__ import annotations

import struct
from unittest.mock import patch

from scripts.clock_drift_monitor import _NTP_EPOCH_OFFSET, query_ntp_offset


def _fake_response(server_unix_time: float) -> bytes:
    """Build a minimal NTPv3 response with the given server transmit time."""
    # 12 32-bit words. Only fields[10]/fields[11] (transmit timestamp)
    # matter for our offset calculation.
    fields = [0] * 12
    secs = int(server_unix_time + _NTP_EPOCH_OFFSET)
    frac_part = (server_unix_time + _NTP_EPOCH_OFFSET) - secs
    frac = int(frac_part * (2**32))
    fields[10] = secs
    fields[11] = frac
    return struct.pack("!12I", *fields)


class TestOffsetParsing:
    def test_zero_offset_when_clocks_aligned(self) -> None:
        """If the server's transmit time matches the round-trip midpoint,
        the computed offset should be ~0."""
        with patch("scripts.clock_drift_monitor.socket.socket") as mock_sock_cls, \
             patch("scripts.clock_drift_monitor.time.time") as mock_time:
            mock_sock = mock_sock_cls.return_value
            mock_time.side_effect = [1000.0, 1000.0]  # t1=t4=1000 → midpoint=1000
            mock_sock.recvfrom.return_value = (_fake_response(1000.0), None)
            offset = query_ntp_offset("fake")
        assert abs(offset) < 1e-3

    def test_positive_offset_when_local_behind(self) -> None:
        """Server says it's 1010, local clock at midpoint says 1000 →
        offset = +10 (we are 10s behind)."""
        with patch("scripts.clock_drift_monitor.socket.socket") as mock_sock_cls, \
             patch("scripts.clock_drift_monitor.time.time") as mock_time:
            mock_sock = mock_sock_cls.return_value
            mock_time.side_effect = [1000.0, 1000.0]
            mock_sock.recvfrom.return_value = (_fake_response(1010.0), None)
            offset = query_ntp_offset("fake")
        assert 9.99 < offset < 10.01

    def test_negative_offset_when_local_ahead(self) -> None:
        """Local clock 1000, server 990 → offset = -10."""
        with patch("scripts.clock_drift_monitor.socket.socket") as mock_sock_cls, \
             patch("scripts.clock_drift_monitor.time.time") as mock_time:
            mock_sock = mock_sock_cls.return_value
            mock_time.side_effect = [1000.0, 1000.0]
            mock_sock.recvfrom.return_value = (_fake_response(990.0), None)
            offset = query_ntp_offset("fake")
        assert -10.01 < offset < -9.99
