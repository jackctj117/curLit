"""CL-i3js: real Yahoo cache replay, offline and isolated from operational caches."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["src.scanners.relative_volume", "src.events.prices"])
def test_download_reuses_calling_thread_and_bounds_cache_descriptors(module: str) -> None:
    # Real yfinance dispatch + SQLite cache; only remote history is replaced.
    # Disable cyclic GC to prove cleanup does not depend on collection timing.
    # Eight batches of 80 symbols exceed the failed service's 256-FD budget if
    # each ticker opens another thread-local SQLite connection.
    script = r"""
import gc, importlib, resource, tempfile, threading
from datetime import UTC, datetime
from unittest.mock import patch
import pandas as pd
import psutil
import yfinance as yf
from yfinance import cache
process = psutil.Process()
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
owner = threading.get_ident()
calls = []
frame = pd.DataFrame({'Close': [100., 101.], 'Volume': [1000, 1100]},
                     index=pd.date_range('2026-09-01', periods=2))
def history(self, *args, **kwargs):
    calls.append(threading.get_ident())
    cache.get_tz_cache().store(self.ticker, 'America/New_York')
    return frame.copy()
with tempfile.TemporaryDirectory(prefix='curlit-yf-offline-') as scratch:
    yf.set_tz_cache_location(scratch)
    cache.get_tz_cache().store('WARM', 'America/New_York')
    gc.collect()
    baseline = process.num_fds()
    gc.disable()
    download = importlib.import_module(MODULE)._yf_download
    with patch.object(yf.Ticker, 'history', history):
        for batch in range(8):
            result = download([f'T{i}' for i in range(80)],
                              datetime(2026, 9, 1, tzinfo=UTC),
                              datetime(2026, 9, 8, tzinfo=UTC))
            assert not result.empty
            assert set(calls) == {owner}, 'per-symbol worker threads retain cache resources'
            pd.testing.assert_frame_equal(result['T0'], frame, check_names=False)
            # Allow cache WAL/SHM and provider housekeeping, not per-symbol growth.
            assert process.num_fds() <= baseline + 8
    gc.enable()
    gc.collect()
"""
    result = subprocess.run(
        [sys.executable, "-c", "MODULE = " + repr(module) + "\n" + script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
