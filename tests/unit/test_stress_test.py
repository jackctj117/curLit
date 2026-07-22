"""Unit tests — stress tester fail-visible missing-data policy (CL-qyav).

Missing prices/factors must never be silently reported as a zero-loss
scenario: unpriceable scenarios are skipped with a WARNING naming what
was missing, and a fully unpriceable stress run raises.
"""

import logging

import pandas as pd
import pytest

from src.risk.stress_test import SCENARIOS, run_stress_test


class _RangeProvider:
    """get_range stub: per-scenario behavior keyed by start date."""

    def __init__(self, frames):  # type: ignore[no-untyped-def]
        self._frames = frames  # {start_date: DataFrame | Exception}

    def get_range(self, start, end):  # type: ignore[no-untyped-def]
        out = self._frames.get(start)
        if isinstance(out, Exception):
            raise out
        if out is None:
            return pd.DataFrame()
        return out


class _LongStrategy:
    def generate_signals(self, data):  # type: ignore[no-untyped-def]
        return pd.Series(1.0, index=data.index)


def _price_frame(start, periods=5, step=1.01):  # type: ignore[no-untyped-def]
    idx = pd.date_range(start=start, periods=periods, freq="D")
    return pd.DataFrame(
        {"close": [100.0 * step**i for i in range(periods)]}, index=idx,
    )


@pytest.mark.unit
def test_missing_scenario_data_is_skipped_not_zero(caplog):  # type: ignore[no-untyped-def]
    """Only the priced scenario appears; missing ones warn, no zero rows."""
    starts = {name: rng[0] for name, rng in SCENARIOS.items()}
    frames = {starts["covid_2020"]: _price_frame(starts["covid_2020"])}
    provider = _RangeProvider(frames)

    with caplog.at_level(logging.WARNING, logger="src.risk.stress_test"):
        results = run_stress_test([_LongStrategy()], provider)

    # Exactly the one priced scenario — nothing fabricated for the rest.
    assert set(results) == {"covid_2020"}
    assert results["covid_2020"]["total_return"] != 0.0

    # Every skipped scenario is named in a WARNING mentioning the skip.
    warned = " ".join(r.message for r in caplog.records)
    for name in SCENARIOS:
        if name != "covid_2020":
            assert name in warned
    assert "NOT counted as zero loss" in warned


@pytest.mark.unit
def test_fetch_exception_skips_scenario_with_warning(caplog):  # type: ignore[no-untyped-def]
    starts = {name: rng[0] for name, rng in SCENARIOS.items()}
    frames = {name_start: _price_frame(name_start) for name_start in starts.values()}
    frames[starts["gfc_2008"]] = RuntimeError("db connection refused")
    provider = _RangeProvider(frames)

    with caplog.at_level(logging.WARNING, logger="src.risk.stress_test"):
        results = run_stress_test([_LongStrategy()], provider)

    assert "gfc_2008" not in results
    assert len(results) == len(SCENARIOS) - 1
    assert any(
        "gfc_2008" in r.message and "fetch failed" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_missing_close_column_skips_scenario(caplog):  # type: ignore[no-untyped-def]
    starts = {name: rng[0] for name, rng in SCENARIOS.items()}
    frames = {s: _price_frame(s) for s in starts.values()}
    bad = _price_frame(starts["brexit_2016"]).rename(columns={"close": "open"})
    frames[starts["brexit_2016"]] = bad
    provider = _RangeProvider(frames)

    with caplog.at_level(logging.WARNING, logger="src.risk.stress_test"):
        results = run_stress_test([_LongStrategy()], provider)

    assert "brexit_2016" not in results
    assert any(
        "brexit_2016" in r.message and "'close' column missing" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_all_scenarios_unpriceable_raises():  # type: ignore[no-untyped-def]
    provider = _RangeProvider({})  # every scenario -> empty frame
    with pytest.raises(RuntimeError, match="no scenario could be priced"):
        run_stress_test([_LongStrategy()], provider)


@pytest.mark.unit
def test_provider_without_get_range_raises():  # type: ignore[no-untyped-def]
    with pytest.raises(TypeError, match="get_range"):
        run_stress_test([_LongStrategy()], object())
