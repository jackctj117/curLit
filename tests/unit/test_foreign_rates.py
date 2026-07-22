"""Tests for the foreign/derived rate-series ingester (CL-gr8o).

Canned payloads only — no live network, sqlite engine. Covers: the three
parsers (Bundesbank CSV incl. BOM/metadata/'.'-gap rows, FRED JSON incl.
'.' missing values, ECB SDMX jsondata), the CVIX realized-vol proxy math,
the full fetch→transform→upsert pipeline (spread derivation on matched
dates, per-series source provenance, window clipping), fail-soft behavior
when a source dies or the FRED key is absent, and upsert idempotence.
"""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.data.foreign_rates import (
    BUNDESBANK_DE2Y_URL,
    ECB_ESTR_3M_URL,
    FRED_OBS_URL,
    SERIES_SOURCES,
    ForeignRatesIngester,
    compute_cvix_proxy,
    parse_bundesbank_csv,
    parse_ecb_sdmx_series,
    parse_fred_observations,
)

# --------------------------------------------------------------------------- #
# canned payloads (shapes captured from the live probes on 2026-07-21)
# --------------------------------------------------------------------------- #

BBK_BODY = "\ufeff" + "\n".join([
    '"",BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A,'
    "BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A_FLAGS",
    '"","Yields, derived from the term structure of interest rates, on '
    'listed Federal securities / residual maturity of 2.0 years / daily data",',
    "Decimals,2,",
    "Time format code,P1D,",
    "unit,PROZENT,",
    "last update,2026-07-21 12:59:57,",
    "2026-07-10,2.70,",
    "2026-07-11,.,No value available",  # weekend gap -> dropped
    "2026-07-12,.,No value available",
    "2026-07-13,2.72,",
    "2026-07-14,2.74,",
    "2026-07-15,2.76,",
])


def _fred_body(series_values: dict[str, str]) -> str:
    return json.dumps({
        "observations": [
            {"realtime_start": "2026-07-21", "realtime_end": "2026-07-21",
             "date": d, "value": v}
            for d, v in series_values.items()
        ],
    })


DGS2_BODY = _fred_body({
    "2026-07-10": "4.20",
    "2026-07-13": "4.22",
    "2026-07-14": "4.21",
    "2026-07-15": ".",       # missing -> dropped -> no spread that day
})

SOFR_BODY = _fred_body({
    "2026-07-13": "3.62",
    "2026-07-14": "3.63",
    "2026-07-15": "3.62675",
})


def _ecb_body(dates_values: dict[str, float]) -> str:
    dates = list(dates_values)
    return json.dumps({
        "dataSets": [{
            "series": {
                "0:0:0": {
                    "observations": {
                        str(i): [v] for i, v in enumerate(dates_values.values())
                    },
                },
            },
        }],
        "structure": {
            "dimensions": {
                "observation": [{"values": [{"id": d} for d in dates]}],
            },
        },
    })


ESTR_BODY = _ecb_body({
    "2026-07-13": 2.031,
    "2026-07-14": 2.030,
    "2026-07-15": 2.02997,
})


def _fake_http(bodies: dict[str, str]):
    """URL-prefix → body shim; unmapped URLs raise (simulated outage)."""
    def _get(url: str) -> str:
        for prefix, body in bodies.items():
            if url.startswith(prefix):
                return body
        raise RuntimeError(f"simulated fetch failure for {url}")
    return _get


def _fred_url(series_id: str) -> str:
    return f"{FRED_OBS_URL}?series_id={series_id}"


ALL_OK = {
    BUNDESBANK_DE2Y_URL: BBK_BODY,
    _fred_url("DGS2"): DGS2_BODY,
    _fred_url("SOFR90DAYAVG"): SOFR_BODY,
    ECB_ESTR_3M_URL: ESTR_BODY,
}

START = datetime(2026, 7, 1)
END = datetime(2026, 7, 21)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def db(tmp_path):  # type: ignore[no-untyped-def]
    """sqlite engine with macro_data + prices shaped like migration 001."""
    url = f"sqlite:///{tmp_path / 'rates.db'}"
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text(
            "CREATE TABLE macro_data (observation_date DATE, "
            "release_date TEXT, series_id TEXT, value FLOAT, "
            "revision INT, source TEXT)",
        ))
        conn.execute(text(
            "CREATE TABLE prices (ts TEXT, symbol TEXT, close FLOAT)",
        ))
        # 40 business days of closes for two pairs -> CVIX has data after
        # the 20-day warmup. Deterministic wiggle so std > 0.
        days = pd.bdate_range("2026-05-25", periods=40)
        for i, d in enumerate(days):
            for sym, base in (("EURUSD", 1.08), ("USDJPY", 155.0)):
                px = base * (1.0 + 0.004 * math.sin(i * 1.7))
                conn.execute(
                    text("INSERT INTO prices VALUES (:t, :s, :c)"),
                    {"t": d.isoformat(), "s": sym, "c": px},
                )
    return url, eng


def _counts(eng):  # type: ignore[no-untyped-def]
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT series_id, COUNT(*), MAX(observation_date), MAX(source) "
            "FROM macro_data GROUP BY series_id",
        )).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #

def test_parse_bundesbank_csv_skips_metadata_and_gaps():
    ser = parse_bundesbank_csv(BBK_BODY)
    assert len(ser) == 4  # 6 date rows minus two '.' gaps
    assert ser[pd.Timestamp("2026-07-10")] == pytest.approx(2.70)
    assert ser[pd.Timestamp("2026-07-15")] == pytest.approx(2.76)
    assert pd.Timestamp("2026-07-11") not in ser.index


def test_parse_bundesbank_csv_malformed():
    assert parse_bundesbank_csv("").empty
    assert parse_bundesbank_csv("<html>ERROR</html>").empty


def test_parse_fred_observations_drops_missing():
    ser = parse_fred_observations(DGS2_BODY)
    assert len(ser) == 3
    assert pd.Timestamp("2026-07-15") not in ser.index  # '.' dropped
    assert ser[pd.Timestamp("2026-07-13")] == pytest.approx(4.22)


def test_parse_fred_observations_malformed():
    assert parse_fred_observations("not json").empty
    assert parse_fred_observations(json.dumps({"observations": []})).empty


def test_parse_ecb_sdmx_series():
    ser = parse_ecb_sdmx_series(ESTR_BODY)
    assert len(ser) == 3
    assert ser[pd.Timestamp("2026-07-15")] == pytest.approx(2.02997)


def test_parse_ecb_sdmx_series_malformed():
    assert parse_ecb_sdmx_series("not json").empty
    assert parse_ecb_sdmx_series(json.dumps({"dataSets": []})).empty


# --------------------------------------------------------------------------- #
# CVIX proxy math
# --------------------------------------------------------------------------- #

def test_compute_cvix_proxy_matches_manual_math():
    idx = pd.bdate_range("2026-01-01", periods=30)
    rng = np.random.default_rng(7)
    closes = pd.DataFrame({
        "EURUSD": 1.08 * np.exp(np.cumsum(rng.normal(0, 0.005, 30))),
        "USDJPY": 155.0 * np.exp(np.cumsum(rng.normal(0, 0.006, 30))),
    }, index=idx)
    out = compute_cvix_proxy(closes)
    # First 20 rows lack a full window -> 30 - 20 obs.
    assert len(out) == 10
    log_ret = np.log(closes / closes.shift(1))
    expected = (
        log_ret.rolling(20).std(ddof=1) * np.sqrt(252.0)
    ).mean(axis=1).dropna() * 100.0
    pd.testing.assert_series_equal(out, expected)
    assert (out > 0).all()
    assert (out < 100).all()  # sane vol-point levels


def test_compute_cvix_proxy_empty():
    assert compute_cvix_proxy(pd.DataFrame()).empty


def test_compute_cvix_proxy_decimal_closes():
    """Postgres NUMERIC arrives as decimal.Decimal (object dtype) — the
    proxy must coerce instead of blowing up in np.log (live-run regression)."""
    from decimal import Decimal
    idx = pd.bdate_range("2026-01-01", periods=25)
    rng = np.random.default_rng(3)
    floats = 1.08 * np.exp(np.cumsum(rng.normal(0, 0.005, 25)))
    closes = pd.DataFrame(
        {"EURUSD": [Decimal(str(round(v, 6))) for v in floats]}, index=idx,
    )
    assert closes.dtypes["EURUSD"] == np.dtype(object)
    out = compute_cvix_proxy(closes)
    assert len(out) == 5
    assert (out > 0).all()


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

def test_full_pipeline_writes_all_series_with_provenance(db):
    url, eng = db
    ing = ForeignRatesIngester(url, http_get=_fake_http(ALL_OK), fred_api_key="k")
    written = ing.run(START, END)
    assert written > 0
    got = _counts(eng)

    assert got["DE2Y"][0] == 4
    assert got["USD_3M_OIS"][0] == 3
    assert got["EUR_3M_ESTR_OIS"][0] == 3
    # Spread only on dates where BOTH legs exist: 07-10, 07-13, 07-14
    # (07-15 DGS2 is "."; 07-11/12 are DE2Y gaps).
    assert got["US2Y_MINUS_DE2Y"][0] == 3
    assert got["CVIX"][0] > 0

    # Spread value = DGS2 - DE2Y on a matched date.
    with eng.connect() as conn:
        v = conn.execute(text(
            "SELECT value FROM macro_data WHERE series_id='US2Y_MINUS_DE2Y' "
            "AND observation_date LIKE '2026-07-13%'",
        )).scalar()
    assert v == pytest.approx(4.22 - 2.72)

    # Per-series source provenance.
    for sid, (_, _, source) in got.items():
        assert source == SERIES_SOURCES[sid], sid


def test_fetch_clips_to_window(db):
    url, _ = db
    ing = ForeignRatesIngester(url, http_get=_fake_http(ALL_OK), fred_api_key="k")
    raw = ing.fetch(datetime(2026, 7, 13), datetime(2026, 7, 14))
    de2y = raw[raw["series_id"] == "DE2Y"]
    assert set(de2y["observation_date"].dt.strftime("%Y-%m-%d")) == {
        "2026-07-13", "2026-07-14",
    }


def test_upsert_idempotent(db):
    url, eng = db
    ing = ForeignRatesIngester(url, http_get=_fake_http(ALL_OK), fred_api_key="k")
    first = ing.run(START, END)
    second = ing.run(START, END)  # rerun: same observations -> 0 new rows
    assert first > 0
    assert second == 0
    got = _counts(eng)
    assert got["DE2Y"][0] == 4  # unchanged


def test_dead_source_fails_soft(db):
    """Bundesbank down -> DE2Y and the spread are absent, everything else
    still ingests (fail-soft per series)."""
    url, eng = db
    bodies = {k: v for k, v in ALL_OK.items() if k != BUNDESBANK_DE2Y_URL}
    ing = ForeignRatesIngester(url, http_get=_fake_http(bodies), fred_api_key="k")
    written = ing.run(START, END)
    assert written > 0
    got = _counts(eng)
    assert "DE2Y" not in got
    assert "US2Y_MINUS_DE2Y" not in got
    assert got["USD_3M_OIS"][0] == 3
    assert got["EUR_3M_ESTR_OIS"][0] == 3


def test_missing_fred_key_skips_fred_legs(db):
    url, eng = db
    ing = ForeignRatesIngester(
        url, http_get=_fake_http(ALL_OK), fred_api_key="",
    )
    ing.fred_api_key = ""  # belt-and-braces: ignore any env fallback
    written = ing.run(START, END)
    assert written > 0
    got = _counts(eng)
    assert "USD_3M_OIS" not in got
    assert "US2Y_MINUS_DE2Y" not in got  # DGS2 leg needs FRED
    assert got["DE2Y"][0] == 4
    assert got["EUR_3M_ESTR_OIS"][0] == 3


def test_dgs2_not_rewritten(db):
    """The DGS2 spread leg must NOT be written under the DGS2 id — the
    FRED ingester owns that series."""
    url, eng = db
    ing = ForeignRatesIngester(url, http_get=_fake_http(ALL_OK), fred_api_key="k")
    ing.run(START, END)
    assert "DGS2" not in _counts(eng)
