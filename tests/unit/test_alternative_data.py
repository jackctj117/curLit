"""Unit tests for the alternative-data layer (CL-6sp).

No network, no API keys — pytrends is replaced with a canned client
double, ENTSO-E with a canned-XML http_get_text shim (same injection
pattern as test_polymarket_ingester).

Covers:
  * registry: registration, lookup, unknown-name error, stub filtering,
    third-party sources can be added (modularity acceptance criterion)
  * base validate: schema enforcement, dropna, dedup, non-finite drop
  * GoogleTrendsSource: fetch parses canned interest_over_time frames,
    one bad geo doesn't kill the run, transform melts to canonical long
    format + drops isPartial, values clamped to [0, 100], symbol naming
  * EntsoeSource: missing token -> ConfigurationError + is_configured
    False, canned GL_MarketDocument XML parses (PT15M/PT60M positions),
    Acknowledgement rejections raise, one bad area doesn't kill the run,
    transform aggregates to daily mean, non-positive load dropped
  * stubs raise NotImplementedError and report unconfigured
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, ClassVar

import pandas as pd
import pytest

from src.data.alternative import (
    CANONICAL_COLUMNS,
    AltDataError,
    AlternativeDataSource,
    ConfigurationError,
    ConsumerSpendingSource,
    EntsoeSource,
    GoogleTrendsSource,
    SatelliteShippingSource,
    available_sources,
    get_source,
    register_source,
    trends_symbol,
)

START = datetime(2026, 6, 1)
END = datetime(2026, 6, 30)


# ---------------------------------------------------------------------- #
# Registry / modularity
# ---------------------------------------------------------------------- #


class TestRegistry:
    def test_builtin_sources_registered(self) -> None:
        names = available_sources()
        assert "google_trends" in names
        assert "entsoe" in names
        assert "satellite_shipping" in names
        assert "consumer_spending" in names

    def test_stub_filtering(self) -> None:
        live = available_sources(include_stubs=False)
        assert "google_trends" in live
        assert "entsoe" in live
        assert "satellite_shipping" not in live
        assert "consumer_spending" not in live

    def test_get_source_returns_class(self) -> None:
        assert get_source("google_trends") is GoogleTrendsSource
        assert get_source("entsoe") is EntsoeSource

    def test_get_source_unknown_name(self) -> None:
        with pytest.raises(KeyError, match="unknown alt-data source"):
            get_source("does_not_exist")

    def test_new_source_can_be_registered(self) -> None:
        """Acceptance: the architecture allows adding new sources."""

        @register_source
        class _WeatherSource(AlternativeDataSource):
            source_name: ClassVar[str] = "_test_weather"

            def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
                return pd.DataFrame([{"ts": start, "temp_c": 21.5}])

            def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
                df = raw.rename(columns={"temp_c": "value"})
                df["symbol"] = "WX:DE:temp"
                df["source"] = self.source_name
                return df[CANONICAL_COLUMNS]

        try:
            assert "_test_weather" in available_sources()
            out = get_source("_test_weather")().run(START, END)
            assert list(out.columns) == CANONICAL_COLUMNS
            assert len(out) == 1
            assert out.loc[0, "value"] == 21.5
        finally:
            # Don't leak the test double into other tests' registry views.
            from src.data import alternative

            alternative._SOURCES.pop("_test_weather", None)

    def test_register_requires_source_name(self) -> None:
        class _Anon(AlternativeDataSource):
            def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
                return pd.DataFrame()

            def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
                return raw

        with pytest.raises(ValueError, match="non-empty source_name"):
            register_source(_Anon)


# ---------------------------------------------------------------------- #
# Base validate
# ---------------------------------------------------------------------- #


class _PassthroughSource(AlternativeDataSource):
    source_name: ClassVar[str] = "_passthrough"

    def fetch(self, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.DataFrame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return raw


class TestBaseValidate:
    def test_missing_canonical_columns_raise(self) -> None:
        df = pd.DataFrame([{"ts": START, "value": 1.0}])  # no symbol/source
        with pytest.raises(AltDataError, match="missing columns"):
            _PassthroughSource().validate(df)

    def test_drops_nulls_and_dupes(self) -> None:
        df = pd.DataFrame(
            {
                "ts": [START, START, START, None],
                "symbol": ["A", "A", None, "B"],
                "value": [1.0, 2.0, 3.0, 4.0],
                "source": ["s"] * 4,
            }
        )
        out = _PassthroughSource().validate(df)
        # Dup (ts, symbol) keeps last; null symbol + null ts rows dropped.
        assert len(out) == 1
        assert out.loc[0, "value"] == 2.0

    def test_drops_non_numeric_values(self) -> None:
        df = pd.DataFrame(
            {
                "ts": [START, END],
                "symbol": ["A", "B"],
                "value": [1.0, "garbage"],
                "source": ["s", "s"],
            }
        )
        out = _PassthroughSource().validate(df)
        assert list(out["symbol"]) == ["A"]

    def test_empty_frame_passes_through(self) -> None:
        out = _PassthroughSource().validate(pd.DataFrame())
        assert out.empty


# ---------------------------------------------------------------------- #
# Google Trends
# ---------------------------------------------------------------------- #


class _FakeTrendReq:
    """Canned pytrends client. Records payloads; returns a wide
    interest_over_time frame like the real library (DatetimeIndex named
    'date', one column per term, plus isPartial)."""

    def __init__(
        self,
        fail_geos: set[str] | None = None,
        empty_geos: set[str] | None = None,
    ) -> None:
        self.fail_geos = fail_geos or set()
        self.empty_geos = empty_geos or set()
        self.payloads: list[dict[str, Any]] = []
        self._terms: list[str] = []
        self._geo = ""

    def build_payload(self, kw_list: list[str], timeframe: str, geo: str) -> None:
        self.payloads.append({"kw_list": kw_list, "timeframe": timeframe, "geo": geo})
        self._terms = kw_list
        self._geo = geo

    def interest_over_time(self) -> pd.DataFrame:
        if self._geo in self.fail_geos:
            msg = "The request failed: Google returned a response with code 429"
            raise RuntimeError(msg)
        if self._geo in self.empty_geos:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(
            [datetime(2026, 6, 7), datetime(2026, 6, 14)],
            name="date",
        )
        data: dict[str, Any] = {t: [40, 60] for t in self._terms}
        data["isPartial"] = [False, True]
        return pd.DataFrame(data, index=idx)


def _trends_source(client: _FakeTrendReq, **kwargs: Any) -> GoogleTrendsSource:
    return GoogleTrendsSource(
        client_factory=lambda: client,
        request_delay_sec=0.0,
        **kwargs,
    )


class TestGoogleTrends:
    def test_fetch_returns_data(self) -> None:
        client = _FakeTrendReq()
        src = _trends_source(client, currencies=["USD", "EUR"])
        raw = src.fetch(START, END)
        assert not raw.empty
        assert set(raw["geo"]) == {"US", "DE"}
        # One payload per currency, correct timeframe string.
        assert len(client.payloads) == 2
        assert client.payloads[0]["timeframe"] == "2026-06-01 2026-06-30"
        assert client.payloads[0]["geo"] == "US"

    def test_one_bad_geo_does_not_kill_run(self) -> None:
        client = _FakeTrendReq(fail_geos={"US"})
        src = _trends_source(client, currencies=["USD", "EUR"])
        raw = src.fetch(START, END)
        assert set(raw["geo"]) == {"DE"}

    def test_empty_geo_skipped(self) -> None:
        client = _FakeTrendReq(empty_geos={"US", "DE"})
        src = _trends_source(client, currencies=["USD", "EUR"])
        assert src.fetch(START, END).empty

    def test_transform_melts_to_canonical(self) -> None:
        src = _trends_source(_FakeTrendReq(), currencies=["USD"])
        out = src.run(START, END)
        assert list(out.columns) == CANONICAL_COLUMNS
        # 3 USD terms x 2 dates
        assert len(out) == 6
        assert (out["source"] == "google_trends").all()
        assert "GT:US:recession" in set(out["symbol"])
        assert "GT:US:unemployment_benefits" in set(out["symbol"])
        # isPartial must not leak through as a symbol.
        assert not any("isPartial" in s for s in out["symbol"])

    def test_validate_drops_out_of_range(self) -> None:
        src = _trends_source(_FakeTrendReq(), currencies=["USD"])
        df = pd.DataFrame(
            {
                "ts": [START, START, START],
                "symbol": ["GT:US:a", "GT:US:b", "GT:US:c"],
                "value": [50.0, -1.0, 101.0],
                "source": ["google_trends"] * 3,
            }
        )
        out = src.validate(df)
        assert list(out["symbol"]) == ["GT:US:a"]

    def test_unknown_currency_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="no trends terms"):
            _trends_source(_FakeTrendReq(), currencies=["XXX"])

    def test_trends_symbol_caps_and_underscores(self) -> None:
        assert trends_symbol("US", "unemployment benefits") == ("GT:US:unemployment_benefits")
        assert len(trends_symbol("US", "x" * 100)) == 64


# ---------------------------------------------------------------------- #
# ENTSO-E
# ---------------------------------------------------------------------- #

_GL_NS = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"

_LOAD_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="{_GL_NS}">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2026-06-01T00:00Z</start>
        <end>2026-06-01T01:00Z</end>
      </timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>41000</quantity></Point>
      <Point><position>2</position><quantity>42000</quantity></Point>
      <Point><position>3</position><quantity>43000</quantity></Point>
      <Point><position>4</position><quantity>44000</quantity></Point>
    </Period>
  </TimeSeries>
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2026-06-02T00:00Z</start>
        <end>2026-06-02T02:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>50000</quantity></Point>
      <Point><position>2</position><quantity>52000</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>
"""

_ACK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument
    xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
  <Reason>
    <code>999</code>
    <text>No matching data found</text>
  </Reason>
</Acknowledgement_MarketDocument>
"""


class TestEntsoe:
    def test_missing_token_not_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
        src = EntsoeSource(areas=["DE"])
        assert not src.is_configured()
        with pytest.raises(ConfigurationError, match="ENTSOE_API_TOKEN"):
            src.fetch(START, END)

    def test_token_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENTSOE_API_TOKEN", "tok-from-env")
        src = EntsoeSource(areas=["DE"])
        assert src.is_configured()
        assert src.api_token == "tok-from-env"

    def test_unknown_area_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown ENTSO-E areas"):
            EntsoeSource(areas=["ZZ"], api_token="t")

    def test_fetch_parses_canned_xml(self) -> None:
        calls: list[dict[str, str]] = []

        def fake_get(url: str, params: dict[str, str]) -> str:
            calls.append(params)
            return _LOAD_XML

        src = EntsoeSource(areas=["DE"], api_token="tok", http_get_text=fake_get)
        raw = src.fetch(START, END)
        assert len(raw) == 6  # 4 quarter-hour + 2 hourly points
        assert (raw["area"] == "DE").all()
        # Position arithmetic: PT15M point 2 is start + 15 min.
        second = raw.iloc[1]
        assert second["ts"] == datetime(2026, 6, 1, 0, 15, tzinfo=UTC)
        assert second["quantity_mw"] == 42000.0
        # Request wiring: A65/A16 + area EIC + token.
        assert calls[0]["documentType"] == "A65"
        assert calls[0]["processType"] == "A16"
        assert calls[0]["outBiddingZone_Domain"] == "10Y1001A1001A83F"
        assert calls[0]["securityToken"] == "tok"
        assert calls[0]["periodStart"] == "202606010000"

    def test_acknowledgement_rejection_skipped_per_area(self) -> None:
        """One rejected area logs a warning and yields no rows, but the
        other areas still return."""

        def fake_get(url: str, params: dict[str, str]) -> str:
            if params["outBiddingZone_Domain"] == "10Y1001A1001A83F":  # DE
                return _ACK_XML
            return _LOAD_XML

        src = EntsoeSource(
            areas=["DE", "FR"],
            api_token="tok",
            http_get_text=fake_get,
        )
        raw = src.fetch(START, END)
        assert set(raw["area"]) == {"FR"}

    def test_acknowledgement_parse_raises_with_reason(self) -> None:
        with pytest.raises(AltDataError, match="No matching data found"):
            EntsoeSource._parse_load_xml(_ACK_XML, "DE")

    def test_transform_daily_mean(self) -> None:
        src = EntsoeSource(
            areas=["DE"],
            api_token="tok",
            http_get_text=lambda u, p: _LOAD_XML,
        )
        out = src.run(START, END)
        assert list(out.columns) == CANONICAL_COLUMNS
        assert (out["symbol"] == "ENTSOE:DE:load").all()
        assert (out["source"] == "entsoe").all()
        by_day = out.set_index("ts")["value"]
        assert len(by_day) == 2
        assert by_day.iloc[0] == pytest.approx(42500.0)  # mean of 41-44k
        assert by_day.iloc[1] == pytest.approx(51000.0)  # mean of 50k, 52k

    def test_validate_drops_non_positive_load(self) -> None:
        src = EntsoeSource(areas=["DE"], api_token="tok")
        df = pd.DataFrame(
            {
                "ts": [START, END],
                "symbol": ["ENTSOE:DE:load"] * 2,
                "value": [42000.0, -5.0],
                "source": ["entsoe"] * 2,
            }
        )
        out = src.validate(df)
        assert list(out["value"]) == [42000.0]

    def test_unsupported_resolution_raises(self) -> None:
        bad = _LOAD_XML.replace("PT15M", "PT7M")
        with pytest.raises(AltDataError, match="unsupported ENTSO-E resolution"):
            EntsoeSource._parse_load_xml(bad, "DE")


# ---------------------------------------------------------------------- #
# Stubs
# ---------------------------------------------------------------------- #


class TestStubs:
    @pytest.mark.parametrize(
        "cls",
        [SatelliteShippingSource, ConsumerSpendingSource],
    )
    def test_stub_fetch_raises(self, cls: type[AlternativeDataSource]) -> None:
        src = cls()
        assert src.status == "stub"
        assert not src.is_configured()
        with pytest.raises(NotImplementedError):
            src.fetch(START, END)


# ---------------------------------------------------------------------- #
# Live smoke test — excluded from default runs. Same env-var gate as
# tests/integration/test_live_external_endpoints.py:
#   CURLIT_RUN_NETWORK_TESTS=1 pytest tests/unit/test_alternative_data.py
# (A live demo also exists: .venv/bin/python -m src.data.alternative)
# ---------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("CURLIT_RUN_NETWORK_TESTS", "0") != "1",
    reason="set CURLIT_RUN_NETWORK_TESTS=1 to enable network tests",
)
def test_live_google_trends_smoke() -> None:
    """Acceptance check: a real Google Trends fetch returns data.
    Anonymous clients get 429'd routinely — skip (not fail) on that."""
    src = GoogleTrendsSource(currencies=["USD"])
    try:
        out = src.run(datetime(2026, 3, 1), datetime(2026, 6, 1))
    except Exception as exc:  # noqa: BLE001 - rate limits shouldn't fail CI
        pytest.skip(f"live Google Trends fetch unavailable: {exc}")
    if out.empty:
        pytest.skip("live Google Trends fetch returned no rows (rate limited?)")
    assert list(out.columns) == CANONICAL_COLUMNS
    assert (out["value"] >= 0).all()
