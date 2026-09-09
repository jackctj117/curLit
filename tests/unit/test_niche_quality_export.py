"""Archived-input reconstruction preserves unknowns and excludes unverified identity."""

import copy

import pytest
from scripts.export_niche_quality_inputs import captured_failure


def audit():
    return {
        "recorded_at": "2026-09-09T00:00:00Z",
        "input_snapshot": {"id": 1, "headline": "Captured event"},
        "report": {
            "discovery": {
                "recorded_at_utc": "2026-09-09T00:00:00Z",
                "sources": [],
                "trace": [
                    {"tool": "check_ticker", "result": {"ticker": "ABC", "exists": True}},
                    {"tool": "check_ticker", "result": {"ticker": "FAKE", "exists": False}},
                    {
                        "tool": "get_company_profile",
                        "arguments": {"ticker": "ABC"},
                        "result": {"sector": "Industrials"},
                    },
                ],
            },
            "candidates": [
                {
                    "ticker": "ABC",
                    "research": {"last_close": 10, "market_observed_at": "2026-09-08T20:00:00Z"},
                }
            ],
        },
    }


def test_capture_uses_original_event_identity_and_measurements_without_invention():
    raw = audit()
    before = copy.deepcopy(raw)
    capture = captured_failure(raw)
    payload = capture.payload()
    assert raw == before
    assert payload["event"] == before["input_snapshot"]
    assert set(payload["symbols"]) == {"ABC"}
    assert payload["symbols"]["ABC"]["profile"] == {"sector": "Industrials"}
    assert payload["market_data"]["ABC"] == {
        "last_close": 10,
        "observed_at": "2026-09-08T20:00:00Z",
        "market_cap": None,
        "avg_dollar_volume": None,
    }
    raw["input_snapshot"]["headline"] = "Later mutation"
    assert capture.payload() == payload
    assert captured_failure(before).input_hash == capture.input_hash


def test_missing_capture_cutoff_is_not_replaced_with_current_time():
    raw = audit()
    raw["recorded_at"] = None
    with pytest.raises(ValueError, match="UTC cutoff"):
        captured_failure(raw)


def test_new_market_receipt_timestamp_round_trips_into_frozen_validation():
    from datetime import UTC, datetime

    from src.events.niche_scoring import NicheIdea, evidence_score

    observed = "2026-09-08T20:00:00+00:00"
    received = "2026-09-08T20:01:00+00:00"
    idea = NicheIdea(
        ticker="ABC",
        company_name="ABC",
        action="long",
        direction="bullish",
        hop_count=1,
        torque_reason="fixture",
        rationale="fixture",
        confidence=0.5,
    )
    measurements = {
        "observed_at": observed,
        "retrieved_at": received,
        "avg_dollar_volume": 100_000_000,
        "last_close": 10,
        "market_cap": 1_000_000_000,
    }
    evidence_score(idea, {"ABC": measurements}, datetime(2026, 9, 9, tzinfo=UTC))
    raw = audit()
    raw["report"]["candidates"] = [idea.to_trade_idea()]
    assert captured_failure(raw).payload()["market_data"]["ABC"] == measurements
