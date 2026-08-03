"""Tests for the fast event triage tier (CL-cunh).

Covers: JSON-array extraction from messy LLM text, batch scoring +
escalation threshold, fail-open on transport/parse failure, malformed-row
tolerance, and env-driven configuration. No live LLM calls — a canned mock
client only.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest

from src.events.triage import (
    EventTriage,
    TriageVerdict,
    extract_json_array,
)


class MockClient:
    """Returns a canned response string; records the calls."""

    def __init__(self, text_out: str, raises: bool = False) -> None:
        self.text_out = text_out
        self.raises = raises
        self.calls: list[dict] = []

    def complete(self, messages: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"messages": messages, "model": model, "kwargs": kwargs})
        if self.raises:
            raise RuntimeError("simulated transport failure")
        return SimpleNamespace(
            text=self.text_out,
            model=model,
            provider="mock",
            input_tokens=10,
            output_tokens=10,
            usd_cost=0.0,
            elapsed_sec=0.01,
        )


def _rows() -> list[dict[str, Any]]:
    return [
        {"id": 1, "theme": "energy_chokepoint", "headline": "Iran moves to close Hormuz"},
        {
            "id": 2,
            "theme": "energy_chokepoint",
            "headline": "Opinion: oil forecasts are usually wrong",
        },
    ]


# --------------------------------------------------------------------------- #
# extract_json_array
# --------------------------------------------------------------------------- #


def test_extract_bare_array():
    assert extract_json_array('[{"id": 1}]') == [{"id": 1}]


def test_extract_fenced_array_with_prose():
    raw = 'Here you go:\n```json\n[{"id": 1, "relevance": 5}]\n```\nThanks!'
    assert extract_json_array(raw) == [{"id": 1, "relevance": 5}]


def test_extract_array_embedded_in_prose():
    raw = 'The scores are [{"id": 7}] as requested.'
    assert extract_json_array(raw) == [{"id": 7}]


def test_extract_no_array_raises():
    with pytest.raises(ValueError, match="no JSON array"):
        extract_json_array("I decline to answer.")


def test_extract_object_only_raises():
    # A bare object has no '[' → treated as "no array present".
    with pytest.raises(ValueError, match="no JSON array"):
        extract_json_array('{"id": 1}')


# --------------------------------------------------------------------------- #
# score_batch
# --------------------------------------------------------------------------- #


def test_score_batch_escalates_high_skips_low():
    resp = json.dumps(
        [
            {"id": 1, "relevance": 9, "tradable": True, "reason": "chokepoint"},
            {"id": 2, "relevance": 1, "tradable": False, "reason": "opinion"},
        ]
    )
    client = MockClient(resp)
    triage = EventTriage(client=client, enabled=True, min_relevance=4)
    verdicts = triage.score_batch(_rows())
    assert verdicts[1].escalate is True
    assert verdicts[1].relevance == 9
    assert verdicts[2].escalate is False
    assert verdicts[2].reason == "opinion"
    # CL-u5cq: same single-shot contract as the impact agent — no toolset.
    assert client.calls[0]["kwargs"].get("no_tools") is True


def test_score_batch_threshold_is_inclusive():
    resp = json.dumps(
        [
            {"id": 1, "relevance": 4},  # exactly at the bar -> escalate
            {"id": 2, "relevance": 3},  # below -> skip
        ]
    )
    triage = EventTriage(client=MockClient(resp), enabled=True, min_relevance=4)
    verdicts = triage.score_batch(_rows())
    assert verdicts[1].escalate is True
    assert verdicts[2].escalate is False


def test_score_batch_fails_open_on_transport_error():
    triage = EventTriage(client=MockClient("", raises=True), enabled=True)
    # Empty map → caller escalates everything (nothing dropped).
    assert triage.score_batch(_rows()) == {}


def test_score_batch_fails_open_on_garbage():
    triage = EventTriage(client=MockClient("not json at all"), enabled=True)
    assert triage.score_batch(_rows()) == {}


def test_score_batch_empty_rows_no_call():
    client = MockClient("[]")
    triage = EventTriage(client=client, enabled=True)
    assert triage.score_batch([]) == {}
    assert client.calls == []  # never bothered the LLM


def test_score_batch_missing_id_absent_from_map():
    # Only id 1 comes back; id 2 is absent → caller will escalate it.
    resp = json.dumps([{"id": 1, "relevance": 8}])
    triage = EventTriage(client=MockClient(resp), enabled=True)
    verdicts = triage.score_batch(_rows())
    assert 1 in verdicts
    assert 2 not in verdicts


def test_score_batch_unparseable_relevance_escalates():
    resp = json.dumps([{"id": 1, "relevance": "high"}])
    triage = EventTriage(client=MockClient(resp), enabled=True, min_relevance=4)
    verdicts = triage.score_batch(_rows())
    assert verdicts[1].relevance == 10  # fail-safe → escalate
    assert verdicts[1].escalate is True


def test_score_batch_skips_non_dict_and_idless_items():
    resp = json.dumps(["garbage", {"relevance": 5}, {"id": 1, "relevance": 6}])
    triage = EventTriage(client=MockClient(resp), enabled=True)
    verdicts = triage.score_batch(_rows())
    assert list(verdicts) == [1]


def test_batch_prompt_lists_every_headline():
    client = MockClient("[]")
    triage = EventTriage(client=client, enabled=True)
    triage.score_batch(_rows())
    user_msg = client.calls[0]["messages"][1].content
    assert "Iran moves to close Hormuz" in user_msg
    assert "[1]" in user_msg and "[2]" in user_msg
    assert client.calls[0]["model"] == triage.model  # the Haiku tier


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("EVENT_TRIAGE_ENABLED", raising=False)
    assert EventTriage(client=MockClient("[]")).enabled is False


def test_enabled_via_env(monkeypatch):
    monkeypatch.setenv("EVENT_TRIAGE_ENABLED", "1")
    assert EventTriage(client=MockClient("[]")).enabled is True


def test_min_relevance_from_env(monkeypatch):
    monkeypatch.setenv("EVENT_TRIAGE_MIN_RELEVANCE", "7")
    assert EventTriage(client=MockClient("[]")).min_relevance == 7


def test_min_relevance_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("EVENT_TRIAGE_MIN_RELEVANCE", "not-an-int")
    assert EventTriage(client=MockClient("[]")).min_relevance == 4


def test_verdict_is_frozen():
    v = TriageVerdict(event_id=1, relevance=5, tradable=True, reason="x", escalate=True)
    with pytest.raises(FrozenInstanceError):
        v.relevance = 9  # type: ignore[misc]
