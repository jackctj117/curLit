"""Tests for the Event Impact Agent (CL-6iu7).

Covers: JSON extraction from messy LLM text, schema validation +
range clamping, equity watch-only enforcement, unreachable-instrument
filtering, and the DB write path (NEW → ASSESSED / DISMISSED) against a
sqlite engine with a mocked LLM client — no live LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.events.impact_agent import (
    EventImpactAgent,
    extract_json_object,
    normalise_assessment,
)
from src.events.playbooks import all_tradable_instruments, load_playbooks

PLAYBOOKS = load_playbooks("configs/event_playbooks.yaml")
FALLBACK = all_tradable_instruments(PLAYBOOKS)
HORMUZ = PLAYBOOKS["energy_chokepoint"]


def _valid_payload(**overrides: Any) -> dict:
    base: dict = {
        "core_event": "Iran announces closure of the Strait of Hormuz",
        "direction": "bearish",
        "urgency": 9,
        "horizon": "hours",
        "confidence": 0.8,
        "affected": [
            {"instrument": "BCO_USD", "kind": "oanda",
             "direction": "long", "reason": "supply risk premium"},
            {"instrument": "FRO", "kind": "equity_watch",
             "direction": "watch", "reason": "tanker rates"},
        ],
        "rationale": "Chokepoint closure is the canonical oil supply shock.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------- #
# extract_json_object
# ---------------------------------------------------------------------- #


class TestExtractJson:
    def test_bare_json(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json_with_prose(self) -> None:
        raw = 'Here is my assessment:\n```json\n{"a": {"b": 2}}\n```\nDone.'
        # Prose before the fence: the brace scan starts at the first '{'.
        assert extract_json_object(raw) == {"a": {"b": 2}}

    def test_json_embedded_in_prose(self) -> None:
        raw = 'Sure! {"a": 1, "s": "brace } in string"} trailing text'
        assert extract_json_object(raw) == {"a": 1, "s": "brace } in string"}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            extract_json_object("I cannot assess this headline.")

    def test_unbalanced_raises(self) -> None:
        with pytest.raises(ValueError, match="unbalanced"):
            extract_json_object('{"a": 1')


# ---------------------------------------------------------------------- #
# normalise_assessment
# ---------------------------------------------------------------------- #


class TestNormalise:
    def _norm(self, payload: dict) -> dict:
        return normalise_assessment(payload, "headline", HORMUZ, FALLBACK)

    def test_happy_path_passes_through(self) -> None:
        out = self._norm(_valid_payload())
        assert out["direction"] == "bearish"
        assert out["urgency"] == 9
        assert out["horizon"] == "hours"
        assert out["confidence"] == 0.8
        assert len(out["affected"]) == 2

    def test_urgency_and_confidence_clamped(self) -> None:
        out = self._norm(_valid_payload(urgency=99, confidence=1.7))
        assert out["urgency"] == 10
        assert out["confidence"] == 1.0
        out = self._norm(_valid_payload(urgency=-3, confidence=-0.4))
        assert out["urgency"] == 1
        assert out["confidence"] == 0.0

    def test_horizon_singular_normalised(self) -> None:
        assert self._norm(_valid_payload(horizon="hour"))["horizon"] == "hours"
        assert self._norm(_valid_payload(horizon="Days"))["horizon"] == "days"

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            self._norm(_valid_payload(direction="to the moon"))

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            self._norm(_valid_payload(horizon="weeks"))

    def test_non_numeric_urgency_raises(self) -> None:
        with pytest.raises(ValueError, match="urgency"):
            self._norm(_valid_payload(urgency="high"))

    def test_equity_forced_to_watch(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "FRO", "kind": "equity_watch",
             "direction": "long", "reason": "tankers"},
        ])
        out = self._norm(payload)
        assert out["affected"][0]["direction"] == "watch"

    def test_unreachable_tradable_dropped(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "brent futures!!", "kind": "oanda",
             "direction": "long", "reason": "x"},
            {"instrument": "BCO_USD", "kind": "oanda",
             "direction": "long", "reason": "ok"},
        ])
        out = self._norm(payload)
        assert [a["instrument"] for a in out["affected"]] == ["BCO_USD"]

    def test_unvetted_oanda_shaped_symbol_demoted_to_watch(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "USD_TRY", "kind": "fx",
             "direction": "long", "reason": "lira stress"},
        ])
        out = self._norm(payload)
        assert out["affected"][0] == {
            "instrument": "USD_TRY", "kind": "fx",
            "direction": "watch", "reason": "lira stress",
        }

    def test_slash_symbol_normalised(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "eur/usd", "kind": "fx",
             "direction": "short", "reason": "x"},
        ])
        out = self._norm(payload)
        assert out["affected"][0]["instrument"] == "EUR_USD"
        assert out["affected"][0]["direction"] == "short"  # vetted (playbooks)

    def test_duplicate_instruments_collapsed(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "BCO_USD", "kind": "oanda",
             "direction": "long", "reason": "a"},
            {"instrument": "BCO_USD", "kind": "oanda",
             "direction": "short", "reason": "b"},
        ])
        assert len(self._norm(payload)["affected"]) == 1

    def test_affected_not_a_list_raises(self) -> None:
        with pytest.raises(ValueError, match="affected"):
            self._norm(_valid_payload(affected="BCO_USD"))

    def test_empty_core_event_falls_back_to_headline(self) -> None:
        out = self._norm(_valid_payload(core_event=""))
        assert out["core_event"] == "headline"


# ---------------------------------------------------------------------- #
# agent + DB round trip (mock LLM, sqlite)
# ---------------------------------------------------------------------- #


class MockLLMClient:
    def __init__(self, text_out: str) -> None:
        self.text_out = text_out
        self.calls: list[dict] = []

    def complete(self, messages: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"messages": messages, "model": model})
        return SimpleNamespace(
            text=self.text_out, model=model, provider="mock",
            input_tokens=10, output_tokens=10,
            usd_cost=0.0, elapsed_sec=0.01,
        )


def _shim(sql: str) -> str:
    return (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("JSONB", "TEXT")
        .replace("BIGSERIAL", "INTEGER")
    )


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'geo.db'}")
    sql = _shim(_strip_sql_comments(
        Path("migrations/005_geo_events.sql").read_text(),
    ))
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


def _insert_event(
    engine: Engine, external_id: str, headline: str,
    theme: str | None = "energy_chokepoint", seen_at: str = "2026-07-14T09:30:00",
) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO geo_events "
            "(seen_at, source, external_id, headline, url, theme, "
            " status, status_updated_at) "
            "VALUES (:seen, 'gdelt', :eid, :hl, 'https://n.test/x', :theme, "
            "'NEW', :seen)",
        ), {"seen": seen_at, "eid": external_id, "hl": headline, "theme": theme})


def _fetch(engine: Engine, external_id: str) -> dict:
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT status, assessment FROM geo_events WHERE external_id=:e",
        ), {"e": external_id}).one()
    return {"status": row[0], "assessment": json.loads(row[1])}


class TestAgentRoundTrip:
    def test_happy_path_marks_assessed(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Iran moves to close Hormuz")
        client = MockLLMClient(json.dumps(_valid_payload()))
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events()

        assert len(results) == 1
        assert results[0].status == "ASSESSED"
        row = _fetch(engine, "e1")
        assert row["status"] == "ASSESSED"
        a = row["assessment"]
        assert a["direction"] == "bearish"
        assert {x["instrument"] for x in a["affected"]} == {"BCO_USD", "FRO"}
        assert "conf=0.80" in results[0].summary_line()
        # Prompt carried the playbook context
        user_msg = client.calls[0]["messages"][1].content
        assert "energy_chokepoint" in user_msg
        assert "BCO_USD" in user_msg

    def test_malformed_llm_output_marks_dismissed(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Some headline")
        client = MockLLMClient("I refuse to answer in JSON, sorry.")
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events()

        assert results[0].status == "DISMISSED"
        row = _fetch(engine, "e1")
        assert row["status"] == "DISMISSED"
        assert "impact agent failure" in row["assessment"]["rationale"]

    def test_schema_violation_marks_dismissed(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Some headline")
        bad = json.dumps(_valid_payload(direction="sideways"))
        agent = EventImpactAgent(engine, client=MockLLMClient(bad))  # type: ignore[arg-type]
        assert agent.assess_new_events()[0].status == "DISMISSED"

    def test_fenced_output_still_parses(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Hormuz headline")
        fenced = f"```json\n{json.dumps(_valid_payload())}\n```"
        agent = EventImpactAgent(engine, client=MockLLMClient(fenced))  # type: ignore[arg-type]
        assert agent.assess_new_events()[0].status == "ASSESSED"

    def test_limit_caps_processing(self, engine: Engine) -> None:
        for i in range(3):
            _insert_event(
                engine, f"e{i}", f"headline {i}",
                seen_at=f"2026-07-14T09:0{i}:00",
            )
        client = MockLLMClient(json.dumps(_valid_payload()))
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events(limit=2)

        assert len(results) == 2
        # Newest first: e2, e1 assessed; e0 still NEW for the next cycle.
        assert {r.event_id for r in results} == {2, 3}
        with engine.connect() as conn:
            remaining = conn.execute(text(
                "SELECT external_id FROM geo_events WHERE status='NEW'",
            )).scalars().all()
        assert remaining == ["e0"]

    def test_unmatched_theme_uses_fallback_whitelist(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Odd headline", theme=None)
        client = MockLLMClient(json.dumps(_valid_payload()))
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events()
        assert results[0].status == "ASSESSED"
        user_msg = client.calls[0]["messages"][1].content
        assert "No playbook matched" in user_msg
