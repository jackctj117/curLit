"""Protocol fixtures derived from CL-uofe failure classes, not model benchmarks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.events.kimi_tool_agent import KimiToolAgent, _source_packet_finalization
from src.events.niche_scoring import parse_niche_ideas
from src.events.niche_shadow import CapturedInput, ModelReply, ResearchBudget, compare_captured
from src.events.passage_contract import PASSAGE_VERSION, PassageRegistry
from src.events.research_evidence import DiscoveryOutcome, SourceDocument


def document(
    text: str = "Acme supplies pumps to Beta. This segment is 12% of revenue.",
) -> SourceDocument:
    return SourceDocument(
        "ACME",
        "https://example.org/filing",
        "2026-09-01T00:00:00Z",
        "2026-09-08T12:00:00Z",
        text,
        "Item 1",
    )


def envelope(ref: str = "S1:P1") -> dict[str, Any]:
    return {
        "schema_version": PASSAGE_VERSION,
        "niche_ideas": [
            {
                "ticker": "ACME",
                "company_name": "Acme",
                "action": "buy_calls",
                "direction": "bullish",
                "hop_count": 1,
                "torque_reason": "Pump exposure",
                "rationale": "Research lead only",
                "confidence": 0.5,
                "claims": [
                    {
                        "kind": "documented_fact",
                        "role": "relationship",
                        "statement": "Acme supplies Beta",
                        "passage_ref": ref,
                    }
                ],
            }
        ],
    }


def test_reference_supplies_exact_captured_quote_and_hash_not_model_transcription() -> None:
    registry = PassageRegistry()
    captured = document()
    shown = registry.render([captured])
    assert shown[0]["passages"] == [{"passage_ref": "S1:P1", "text": captured.text}]
    normalized = json.loads(registry.normalize(envelope()))
    claim = normalized["niche_ideas"][0]["claims"][0]
    assert claim["source_id"] == captured.source_id
    assert claim["passage"] == captured.text
    assert (
        parse_niche_ideas(json.dumps(normalized))[0]
        .claims[0]
        .backed([captured], datetime(2026, 9, 9, tzinfo=UTC), "ACME")
    )


@given(st.text(min_size=1, max_size=2400))
def test_passage_chunks_preserve_all_captured_characters(text: str) -> None:
    registry = PassageRegistry()
    result = registry.render([document(text)])[0]
    assert "".join(p["text"] for p in result["passages"]) == text
    assert registry.render([document(text)]) == [result]


def test_registry_never_resolves_an_unretrieved_or_previous_invocation_source() -> None:
    first = PassageRegistry()
    first.render([document()])
    with pytest.raises(ValueError, match="unknown_passage_reference"):
        PassageRegistry().normalize(envelope())
    with pytest.raises(ValueError, match="unknown_passage_reference"):
        first.normalize(envelope("S2:P1"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "fact"),
        ("role", "story"),
        ("statement", " "),
        ("passage_ref", "invented-hash"),
        ("source_id", "invented-hash"),
        ("passage", "a convenient fabricated quote"),
    ],
)
def test_schema_rejects_malformed_claims_and_model_supplied_provenance(
    field: str, value: str
) -> None:
    registry = PassageRegistry()
    registry.render([document()])
    raw = envelope()
    raw["niche_ideas"][0]["claims"][0][field] = value
    with pytest.raises(ValueError):
        registry.normalize(raw)


def test_wrong_company_and_future_source_still_fail_existing_provenance_gate() -> None:
    registry = PassageRegistry()
    registry.render([document()])
    claim = parse_niche_ideas(registry.normalize(envelope()))[0].claims[0]
    assert not claim.backed([document()], datetime(2026, 9, 9, tzinfo=UTC), "OTHER")
    assert not claim.backed([document()], datetime(2026, 8, 1, tzinfo=UTC), "ACME")


def test_correct_reference_does_not_verify_semantic_support_or_review() -> None:
    registry = PassageRegistry()
    registry.render([document()])
    raw = envelope()
    raw["niche_ideas"][0]["claims"][0]["statement"] = "Acme exclusively supplies aircraft engines"
    idea = parse_niche_ideas(registry.normalize(raw))[0]
    assert idea.review_status != "supported"
    assert not idea.research_eligible


def test_fact_and_inference_are_preserved_and_abstention_has_a_schema() -> None:
    registry = PassageRegistry()
    registry.render([document()])
    raw = envelope()
    raw["niche_ideas"][0]["claims"][0]["kind"] = "inference"
    assert parse_niche_ideas(registry.normalize(raw))[0].claims[0].kind == "inference"
    assert (
        json.loads(registry.normalize({"schema_version": PASSAGE_VERSION, "niche_ideas": []}))[
            "niche_ideas"
        ]
        == []
    )
    with pytest.raises(ValueError):
        registry.normalize({"niche_ideas": []})


@pytest.mark.parametrize("ref,status", [("S1:P1", "completed"), ("S99:P1", "invalid_output")])
def test_native_kimi_loop_resolves_references_and_keeps_raw_response(ref: str, status: str) -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        tc = SimpleNamespace(
            id="tool",
            function=SimpleNamespace(name="get_sec_filing", arguments='{"ticker":"ACME"}'),
        )
        msg = SimpleNamespace(
            content=None if len(calls) == 1 else json.dumps(envelope(ref)),
            tool_calls=[tc] if len(calls) == 1 else [],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    tools = SimpleNamespace(filing_documents=lambda *args: [document()])
    universe = SimpleNamespace(get_cik=lambda ticker: 1)
    agent = KimiToolAgent(
        universe, tools, api_key="fixture", create_fn=create, passage_references=True
    )
    result = agent.discover_result({"id": 1, "headline": "Pump supply event"})
    assert result.status == status
    assert result.prompt_version.endswith(PASSAGE_VERSION)
    tool_message = json.loads(calls[1]["messages"][-1]["content"])
    assert tool_message["sources"][0]["passages"][0]["passage_ref"] == "S1:P1"
    assert result.trace[-1]["response"]["content"] == json.dumps(envelope(ref))
    if status == "completed":
        assert parse_niche_ideas(result.text)[0].claims[0].passage == document().text


def test_context_finalization_preserves_original_reference_labels() -> None:
    registry = PassageRegistry()
    docs = [document(), document("Contrary evidence")]
    registry.render(docs)
    outcome = DiscoveryOutcome("unavailable", sources=docs)
    messages = _source_packet_finalization([], outcome, registry)
    packet = json.loads(messages[-2]["content"])
    assert packet["sources"] == [d.to_dict() for d in docs]
    assert packet["passage_sources"][1]["passages"][0]["passage_ref"] == "S2:P1"
    assert (
        json.loads(registry.normalize(envelope("S2:P1")))["niche_ideas"][0]["claims"][0]["passage"]
        == docs[1].text
    )


def test_equivalent_tool_shadow_arms_use_same_passage_contract_and_unknown_liquidity() -> None:
    capture = CapturedInput.capture(
        {
            "captured_at": "2026-09-09T00:00:00Z",
            "event": {"headline": "Pumps"},
            "symbols": {"ACME": {"exchange": "NYSE"}},
            "market_data": {},
            "sources": [document().to_dict()],
        }
    )

    class Model:
        provider = "fixture"

        def __init__(self, model: str) -> None:
            self.model = model
            self.requests: list[list[dict[str, str]]] = []

        def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
            self.requests.append(messages)
            raw = (
                {"tool_requests": [{"name": "get_sec_filing", "arguments": {"ticker": "ACME"}}]}
                if len(self.requests) == 1
                else envelope()
            )
            return ModelReply(json.dumps(raw), self.model, 1, 1)

    baseline, challenger = Model("baseline"), Model("challenger")
    report = compare_captured(
        capture,
        [baseline, challenger],
        passage_references=True,
        budget=ResearchBudget(model_calls=2),
    )
    assert report["mode"] == "shadow_only"
    assert baseline.requests == challenger.requests
    assert all(t["discovery"]["status"] == "completed" for t in report["trials"])
    assert all(t["discovery"]["billed_cost_usd"] is None for t in report["trials"])
    for trial in report["trials"]:
        assert trial["metrics"]["supported_after_critic"] == 0
        assert trial["raw_candidates"][0]["research"]["liquidity_status"] == "unknown"
