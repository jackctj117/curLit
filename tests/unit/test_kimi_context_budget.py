"""CL-lu3d: independent serialized-size and exact-source oracles, no providers."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from src.events.kimi_tool_agent import KimiToolAgent
from src.events.research_evidence import SourceDocument


def source(text: str, number: int = 1) -> SourceDocument:
    return SourceDocument(
        "ACME",
        f"https://www.sec.gov/Archives/fixture-{number}.htm",
        "2026-08-01T00:00:00+00:00",
        "2026-09-08T12:00:00+00:00",
        text,
        "fixture: business and risk factors",
    )


def response(reasoning: str = "", count: int = 1, final: bool = False) -> Any:
    calls = (
        []
        if final
        else [
            SimpleNamespace(
                id=f"c{i}",
                function=SimpleNamespace(
                    name="get_sec_filing",
                    arguments=json.dumps({"ticker": "ACME", "query": str(i)}),
                ),
            )
            for i in range(count)
        ]
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"niche_ideas": []}' if final else "",
                    reasoning_content=reasoning,
                    tool_calls=calls,
                ),
                finish_reason="stop" if final else "tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def test_oversized_history_finalizes_with_complete_deduplicated_evidence() -> None:
    documents = [source("Exact factual passage. " + "x" * 8000, n) for n in range(3)]
    requests = []
    tool_result = {
        "sources": [s.to_dict() for s in documents],
        "contrary_fact": "Customer delayed delivery",
        "status": "captured_only",
        "source_ids": ["pre-existing metadata field"],
    }
    original = copy.deepcopy(tool_result)

    def create(**kwargs: Any) -> Any:
        # External boundary fails independently if the implementation sends too much.
        assert len(json.dumps(kwargs["messages"])) <= 100000
        requests.append(copy.deepcopy(kwargs))
        if len(requests) == 1:
            return response("r" * 60000, count=3)
        assert kwargs["tool_choice"] == "none"
        return response(final=True)

    agent = KimiToolAgent(None, None, api_key="fixture", create_fn=create)
    agent._dispatch = lambda name, args: copy.deepcopy(tool_result)
    outcome = agent.discover_result({"id": 1, "headline": "Customer order changed"})
    assert outcome.status == "abstained"
    assert len(requests) == 2
    final_messages = requests[-1]["messages"]
    assert all(m["role"] in {"system", "user"} for m in final_messages)
    packet = json.loads(final_messages[-2]["content"])
    assert packet["sources"] == [s.to_dict() for s in documents]
    assert len(packet["tool_results"]) == 3
    assert all(
        r["result"]["contrary_fact"] == "Customer delayed delivery" for r in packet["tool_results"]
    )
    assert all(
        r["result"]["source_ids"] == ["pre-existing metadata field"] for r in packet["tool_results"]
    )
    assert all(r["source_ids"] == [s.source_id for s in documents] for r in packet["tool_results"])
    assert tool_result == original
    assert outcome.sources == documents
    assert outcome.trace[0]["response"]["reasoning_content"] == "r" * 60000
    transition = next(t for t in outcome.trace if "context_transition" in t)
    assert transition["original_prompt_chars"] > 100000
    assert transition["final_prompt_chars"] <= 100000
    assert sum(r["max_tokens"] for r in requests) <= 8 * 4096


def test_evidence_packet_that_cannot_fit_fails_explicitly_without_truncating() -> None:
    document = source("x" * 110000)
    requests = []

    def create(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return response() if len(requests) == 1 else response(final=True)

    agent = KimiToolAgent(None, None, api_key="fixture", create_fn=create)
    agent._dispatch = lambda name, args: {"sources": [document.to_dict()]}
    outcome = agent.discover_result({"id": 1})
    assert outcome.status == "budget_exhausted"
    assert outcome.reason == "prompt_char_limit"
    assert len(requests) == 1
    assert outcome.sources == [document]
    assert not outcome.text  # Do not manufacture abstention after losing evidence.


def test_oversized_initial_event_never_reaches_provider() -> None:
    requests = []
    agent = KimiToolAgent(
        None,
        None,
        api_key="fixture",
        create_fn=lambda **kw: requests.append(kw) or response(final=True),
    )
    outcome = agent.discover_result({"headline": "x" * 100000})
    assert outcome.status == "budget_exhausted" and outcome.reason == "prompt_char_limit"
    assert requests == []


@given(st.integers(min_value=0, max_value=15000), st.sampled_from(["é", "💵", "\n", '"']))
def test_serialized_unicode_history_obeys_cap_and_keeps_full_reasoning_when_continuing(
    repetitions: int,
    character: str,
) -> None:
    requests = []
    reasoning = character * repetitions

    def create(**kwargs: Any) -> Any:
        requests.append(copy.deepcopy(kwargs))
        return response(reasoning) if len(requests) == 1 else response(final=True)

    agent = KimiToolAgent(None, None, api_key="fixture", create_fn=create)
    agent._dispatch = lambda name, args: {
        "sources": [source("A documented relationship.").to_dict()]
    }
    outcome = agent.discover_result({"id": 1})
    assert outcome.status == "abstained"
    assert all(len(json.dumps(r["messages"])) <= 100000 for r in requests)
    for r in requests:
        for m in r["messages"]:
            if m["role"] == "assistant":
                assert m["reasoning_content"] == reasoning
