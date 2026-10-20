"""CL-5hj5: quoted evidence is data, not JSON syntax or implicit approval.

The quotation mirrors the captured October 20 canary failure. Provider calls
are mocked; expected statuses follow the evidence contract, not model opinion.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.events.adversarial_critic import AdversarialCritic
from src.events.niche_scoring import NicheIdea
from src.events.research_evidence import RelationshipClaim, SourceDocument

NOW = datetime(2026, 10, 20, tzinfo=UTC)
PASSAGE = 'AeroVironment, Inc. (the "Company") supplies drones.'
SOURCE = SourceDocument(
    "AVAV", "https://example.org/filing", "2026-09-09T00:00:00Z",
    "2026-10-19T00:00:00Z", PASSAGE, "synthetic quoted passage",
)


def candidate() -> NicheIdea:
    return NicheIdea(
        ticker="AVAV", company_name="AeroVironment", action="long", direction="bullish",
        hop_count=2, torque_reason="supplier", rationale="research hypothesis", confidence=0.7,
        verified=True, evidence_status="source_backed", sources=[SOURCE],
        claims=[RelationshipClaim("documented_fact", "relationship", "Supplies drones",
                                  SOURCE.source_id, PASSAGE)],
    )


def payload() -> dict[str, Any]:
    return {"verdicts": [{
        "ticker": "AVAV", "verdict": "supported", "strongest_attack": "Exposure uncertain",
        "adjusted_confidence": 0.6,
        "citations": [{"source_id": SOURCE.source_id, "passage": PASSAGE}],
    }]}


def review(raw: str) -> tuple[NicheIdea, Mock]:
    client = Mock()
    client.complete.return_value = SimpleNamespace(text=raw, model="claude-sonnet-4-6")
    idea = candidate()
    AdversarialCritic(client=client, enabled=True, subscription_only=True).apply(
        [idea], {"id": 1}, as_of=NOW,
    )
    return idea, client


def test_quoted_passage_roundtrips_with_schema_and_unknown_liquidity() -> None:
    raw = json.dumps(payload())
    idea, client = review(raw)
    assert idea.review_status == "supported"
    assert idea.review_provenance["response_text"] == raw
    assert idea.review_provenance["citations"][0]["passage"] == PASSAGE
    assert not idea.research_eligible  # Unknown liquidity is still not approval.
    assert client.complete.call_count == 1  # No application repair loop.
    args = client.complete.call_args.kwargs
    assert args["subscription_only"] and args["no_tools"]
    assert args["json_schema"]["additionalProperties"] is False


def test_captured_unescaped_quote_failure_is_retained_not_repaired() -> None:
    raw = json.dumps(payload()).replace('\\"Company\\"', '"Company"')
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    idea, client = review(raw)
    assert idea.review_status == "review_unavailable"
    assert idea.review_reason == "invalid_review_output"
    assert idea.review_provenance["response_text"] == raw
    assert idea.review_provenance["actual_model"] == "claude-sonnet-4-6"
    assert not idea.red_team_verdict and not idea.research_eligible
    assert client.complete.call_count == 1


@pytest.mark.parametrize("mutation", [
    "duplicate", "missing", "unknown_ticker", "extra_field", "missing_citations",
    "string_confidence", "boolean_confidence", "nan_confidence", "extra_citation_field",
])
def test_invalid_review_contract_never_approves(mutation: str) -> None:
    body = payload()
    verdict = body["verdicts"][0]
    if mutation == "duplicate":
        body["verdicts"].append(copy.deepcopy(verdict))
    elif mutation == "missing":
        body["verdicts"] = []
    elif mutation == "unknown_ticker":
        verdict["ticker"] = "OTHER"
    elif mutation == "extra_field":
        verdict["approved"] = True
    elif mutation == "missing_citations":
        del verdict["citations"]
    elif mutation == "extra_citation_field":
        verdict["citations"][0]["invented"] = True
    else:
        verdict["adjusted_confidence"] = {
            "string_confidence": "0.6", "boolean_confidence": True,
            "nan_confidence": float("nan"),
        }[mutation]
    idea, _ = review(json.dumps(body))
    assert idea.review_status == "review_unavailable"
    assert idea.review_provenance["response_text"] == json.dumps(body)
    assert not idea.red_team_verdict and not idea.research_eligible


@pytest.mark.parametrize("mutation", ["unknown_source", "changed_passage", "insufficient"])
def test_valid_format_does_not_override_evidence(mutation: str) -> None:
    body = payload()
    verdict = body["verdicts"][0]
    if mutation == "unknown_source":
        verdict["citations"][0]["source_id"] = "not-captured"
    elif mutation == "changed_passage":
        verdict["citations"][0]["passage"] = "Guaranteed revenue growth"
    else:
        verdict["verdict"] = "insufficient_evidence"
    idea, _ = review(json.dumps(body))
    assert idea.review_status == "insufficient_evidence"
    assert not idea.red_team_verdict and not idea.research_eligible


@given(st.text(min_size=1))
def test_schema_roundtrip_never_rewrites_a_quotation(passage: str) -> None:
    from src.events.critic_contract import ReviewBatch

    body = payload()
    body["verdicts"][0]["citations"][0]["passage"] = passage
    parsed = ReviewBatch.model_validate_json(json.dumps(body))
    assert parsed.verdicts[0].citations[0].passage == passage
