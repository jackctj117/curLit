"""Independent evidence/status oracles; synthetic sources, no providers or DB."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime, tzinfo
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.events.adversarial_critic import AdversarialCritic
from src.events.kimi_tool_agent import KimiToolAgent
from src.events.niche_agent import NicheAgent
from src.events.niche_scoring import NicheIdea, evidence_score, parse_niche_ideas
from src.events.niche_shadow import (
    CapturedInput,
    FrozenTools,
    ModelReply,
    ResearchBudget,
    blinded_candidates,
    compare_captured,
    discover_captured,
)
from src.events.research_evidence import DiscoveryOutcome, RelationshipClaim, SourceDocument
from src.events.research_tools import ResearchTools

NOW = datetime(2026, 9, 8, 18, tzinfo=UTC)


@pytest.fixture(autouse=True)
def fixed_evidence_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freshness tests use the capture's clock, not the date CI happens to run."""

    class EvidenceClock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr("src.events.niche_agent.datetime", EvidenceClock)
    monkeypatch.setattr("src.events.adversarial_critic.datetime", EvidenceClock)


PASSAGE = (
    "Acme supplies pumps to Beta. Beta accounts for 40% of revenue. Delivery begins October 2026."
)
SOURCE = SourceDocument(
    "ACME",
    "https://www.sec.gov/Archives/acme.htm",
    "2026-08-01T00:00:00+00:00",
    "2026-09-08T12:00:00+00:00",
    PASSAGE,
    "10-Q; normalized-text chars 100:200",
)
MARKET = {
    "ACME": {
        "avg_dollar_volume": 5_000_000,
        "market_cap": 200_000_000,
        "observed_at": "2026-09-08T00:00:00+00:00",
        "retrieved_at": "2026-09-08T12:00:00+00:00",
    }
}


def idea() -> NicheIdea:
    claims = [
        RelationshipClaim("documented_fact", role, statement, SOURCE.source_id, PASSAGE)
        for role, statement in (
            ("relationship", "Acme supplies Beta"),
            ("exposure", "Beta is 40% of revenue"),
            ("catalyst", "Delivery starts October 2026"),
        )
    ]
    return NicheIdea(
        "ACME",
        "Acme Corp",
        "long",
        "bullish",
        2,
        "pumps",
        "delivery thesis",
        0.6,
        verified=True,
        claims=claims,
        sources=[SOURCE],
        discovery_status="completed",
    )


def snapshot() -> CapturedInput:
    return CapturedInput.capture(
        {
            "captured_at": NOW.isoformat(),
            "event": {"id": 1, "headline": "Beta orders pumps"},
            "symbols": {"ACME": {"security_name": "Acme Corp", "exchange": "NYSE"}},
            "sources": [SOURCE.to_dict()],
            "market_data": copy.deepcopy(MARKET),
        }
    )


def final_text() -> str:
    candidate = idea()
    return json.dumps(
        {"niche_ideas": [{k: v for k, v in asdict(candidate).items() if k not in ("sources",)}]}
    )


class CriticClient:
    def __init__(
        self, verdict: str = "supported", citations: list[dict[str, str]] | None = None
    ) -> None:
        self.verdict = verdict
        self.citations = (
            citations
            if citations is not None
            else [{"source_id": SOURCE.source_id, "passage": PASSAGE}]
        )
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=json.dumps(
                {
                    "verdicts": [
                        {
                            "ticker": "ACME",
                            "verdict": self.verdict,
                            "strongest_attack": "Beta concentration risk",
                            "citations": self.citations,
                            "adjusted_confidence": 0.5,
                        }
                    ]
                }
            )
        )


class ScriptedModel:
    provider = "scripted"
    model = "scripted-v1"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
        self.calls.append(copy.deepcopy(messages))
        return ModelReply(
            self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)],
            self.model,
            20,
            min(10, max_tokens),
        )


def test_source_roundtrip_and_fabricated_hash_rejected() -> None:
    assert SourceDocument.from_dict(SOURCE.to_dict()) == SOURCE
    raw = SOURCE.to_dict()
    raw["text"] = "Acme has no customers"
    with pytest.raises(ValueError, match="hash"):
        SourceDocument.from_dict(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_id": "invented"},
        {"passage": "Beta is 90% of revenue"},
        {"passage": ""},
        {"kind": "inference"},
    ],
)
def test_critical_claim_without_captured_fact_is_incomplete(mutation: dict[str, str]) -> None:
    candidate = idea()
    candidate.claims[0] = replace(candidate.claims[0], **mutation)
    evidence_score(candidate, MARKET, NOW)
    assert candidate.evidence_status == "insufficient_evidence"
    assert not candidate.research_eligible


@pytest.mark.parametrize(
    "mutation",
    [
        {"published_at": "2020-01-01T00:00:00+00:00"},
        {"published_at": ""},
        {"retrieved_at": "2026-09-09T00:00:00+00:00"},
        {"symbol": "OTHER"},
        {"url": "file:///secrets"},
    ],
)
def test_unusable_sources_cannot_pass(mutation: dict[str, str]) -> None:
    candidate = idea()
    doc = replace(SOURCE, **mutation)
    candidate.sources = [doc]
    candidate.claims = [replace(c, source_id=doc.source_id) for c in candidate.claims]
    evidence_score(candidate, MARKET, NOW)
    assert candidate.evidence_status == "insufficient_evidence"


@pytest.mark.parametrize("adv", [None, "bad", float("nan"), float("inf"), -1, True])
def test_unknown_adv_never_passes(adv: object) -> None:
    market = copy.deepcopy(MARKET)
    market["ACME"]["avg_dollar_volume"] = adv
    candidate = evidence_score(idea(), market, NOW)
    assert candidate.avg_dollar_volume is None
    assert candidate.liquidity_status == "unknown"
    assert candidate.dropped_reason == "liquidity_unknown"


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed_at", None),
        ("retrieved_at", None),
        ("observed_at", "2026-01-01T00:00:00+00:00"),
        ("observed_at", "2026-09-09T00:00:00+00:00"),
    ],
)
def test_unknown_or_stale_market_timestamps_block(field: str, value: object) -> None:
    market = copy.deepcopy(MARKET)
    market["ACME"][field] = value
    assert evidence_score(idea(), market, NOW).liquidity_status == "unknown"


@given(st.integers(1, 100), st.sampled_from(["", "sole supplier", "levered junior pure-play"]))
def test_story_elaboration_cannot_change_score(hops: int, prose: str) -> None:
    candidate = idea()
    candidate.hop_count, candidate.torque_reason = hops, prose
    evidence_score(candidate, MARKET, NOW)
    assert candidate.asymmetry_score == 3 / 4  # Three documented roles, one unavailable.
    assert candidate.evidence_status == "source_backed"


@given(st.floats(min_value=0, max_value=1e12, allow_nan=False, allow_infinity=False))
def test_liquidity_floor_has_independent_numeric_oracle(adv: float) -> None:
    market = copy.deepcopy(MARKET)
    market["ACME"]["avg_dollar_volume"] = adv
    candidate = evidence_score(idea(), market, NOW)
    assert candidate.liquidity_status == ("sufficient" if adv >= 2_000_000 else "insufficient")
    assert 0 <= candidate.asymmetry_score <= 1


def test_critic_gets_sources_measurements_and_no_external_tools() -> None:
    client = CriticClient()
    critic = AdversarialCritic(client=client, enabled=True)
    candidate = evidence_score(idea(), MARKET, NOW)
    critic.apply([candidate], {"headline": "order"}, as_of=NOW)
    assert candidate.review_status == "supported" and candidate.research_eligible
    assert candidate.red_team_verdict == "confirmed"
    assert candidate.confidence == 0.5
    assert client.calls[0]["no_tools"] is True
    prompt = client.calls[0]["messages"][1].content
    assert PASSAGE in prompt and SOURCE.url in prompt and "5000000" in prompt


@pytest.mark.parametrize("verdict", ["supported", "contradicted", "confirmed", "refuted"])
def test_unsourced_verdict_cannot_approve_or_remove_lead(verdict: str) -> None:
    candidate = evidence_score(idea(), MARKET, NOW)
    critic = AdversarialCritic(client=CriticClient(verdict, []), enabled=True)
    assert critic.apply([candidate], {}, as_of=NOW) == [candidate]
    assert candidate.review_status == "insufficient_evidence"
    assert not candidate.red_team_verdict and not candidate.research_eligible


def test_contradiction_requires_an_actual_passage() -> None:
    candidate = evidence_score(idea(), MARKET, NOW)
    critic = AdversarialCritic(client=CriticClient("contradicted"), enabled=True)
    assert critic.apply([candidate], {}, as_of=NOW) == []
    assert candidate.review_status == "contradicted"
    assert candidate.dropped_reason == "evidence_contradicted"


def test_valid_citation_cannot_launder_bad_quote_with_same_id() -> None:
    client = CriticClient(
        citations=[
            {"source_id": SOURCE.source_id, "passage": PASSAGE},
            {"source_id": SOURCE.source_id, "passage": "invented"},
        ]
    )
    candidate = evidence_score(idea(), MARKET, NOW)
    AdversarialCritic(client=client, enabled=True).apply([candidate], {}, as_of=NOW)
    assert candidate.review_status == "insufficient_evidence"


def test_failed_review_clears_old_approval() -> None:
    class Failed:
        def complete(self, **kwargs: Any) -> Any:
            raise RuntimeError("provider failure")

    candidate = evidence_score(idea(), MARKET, NOW)
    candidate.review_status, candidate.red_team_verdict = "supported", "confirmed"
    AdversarialCritic(client=Failed(), enabled=True).apply([candidate], {}, as_of=NOW)
    assert candidate.review_status == "review_unavailable"
    assert not candidate.red_team_verdict and not candidate.research_eligible


@pytest.mark.parametrize(
    "text,status",
    [("garbage", "invalid_output"), ("{}", "invalid_output"), ('{"niche_ideas": []}', "abstained")],
)
def test_native_discovery_empty_and_invalid_are_distinct(text: str, status: str) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=[]), finish_reason="stop"
            )
        ]
    )
    agent = KimiToolAgent(None, None, api_key="fixture", create_fn=lambda **kw: response)
    assert agent.discover_result({}).status == status


def test_native_unfunded_error_is_redacted_and_unavailable(caplog: Any) -> None:
    def fail(**kwargs: Any) -> Any:
        raise RuntimeError("account PRIVATE_ID suspended due to insufficient balance")

    result = KimiToolAgent(None, None, api_key="fixture", create_fn=fail).discover_result({})
    assert result.status == "unavailable" and result.reason == "insufficient_balance"
    assert "PRIVATE_ID" not in caplog.text and "PRIVATE_ID" not in json.dumps(result.to_dict())


def test_native_budget_finalization_preserves_sources_and_all_downstream_gates() -> None:
    """Synthetic captured tool packet proves transport recovery, not model quality."""
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if len(calls) == 1:
            tool_calls = [
                SimpleNamespace(
                    id=f"filing-{i}",
                    function=SimpleNamespace(
                        name="get_sec_filing",
                        arguments='{"ticker":"ACME"}',
                    ),
                )
                for i in range(25)
            ]
            message = SimpleNamespace(content="", tool_calls=tool_calls)
            finish = "tool_calls"
        else:
            assert kwargs["tool_choice"] == "none"
            message = SimpleNamespace(content=final_text(), tool_calls=[])
            finish = "stop"
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])

    tools = SimpleNamespace(filing_documents=lambda *args: [SOURCE])
    universe = SimpleNamespace(get_cik=lambda ticker: 1)
    result = KimiToolAgent(universe, tools, api_key="fixture", create_fn=create).discover_result(
        snapshot().payload()["event"],
    )
    assert result.status == "completed"
    assert result.sources == [SOURCE]  # Exact source identity survives finalization and caching.
    candidates = parse_niche_ideas(result.text)
    assert len(candidates) == 1
    candidate = candidates[0]
    candidate.sources, candidate.discovery_status, candidate.verified = (
        result.sources,
        result.status,
        True,
    )
    evidence_score(candidate, MARKET, NOW)
    assert candidate.evidence_status == "source_backed"
    assert not candidate.research_eligible  # A final answer still needs independent review.
    AdversarialCritic(client=CriticClient(), enabled=True).apply([candidate], {}, as_of=NOW)
    assert candidate.research_eligible
    evidence_score(candidate, {}, NOW)
    assert candidate.liquidity_status == "unknown"
    assert not candidate.research_eligible


def test_equivalent_tool_trials_preserve_raw_before_common_critic() -> None:
    request = json.dumps(
        {"tool_requests": [{"name": "get_sec_filing", "arguments": {"ticker": "ACME"}}]}
    )
    a, b = ScriptedModel([request, final_text()]), ScriptedModel([request, final_text()])
    report = compare_captured(
        snapshot(), [a, b], critic=AdversarialCritic(client=CriticClient(), enabled=True)
    )
    assert a.calls == b.calls
    for trial in report["trials"]:
        assert trial["raw_candidates"][0]["research"]["review_status"] == "not_requested"
        assert trial["post_critic_candidates"][0]["research"]["review_status"] == "supported"
        assert trial["discovery"]["input_tokens"] == 40
        assert trial["discovery"]["billed_cost_usd"] is None
    blind = blinded_candidates(report)
    assert all(set(row) == {"blind_id", "candidates"} for row in blind)
    assert "scripted-v1" not in json.dumps(blind)
    assert report["mode"] == "shadow_only"


def test_snapshot_has_no_live_fallback_and_is_immutable() -> None:
    capture = snapshot()
    original = capture.input_hash
    payload = capture.payload()
    payload["sources"][0]["text"] = "future changed data"
    assert capture.input_hash == original
    tools = FrozenTools(capture)
    for forbidden in ("submit_order", "execute_sql", "fetch_url", "read_file"):
        assert tools.dispatch(forbidden, {"url": "https://broker.invalid"}) == {
            "error": "tool_not_allowed"
        }
    assert tools.dispatch("get_sec_filing", {"ticker": "UNKNOWN"})["sources"] == []
    with pytest.raises(ValueError, match="cutoff"):
        CapturedInput.capture({**capture.payload(), "captured_at": "2026-01-01T00:00:00+00:00"})


def test_shadow_budget_stops_an_endless_tool_loop() -> None:
    request = '{"tool_requests":[{"name":"check_ticker","arguments":{"ticker":"ACME"}}]}'
    model = ScriptedModel([request])
    result = discover_captured(snapshot(), model, ResearchBudget(model_calls=5, tool_calls=1))
    assert result.status == "budget_exhausted" and result.reason == "tool_call_limit"
    assert len(model.calls) == 2
    assert len([t for t in result.trace if "tool" in t]) == 1


def test_parallel_reports_do_not_share_failure_state() -> None:
    class Provider:
        def discover_result(self, row: dict[str, Any], playbook: Any) -> DiscoveryOutcome:
            return DiscoveryOutcome("unavailable" if row["id"] == 1 else "abstained", text="")

    agent = NicheAgent(FrozenTools(snapshot()), client=CriticClient(), tool_agent=Provider())
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(agent.run_report, [{"id": 1}, {"id": 2}]))
    assert [r.discovery.status for r in reports] == ["unavailable", "abstained"]
    assert reports[0].discovery is not reports[1].discovery


def test_recent_targeted_filings_do_not_prefer_old_annual() -> None:
    index = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "8-K"],
                "accessionNumber": ["1-1", "1-2", "1-3"],
                "primaryDocument": ["annual.htm", "quarter.htm", "news.htm"],
                "filingDate": ["2025-12-01", "2026-06-01", "2026-08-01"],
            }
        }
    }
    calls = []

    def fetch(url: str, headers: dict[str, str]) -> str:
        calls.append(url)
        return (
            json.dumps(index)
            if "submissions" in url
            else "<p>" + "Background. " * 30 + PASSAGE + "</p>"
        )

    docs = ResearchTools(sec_http_get=fetch).filing_documents(1, "ACME", "Beta", limit=2)
    assert len(docs) == 2
    assert [d.url.rsplit("/", 1)[1] for d in docs] == ["news.htm", "quarter.htm"]
    assert all("Beta" in d.text and "normalized-text chars" in d.locator for d in docs)
    assert len(calls) == 3  # Index and two requested documents; no annual preference.


def test_parse_cannot_accept_model_claimed_verification() -> None:
    raw = json.loads(final_text())
    raw["niche_ideas"][0].update(evidence_status="source_backed", review_status="supported")
    parsed = parse_niche_ideas(json.dumps(raw))[0]
    assert parsed.review_status == "not_requested" and not parsed.research_eligible


def test_complete_pipeline_requires_evidence_review_before_merge() -> None:
    class Provider:
        def discover_result(self, row: dict[str, Any], playbook: Any) -> DiscoveryOutcome:
            return DiscoveryOutcome("completed", text=final_text(), sources=[SOURCE])

    client = CriticClient()
    agent = NicheAgent(
        FrozenTools(snapshot()),
        client=client,
        tool_agent=Provider(),
        market_data_fn=lambda tickers: copy.deepcopy(MARKET),
        critic=AdversarialCritic(client=client, enabled=True),
    )
    report = agent.run_report({"id": 1, "headline": "pump orders"})
    assert len(report.eligible) == 1
    assessment: dict[str, Any] = {}
    assert agent.merge_into_assessment(assessment, report.eligible) == 1
    merged = assessment["trade_ideas"][0]
    assert merged["research"]["review_status"] == "supported"
    assert merged["research"]["sources"][0]["source_id"] == SOURCE.source_id
    report.candidates[0].review_status = "review_unavailable"
    assert agent.merge_into_assessment({}, report.candidates) == 0


def test_summary_counts_only_post_review_eligible_ideas(caplog: pytest.LogCaptureFixture) -> None:
    class Provider:
        def discover_result(self, row: dict[str, Any], playbook: Any) -> DiscoveryOutcome:
            return DiscoveryOutcome("completed", text=final_text(), sources=[SOURCE])

    client = CriticClient("review_unavailable")
    agent = NicheAgent(
        FrozenTools(snapshot()),
        client=client,
        tool_agent=Provider(),
        market_data_fn=lambda tickers: copy.deepcopy(MARKET),
        critic=AdversarialCritic(client=client, enabled=True),
    )
    with caplog.at_level("INFO", logger="src.events.niche_agent"):
        report = agent.run_report({"id": 1, "headline": "pump orders"})
    assert len(report.candidates) == 1
    assert report.candidates[0].evidence_status == "source_backed"
    assert not report.eligible
    assert "1 gated, 0 surfaced after red-team" in caplog.text


@pytest.mark.parametrize("raw", ["{}", "not JSON", '{"niche_ideas": [null]}'])
def test_shadow_invalid_output_is_not_abstention(raw: str) -> None:
    result = discover_captured(snapshot(), ScriptedModel([raw]), ResearchBudget())
    assert result.status == "invalid_output"


def test_duplicate_critic_verdict_cannot_choose_favorable_last_answer() -> None:
    class DuplicateClient:
        def complete(self, **kwargs: Any) -> SimpleNamespace:
            verdict = {
                "ticker": "ACME",
                "verdict": "supported",
                "citations": [{"source_id": SOURCE.source_id, "passage": PASSAGE}],
            }
            return SimpleNamespace(text=json.dumps({"verdicts": [verdict, verdict]}))

    candidate = evidence_score(idea(), MARKET, NOW)
    AdversarialCritic(client=DuplicateClient(), enabled=True).apply([candidate], {}, as_of=NOW)
    assert candidate.review_status == "review_unavailable"
    assert not candidate.research_eligible


def test_partial_generation_cannot_be_success_even_with_valid_json() -> None:
    class Truncated(ScriptedModel):
        def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
            return ModelReply('{"niche_ideas": []}', self.model, 20, 10, "partial")

    result = discover_captured(snapshot(), Truncated([]), ResearchBudget())
    assert result.status == "partial" and result.reason == "incomplete_generation"


def test_malformed_extra_claim_cannot_disappear_from_evidence_gate() -> None:
    raw = json.loads(final_text())
    raw["niche_ideas"][0]["claims"].append({"statement": "invented customer"})
    candidate = parse_niche_ideas(json.dumps(raw))[0]
    candidate.sources = [SOURCE]
    evidence_score(candidate, MARKET, NOW)
    assert candidate.claims_parse_error
    assert candidate.evidence_status == "insufficient_evidence"


def test_provider_overrun_and_substitution_are_not_comparable() -> None:
    class Overrun(ScriptedModel):
        def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
            return ModelReply('{"niche_ideas": []}', self.model, 1, max_tokens + 1)

    trial = compare_captured(snapshot(), [Overrun([])])["trials"][0]
    assert trial["discovery"]["status"] == "budget_exhausted" and not trial["comparable"]

    class Substituted(ScriptedModel):
        def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
            return ModelReply('{"niche_ideas": []}', "unexpected-model", 1, 1)

    trial = compare_captured(snapshot(), [Substituted([])])["trials"][0]
    assert trial["discovery"]["model"] == "unexpected-model" and not trial["comparable"]


def test_claude_adapter_uses_explicit_cap_and_preserves_stop_reason() -> None:
    from src.events.niche_shadow import ClaudeResearchModel

    calls = []

    class Client:
        def complete(self, *args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                text="{}",
                model="test-model",
                input_tokens=1,
                output_tokens=2,
                raw=SimpleNamespace(
                    stop_reason="max_tokens", usage=SimpleNamespace(input_tokens=1, output_tokens=2)
                ),
            )

    # Inject only the transport; exercise the real adapter without auth/network.
    adapter = object.__new__(ClaudeResearchModel)
    adapter.model, adapter.client = "test-model", Client()
    reply = adapter.complete([{"role": "user", "content": "research"}], max_tokens=123)
    assert calls[0]["max_tokens"] == 123 and calls[0]["no_tools"] is True
    assert reply.finish_status == "partial"
