"""End-to-end hop-graph shadow replay on a frozen capture (CL-ynuh).

Runs scripts/niche_hopgraph_shadow.py with injected fakes only: no provider,
network, broker or production database. The oracle is the fixture world
(ACME is the one fully evidenced 2-hop candidate; FRO lacks exposure and
catalyst facts) and the baseline's scripted output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scripts import niche_hopgraph_shadow as shadow

from src.events.hop_graph_shadow import (
    _comparison_coverage,
    captured_retriever,
    compare_events,
    validate_capture_cohort,
)
from src.events.niche_shadow import CapturedInput, ModelReply
from src.events.subscription_shadow_budget import SUBSCRIPTION_MODEL, SubscriptionResearchModel
from tests.unit._hop_graph_fixtures import (
    ACME_10K,
    REL_PASSAGE,
    FakeTraversal,
    SupportiveCritic,
    capture_payload,
    reply,
    world_edges,
    world_entailment,
)


class Baseline:
    """Scripted equivalent-tool baseline: abstains (valid empty output)."""

    provider = "scripted"
    model = "scripted-v1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
        self.calls += 1
        return ModelReply('{"niche_ideas": []}', self.model, 20, 5)


def _write_capture(tmp_path: Path) -> Path:
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "event7.json").write_text(json.dumps(capture_payload()))
    return captures


class StrictFixtureClient:
    """Assert transport policy, then adapt the existing synthetic world fixture."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    @property
    def calls(self) -> Any:
        return self.inner.calls

    def complete(self, **kwargs: Any) -> Any:
        assert kwargs.pop("subscription_only") is True
        assert kwargs["no_tools"] is True
        assert kwargs["model"] == SUBSCRIPTION_MODEL
        kwargs.pop("json_schema", None)
        result = self.inner.complete(**kwargs)
        return reply(result.text, SUBSCRIPTION_MODEL)


class SubscriptionBaselineFixture:
    def __init__(self, baseline: Baseline) -> None:
        self.baseline = baseline

    def complete(self, **kwargs: Any) -> Any:
        assert kwargs["subscription_only"] is True and kwargs["no_tools"] is True
        self.baseline.calls += 1
        return reply('{"niche_ideas": []}', SUBSCRIPTION_MODEL)


def _clients(baseline: Baseline | None = None) -> shadow.ShadowClients:
    return shadow.ShadowClients(
        traversal=StrictFixtureClient(FakeTraversal(world_edges())),
        entailment=StrictFixtureClient(world_entailment()),
        critic=StrictFixtureClient(SupportiveCritic()),
        baseline=SubscriptionResearchModel(SubscriptionBaselineFixture(baseline))
        if baseline
        else None,
    )


def test_shadow_script_runs_end_to_end_and_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "report" / "hopgraph.json"
    baseline = Baseline()
    code = shadow.main(
        [
            "--captures",
            str(_write_capture(tmp_path)),
            "--output",
            str(out),
            "--baseline",
            "claude-subscription",
            "--memory-sqlite",
            str(tmp_path / "memory.sqlite"),
            "--max-model-calls-per-event",
            "32",  # Synthetic complete graph plus baseline, not a deployment default.
            "--allow-model-calls",
        ],
        clients=_clients(baseline),
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["evidence_coverage"]["status"] == "exploratory"
    assert report["evidence_coverage"]["completed_pairs"] == 1
    assert report["mode"] == "shadow_only" and report["baseline_run"] is True
    [event] = report["events"]
    hop = event["hop_graph"]
    assert hop["traversal_model"] == SUBSCRIPTION_MODEL
    assert hop["entailment_model"] == SUBSCRIPTION_MODEL
    assert event["subscription_budget"]["stop_reason"] == ""
    assert set(event["subscription_budget"]["calls_by_role"]) == {
        "traversal",
        "entailment",
        "critic",
        "baseline",
    }
    metrics = hop["metrics"]
    assert metrics["candidates"] == 2  # FRO (hop 1) and ACME (hop 2)
    assert metrics["evidence_gate_passes"] == 1
    assert metrics["eligible"] == 1
    assert metrics["hop_depth_candidates"] == {"1": 1, "2": 1}
    assert metrics["hop_depth_eligible"] == {"2": 1}
    assert metrics["traversal_calls"] == 3  # one per hop; hop 3 frontier = ACME
    assert metrics["tool_calls"] > 0 and metrics["entailment_calls"] > 0
    # Every fake reply reports 10 input / 5 output tokens: traversal + entailment.
    calls = metrics["traversal_calls"] + metrics["entailment_calls"]
    assert hop["cost"]["input_tokens"] == 10 * calls
    assert hop["cost"]["output_tokens"] == 5 * calls
    assert hop["cost"]["usd_cost"] is None  # subscription: unknown, never $0
    assert hop["cost"]["cost_provenance"] == ["subscription_unmetered"]
    eligible = [c for c in hop["candidates"] if c["research"]["eligible"]]
    assert [c["ticker"] for c in eligible] == ["ACME"]
    claims = eligible[0]["research"]["claims"]
    assert {"relationship", "exposure", "catalyst"} <= {c["role"] for c in claims}
    assert any(c["passage"] == REL_PASSAGE for c in claims)
    base = event["baseline"]
    assert baseline.calls == 1 and base["discovery_status"] == "abstained"
    assert base["metrics"]["eligible"] == 0 and base["metrics"]["model_calls"] == 1
    summary = report["summary"]
    assert summary["hop_graph"]["eligible_per_event"] == 1.0
    assert summary["baseline"]["eligible_per_event"] == 0.0
    assert summary["hop_graph"]["gate_pass_rate"] == 0.5


def test_shadow_script_refuses_model_calls_without_flag(tmp_path: Path) -> None:
    clients = _clients()
    with pytest.raises(SystemExit) as exc:
        shadow.main(
            ["--captures", str(_write_capture(tmp_path)), "--output", str(tmp_path / "r.json")],
            clients=clients,
        )
    assert exc.value.code == 2
    assert clients.traversal.calls == [] and clients.entailment.calls == []
    assert not (tmp_path / "r.json").exists()


@pytest.mark.parametrize(
    "override",
    [
        ["--baseline", "moonshot"],
        ["--baseline", "claude"],
        ["--traversal-model", "other"],
        ["--entailment-model", "other"],
        ["--critic-model", "other"],
        ["--baseline-model", "other"],
        ["--max-model-calls-per-event", "0"],
    ],
)
def test_shadow_cli_rejects_paid_or_unapproved_policy_before_calls(
    tmp_path: Path, override: list[str]
) -> None:
    clients = _clients()
    with pytest.raises(SystemExit):
        shadow.main(
            [
                "--captures",
                str(_write_capture(tmp_path)),
                "--output",
                str(tmp_path / "r.json"),
                "--allow-model-calls",
                *override,
            ],
            clients=clients,
        )
    assert clients.traversal.calls == [] and clients.entailment.calls == []
    assert clients.critic.calls == 0


def test_default_shared_budget_is_explicitly_incomplete(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    shadow.main(
        ["--captures", str(_write_capture(tmp_path)), "--output", str(out), "--allow-model-calls"],
        clients=_clients(),
    )
    report = json.loads(out.read_text())
    [event] = report["events"]
    assert event["subscription_budget"]["calls"] == 3
    assert event["subscription_budget"]["limit"] == 3
    arm = event["hop_graph"]
    assert arm["discovery_status"] == "budget_exhausted"
    assert arm["graph"]["outcome"]["status"] == "budget_exhausted"
    assert arm["metrics"]["eligible"] == 0
    assert not any(c["research"]["eligible"] for c in arm["candidates"])
    assert report["evidence_coverage"]["status"] == "exploratory"


def test_late_baseline_failure_revokes_earlier_graph_approval(tmp_path: Path) -> None:
    class FailedBaseline:
        def complete(self, **kwargs: Any) -> Any:
            raise RuntimeError("synthetic transport unavailable")

    clients = _clients()
    clients.baseline = SubscriptionResearchModel(FailedBaseline())
    out = tmp_path / "r.json"
    shadow.main(
        [
            "--captures",
            str(_write_capture(tmp_path)),
            "--output",
            str(out),
            "--allow-model-calls",
            "--baseline",
            "claude-subscription",
            "--max-model-calls-per-event",
            "32",
        ],
        clients=clients,
    )
    [event] = json.loads(out.read_text())["events"]
    graph = event["hop_graph"]
    # The independent synthetic world supports ACME, but the paired experiment
    # cannot retain approval after its shared event transport fails.
    assert graph["metrics"]["evidence_gate_passes"] == 1
    assert graph["graph"]["component_outcome_before_shared_budget"]["status"] == "completed"
    assert graph["graph"]["paths"]
    for arm in (graph, event["baseline"]):
        assert arm["discovery_status"] == "unavailable"
        assert arm["metrics"]["eligible"] == 0
        assert arm["comparable"] is False
        assert not any(c["research"]["eligible"] for c in arm["candidates"])


def test_shadow_script_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        shadow.main(
            [
                "--captures",
                str(_write_capture(tmp_path)),
                "--output",
                str(out),
                "--allow-model-calls",
            ],
            clients=_clients(),
        )
    assert out.read_text() == "{}"


def test_captured_retriever_has_no_live_fallback() -> None:
    capture = CapturedInput.capture(capture_payload())
    retrieve = captured_retriever(capture)
    assert ACME_10K in retrieve("ACME", "revenue coatings")
    assert retrieve("ZZZZ", "anything") == []


def test_shadow_memory_carries_edges_across_events(tmp_path: Path) -> None:
    captures = _write_capture(tmp_path)
    second: dict[str, Any] = capture_payload()
    second["captured_at"] = "2026-09-09T18:00:00+00:00"
    second["event"] = {**second["event"], "id": 8}
    second["event"]["headline"] = "New follow-up disruption event, not a copied observation"
    (captures / "event8.json").write_text(json.dumps(second))
    out = tmp_path / "r.json"
    shadow.main(
        [
            "--captures",
            str(captures),
            "--output",
            str(out),
            "--memory-sqlite",
            str(tmp_path / "m.sqlite"),
            "--allow-model-calls",
            "--max-model-calls-per-event",
            "32",  # Explicit synthetic fixture allocation.
        ],
        clients=_clients(),
    )
    events = json.loads(out.read_text())["events"]
    assert [e["event_id"] for e in events] == [7, 8]
    assert events[0]["hop_graph"]["metrics"]["edges_from_memory"] == 0
    assert events[1]["hop_graph"]["metrics"]["edges_from_memory"] == 2
    assert events[1]["hop_graph"]["metrics"]["eligible"] == 1
    assert (
        events[1]["hop_graph"]["metrics"]["tool_calls"]
        < events[0]["hop_graph"]["metrics"]["tool_calls"]
    )


def test_failed_traversal_call_makes_usage_unknown() -> None:
    from src.events.hop_graph_shadow import run_hopgraph_arm

    class FailSecond(FakeTraversal):
        def complete(self, **kwargs: Any) -> Any:
            if self.calls:
                raise TimeoutError
            return super().complete(**kwargs)

    arm = run_hopgraph_arm(
        CapturedInput.capture(capture_payload()),
        traversal_client=FailSecond(world_edges()),
        entailment_client=world_entailment(),
    )
    assert arm["discovery_status"] == "partial"
    assert arm["metrics"]["traversal_calls"] == 2
    assert arm["cost"]["input_tokens"] is None and arm["cost"]["output_tokens"] is None
    assert arm["graph"]["outcome"]["input_tokens"] is None


def test_failed_baseline_call_counts_as_an_attempt_with_unknown_usage() -> None:
    from src.events.hop_graph_shadow import run_baseline_arm

    class Down(Baseline):
        def complete(self, messages: list[dict[str, str]], max_tokens: int) -> ModelReply:
            self.calls += 1
            raise TimeoutError

    arm = run_baseline_arm(CapturedInput.capture(capture_payload()), Down())
    assert arm["discovery_status"] == "unavailable"
    assert arm["metrics"]["model_calls"] == 1 and arm["metrics"]["failed_model_calls"] == 1
    assert arm["cost"]["input_tokens"] is None and arm["cost"]["output_tokens"] is None
    ok = run_baseline_arm(CapturedInput.capture(capture_payload()), Baseline())
    assert ok["metrics"]["model_calls"] == 1 and ok["cost"]["input_tokens"] == 20


@pytest.mark.parametrize("alias", [7, "7", "007", " 7 ", "+7"])
def test_duplicate_event_ids_fail_before_any_model_call(alias: Any) -> None:
    first = CapturedInput.capture(capture_payload())
    second = capture_payload()
    second["event"]["id"] = alias
    second["event"]["headline"] = "Changed content does not change the event identity"
    baseline = Baseline()
    clients = _clients(baseline)
    with pytest.raises(ValueError, match="duplicate event identity"):
        compare_events(
            [first, CapturedInput.capture(second)],
            traversal_client=clients.traversal,
            entailment_client=clients.entailment,
            baseline_model=clients.baseline,
            critic=None,
        )
    assert clients.traversal.calls == []
    assert clients.entailment.calls == []
    assert baseline.calls == 0


def test_copied_event_with_new_id_is_not_a_distinct_observation() -> None:
    first = capture_payload()
    second = capture_payload()
    second["event"]["id"] = 8
    with pytest.raises(ValueError, match="duplicate event content"):
        validate_capture_cohort([CapturedInput.capture(first), CapturedInput.capture(second)])


@pytest.mark.parametrize("event_id", [None, "", "  ", True, 0, -1, 7.5, "-1", "+0"])
def test_missing_or_malformed_event_identity_is_not_counted(event_id: Any) -> None:
    payload = capture_payload()
    payload["event"]["id"] = event_id
    with pytest.raises(ValueError, match="event identity"):
        validate_capture_cohort([CapturedInput.capture(payload)])


@pytest.mark.parametrize("event", [{}, {"id": 7}])
def test_empty_event_record_cannot_count_toward_coverage(event: dict[str, Any]) -> None:
    payload = capture_payload()
    payload["event"] = event
    with pytest.raises(ValueError, match="capture needs"):
        validate_capture_cohort([CapturedInput.capture(payload)])


def test_opaque_event_identity_is_preserved() -> None:
    payload = capture_payload()
    payload["event"]["id"] = "7f847321-d76f-4d1f-b38b-572f8acc6424"
    manifest = validate_capture_cohort([CapturedInput.capture(payload)])
    assert manifest["event_ids"] == [payload["event"]["id"]]


def test_loader_rejects_same_file_supplied_twice(tmp_path: Path) -> None:
    folder = _write_capture(tmp_path)
    with pytest.raises(ValueError, match="duplicate event identity"):
        shadow.load_captures([folder, folder / "event7.json"])


def test_validate_only_never_constructs_clients_or_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("offline validation must not construct model or memory")

    monkeypatch.setattr(shadow, "_live_clients", forbidden)
    monkeypatch.setattr(shadow, "sqlite_memory", forbidden)
    output = tmp_path / "validation.json"
    assert (
        shadow.main(
            [
                "--captures",
                str(_write_capture(tmp_path)),
                "--output",
                str(output),
                "--validate-only",
                "--memory-sqlite",
                str(tmp_path / "must-not-exist.sqlite"),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["model_calls"] == 0 and report["comparison_run"] is False
    assert report["quality_acceptance"] == "not_evaluated"
    assert report["cohort"]["distinct_event_count"] == 1
    assert report["cohort"]["sample_size_met"] is False
    assert not (tmp_path / "must-not-exist.sqlite").exists()


@given(st.integers(min_value=1, max_value=30))
def test_paired_coverage_boundary_does_not_claim_quality_or_equal_budgets(count: int) -> None:
    rows = [
        {"hop_graph": {"discovery_status": "completed"}, "baseline": {"comparable": True}}
        for _ in range(count)
    ]
    result = _comparison_coverage(rows, critic_present=True)
    assert result["completed_pairs"] == count
    assert result["status"] == (
        "minimum_completed_paired_coverage" if count >= 20 else "exploratory"
    )
    assert result["quality_acceptance"] == "not_determined_by_sample_coverage"
    assert result["budget_equivalence"] == "not_established"


def test_twenty_rows_without_baseline_or_critic_are_exploratory() -> None:
    rows = [{"hop_graph": {"discovery_status": "completed"}, "baseline": None}] * 20
    result = _comparison_coverage(rows, critic_present=False)
    assert result["status"] == "exploratory"
    assert result["completed_pairs"] == 0
    assert "missing_baseline_arm" in result["reasons"]
    assert "critic_not_run" in result["reasons"]


def test_incomplete_discovery_does_not_count_as_completed_pair() -> None:
    rows = [
        {"hop_graph": {"discovery_status": "completed"}, "baseline": {"comparable": True}}
        for _ in range(20)
    ]
    rows[-1]["hop_graph"]["discovery_status"] = "partial"
    result = _comparison_coverage(rows, critic_present=True)
    assert result["status"] == "exploratory"
    assert result["completed_pairs"] == 19
    assert result["incomplete_pairs"] == 1
