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
from scripts import niche_hopgraph_shadow as shadow

from src.events.hop_graph_shadow import captured_retriever
from src.events.niche_shadow import CapturedInput, ModelReply
from tests.unit._hop_graph_fixtures import (
    ACME_10K,
    REL_PASSAGE,
    FakeTraversal,
    SupportiveCritic,
    capture_payload,
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


def _clients(baseline: Baseline | None = None) -> shadow.ShadowClients:
    return shadow.ShadowClients(
        traversal=FakeTraversal(world_edges()),
        entailment=world_entailment(),
        critic=SupportiveCritic(),
        baseline=baseline,
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
            "claude",
            "--baseline-model",
            "scripted-v1",
            "--memory-sqlite",
            str(tmp_path / "memory.sqlite"),
            "--traversal-model",
            "trav-x",
            "--allow-model-calls",
        ],
        clients=_clients(baseline),
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["mode"] == "shadow_only" and report["baseline_run"] is True
    [event] = report["events"]
    hop = event["hop_graph"]
    assert hop["traversal_model"] == "trav-x"
    assert hop["entailment_model"] == "claude-haiku-4-5-20251001"
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
