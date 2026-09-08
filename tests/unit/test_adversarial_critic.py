"""Tests for the adversarial red-team critic (CL-3v56).

Mocked LLM only. Covers verdict parsing, the apply() filter (refuted dropped;
weakened annotated + confidence lowered; confirmed kept; no-verdict survives),
and fail-open on transport / garbage.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.events.adversarial_critic import AdversarialCritic
from src.events.niche_agent import NicheIdea


class MockLLMClient:
    def __init__(self, text_out: str, raises: bool = False) -> None:
        self.text_out = text_out
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"messages": messages, "model": model})
        if self.raises:
            raise RuntimeError("critic transport down")
        return SimpleNamespace(
            text=self.text_out,
            model=model,
            provider="mock",
            input_tokens=10,
            output_tokens=10,
            usd_cost=0.0,
            elapsed_sec=0.01,
        )


def _idea(ticker: str, conf: float = 0.7) -> NicheIdea:
    return NicheIdea(
        ticker=ticker,
        company_name=f"{ticker} Corp",
        action="long",
        direction="bullish",
        hop_count=3,
        torque_reason="junior",
        rationale="hop chain",
        confidence=conf,
        verified=True,
    )


def _verdicts(*vs: dict[str, Any]) -> str:
    return json.dumps({"verdicts": list(vs)})


def _event() -> dict[str, Any]:
    return {"id": 1, "headline": "rare-earth ban", "theme": "sanctions_trade"}


# --------------------------------------------------------------------------- #
# critique parse
# --------------------------------------------------------------------------- #


def test_critique_parses_verdicts():
    body = _verdicts(
        {
            "ticker": "AAA",
            "verdict": "confirmed",
            "strongest_attack": "none",
            "adjusted_confidence": 0.7,
        },
        {
            "ticker": "BBB",
            "verdict": "refuted",
            "strongest_attack": "already up 30%",
            "adjusted_confidence": 0.1,
        },
    )
    critic = AdversarialCritic(client=MockLLMClient(body), enabled=True)
    out = critic.critique([_idea("AAA"), _idea("BBB")], _event())
    assert out["AAA"].survives is True
    assert out["BBB"].survives is True  # An unvalidated raw refutation is not an elimination.
    assert out["BBB"].strongest_attack == "already up 30%"


def test_critique_drops_bad_verdict_values():
    body = _verdicts({"ticker": "AAA", "verdict": "maybe"})  # invalid verdict
    critic = AdversarialCritic(client=MockLLMClient(body), enabled=True)
    assert critic.critique([_idea("AAA")], _event()) == {}


def test_critique_empty_ideas_no_call():
    client = MockLLMClient(_verdicts())
    critic = AdversarialCritic(client=client, enabled=True)
    assert critic.critique([], _event()) == {}
    assert client.calls == []


def test_critique_fail_open_on_transport():
    critic = AdversarialCritic(client=MockLLMClient("", raises=True), enabled=True)
    assert critic.critique([_idea("AAA")], _event()) == {}


def test_critique_fail_open_on_garbage():
    critic = AdversarialCritic(client=MockLLMClient("not json"), enabled=True)
    assert critic.critique([_idea("AAA")], _event()) == {}


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #


def test_apply_drops_refuted_keeps_others():
    body = _verdicts(
        {"ticker": "AAA", "verdict": "confirmed", "strongest_attack": "weak bear case"},
        {"ticker": "BBB", "verdict": "refuted", "strongest_attack": "priced in"},
    )
    critic = AdversarialCritic(client=MockLLMClient(body), enabled=True)
    survivors = critic.apply([_idea("AAA"), _idea("BBB")], _event())
    assert [i.ticker for i in survivors] == ["AAA", "BBB"]
    # Changed requirement: unsupported praise OR objections remain research-only.
    assert all(i.review_status == "insufficient_evidence" for i in survivors)
    assert all(not i.red_team_verdict and not i.research_eligible for i in survivors)


def test_apply_weakened_annotates_and_lowers_confidence():
    body = _verdicts(
        {
            "ticker": "AAA",
            "verdict": "weakened",
            "strongest_attack": "thin borrow",
            "adjusted_confidence": 0.4,
        }
    )
    critic = AdversarialCritic(client=MockLLMClient(body), enabled=True)
    idea = _idea("AAA", conf=0.8)
    survivors = critic.apply([idea], _event())
    assert survivors[0].red_team_note == "thin borrow"
    assert survivors[0].confidence == 0.4  # lowered from 0.8
    # And it surfaces in the operator-visible notes.
    assert "red-team" in survivors[0].to_trade_idea()["notes"]


def test_apply_confidence_never_raised():
    body = _verdicts(
        {
            "ticker": "AAA",
            "verdict": "confirmed",
            "adjusted_confidence": 0.99,
        }
    )
    critic = AdversarialCritic(client=MockLLMClient(body), enabled=True)
    idea = _idea("AAA", conf=0.5)
    survivors = critic.apply([idea], _event())
    assert survivors[0].confidence == 0.5  # min() — never inflated


def test_apply_no_verdict_survives_untouched():
    # Critic only ruled on AAA; BBB has no verdict → survives (fail-open).
    body = _verdicts({"ticker": "AAA", "verdict": "refuted"})
    critic = AdversarialCritic(client=MockLLMClient(body), enabled=True)
    survivors = critic.apply([_idea("AAA"), _idea("BBB")], _event())
    assert [i.ticker for i in survivors] == ["AAA", "BBB"]
    assert survivors[0].review_status == "insufficient_evidence"
    assert survivors[1].review_status == "review_unavailable"


def test_apply_fail_open_keeps_all():
    critic = AdversarialCritic(client=MockLLMClient("", raises=True), enabled=True)
    ideas = [_idea("AAA"), _idea("BBB")]
    assert critic.apply(ideas, _event()) is ideas  # unchanged list on failure
    assert all(i.review_status == "review_unavailable" for i in ideas)
    assert all(not i.red_team_verdict and not i.research_eligible for i in ideas)
