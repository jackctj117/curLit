"""Tests for the agentic Kimi tool-loop (CL-ddzt).

Covers tool dispatch against a fake universe / research tools, the
call → tool → call loop via an injected create_fn (no live Kimi API), and
the fail-soft branches (no key, create error, iteration budget exhausted).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from src.events.kimi_tool_agent import KimiToolAgent


class _FakeUniverse:
    def exists(self, t: str) -> bool:
        return (t or "").upper() in {"MP", "FRO"}

    def get(self, t: str) -> dict[str, Any] | None:
        return {"exchange": "NYSE"} if self.exists(t) else None

    def robinhood_tradeable(self, t: str) -> bool:
        return self.exists(t)

    def resolve_name(self, q: str, limit: int = 5) -> list[dict[str, Any]]:
        if "materials" in (q or "").lower():
            return [{"symbol": "MP", "security_name": "MP Materials", "exchange": "NYSE"}]
        return []

    def get_cik(self, t: str) -> int | None:
        return {"MP": 1801368}.get((t or "").upper())


class _FakeTools:
    def __init__(self) -> None:
        self._profile_fn = lambda t: {
            "sector": "Materials",
            "industry": "Mining",
            "summary": "Rare-earth producer.",
        }

    def sec_excerpt(self, cik: int) -> str | None:
        return f"[10-K] cik {cik}: depends on a single customer, Acme Corp."


def _tc(cid: str, name: str, args: str) -> SimpleNamespace:
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=args))


def _resp(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class _FakeCreate:
    """Returns a scripted sequence of responses; records the calls."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        idx = min(len(self.calls), len(self.responses) - 1)
        self.calls.append(kwargs)
        return self.responses[idx]


def _agent(create_fn: Any, **kw: Any) -> KimiToolAgent:
    return KimiToolAgent(
        universe=_FakeUniverse(),
        tools=_FakeTools(),
        api_key="sk-test",
        create_fn=create_fn,
        **kw,
    )


# --------------------------------------------------------------------------- #
# tool dispatch
# --------------------------------------------------------------------------- #


def test_dispatch_check_ticker():
    a = _agent(create_fn=lambda **k: _resp("{}"))
    assert a._dispatch("check_ticker", {"ticker": "MP"}) == {
        "ticker": "MP",
        "exists": True,
        "exchange": "NYSE",
        "robinhood_tradeable": True,
    }
    miss = a._dispatch("check_ticker", {"ticker": "ZZZZ"})
    assert miss["exists"] is False


def test_dispatch_resolve_company():
    a = _agent(create_fn=lambda **k: _resp("{}"))
    out = a._dispatch("resolve_company", {"company_name": "MP Materials"})
    assert out["matches"][0]["symbol"] == "MP"


def test_dispatch_get_sec_filing():
    a = _agent(create_fn=lambda **k: _resp("{}"))
    out = a._dispatch("get_sec_filing", {"ticker": "MP"})
    assert out["cik"] == 1801368
    assert "Acme Corp" in out["excerpt"]


def test_dispatch_unknown_tool():
    a = _agent(create_fn=lambda **k: _resp("{}"))
    assert "error" in a._dispatch("nope", {})


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #


def test_loop_runs_tool_then_returns_json():
    final = (
        '{"niche_ideas": [{"ticker": "MP", "company_name": "MP Materials", '
        '"action": "long", "hop_count": 3, "torque_reason": "single-asset", '
        '"rationale": "chain", "confidence": 0.6}]}'
    )
    create = _FakeCreate(
        [
            _resp(tool_calls=[_tc("c1", "get_sec_filing", '{"ticker": "MP"}')]),
            _resp(content=final),
        ]
    )
    out = _agent(create).discover({"id": 1, "headline": "rare-earth ban"})
    assert "niche_ideas" in out
    assert create.calls and len(create.calls) == 2
    # The 2nd call carried the tool result back to the model.
    second_msgs = create.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "Acme Corp" in m.get("content", "") for m in second_msgs)


def test_loop_no_toolcalls_returns_immediately():
    create = _FakeCreate([_resp(content='{"niche_ideas": []}')])
    out = _agent(create).discover({"id": 1, "headline": "x"})
    assert out == '{"niche_ideas": []}'
    assert len(create.calls) == 1


def test_no_key_returns_empty():
    a = KimiToolAgent(
        universe=_FakeUniverse(), tools=_FakeTools(), api_key="", create_fn=lambda **k: _resp("{}")
    )
    assert a.discover({"id": 1, "headline": "x"}) == ""
    assert a.configured is False


def test_create_failure_is_fail_soft():
    def boom(**kwargs):
        raise RuntimeError("kimi 500")

    assert _agent(boom).discover({"id": 1, "headline": "x"}) == ""


def test_max_iterations_exhausted_returns_empty():
    # Always asks for a tool → never finishes → budget exhausted → "".
    always_tool = _FakeCreate(
        [
            _resp(tool_calls=[_tc("c1", "check_ticker", '{"ticker": "MP"}')]),
        ]
    )
    out = _agent(always_tool, max_iterations=3).discover({"id": 1, "headline": "x"})
    assert out == ""
    assert len(always_tool.calls) == 3  # capped at the budget


def test_bad_tool_arguments_are_tolerated():
    create = _FakeCreate(
        [
            _resp(tool_calls=[_tc("c1", "check_ticker", "not-json")]),
            _resp(content='{"niche_ideas": []}'),
        ]
    )
    out = _agent(create).discover({"id": 1, "headline": "x"})
    assert out == '{"niche_ideas": []}'  # bad args → {} → tool runs, loop continues


def test_reasoning_is_preserved_and_k3_effort_is_explicit() -> None:
    first = _resp(tool_calls=[_tc("c1", "check_ticker", '{"ticker": "MP"}')])
    first.choices[0].message.reasoning_content = "fixture reasoning for tool selection"
    create = _FakeCreate([first, _resp('{"niche_ideas": []}')])
    _agent(create, model="kimi-k3").discover_result({"id": 1})
    assistant = next(m for m in create.calls[1]["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == first.choices[0].message.reasoning_content
    assert all(c["reasoning_effort"] == "low" for c in create.calls)


def test_tool_limit_reserves_final_answer_and_answers_every_call() -> None:
    # More requests than the fixed 24-tool ceiling must not discard collected evidence.
    batch = [_tc(str(i), "check_ticker", '{"ticker": "MP"}') for i in range(25)]
    create = _FakeCreate([_resp(tool_calls=batch), _resp('{"niche_ideas": []}')])
    agent = _agent(create)
    dispatched: list[str] = []
    agent._dispatch = lambda name, args: dispatched.append(name) or {"exists": True}
    result = agent.discover_result({"id": 1})
    assert result.status == "abstained"
    assert len(dispatched) <= 24
    assert len(create.calls) == 2
    assert create.calls[1]["tool_choice"] == "none"
    replies = [m for m in create.calls[1]["messages"] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in replies} == {str(i) for i in range(25)}
    assert "tool_call_limit" in replies[-1]["content"]


def test_last_model_call_is_reserved_for_finalization() -> None:
    create = _FakeCreate(
        [
            _resp(tool_calls=[_tc("c1", "check_ticker", '{"ticker": "MP"}')]),
            _resp('{"niche_ideas": []}'),
        ]
    )
    result = _agent(create, max_iterations=2).discover_result({"id": 1})
    assert result.status == "abstained"
    assert create.calls[-1]["tool_choice"] == "none"


def test_truncation_gets_one_bounded_finalization_not_false_approval() -> None:
    truncated = _resp('{"niche_ideas": [')
    truncated.choices[0].finish_reason = "length"
    truncated.usage.completion_tokens = 4096
    create = _FakeCreate([truncated, _resp('{"niche_ideas": []}')])
    result = _agent(create).discover_result({"id": 1})
    assert result.status == "abstained"
    assert len(create.calls) == 2
    assert create.calls[1]["tool_choice"] == "none"
    assert create.calls[1]["max_tokens"] > create.calls[0]["max_tokens"]
    assert sum(c["max_tokens"] for c in create.calls) <= 8 * 4096
    assert result.trace[0]["finish_reason"] == "length"
    assert result.trace[0]["output_tokens"] == 4096

    never_finishes = _FakeCreate([truncated])
    partial = _agent(never_finishes).discover_result({"id": 1})
    assert partial.status == "partial"
    assert len(never_finishes.calls) == 2  # no indefinite retry or invented empty result


def test_truncated_tool_call_is_not_executed() -> None:
    truncated = _resp(tool_calls=[_tc("c1", "check_ticker", '{"ticker": "MP"}')])
    truncated.choices[0].finish_reason = "length"
    create = _FakeCreate([truncated, _resp('{"niche_ideas": []}')])
    agent = _agent(create)
    dispatched: list[str] = []
    agent._dispatch = lambda name, args: dispatched.append(name) or {}
    agent.discover_result({"id": 1})
    assert dispatched == []


@given(st.integers(min_value=1, max_value=12), st.integers(min_value=1, max_value=8192))
def test_finalization_never_increases_requested_token_envelope(
    iterations: int, tokens: int
) -> None:
    create = _FakeCreate([_resp(tool_calls=[_tc("c1", "check_ticker", '{"ticker": "MP"}')])])
    result = _agent(create, max_iterations=iterations, max_tokens=tokens).discover_result({"id": 1})
    assert result.status == "budget_exhausted"
    assert len(create.calls) <= iterations
    assert sum(c["max_tokens"] for c in create.calls) <= iterations * tokens
    assert all(c["max_tokens"] > 0 for c in create.calls)
    assert create.calls[-1]["tool_choice"] == "none"


def test_duplicate_lookups_reuse_result_only_inside_one_invocation() -> None:
    first = _resp(
        tool_calls=[
            _tc("c1", "get_company_profile", '{"ticker": "MP"}'),
            _tc("c2", "get_company_profile", '{"ticker": "MP"}'),
        ]
    )
    create = _FakeCreate([first, _resp('{"niche_ideas": []}'), first, _resp('{"niche_ideas": []}')])
    agent = _agent(create)
    dispatches: list[str] = []
    agent._dispatch = lambda name, args: dispatches.append(name) or {"error": "unavailable"}
    for _ in range(2):
        report = agent.discover_result({"id": 1})
        tool_trace = [t for t in report.trace if "tool" in t]
        assert [t["cached"] for t in tool_trace] == [False, True]
    assert len(dispatches) == 2


def test_other_model_does_not_receive_k3_only_effort() -> None:
    create = _FakeCreate([_resp('{"niche_ideas": []}')])
    _agent(create, model="kimi-k2.6").discover_result({"id": 1})
    assert "reasoning_effort" not in create.calls[0]


def test_provider_failure_after_truncation_stays_unavailable() -> None:
    truncated = _resp("")
    truncated.choices[0].finish_reason = "length"
    calls = 0

    def create(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return truncated
        raise RuntimeError("account suspended due to insufficient balance")

    report = _agent(create).discover_result({"id": 1})
    assert report.status == "unavailable"
    assert report.reason == "insufficient_balance"
