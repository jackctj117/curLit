"""Tests for the agentic Kimi tool-loop (CL-ddzt).

Covers tool dispatch against a fake universe / research tools, the
call → tool → call loop via an injected create_fn (no live Kimi API), and
the fail-soft branches (no key, create error, iteration budget exhausted).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
            return [{"symbol": "MP", "security_name": "MP Materials",
                     "exchange": "NYSE"}]
        return []

    def get_cik(self, t: str) -> int | None:
        return {"MP": 1801368}.get((t or "").upper())


class _FakeTools:
    def __init__(self) -> None:
        self._profile_fn = lambda t: {"sector": "Materials", "industry": "Mining",
                                      "summary": "Rare-earth producer."}

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
        universe=_FakeUniverse(), tools=_FakeTools(), api_key="sk-test",
        create_fn=create_fn, **kw,
    )


# --------------------------------------------------------------------------- #
# tool dispatch
# --------------------------------------------------------------------------- #


def test_dispatch_check_ticker():
    a = _agent(create_fn=lambda **k: _resp("{}"))
    assert a._dispatch("check_ticker", {"ticker": "MP"}) == {
        "ticker": "MP", "exists": True, "exchange": "NYSE",
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
    final = '{"niche_ideas": [{"ticker": "MP", "company_name": "MP Materials", ' \
            '"action": "long", "hop_count": 3, "torque_reason": "single-asset", ' \
            '"rationale": "chain", "confidence": 0.6}]}'
    create = _FakeCreate([
        _resp(tool_calls=[_tc("c1", "get_sec_filing", '{"ticker": "MP"}')]),
        _resp(content=final),
    ])
    out = _agent(create).discover({"id": 1, "headline": "rare-earth ban"})
    assert "niche_ideas" in out
    assert create.calls and len(create.calls) == 2
    # The 2nd call carried the tool result back to the model.
    second_msgs = create.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "Acme Corp" in m.get("content", "")
               for m in second_msgs)


def test_loop_no_toolcalls_returns_immediately():
    create = _FakeCreate([_resp(content='{"niche_ideas": []}')])
    out = _agent(create).discover({"id": 1, "headline": "x"})
    assert out == '{"niche_ideas": []}'
    assert len(create.calls) == 1


def test_no_key_returns_empty():
    a = KimiToolAgent(universe=_FakeUniverse(), tools=_FakeTools(),
                      api_key="", create_fn=lambda **k: _resp("{}"))
    assert a.discover({"id": 1, "headline": "x"}) == ""
    assert a.configured is False


def test_create_failure_is_fail_soft():
    def boom(**kwargs):
        raise RuntimeError("kimi 500")
    assert _agent(boom).discover({"id": 1, "headline": "x"}) == ""


def test_max_iterations_exhausted_returns_empty():
    # Always asks for a tool → never finishes → budget exhausted → "".
    always_tool = _FakeCreate([
        _resp(tool_calls=[_tc("c1", "check_ticker", '{"ticker": "MP"}')]),
    ])
    out = _agent(always_tool, max_iterations=3).discover({"id": 1, "headline": "x"})
    assert out == ""
    assert len(always_tool.calls) == 3  # capped at the budget


def test_bad_tool_arguments_are_tolerated():
    create = _FakeCreate([
        _resp(tool_calls=[_tc("c1", "check_ticker", "not-json")]),
        _resp(content='{"niche_ideas": []}'),
    ])
    out = _agent(create).discover({"id": 1, "headline": "x"})
    assert out == '{"niche_ideas": []}'  # bad args → {} → tool runs, loop continues
