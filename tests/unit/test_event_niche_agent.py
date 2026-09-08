"""Tests for the multi-hop niche/asymmetry opportunity agent (CL-u2ph).

Covers, with a mocked LLM + mocked SymbolUniverse + mocked market data
(no live LLM, no network, no DB):
  * multi-hop JSON parse (fences/prose tolerated, malformed dropped);
  * VERIFICATION — exists→keep, resolve_name→correct+keep, neither→drop;
    a hallucinated ticker is dropped (NO unverified survives);
  * asymmetry scoring — hop/torque/smallness combine; the liquidity
    floor penalizes/hard-drops illiquid names; the threshold gates;
  * quota gate — the niche pass is skipped for urgency < min;
  * the merge into an assessment's trade_ideas (deduped, tagged niche).

The pure pieces (parse/verify/score) live in ``src.events.niche_scoring``
since CL-ikz2 (review §6.2.2) and are imported from there; ``niche_agent``
re-exports them, which ``TestModuleSplit`` pins down.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.events.niche_agent import NicheAgent
from src.events.niche_scoring import (
    AsymmetryConfig,
    NicheIdea,
    asymmetry_score,
    parse_niche_ideas,
    score_and_gate,
    torque_from_reason,
    verify_ideas,
)

# --------------------------------------------------------------------- #
# Mocks / fixtures
# --------------------------------------------------------------------- #


class MockLLMClient:
    """Returns a canned response text; records the calls it saw."""

    def __init__(self, text_out: str) -> None:
        self.text_out = text_out
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"messages": messages, "model": model, "kwargs": kwargs})
        return SimpleNamespace(
            text=self.text_out,
            model=model,
            provider="mock",
            input_tokens=10,
            output_tokens=10,
            usd_cost=0.0,
            elapsed_sec=0.01,
        )


class _RaisingLLMClient:
    def complete(self, messages: Any, model: str, **kwargs: Any) -> Any:
        raise RuntimeError("simulated LLM transport failure")


class MultiResponseClient:
    """Returns successive canned texts, one per call (the last one repeats)."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        idx = min(len(self.calls), len(self.texts) - 1)
        self.calls.append({"messages": messages, "model": model})
        return SimpleNamespace(
            text=self.texts[idx],
            model=model,
            provider="mock",
            input_tokens=10,
            output_tokens=10,
            usd_cost=0.0,
            elapsed_sec=0.01,
        )


@pytest.fixture(autouse=True)
def _default_single_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the niche agent single-pass + tool-free (CL-2dnf / CL-2czc)
    regardless of a leaked NICHE_MAX_CYCLES / NICHE_TOOLS_ENABLED from the
    operator's .env; the iterative/tool tests set them explicitly."""
    monkeypatch.delenv("NICHE_MAX_CYCLES", raising=False)
    monkeypatch.delenv("NICHE_TOOLS_ENABLED", raising=False)
    monkeypatch.delenv("NICHE_TOOL_AGENT_ENABLED", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("NICHE_CRITIC_ENABLED", raising=False)


def _liquid_md(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Market-data stub: every ticker small-cap + liquid (clears the gate)."""
    return {t: {"market_cap": 2e8, "avg_dollar_volume": 5e6} for t in tickers}


class FakeUniverse:
    """Minimal SymbolUniverse stand-in over a fixed symbol set."""

    def __init__(self, symbols: dict[str, dict[str, Any]] | None = None) -> None:
        self._symbols = symbols or {
            "REAL": {
                "symbol": "REAL",
                "security_name": "Real Co Inc",
                "exchange": "NASDAQ",
                "is_etf": False,
            },
            "FRO": {
                "symbol": "FRO",
                "security_name": "Frontline Ltd",
                "exchange": "NYSE",
                "is_etf": False,
            },
        }

    def exists(self, ticker: str) -> bool:
        return bool(ticker) and ticker.strip().upper() in self._symbols

    def get(self, ticker: str) -> dict[str, Any] | None:
        return self._symbols.get((ticker or "").strip().upper())

    def robinhood_tradeable(self, ticker: str) -> bool:
        return self.exists(ticker)

    def resolve_name(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        q = (query or "").strip().lower()
        if not q:
            return []
        out = [
            row
            for row in self._symbols.values()
            if row["security_name"] and q in row["security_name"].lower()
        ]
        return out[:limit]

    def get_cik(self, ticker: str) -> int | None:
        return {"REAL": 111, "FRO": 222}.get((ticker or "").strip().upper())


class FakeTools:
    """Records enrich() calls and returns a canned grounding block."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def enrich(self, ideas: Any, universe: Any) -> list[str]:
        self.calls.append([getattr(i, "ticker", "") for i in ideas])
        return [f"--- grounded {getattr(i, 'ticker', '')} ---" for i in ideas]


class FakeToolAgent:
    """Stand-in for the Kimi tool-loop: returns a canned final JSON payload."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[Any] = []

    def discover(self, event_row: Any, playbook: Any = None) -> str:
        self.calls.append(event_row.get("id"))
        return self.payload


class FakeCritic:
    """Red-team stand-in: drops any idea whose ticker is in ``refute``."""

    enabled = True

    def __init__(self, refute: set[str]) -> None:
        self.refute = {t.upper() for t in refute}
        self.calls: list[Any] = []

    def apply(self, ideas: Any, event_row: Any) -> list[Any]:
        self.calls.append(event_row.get("id"))
        out = []
        for i in ideas:
            if i.ticker.upper() in self.refute:
                continue
            i.red_team_note = "survived"
            out.append(i)
        return out


def _payload(*ideas: dict[str, Any]) -> str:
    return json.dumps({"niche_ideas": list(ideas)})


def _idea_dict(**over: Any) -> dict[str, Any]:
    base = {
        "ticker": "REAL",
        "company_name": "Real Co Inc",
        "action": "buy_calls",
        "direction": "bullish",
        "hop_count": 4,
        "torque_reason": "single-asset junior, high operating leverage",
        "rationale": "hop1 obvious -> hop2 supplier -> REAL",
        "confidence": 0.6,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------- #


class TestParse:
    def test_multi_hop_parse_from_fenced_prose(self) -> None:
        raw = (
            "Sure, here's the analysis:\n```json\n"
            + _payload(_idea_dict(), _idea_dict(ticker="FRO", hop_count=2))
            + "\n```\nHope that helps."
        )
        ideas = parse_niche_ideas(raw)
        assert [i.ticker for i in ideas] == ["REAL", "FRO"]
        assert ideas[0].hop_count == 4
        assert ideas[0].direction == "bullish"

    def test_bad_action_dropped_individually(self) -> None:
        ideas = parse_niche_ideas(
            _payload(
                _idea_dict(action="hodl"),  # invalid → dropped
                _idea_dict(ticker="FRO", action="short"),  # valid
            )
        )
        assert [i.ticker for i in ideas] == ["FRO"]

    def test_missing_ticker_and_company_dropped(self) -> None:
        ideas = parse_niche_ideas(
            _payload(
                _idea_dict(ticker="", company_name=""),
                _idea_dict(ticker="FRO"),
            )
        )
        assert [i.ticker for i in ideas] == ["FRO"]

    def test_direction_inferred_from_action(self) -> None:
        ideas = parse_niche_ideas(
            _payload(
                _idea_dict(action="buy_puts", direction="garbage"),
            )
        )
        assert ideas[0].direction == "bearish"

    def test_no_json_returns_empty(self) -> None:
        assert parse_niche_ideas("I cannot help with that.") == []

    def test_hop_count_clamped(self) -> None:
        ideas = parse_niche_ideas(_payload(_idea_dict(hop_count=99)))
        assert ideas[0].hop_count == 8  # clamp ceiling


# --------------------------------------------------------------------- #
# Verification — the anti-hallucination guard
# --------------------------------------------------------------------- #


class TestVerification:
    def test_existing_ticker_kept_and_tagged(self) -> None:
        ideas = parse_niche_ideas(_payload(_idea_dict(ticker="REAL")))
        surv = verify_ideas(ideas, FakeUniverse())
        assert len(surv) == 1
        assert surv[0].ticker == "REAL"
        assert surv[0].verified and not surv[0].corrected
        assert surv[0].exchange == "NASDAQ"
        assert surv[0].robinhood_tradeable is True

    def test_wrong_ticker_corrected_via_company_name(self) -> None:
        # Ticker "XXXX" doesn't exist, but "Frontline" resolves to FRO.
        ideas = parse_niche_ideas(
            _payload(
                _idea_dict(ticker="XXXX", company_name="Frontline"),
            )
        )
        surv = verify_ideas(ideas, FakeUniverse())
        assert len(surv) == 1
        assert surv[0].ticker == "FRO"
        assert surv[0].corrected is True
        assert surv[0].exchange == "NYSE"

    def test_hallucinated_ticker_dropped(self) -> None:
        # Neither the ticker nor the company resolves → DROPPED.
        ideas = parse_niche_ideas(
            _payload(
                _idea_dict(ticker="ZZZQ", company_name="Fictional Vapor Mining"),
            )
        )
        surv = verify_ideas(ideas, FakeUniverse())
        assert surv == []  # NO unverified ticker survives

    def test_no_unverified_survives_in_mixed_batch(self) -> None:
        ideas = parse_niche_ideas(
            _payload(
                _idea_dict(ticker="REAL"),  # keep
                _idea_dict(ticker="XXXX", company_name="Frontline"),  # correct
                _idea_dict(ticker="ZZZQ", company_name="Fictional Mining"),  # drop
            )
        )
        surv = verify_ideas(ideas, FakeUniverse())
        tickers = {i.ticker for i in surv}
        assert tickers == {"REAL", "FRO"}
        # Every survivor resolves in the universe — nothing unverified.
        u = FakeUniverse()
        assert all(u.exists(i.ticker) for i in surv)


# --------------------------------------------------------------------- #
# Asymmetry + liquidity scoring
# --------------------------------------------------------------------- #


class TestTorque:
    def test_leverage_keywords_score_high(self) -> None:
        assert torque_from_reason("single-asset pure-play miner") == 1.0
        assert torque_from_reason("sole supplier bottleneck") == 1.0

    def test_bland_reason_is_neutral(self) -> None:
        assert torque_from_reason("a diversified large company") == 0.4
        assert torque_from_reason("") == 0.4


class TestAsymmetryScoring:
    def _idea(self, **over: Any) -> NicheIdea:
        d = {
            "ticker": "AAA",
            "company_name": "Alpha",
            "action": "buy_calls",
            "direction": "bullish",
            "hop_count": 4,
            "torque_reason": "single-asset junior, high operating leverage",
            "rationale": "chain",
            "confidence": 0.6,
        }
        d.update(over)
        return NicheIdea(**d)

    def test_liquid_smallcap_high_torque_scores_high(self) -> None:
        i = self._idea()
        asymmetry_score(i, {"AAA": {"market_cap": 200e6, "avg_dollar_volume": 5e6}})
        assert i.asymmetry_score is not None and i.asymmetry_score > 0.8
        assert i.liquidity_flag is False

    def test_hop_count_raises_score(self) -> None:
        near = asymmetry_score(
            self._idea(hop_count=1),
            {"AAA": {"market_cap": 200e6, "avg_dollar_volume": 5e6}},
        ).asymmetry_score
        far = asymmetry_score(
            self._idea(hop_count=5),
            {"AAA": {"market_cap": 200e6, "avg_dollar_volume": 5e6}},
        ).asymmetry_score
        assert far > near

    def test_smaller_cap_raises_score(self) -> None:
        small = asymmetry_score(
            self._idea(),
            {"AAA": {"market_cap": 150e6, "avg_dollar_volume": 5e6}},
        ).asymmetry_score
        big = asymmetry_score(
            self._idea(),
            {"AAA": {"market_cap": 30e9, "avg_dollar_volume": 5e6}},
        ).asymmetry_score
        assert small > big

    def test_illiquid_name_flagged_and_penalized(self) -> None:
        liquid = asymmetry_score(
            self._idea(),
            {"AAA": {"market_cap": 200e6, "avg_dollar_volume": 5e6}},
        ).asymmetry_score
        illiquid = self._idea()
        asymmetry_score(
            illiquid,
            {"AAA": {"market_cap": 200e6, "avg_dollar_volume": 500_000}},
        )
        assert illiquid.liquidity_flag is True
        assert illiquid.asymmetry_score < liquid  # heavily penalized

    def test_thin_shell_hard_dropped(self) -> None:
        i = self._idea()
        asymmetry_score(
            i,
            {"AAA": {"market_cap": 50e6, "avg_dollar_volume": 100_000}},
        )
        assert i.asymmetry_score == 0.0
        assert i.dropped_reason is not None
        assert i.liquidity_flag is True

    def test_missing_market_data_scores_data_free(self) -> None:
        # No market data → hop/torque only, neutral smallness, no flag.
        i = self._idea()
        asymmetry_score(i, {})
        assert i.asymmetry_score is not None
        assert i.liquidity_flag is False

    def test_threshold_gates_surfacing(self) -> None:
        # A low-torque obvious megacap should be logged, not surfaced.
        obvious = self._idea(
            ticker="MMM",
            hop_count=1,
            torque_reason="large diversified",
        )
        niche = self._idea(ticker="AAA")
        surviving, logged = score_and_gate(
            [obvious, niche],
            {
                "MMM": {"market_cap": 50e9, "avg_dollar_volume": 500e6},
                "AAA": {"market_cap": 200e6, "avg_dollar_volume": 5e6},
            },
        )
        assert [i.ticker for i in surviving] == ["AAA"]
        assert "MMM" in {i.ticker for i in logged}

    def test_illiquid_dropped_from_surfacing(self) -> None:
        illiquid = self._idea(ticker="AAA")
        surviving, logged = score_and_gate(
            [illiquid],
            {"AAA": {"market_cap": 200e6, "avg_dollar_volume": 400_000}},
        )
        assert surviving == []
        assert logged and logged[0].ticker == "AAA"

    def test_config_threshold_configurable(self) -> None:
        i = self._idea()
        # A punishing threshold suppresses even a strong idea.
        surviving, logged = score_and_gate(
            [i],
            {"AAA": {"market_cap": 200e6, "avg_dollar_volume": 5e6}},
            AsymmetryConfig(asymmetry_threshold=0.99),
        )
        assert surviving == []
        assert logged


# --------------------------------------------------------------------- #
# Agent.run — full pass wiring (mocked LLM + universe + market data)
# --------------------------------------------------------------------- #


def _mock_market_data(values: dict[str, dict[str, Any]]):
    def _fn(tickers: list[str]) -> dict[str, dict[str, Any]]:
        return {t: values.get(t, {"market_cap": None, "avg_dollar_volume": None}) for t in tickers}

    return _fn


class TestAgentRun:
    def test_run_verifies_scores_and_surfaces(self) -> None:
        client = MockLLMClient(
            _payload(
                _idea_dict(ticker="REAL"),  # verified, liquid
                _idea_dict(ticker="ZZZQ", company_name="Fictional"),  # hallucination
            )
        )
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_mock_market_data(
                {
                    "REAL": {"market_cap": 200e6, "avg_dollar_volume": 5e6},
                }
            ),
        )
        report = agent.run_report(
            {
                "id": 1,
                "headline": "China restricts rare-earth exports",
                "theme": "critical_minerals",
                "assessment": {},
            }
        )
        assert [i.ticker for i in report.candidates if i.verified] == ["REAL"]
        assert report.candidates[1].dropped_reason == "unverified"
        assert report.candidates[0].asymmetry_score == 0  # No sourced claims, not a hop bonus.
        assert report.eligible == []
        assert client.calls  # the LLM was actually called

    def test_run_drops_all_when_all_hallucinated(self) -> None:
        client = MockLLMClient(
            _payload(
                _idea_dict(ticker="ZZZQ", company_name="Fictional Vapor"),
            )
        )
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_mock_market_data({}),
        )
        assert agent.run({"id": 1, "headline": "x", "theme": None, "assessment": {}}) == []

    def test_run_fail_soft_on_transport_error(self) -> None:
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=_RaisingLLMClient(),  # type: ignore[arg-type]
            market_data_fn=_mock_market_data({}),
        )
        assert agent.run({"id": 1, "headline": "x", "theme": None, "assessment": {}}) == []

    def test_run_fail_soft_on_market_data_error(self) -> None:
        def _boom(tickers: list[str]) -> dict[str, Any]:
            raise RuntimeError("yfinance down")

        client = MockLLMClient(_payload(_idea_dict(ticker="REAL")))
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_boom,
        )
        # Changed requirement: retain the lead, never call missing liquidity passed.
        report = agent.run_report({"id": 1, "headline": "x", "theme": None, "assessment": {}})
        assert [i.ticker for i in report.candidates] == ["REAL"]
        assert report.candidates[0].liquidity_status == "unknown"
        assert report.eligible == []


# --------------------------------------------------------------------- #
# Merge into assessment
# --------------------------------------------------------------------- #


class TestMerge:
    def test_merge_tags_niche_and_dedups(self) -> None:
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
        )
        assessment: dict[str, Any] = {
            "trade_ideas": [
                {"ticker": "REAL", "action": "buy_calls"},  # already present
            ],
        }
        i1 = NicheIdea(
            "REAL", "Real Co", "buy_calls", "bullish", 4, "junior", "chain", 0.6
        )  # dup → skipped
        i2 = NicheIdea(
            "FRO", "Frontline", "long", "bullish", 3, "levered tanker", "chain", 0.5
        )  # new → merged
        i2.asymmetry_score = 0.7
        for idea in (i1, i2):
            # Merge-unit precondition; end-to-end evidence checks are covered separately.
            idea.verified, idea.discovery_status = True, "completed"
            idea.evidence_status, idea.review_status = "source_backed", "supported"
            idea.liquidity_status = "sufficient"
        added = agent.merge_into_assessment(assessment, [i1, i2])
        assert added == 1
        merged = assessment["trade_ideas"]
        assert len(merged) == 2
        fro = next(x for x in merged if x["ticker"] == "FRO")
        assert fro["niche"] is True
        assert fro["hop_count"] == 3
        assert fro["torque_reason"] == "levered tanker"
        assert fro["asymmetry_score"] == 0.7

    def test_merge_respects_max_total(self) -> None:
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
        )
        assessment: dict[str, Any] = {"trade_ideas": []}
        ideas = [
            NicheIdea(f"T{i}", f"Co{i}", "long", "bullish", 3, "x", "y", 0.5) for i in range(10)
        ]
        for idea in ideas:
            idea.verified, idea.discovery_status = True, "completed"
            idea.evidence_status, idea.review_status = "source_backed", "supported"
            idea.liquidity_status = "sufficient"
        added = agent.merge_into_assessment(assessment, ideas, max_total=3)
        assert added == 3
        assert len(assessment["trade_ideas"]) == 3


# --------------------------------------------------------------------- #
# Iterative multi-cycle hopping (CL-2dnf)
# --------------------------------------------------------------------- #


class TestIterativeHopping:
    def _event(self) -> dict[str, Any]:
        return {
            "id": 1,
            "headline": "China restricts rare-earth exports",
            "theme": None,
            "assessment": {},
        }

    def test_default_is_single_cycle(self) -> None:
        client = MockLLMClient(_payload(_idea_dict(ticker="REAL")))
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
        )
        assert agent.max_cycles == 1
        agent.run(self._event())
        assert len(client.calls) == 1  # exactly one pass, original behavior
        # CL-scup: text-protocol cycles — the claude-code toolset must be
        # stripped (sanctioned tool use lives in kimi_tool_agent).
        assert client.calls[0]["kwargs"].get("no_tools") is True

    def test_accumulates_new_names_across_cycles(self) -> None:
        c1 = _payload(_idea_dict(ticker="REAL", company_name="Real Co Inc"))
        c2 = _payload(_idea_dict(ticker="FRO", company_name="Frontline Ltd", hop_count=5))
        client = MultiResponseClient([c1, c2])
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            max_cycles=2,
        )
        report = agent.run_report(self._event())
        ideas = report.candidates  # Incomplete leads survive for research, not execution.
        assert report.eligible == []
        assert {i.ticker for i in ideas} == {"REAL", "FRO"}
        assert len(client.calls) == 2  # cycle 1 + a genuine deeper cycle

    def test_early_stop_when_cycle_adds_nothing(self) -> None:
        same = _payload(_idea_dict(ticker="REAL"))
        client = MultiResponseClient([same])  # every call returns REAL
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            max_cycles=3,
        )
        report = agent.run_report(self._event())
        ideas = report.candidates  # Incomplete leads survive for research, not execution.
        assert report.eligible == []
        assert {i.ticker for i in ideas} == {"REAL"}
        # cycle 2 was dry → stopped early (2 calls, not the full 3).
        assert len(client.calls) == 2

    def test_later_cycle_failure_keeps_earlier(self) -> None:
        class _FailSecond:
            def __init__(self, first: str) -> None:
                self.first = first
                self.calls: list[int] = []

            def complete(self, messages: Any, model: str, **kwargs: Any) -> Any:
                self.calls.append(1)
                if len(self.calls) == 1:
                    return SimpleNamespace(
                        text=self.first,
                        model=model,
                        provider="mock",
                        input_tokens=10,
                        output_tokens=10,
                        usd_cost=0.0,
                        elapsed_sec=0.01,
                    )
                raise RuntimeError("cycle 2 transport down")

        client = _FailSecond(_payload(_idea_dict(ticker="REAL")))
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            max_cycles=2,
        )
        report = agent.run_report(self._event())
        ideas = report.candidates  # Incomplete leads survive for research, not execution.
        assert report.eligible == []
        assert {i.ticker for i in ideas} == {"REAL"}  # cycle-1 result survives

    def test_followup_prompt_lists_discovered_and_branches(self) -> None:
        c1 = _payload(_idea_dict(ticker="REAL", company_name="Real Co Inc"))
        client = MultiResponseClient([c1, _payload()])  # cycle 2 empty
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            max_cycles=2,
        )
        agent.run(
            {"id": 1, "headline": "Rare-earth ban", "theme": "sanctions_trade", "assessment": {}}
        )
        followup = client.calls[1]["messages"][1].content
        # It feeds back what was found and pushes the tree-of-thought branches.
        assert "Real Co Inc" in followup and "REAL" in followup
        assert "Upstream" in followup and "Downstream" in followup
        assert "Substitutes" in followup and "Financial" in followup

    def test_max_cycles_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NICHE_MAX_CYCLES", "3")
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
        )
        assert agent.max_cycles == 3

    def test_max_cycles_clamped(self) -> None:
        hi = NicheAgent(
            FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
            max_cycles=99,
        )
        lo = NicheAgent(
            FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
            max_cycles=0,
        )
        assert hi.max_cycles == 5
        assert lo.max_cycles == 1


# --------------------------------------------------------------------- #
# Tool-augmented hopping (CL-2czc)
# --------------------------------------------------------------------- #


class TestToolAugmentedHopping:
    def _event(self) -> dict[str, Any]:
        return {
            "id": 1,
            "headline": "China restricts rare-earth exports",
            "theme": None,
            "assessment": {},
        }

    def test_tools_ground_the_next_cycle(self) -> None:
        c1 = _payload(_idea_dict(ticker="REAL", company_name="Real Co Inc"))
        c2 = _payload(_idea_dict(ticker="FRO", company_name="Frontline Ltd"))
        client = MultiResponseClient([c1, c2])
        tools = FakeTools()
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            max_cycles=2,
            tools=tools,
        )
        agent.run(self._event())
        # Cycle-1's REAL (a valid ticker) was enriched before cycle 2...
        assert tools.calls == [["REAL"]]
        # ...and its grounding block landed in the cycle-2 prompt.
        followup = client.calls[1]["messages"][1].content
        assert "grounded REAL" in followup
        assert "REAL RESEARCH DATA" in followup

    def test_tools_respect_max_entities(self) -> None:
        c1 = _payload(
            _idea_dict(ticker="REAL", company_name="Real Co Inc"),
            _idea_dict(ticker="FRO", company_name="Frontline Ltd"),
        )
        client = MultiResponseClient([c1, _payload()])
        tools = FakeTools()
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            max_cycles=2,
            tools=tools,
            tools_max_entities=1,
        )
        agent.run(self._event())
        assert tools.calls == [["REAL"]]  # capped at 1 entity despite 2 fresh

    def test_no_tools_no_enrichment(self) -> None:
        c1 = _payload(_idea_dict(ticker="REAL"))
        client = MultiResponseClient([c1, _payload()])
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            max_cycles=2,  # tools default off
        )
        assert agent.tools is None
        agent.run(self._event())  # no crash, no grounding
        followup = client.calls[1]["messages"][1].content
        assert "REAL RESEARCH DATA" not in followup

    def test_tools_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NICHE_TOOLS_ENABLED", "1")
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
        )
        # A real ResearchTools is constructed when enabled.
        assert agent.tools is not None

    def test_tools_enrichment_failure_is_soft(self) -> None:
        class _BoomTools:
            def enrich(self, ideas: Any, universe: Any) -> list[str]:
                raise RuntimeError("sec exploded")

        c1 = _payload(_idea_dict(ticker="REAL"))
        c2 = _payload(_idea_dict(ticker="FRO", company_name="Frontline Ltd"))
        client = MultiResponseClient([c1, c2])
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            max_cycles=2,
            tools=_BoomTools(),
        )
        report = agent.run_report(self._event())
        ideas = report.candidates  # Incomplete leads survive for research, not execution.
        assert report.eligible == []
        # Enrichment blew up but the run still completes with both names.
        assert {i.ticker for i in ideas} == {"REAL", "FRO"}


# --------------------------------------------------------------------- #
# Agentic Kimi tool-loop delegation (CL-ddzt)
# --------------------------------------------------------------------- #


class TestKimiToolAgentDelegation:
    def _event(self) -> dict[str, Any]:
        return {
            "id": 1,
            "headline": "China restricts rare-earth exports",
            "theme": None,
            "assessment": {},
        }

    def test_tool_agent_replaces_cycles(self) -> None:
        payload = _payload(_idea_dict(ticker="REAL", company_name="Real Co Inc"))
        # The claude-code client must NOT be used when a tool_agent is set.
        client = MockLLMClient(_payload(_idea_dict(ticker="FRO")))
        tool_agent = FakeToolAgent(payload)
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=client,  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            tool_agent=tool_agent,
        )
        report = agent.run_report(self._event())
        ideas = report.candidates  # Incomplete leads survive for research, not execution.
        assert report.eligible == []
        assert {i.ticker for i in ideas} == {"REAL"}
        assert tool_agent.calls == [1]  # the Kimi agent did the discovery
        assert client.calls == []  # cycles path was bypassed

    def test_tool_agent_empty_yields_nothing(self) -> None:
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            tool_agent=FakeToolAgent(""),
        )
        assert agent.run(self._event()) == []

    def test_tool_agent_output_still_verified(self) -> None:
        # A hallucinated ticker from the tool agent is still dropped.
        payload = _payload(_idea_dict(ticker="FAKE", company_name="Nowhere Inc"))
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            tool_agent=FakeToolAgent(payload),
        )
        assert agent.run(self._event()) == []  # FAKE not in universe → dropped

    def test_tool_agent_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NICHE_TOOL_AGENT_ENABLED", "1")
        monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
        )
        assert agent.tool_agent is not None

    def test_tool_agent_env_without_key_stays_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NICHE_TOOL_AGENT_ENABLED", "1")
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
        )
        assert agent.tool_agent is not None  # Missing key must not silently switch providers.
        report = agent.run_report(self._event())
        assert report.discovery.status == "unavailable"
        assert report.discovery.reason == "credentials_missing"


# --------------------------------------------------------------------- #
# Adversarial red-team critic integration (CL-3v56)
# --------------------------------------------------------------------- #


class TestRedTeamCritic:
    def _event(self) -> dict[str, Any]:
        return {"id": 1, "headline": "rare-earth ban", "theme": None, "assessment": {}}

    def test_critic_drops_refuted_ideas(self) -> None:
        c1 = _payload(
            _idea_dict(ticker="REAL", company_name="Real Co Inc"),
            _idea_dict(ticker="FRO", company_name="Frontline Ltd"),
        )
        critic = FakeCritic(refute={"FRO"})
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient(c1),  # type: ignore[arg-type]
            market_data_fn=_liquid_md,
            critic=critic,
        )
        report = agent.run_report(self._event())
        ideas = report.candidates  # Incomplete leads survive for research, not execution.
        assert report.eligible == []
        assert {i.ticker for i in ideas} == {"REAL", "FRO"}  # Retain the critic's research inputs.
        assert all(
            i.review_status != "supported" for i in ideas
        )  # Legacy filtering is not approval.
        assert critic.calls == [1]

    def test_no_critic_keeps_all(self) -> None:
        c1 = _payload(
            _idea_dict(ticker="REAL", company_name="Real Co Inc"),
            _idea_dict(ticker="FRO", company_name="Frontline Ltd"),
        )
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient(c1),  # type: ignore[arg-type]
            market_data_fn=_liquid_md,  # critic off by default
        )
        assert agent.critic is None
        report = agent.run_report(self._event())
        assert {i.ticker for i in report.candidates} == {"REAL", "FRO"}
        assert all(i.review_status == "not_requested" for i in report.candidates)
        assert report.eligible == []

    def test_critic_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NICHE_CRITIC_ENABLED", "1")
        agent = NicheAgent(
            universe=FakeUniverse(),
            client=MockLLMClient("{}"),  # type: ignore[arg-type]
        )
        assert agent.critic is not None


# --------------------------------------------------------------------- #
# Module split (CL-ikz2 — review §6.2.2)
# --------------------------------------------------------------------- #


class TestModuleSplit:
    """Pure scoring moved to ``niche_scoring``; ``niche_agent`` must keep
    re-exporting the SAME objects so every existing import (tests, scripts,
    monkeypatch targets) keeps working unchanged."""

    def test_niche_agent_reexports_scoring_surface(self) -> None:
        from src.events import niche_agent, niche_scoring

        for name in (
            "MAX_NICHE_IDEAS",
            "VALID_NICHE_ACTIONS",
            "AsymmetryConfig",
            "NicheIdea",
            "asymmetry_score",
            "parse_niche_ideas",
            "score_and_gate",
            "torque_from_reason",
            "verify_ideas",
        ):
            assert getattr(niche_agent, name) is getattr(niche_scoring, name), name

    def test_legacy_import_surface_intact(self) -> None:
        # The exact import shape used across the repo before the split.
        from src.events.niche_agent import (  # noqa: F401
            DEFAULT_MIN_URGENCY,
            AsymmetryConfig,
            NicheAgent,
            NicheIdea,
            asymmetry_score,
            parse_niche_ideas,
            score_and_gate,
            verify_ideas,
        )
