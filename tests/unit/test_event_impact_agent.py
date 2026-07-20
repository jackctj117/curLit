"""Tests for the Event Impact Agent (CL-6iu7).

Covers: JSON extraction from messy LLM text, schema validation +
range clamping, equity watch-only enforcement, unreachable-instrument
filtering, and the DB write path (NEW → ASSESSED / DISMISSED) against a
sqlite engine with a mocked LLM client — no live LLM calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.events.impact_agent import (
    EventImpactAgent,
    extract_json_object,
    normalise_assessment,
)
from src.events.playbooks import all_tradable_instruments, load_playbooks

PLAYBOOKS = load_playbooks("configs/event_playbooks.yaml")
FALLBACK = all_tradable_instruments(PLAYBOOKS)
HORMUZ = PLAYBOOKS["energy_chokepoint"]


def _valid_payload(**overrides: Any) -> dict:
    base: dict = {
        "core_event": "Iran announces closure of the Strait of Hormuz",
        "direction": "bearish",
        "urgency": 9,
        "horizon": "hours",
        "confidence": 0.8,
        "affected": [
            {"instrument": "BCO_USD", "kind": "oanda",
             "direction": "long", "reason": "supply risk premium"},
            {"instrument": "FRO", "kind": "equity_watch",
             "direction": "watch", "reason": "tanker rates"},
        ],
        "rationale": "Chokepoint closure is the canonical oil supply shock.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------- #
# extract_json_object
# ---------------------------------------------------------------------- #


class TestExtractJson:
    def test_bare_json(self) -> None:
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json_with_prose(self) -> None:
        raw = 'Here is my assessment:\n```json\n{"a": {"b": 2}}\n```\nDone.'
        # Prose before the fence: the brace scan starts at the first '{'.
        assert extract_json_object(raw) == {"a": {"b": 2}}

    def test_json_embedded_in_prose(self) -> None:
        raw = 'Sure! {"a": 1, "s": "brace } in string"} trailing text'
        assert extract_json_object(raw) == {"a": 1, "s": "brace } in string"}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            extract_json_object("I cannot assess this headline.")

    def test_unbalanced_raises(self) -> None:
        with pytest.raises(ValueError, match="unbalanced"):
            extract_json_object('{"a": 1')


# ---------------------------------------------------------------------- #
# normalise_assessment
# ---------------------------------------------------------------------- #


class TestNormalise:
    def _norm(self, payload: dict) -> dict:
        return normalise_assessment(payload, "headline", HORMUZ, FALLBACK)

    def test_happy_path_passes_through(self) -> None:
        out = self._norm(_valid_payload())
        assert out["direction"] == "bearish"
        assert out["urgency"] == 9
        assert out["horizon"] == "hours"
        assert out["confidence"] == 0.8
        assert len(out["affected"]) == 2

    def test_urgency_and_confidence_clamped(self) -> None:
        out = self._norm(_valid_payload(urgency=99, confidence=1.7))
        assert out["urgency"] == 10
        assert out["confidence"] == 1.0
        out = self._norm(_valid_payload(urgency=-3, confidence=-0.4))
        assert out["urgency"] == 1
        assert out["confidence"] == 0.0

    def test_horizon_singular_normalised(self) -> None:
        assert self._norm(_valid_payload(horizon="hour"))["horizon"] == "hours"
        assert self._norm(_valid_payload(horizon="Days"))["horizon"] == "days"

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            self._norm(_valid_payload(direction="to the moon"))

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            self._norm(_valid_payload(horizon="weeks"))

    def test_non_numeric_urgency_raises(self) -> None:
        with pytest.raises(ValueError, match="urgency"):
            self._norm(_valid_payload(urgency="high"))

    def test_equity_forced_to_watch(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "FRO", "kind": "equity_watch",
             "direction": "long", "reason": "tankers"},
        ])
        out = self._norm(payload)
        assert out["affected"][0]["direction"] == "watch"

    def test_unreachable_tradable_dropped(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "brent futures!!", "kind": "oanda",
             "direction": "long", "reason": "x"},
            {"instrument": "BCO_USD", "kind": "oanda",
             "direction": "long", "reason": "ok"},
        ])
        out = self._norm(payload)
        assert [a["instrument"] for a in out["affected"]] == ["BCO_USD"]

    def test_unvetted_oanda_shaped_symbol_demoted_to_watch(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "USD_TRY", "kind": "fx",
             "direction": "long", "reason": "lira stress"},
        ])
        out = self._norm(payload)
        assert out["affected"][0] == {
            "instrument": "USD_TRY", "kind": "fx",
            "direction": "watch", "reason": "lira stress",
        }

    def test_slash_symbol_normalised(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "eur/usd", "kind": "fx",
             "direction": "short", "reason": "x"},
        ])
        out = self._norm(payload)
        assert out["affected"][0]["instrument"] == "EUR_USD"
        assert out["affected"][0]["direction"] == "short"  # vetted (playbooks)

    def test_duplicate_instruments_collapsed(self) -> None:
        payload = _valid_payload(affected=[
            {"instrument": "BCO_USD", "kind": "oanda",
             "direction": "long", "reason": "a"},
            {"instrument": "BCO_USD", "kind": "oanda",
             "direction": "short", "reason": "b"},
        ])
        assert len(self._norm(payload)["affected"]) == 1

    def test_affected_not_a_list_raises(self) -> None:
        with pytest.raises(ValueError, match="affected"):
            self._norm(_valid_payload(affected="BCO_USD"))

    def test_empty_core_event_falls_back_to_headline(self) -> None:
        out = self._norm(_valid_payload(core_event=""))
        assert out["core_event"] == "headline"

    def test_advisory_fields_default_to_empty_lists(self) -> None:
        # Optional fields absent -> stable empty-list shape downstream.
        out = self._norm(_valid_payload())
        assert out["trade_ideas"] == []
        assert out["fade_candidates"] == []


class TestTradeIdeas:
    """CL-01zt advisory trade ideas — additive, operator-facing; the
    machine-traded 'affected' array must be untouched by them."""

    def _idea(self, **overrides: Any) -> dict:
        base: dict = {
            "ticker": "GOLD",
            "action": "buy_puts",
            "direction": "bearish",
            "confidence": 0.7,
            "rationale": "Loulo-Gounkoto (Mali) seizure risk hits Barrick NAV",
            "time_horizon": "medium",
            "holding_period_days": "5-20",
            "time_stop_days": 20,
            "suggested_entry": "on any bounce",
            "preferred_instrument": "puts, 1-2 month expiry",
            "notes": "training-data vintage — verify mine status",
        }
        base.update(overrides)
        return base

    def _norm(self, **payload_overrides: Any) -> dict:
        return normalise_assessment(
            _valid_payload(**payload_overrides), "headline", HORMUZ, FALLBACK,
        )

    def test_valid_idea_passes_through(self) -> None:
        out = self._norm(trade_ideas=[self._idea()])
        assert len(out["trade_ideas"]) == 1
        idea = out["trade_ideas"][0]
        assert idea["ticker"] == "GOLD"
        assert idea["action"] == "buy_puts"
        assert idea["time_horizon"] == "medium"
        assert idea["time_stop_days"] == 20
        # affected is untouched by advisory extras
        assert {a["instrument"] for a in out["affected"]} == {"BCO_USD", "FRO"}

    def test_bad_action_dropped_assessment_survives(self) -> None:
        out = self._norm(trade_ideas=[
            self._idea(action="yolo_leaps"),
            self._idea(ticker="BTG", action="short"),
        ])
        assert [i["ticker"] for i in out["trade_ideas"]] == ["BTG"]
        assert out["direction"] == "bearish"  # whole assessment intact

    def test_bad_horizon_dropped(self) -> None:
        out = self._norm(trade_ideas=[self._idea(time_horizon="forever")])
        assert out["trade_ideas"] == []

    def test_missing_ticker_dropped(self) -> None:
        out = self._norm(trade_ideas=[self._idea(ticker="  ")])
        assert out["trade_ideas"] == []

    def test_confidence_clamped_and_defaulted(self) -> None:
        out = self._norm(trade_ideas=[
            self._idea(confidence=1.8),
            self._idea(ticker="B", confidence="very high"),
        ])
        assert out["trade_ideas"][0]["confidence"] == 1.0
        assert out["trade_ideas"][1]["confidence"] == 0.5

    def test_time_stop_defaults_by_horizon(self) -> None:
        out = self._norm(trade_ideas=[
            self._idea(time_stop_days=None, time_horizon="immediate"),
            self._idea(ticker="B", time_stop_days="soon", time_horizon="medium"),
        ])
        assert out["trade_ideas"][0]["time_stop_days"] == 3
        assert out["trade_ideas"][1]["time_stop_days"] == 20

    def test_direction_derived_from_action_when_invalid(self) -> None:
        out = self._norm(trade_ideas=[
            self._idea(direction="sideways", action="buy_calls"),
            self._idea(ticker="B", direction="", action="short"),
        ])
        assert out["trade_ideas"][0]["direction"] == "bullish"
        assert out["trade_ideas"][1]["direction"] == "bearish"

    def test_non_list_trade_ideas_ignored_not_dismissed(self) -> None:
        out = self._norm(trade_ideas="buy everything")
        assert out["trade_ideas"] == []
        assert out["direction"] == "bearish"

    def test_idea_count_capped(self) -> None:
        ideas = [self._idea(ticker=f"T{i}") for i in range(12)]
        out = self._norm(trade_ideas=ideas)
        assert len(out["trade_ideas"]) == 8

    # -- concrete, actionable level fields (CL-jiqq) ------------------- #

    def test_concrete_levels_parsed(self) -> None:
        out = self._norm(trade_ideas=[self._idea(
            stop_loss_pct=0.07,
            target_pct=[0.08, 0.15],
            entry_trigger="on confirmed blockade language",
            invalidation="official denial of the seizure",
        )])
        idea = out["trade_ideas"][0]
        assert idea["stop_loss_pct"] == pytest.approx(0.07)
        assert idea["target_pct"] == [0.08, 0.15]
        assert idea["entry_trigger"] == "on confirmed blockade language"
        assert idea["invalidation"] == "official denial of the seizure"

    def test_level_fields_default_when_omitted(self) -> None:
        # Absent levels are honest defaults, never fabricated — the
        # enrichment layer fills them. The idea still survives.
        idea = self._idea()
        for k in ("stop_loss_pct", "target_pct", "entry_trigger", "invalidation"):
            idea.pop(k, None)
        out = self._norm(trade_ideas=[idea])
        parsed = out["trade_ideas"][0]
        assert parsed["stop_loss_pct"] is None
        assert parsed["target_pct"] == []
        assert parsed["entry_trigger"] == ""
        assert parsed["invalidation"] == ""

    def test_stop_loss_pct_clamped_and_bad_dropped(self) -> None:
        out = self._norm(trade_ideas=[
            self._idea(stop_loss_pct=5.0),          # over the 0.90 cap
            self._idea(ticker="B", stop_loss_pct="nope"),  # unparseable
            self._idea(ticker="C", stop_loss_pct=-0.1),    # <= 0 → unset
        ])
        assert out["trade_ideas"][0]["stop_loss_pct"] == pytest.approx(0.90)
        assert out["trade_ideas"][1]["stop_loss_pct"] is None
        assert out["trade_ideas"][2]["stop_loss_pct"] is None

    def test_targets_cleaned_sorted_capped(self) -> None:
        out = self._norm(trade_ideas=[self._idea(
            target_pct=[0.30, 0.05, "junk", 0.30, 5.0, -1.0, 0.15],
        )])
        # dedup + sort + clamp (5.0 → 3.0 cap) + drop junk/neg + cap to 2
        assert out["trade_ideas"][0]["target_pct"] == [0.05, 0.15]

    def test_scalar_target_accepted(self) -> None:
        out = self._norm(trade_ideas=[self._idea(target_pct=0.12)])
        assert out["trade_ideas"][0]["target_pct"] == [0.12]

    def test_bad_levels_never_drop_the_idea(self) -> None:
        # Malformed level fields degrade individually; the idea itself
        # (valid ticker/action/horizon) survives — levels are advisory.
        out = self._norm(trade_ideas=[self._idea(
            stop_loss_pct="", target_pct="all of it",
        )])
        assert len(out["trade_ideas"]) == 1
        assert out["trade_ideas"][0]["stop_loss_pct"] is None
        assert out["trade_ideas"][0]["target_pct"] == []

    def test_fade_candidates_parsed_and_bad_dropped(self) -> None:
        out = self._norm(fade_candidates=[
            {"ticker": "FXI", "action": "fade the spike",
             "reason": "routine drills, knee-jerk China risk-off"},
            {"action": "fade", "reason": "no ticker"},
            "not a dict",
        ])
        assert out["fade_candidates"] == [{
            "ticker": "FXI", "action": "fade the spike",
            "reason": "routine drills, knee-jerk China risk-off",
        }]


# ---------------------------------------------------------------------- #
# agent + DB round trip (mock LLM, sqlite)
# ---------------------------------------------------------------------- #


class MockLLMClient:
    def __init__(self, text_out: str) -> None:
        self.text_out = text_out
        self.calls: list[dict] = []

    def complete(self, messages: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"messages": messages, "model": model})
        return SimpleNamespace(
            text=self.text_out, model=model, provider="mock",
            input_tokens=10, output_tokens=10,
            usd_cost=0.0, elapsed_sec=0.01,
        )


def _shim(sql: str) -> str:
    return (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("JSONB", "TEXT")
        .replace("BIGSERIAL", "INTEGER")
    )


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'geo.db'}")
    sql = _shim(_strip_sql_comments(
        Path("migrations/005_geo_events.sql").read_text(),
    ))
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


def _insert_event(
    engine: Engine, external_id: str, headline: str,
    theme: str | None = "energy_chokepoint", seen_at: str = "2026-07-14T09:30:00",
) -> None:
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO geo_events "
            "(seen_at, source, external_id, headline, url, theme, "
            " status, status_updated_at) "
            "VALUES (:seen, 'gdelt', :eid, :hl, 'https://n.test/x', :theme, "
            "'NEW', :seen)",
        ), {"seen": seen_at, "eid": external_id, "hl": headline, "theme": theme})


def _fetch(engine: Engine, external_id: str) -> dict:
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT status, assessment FROM geo_events WHERE external_id=:e",
        ), {"e": external_id}).one()
    return {"status": row[0], "assessment": json.loads(row[1])}


class TestAgentRoundTrip:
    def test_happy_path_marks_assessed(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Iran moves to close Hormuz")
        client = MockLLMClient(json.dumps(_valid_payload()))
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events()

        assert len(results) == 1
        assert results[0].status == "ASSESSED"
        row = _fetch(engine, "e1")
        assert row["status"] == "ASSESSED"
        a = row["assessment"]
        assert a["direction"] == "bearish"
        assert {x["instrument"] for x in a["affected"]} == {"BCO_USD", "FRO"}
        assert "conf=0.80" in results[0].summary_line()
        # Prompt carried the playbook context
        user_msg = client.calls[0]["messages"][1].content
        assert "energy_chokepoint" in user_msg
        assert "BCO_USD" in user_msg

    def test_malformed_llm_output_marks_dismissed(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Some headline")
        client = MockLLMClient("I refuse to answer in JSON, sorry.")
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events()

        assert results[0].status == "DISMISSED"
        row = _fetch(engine, "e1")
        assert row["status"] == "DISMISSED"
        assert "impact agent failure" in row["assessment"]["rationale"]

    def test_schema_violation_marks_dismissed(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Some headline")
        bad = json.dumps(_valid_payload(direction="sideways"))
        agent = EventImpactAgent(engine, client=MockLLMClient(bad))  # type: ignore[arg-type]
        assert agent.assess_new_events()[0].status == "DISMISSED"

    def test_fenced_output_still_parses(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Hormuz headline")
        fenced = f"```json\n{json.dumps(_valid_payload())}\n```"
        agent = EventImpactAgent(engine, client=MockLLMClient(fenced))  # type: ignore[arg-type]
        assert agent.assess_new_events()[0].status == "ASSESSED"

    def test_limit_caps_processing(self, engine: Engine) -> None:
        for i in range(3):
            _insert_event(
                engine, f"e{i}", f"headline {i}",
                seen_at=f"2026-07-14T09:0{i}:00",
            )
        client = MockLLMClient(json.dumps(_valid_payload()))
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events(limit=2)

        assert len(results) == 2
        # Newest first: e2, e1 assessed; e0 still NEW for the next cycle.
        assert {r.event_id for r in results} == {2, 3}
        with engine.connect() as conn:
            remaining = conn.execute(text(
                "SELECT external_id FROM geo_events WHERE status='NEW'",
            )).scalars().all()
        assert remaining == ["e0"]

    def test_unmatched_theme_uses_fallback_whitelist(self, engine: Engine) -> None:
        _insert_event(engine, "e1", "Odd headline", theme=None)
        client = MockLLMClient(json.dumps(_valid_payload()))
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events()
        assert results[0].status == "ASSESSED"
        user_msg = client.calls[0]["messages"][1].content
        assert "No playbook matched" in user_msg


# ---------------------------------------------------------------------- #
# CL-01zt prompt guidance
# ---------------------------------------------------------------------- #


class TestPromptGuidance:
    """The analyst-guidance upgrade must land in the system prompt
    WITHOUT touching the JSON contract (confluence/digest depend on it)."""

    def test_system_prompt_has_operator_guidance(self) -> None:
        from src.events.impact_agent import _SYSTEM_PROMPT

        low = _SYSTEM_PROMPT.lower()
        assert "conservative" in low
        assert "second-order" in low
        assert "territory" in low  # territorial asset naming
        assert "training-data" in low  # vintage honesty
        assert "liquid" in low
        assert "tsmc" in low  # China-Taiwan guidance
        # advisory trade-idea guidance (second refinement)
        assert "trade_ideas" in low
        assert "buy_puts" in low
        assert "time_stop_days" in low
        assert "fade_candidates" in low

    def test_json_schema_keys_unchanged(self) -> None:
        from src.events.impact_agent import _SYSTEM_PROMPT

        for key in (
            '"core_event"', '"direction"', '"urgency"', '"horizon"',
            '"confidence"', '"affected"', '"instrument"', '"kind"',
            '"reason"', '"rationale"',
        ):
            assert key in _SYSTEM_PROMPT

    def test_territorial_playbook_context_reaches_llm(self, engine: Engine) -> None:
        # A DRC event's user prompt must carry the mine/territory notes
        # so the model can name whose assets sit in the territory.
        _insert_event(
            engine, "e1", "Kolwezi export halt announced",
            theme="drc_copper_cobalt",
        )
        client = MockLLMClient(json.dumps(_valid_payload()))
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        agent.assess_new_events()
        user_msg = client.calls[0]["messages"][1].content
        assert "Kamoa-Kakula" in user_msg
        assert "IVN.TO" in user_msg
        assert "XCU_USD" in user_msg


class _RaisingLLMClient:
    """LLM client whose complete() always raises (transport failure)."""

    def complete(self, messages: Any, model: str, **kwargs: Any) -> Any:
        raise RuntimeError("claude -p exited 1: usage limit reached")


class TestTransportFailureRetrySemantics:
    """Transport failures must leave rows NEW (retried next cycle) —
    a quota outage once terminally DISMISSED ~175 healthy events."""

    def test_llm_transport_error_leaves_row_new(self, engine: Engine) -> None:
        _insert_event(engine, "tf1", "Iran moves to close Hormuz")
        agent = EventImpactAgent(engine, client=_RaisingLLMClient())  # type: ignore[arg-type]
        results = agent.assess_new_events()
        assert len(results) == 1
        assert results[0].status == "NEW"
        assert results[0].assessment == {}
        # Row untouched in the DB — still queued, no assessment written.
        # (_fetch json.loads()es the assessment, so query raw here.)
        with engine.connect() as conn:
            raw = conn.execute(text(
                "SELECT status, assessment FROM geo_events "
                "WHERE external_id='tf1'",
            )).one()
        assert raw[0] == "NEW"
        assert raw[1] is None

    def test_content_failure_still_dismisses(self, engine: Engine) -> None:
        _insert_event(engine, "tf2", "Iran moves to close Hormuz")
        client = MockLLMClient("this is not json at all")
        agent = EventImpactAgent(engine, client=client)  # type: ignore[arg-type]
        results = agent.assess_new_events()
        assert results[0].status == "DISMISSED"
        assert "impact agent failure" in results[0].assessment["rationale"]
