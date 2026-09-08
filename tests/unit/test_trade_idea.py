"""Unit tests — typed TradeIdea model (CL-59mk).

This is a PURE TYPING refactor of dicts that ride the money path
(``geo_events.assessment["trade_ideas"]`` → the ``trade_ideas`` table →
the Alpaca options executor), so the tests here are shape contracts, not
behaviour tests:

  * EXACT round-trip on the two real producer shapes — the impact
    agent's normalised idea and the niche pass's merged idea. Both are
    built from the LIVE producers (``_normalise_trade_ideas`` /
    ``NicheIdea.to_trade_idea``), not from hand-copied literals, so a
    producer change that this model doesn't follow fails here.
  * Key set + key ORDER parity (``json.dumps`` must stay byte-identical).
  * Tolerant-reader parity: a minimal / legacy dict defaults exactly the
    way today's readers default it, and clamping/enum validation is NOT
    duplicated out of ``_normalise_trade_ideas``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.events.impact_agent import (
    MAX_TRADE_IDEAS,
    VALID_IDEA_ACTIONS,
    VALID_IDEA_HORIZONS,
    _normalise_trade_ideas,
)
from src.events.niche_scoring import NicheIdea
from src.events.trade_idea import (
    BULLISH_ACTIONS,
    CORE_KEYS,
    NICHE_KEYS,
    OPTION_ACTIONS,
    TradeIdea,
)

# --------------------------------------------------------------------- #
# Real producer shapes
# --------------------------------------------------------------------- #


def _raw_idea(**overrides: Any) -> dict[str, Any]:
    """A full-fat LLM idea payload (every documented key populated)."""
    raw: dict[str, Any] = {
        "ticker": "TSM",
        "action": "buy_puts",
        "direction": "bearish",
        "confidence": 0.72,
        "rationale": "advanced-node concentration risk",
        "time_horizon": "short",
        "holding_period_days": "2-6",
        "time_stop_days": 5,
        "stop_loss_pct": 0.07,
        "target_pct": [0.10, 0.18],
        "entry_trigger": "on confirmed blockade language",
        "invalidation": "official denial of the strike",
        "suggested_entry": "on any bounce",
        "preferred_instrument": "slightly OTM puts, 2-4 weeks to expiry",
        "notes": "vol crush after the spike; training-data vintage",
    }
    raw.update(overrides)
    return raw


def _normalised(**overrides: Any) -> dict[str, Any]:
    """One idea as the impact agent actually persists it."""
    ideas = _normalise_trade_ideas([_raw_idea(**overrides)])
    assert len(ideas) == 1
    return ideas[0]


def _niche_merged(**overrides: Any) -> dict[str, Any]:
    """One idea as the niche pass actually merges it (niche=true shape)."""
    idea = NicheIdea(
        ticker="ABC",
        company_name="Abcdaria Mining Corp",
        action="buy_calls",
        direction="bullish",
        hop_count=4,
        torque_reason="single-asset junior, high operating leverage",
        rationale="only Western producer of the cut-off feedstock",
        confidence=0.61,
    )
    idea.verified = True
    idea.exchange = "NYSE American"
    idea.robinhood_tradeable = True
    idea.asymmetry_score = 0.71
    idea.liquidity_flag = False
    idea.red_team_verdict = "survived"
    for key, value in overrides.items():
        setattr(idea, key, value)
    return idea.to_trade_idea()


# --------------------------------------------------------------------- #
# Key set — derived from the producers, not from literals
# --------------------------------------------------------------------- #


class TestKeySet:
    def test_core_keys_match_the_impact_agent_shape(self) -> None:
        assert tuple(_normalised()) == CORE_KEYS

    def test_niche_shape_is_core_plus_the_niche_block(self) -> None:
        assert tuple(_niche_merged()) == CORE_KEYS + NICHE_KEYS + ("research",)

    def test_core_key_count_is_stable(self) -> None:
        # The persisted contract: 15 core keys, 9 additive niche keys.
        assert len(CORE_KEYS) == 15
        assert len(NICHE_KEYS) == 9
        assert not set(CORE_KEYS) & set(NICHE_KEYS)


# --------------------------------------------------------------------- #
# Round-trip: to_dict(from_dict(d)) == d, byte-for-byte
# --------------------------------------------------------------------- #


class TestRoundTrip:
    def test_full_key_dict_round_trips_exactly(self) -> None:
        d = _normalised()
        assert TradeIdea.from_dict(d).to_dict() == d

    def test_full_key_dict_round_trips_byte_identically(self) -> None:
        # json.dumps is what actually hits geo_events.assessment, so key
        # ORDER matters, not just key/value equality.
        d = _normalised()
        assert json.dumps(TradeIdea.from_dict(d).to_dict()) == json.dumps(d)

    def test_niche_merged_dict_round_trips_exactly(self) -> None:
        d = _niche_merged()
        assert TradeIdea.from_dict(d).to_dict() == d
        assert json.dumps(TradeIdea.from_dict(d).to_dict()) == json.dumps(d)

    def test_niche_optional_nones_survive_the_round_trip(self) -> None:
        # exchange / red_team_verdict / asymmetry_score are None-able and
        # None must NOT collapse to "" (the digest renders "?" off None).
        d = _niche_merged(
            exchange=None,
            red_team_verdict="",  # to_trade_idea maps "" -> None
            asymmetry_score=None,
        )
        assert d["exchange"] is None
        assert d["red_team_verdict"] is None
        assert d["asymmetry_score"] is None
        assert TradeIdea.from_dict(d).to_dict() == d

    def test_minimal_key_dict_round_trips_exactly(self) -> None:
        # The normaliser fills every key even from a bare LLM entry, so
        # the minimal PERSISTED shape is still the canonical 15.
        d = _normalise_trade_ideas([{"ticker": "FRO", "action": "long", "time_horizon": "short"}])[
            0
        ]
        assert TradeIdea.from_dict(d).to_dict() == d
        assert json.dumps(TradeIdea.from_dict(d).to_dict()) == json.dumps(d)

    def test_niche_block_absent_on_a_plain_idea(self) -> None:
        # A non-niche idea must not GROW niche keys — that would change
        # every persisted assessment payload.
        out = TradeIdea.from_dict(_normalised()).to_dict()
        assert not set(NICHE_KEYS) & set(out)

    def test_dataclass_construction_matches_from_dict(self) -> None:
        d = _normalised()
        built = TradeIdea(
            ticker="TSM",
            action="buy_puts",
            direction="bearish",
            confidence=0.72,
            rationale="advanced-node concentration risk",
            time_horizon="short",
            holding_period_days="2-6",
            time_stop_days=5,
            stop_loss_pct=0.07,
            target_pct=[0.10, 0.18],
            entry_trigger="on confirmed blockade language",
            invalidation="official denial of the strike",
            suggested_entry="on any bounce",
            preferred_instrument="slightly OTM puts, 2-4 weeks to expiry",
            notes="vol crush after the spike; training-data vintage",
        )
        assert built == TradeIdea.from_dict(d)
        assert built.to_dict() == d


# --------------------------------------------------------------------- #
# Tolerant reader: defaults match today's hand-rolled reads
# --------------------------------------------------------------------- #


class TestTolerantReader:
    def test_empty_dict_gives_the_reader_defaults(self) -> None:
        idea = TradeIdea.from_dict({})
        assert idea.ticker == ""
        assert idea.action == ""
        assert idea.direction == ""
        assert idea.confidence is None
        assert idea.time_stop_days is None
        assert idea.stop_loss_pct is None
        assert idea.target_pct == []
        assert idea.niche is False
        assert idea.hop_count is None
        assert idea.exchange is None
        assert idea.red_team_verdict is None

    def test_legacy_pre_levels_dict_adds_only_canonical_keys(self) -> None:
        # A row persisted before CL-jiqq added stop/target/trigger fields.
        legacy = {
            k: v
            for k, v in _normalised().items()
            if k not in ("stop_loss_pct", "target_pct", "entry_trigger", "invalidation")
        }
        out = TradeIdea.from_dict(legacy).to_dict()
        # Nothing dropped, nothing invented beyond the canonical shape.
        assert tuple(out) == CORE_KEYS
        for key, value in legacy.items():
            assert out[key] == value
        # The four late-added keys default exactly the way the normaliser
        # defaults them when the LLM omits them.
        assert out["stop_loss_pct"] is None
        assert out["target_pct"] == []
        assert out["entry_trigger"] == ""
        assert out["invalidation"] == ""

    def test_unknown_keys_are_ignored_not_carried(self) -> None:
        # The digest attaches a local ``corroboration`` block to a copy of
        # the idea; parsing must neither choke on it nor persist it.
        d = dict(_normalised())
        d["corroboration"] = {"count": 3, "themes": ["taiwan_strait"]}
        assert TradeIdea.from_dict(d).to_dict() == _normalised()

    def test_none_values_read_as_absent(self) -> None:
        d = _normalised()
        d.update(
            {
                "rationale": None,
                "notes": None,
                "confidence": None,
                "time_stop_days": None,
                "target_pct": None,
            }
        )
        idea = TradeIdea.from_dict(d)
        assert idea.rationale == ""
        assert idea.notes == ""
        assert idea.confidence is None
        assert idea.time_stop_days is None
        assert idea.target_pct == []

    def test_string_reads_are_lossless_never_stripped(self) -> None:
        # Readers DISAGREE about stripping (the ledger strips ticker /
        # action / preferred_instrument / notes and nothing else), so the
        # model refuses to pick: it carries the value through untouched
        # and the callers that stripped still strip. This is also what
        # keeps to_dict(from_dict(d)) == d true for a padded legacy row.
        d = {"ticker": "  TSM  ", "rationale": "  padded  ", "notes": "  n  "}
        idea = TradeIdea.from_dict(d)
        assert idea.ticker == "  TSM  "
        assert idea.rationale == "  padded  "
        assert idea.to_dict()["rationale"] == "  padded  "

    def test_falsy_non_none_values_read_as_empty_string(self) -> None:
        # ``str(x or "")``, NOT ``str(x)``: an entry_trigger of 0 has
        # always persisted as NULL, not as the string "0".
        idea = TradeIdea.from_dict(
            {"entry_trigger": 0, "notes": False, "invalidation": [], "rationale": 0.0},
        )
        assert idea.entry_trigger == ""
        assert idea.notes == ""
        assert idea.invalidation == ""
        assert idea.rationale == ""

    def test_unparseable_numbers_degrade_to_none_never_raise(self) -> None:
        idea = TradeIdea.from_dict(
            {
                "ticker": "X",
                "confidence": "very high",
                "time_stop_days": "soon",
                "stop_loss_pct": "",
                "asymmetry_score": "n/a",
                "hop_count": [],
            }
        )
        assert idea.confidence is None
        assert idea.time_stop_days is None
        assert idea.stop_loss_pct is None
        assert idea.asymmetry_score is None
        assert idea.hop_count is None

    def test_target_pct_accepts_scalar_and_drops_junk_preserving_order(self) -> None:
        assert TradeIdea.from_dict({"target_pct": 0.12}).target_pct == [0.12]
        assert TradeIdea.from_dict({"target_pct": "nope"}).target_pct == []
        # Order preserved (dedupe/sort/cap belong to the producer).
        assert TradeIdea.from_dict(
            {"target_pct": [0.18, "junk", 0.10]},
        ).target_pct == [0.18, 0.10]

    def test_niche_flag_is_truthiness_like_every_reader(self) -> None:
        assert TradeIdea.from_dict({"niche": True}).niche is True
        assert TradeIdea.from_dict({"niche": False}).niche is False
        assert TradeIdea.from_dict({}).niche is False


# --------------------------------------------------------------------- #
# The model does NOT own validation — the normaliser still does
# --------------------------------------------------------------------- #


class TestValidationStaysInTheNormaliser:
    """CL-59mk is typing only: enum validation, clamping and horizon
    defaulting must remain in ``_normalise_trade_ideas``. These pin that
    the model neither duplicates nor bypasses it."""

    def test_from_dict_does_not_clamp_or_validate(self) -> None:
        idea = TradeIdea.from_dict(
            {
                "ticker": "X",
                "action": "yolo_leaps",  # not a valid action
                "time_horizon": "forever",  # not a valid horizon
                "confidence": 1.8,  # out of [0, 1]
                "stop_loss_pct": 5.0,  # over the 0.90 cap
                "time_stop_days": 900,  # over the 120 cap
            }
        )
        assert idea.action == "yolo_leaps"
        assert idea.time_horizon == "forever"
        assert idea.confidence == pytest.approx(1.8)
        assert idea.stop_loss_pct == pytest.approx(5.0)
        assert idea.time_stop_days == 900

    def test_normaliser_still_clamps_confidence_and_time_stop(self) -> None:
        out = _normalise_trade_ideas(
            [
                _raw_idea(confidence=1.8, time_stop_days=900),
                _raw_idea(
                    ticker="B",
                    confidence="very high",
                    time_stop_days=None,
                    time_horizon="medium",
                ),
            ]
        )
        assert out[0]["confidence"] == 1.0
        assert out[0]["time_stop_days"] == 120
        assert out[1]["confidence"] == 0.5  # advisory default
        assert out[1]["time_stop_days"] == 20  # medium-horizon default

    def test_normaliser_still_drops_bad_enums_and_caps_the_list(self) -> None:
        assert _normalise_trade_ideas([_raw_idea(action="yolo_leaps")]) == []
        assert _normalise_trade_ideas([_raw_idea(time_horizon="forever")]) == []
        assert _normalise_trade_ideas([_raw_idea(ticker="  ")]) == []
        many = [_raw_idea(ticker=f"T{i}") for i in range(MAX_TRADE_IDEAS + 4)]
        assert len(_normalise_trade_ideas(many)) == MAX_TRADE_IDEAS

    def test_every_valid_action_and_horizon_round_trips(self) -> None:
        for action in sorted(VALID_IDEA_ACTIONS):
            for horizon in sorted(VALID_IDEA_HORIZONS):
                d = _normalised(action=action, time_horizon=horizon, direction="")
                assert TradeIdea.from_dict(d).to_dict() == d


# --------------------------------------------------------------------- #
# Small typed helpers (no new business logic)
# --------------------------------------------------------------------- #


class TestHelpers:
    @pytest.mark.parametrize(
        ("action", "expected"),
        [("long", True), ("buy_calls", True), ("short", False), ("buy_puts", False)],
    )
    def test_is_bullish_reads_action_first(self, action: str, expected: bool) -> None:
        # Action wins even when ``direction`` disagrees (same precedence
        # the trade-card grounder has always used).
        idea = TradeIdea(ticker="X", action=action, direction="bearish")
        assert idea.is_bullish is expected

    def test_is_bullish_falls_back_to_direction(self) -> None:
        assert TradeIdea(ticker="X", action="", direction="bearish").is_bullish is False
        assert TradeIdea(ticker="X", action="", direction="bullish").is_bullish is True
        assert TradeIdea(ticker="X").is_bullish is True  # arbitrary but pinned

    def test_is_options_action(self) -> None:
        assert TradeIdea(ticker="X", action="buy_calls").is_options_action is True
        assert TradeIdea(ticker="X", action="buy_puts").is_options_action is True
        assert TradeIdea(ticker="X", action="long").is_options_action is False
        assert TradeIdea(ticker="X", action="short").is_options_action is False

    def test_helper_sets_agree_with_the_enum_the_agent_validates(self) -> None:
        assert BULLISH_ACTIONS < VALID_IDEA_ACTIONS
        assert OPTION_ACTIONS < VALID_IDEA_ACTIONS
