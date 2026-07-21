"""Tests for the reflective self-tuning loop (CL-g8jl).

Sqlite idea_outcomes fixture + mocked LLM. Covers dimension aggregation, the
insufficient-data guard, the LLM proposal path, fail-soft parse, and the
Telegram rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, text

from src.events.reflective_review import (
    ReflectiveReviewer,
    aggregate_outcomes,
)


class MockLLMClient:
    def __init__(self, text_out: str, raises: bool = False) -> None:
        self.text_out = text_out
        self.raises = raises
        self.calls: list[Any] = []

    def complete(self, messages: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(model)
        if self.raises:
            raise RuntimeError("review LLM down")
        return SimpleNamespace(
            text=self.text_out, model=model, provider="mock",
            input_tokens=10, output_tokens=10, usd_cost=0.0, elapsed_sec=0.01,
        )


@pytest.fixture
def engine(tmp_path):  # type: ignore[no-untyped-def]
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'r.db'}")
    sql = _strip_sql_comments(Path("migrations/013_idea_outcomes.sql").read_text())
    sql = (sql.replace("TIMESTAMPTZ", "TEXT").replace("NUMERIC", "FLOAT")
           .replace("DEFAULT FALSE", "DEFAULT 0"))
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


def _seed(engine, idea_id, outcome, ret, *, theme="hormuz", action="long",
          direction="bullish", confidence=0.6, is_niche=0, hop=None,
          red_team=0):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO idea_outcomes (idea_id, ticker, action, direction, theme,
                confidence, is_niche, hop_count, red_team_survived, return_pct,
                outcome, updated_at)
            VALUES (:id,'T',:ac,:d,:th,:c,:n,:h,:rt,:r,:o,'2026-07-21')
        """), {"id": idea_id, "ac": action, "d": direction, "th": theme,
               "c": confidence, "n": is_niche, "h": hop, "rt": red_team,
               "r": ret, "o": outcome})


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_overall_and_dimensions(engine):
    _seed(engine, "1", "win", 0.10, theme="hormuz")
    _seed(engine, "2", "loss", -0.08, theme="hormuz")
    _seed(engine, "3", "win", 0.06, theme="taiwan", is_niche=1, hop=4)
    _seed(engine, "4", "flat", 0.01, theme="taiwan")
    agg = aggregate_outcomes(engine)
    assert agg["sample"] == 4
    assert agg["overall"]["wins"] == 2
    assert agg["overall"]["win_rate"] == 0.5
    assert agg["by_theme"]["hormuz"]["n"] == 2
    assert agg["by_theme"]["hormuz"]["win_rate"] == 0.5
    assert agg["by_hop"]["3+"]["n"] == 1
    assert agg["by_niche"]["niche"]["n"] == 1
    assert agg["by_niche"]["core"]["n"] == 3


def test_aggregate_excludes_open(engine):
    _seed(engine, "1", "win", 0.10)
    _seed(engine, "2", "open", None)  # not finalised → excluded
    assert aggregate_outcomes(engine)["sample"] == 1


def test_confidence_buckets(engine):
    _seed(engine, "1", "win", 0.1, confidence=0.9)
    _seed(engine, "2", "loss", -0.1, confidence=0.55)
    _seed(engine, "3", "flat", 0.0, confidence=0.3)
    agg = aggregate_outcomes(engine)
    assert set(agg["by_confidence"]) == {"high", "medium", "low"}


# --------------------------------------------------------------------------- #
# reviewer
# --------------------------------------------------------------------------- #


def test_insufficient_data_short_circuits(engine):
    _seed(engine, "1", "win", 0.1)
    client = MockLLMClient("{}")
    result = ReflectiveReviewer(engine, client=client, min_sample=20).review()
    assert result.status == "insufficient_data"
    assert result.sample == 1
    assert client.calls == []  # never bothered the LLM
    assert "Insufficient" in result.to_telegram()


def test_review_produces_proposal(engine):
    for i in range(6):
        _seed(engine, f"w{i}", "win", 0.08, action="long")
    for i in range(6):
        _seed(engine, f"l{i}", "loss", -0.09, action="short")
    proposal = json.dumps({
        "overall_read": "Longs work, shorts don't.",
        "findings": ["short win rate 0%"],
        "proposed_changes": [
            {"knob": "min_confidence", "change": "raise shorts to 0.85",
             "rationale": "0/6 shorts won"}],
    })
    result = ReflectiveReviewer(
        engine, client=MockLLMClient(proposal), min_sample=5).review()
    assert result.status == "ok"
    assert result.sample == 12
    assert result.proposal["proposed_changes"][0]["knob"] == "min_confidence"
    tg = result.to_telegram()
    assert "min_confidence" in tg and "Proposed tuning" in tg


def test_review_fail_soft_on_bad_llm(engine):
    for i in range(6):
        _seed(engine, f"x{i}", "win", 0.05)
    result = ReflectiveReviewer(
        engine, client=MockLLMClient("not json"), min_sample=5).review()
    assert result.status == "ok"       # aggregation still returned
    assert result.proposal is None     # but no proposal
    assert result.sample == 6


def test_review_fail_soft_on_transport(engine):
    for i in range(6):
        _seed(engine, f"x{i}", "win", 0.05)
    result = ReflectiveReviewer(
        engine, client=MockLLMClient("", raises=True), min_sample=5).review()
    assert result.status == "ok"
    assert result.proposal is None
