"""Tests for the Polymarket prediction-market SIGNAL (CL-r1ep).

Covers, all with a mocked HTTP shim + sqlite fixtures / injected engine:
  * gamma parse of JSON-string arrays (outcomePrices / clobTokenIds)
  * current-prob fetch (gamma primary, clob midpoint fallback)
  * prob persistence
  * shift detection (delta threshold, window boundary, no-shift,
    insufficient history, dedup — same shift not re-notified)
  * shift-alert HTML (escaping, ↑/↓, delta sign, implied read)
  * latest_prob_for_theme
  * config load / tracked-market parsing
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Engine, create_engine, text

from src.events.polymarket_signal import (
    PolymarketSignal,
    ProbShift,
    TrackedMarket,
    _implied_read,
    _parse_outcome_yes,
    build_shift_alert,
    load_tracked_markets,
)

# ---------------------------------------------------------------------- #
# sqlite engine mirroring migration 009
# ---------------------------------------------------------------------- #


def _shim(sql: str) -> str:
    return (
        sql.replace("TIMESTAMPTZ", "TEXT")
        .replace("BIGSERIAL", "INTEGER")
        .replace("DOUBLE PRECISION", "REAL")
    )


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    from migrations.run import _strip_sql_comments

    eng = create_engine(f"sqlite:///{tmp_path / 'poly.db'}")
    sql = _shim(_strip_sql_comments(
        Path("migrations/009_poly_market_probs.sql").read_text(),
    ))
    with eng.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    return eng


class FakeHttp:
    """Injectable HTTP shim. Maps (url-substring, param) → canned JSON."""

    def __init__(self, responses: dict[str, Any]) -> None:
        # keyed by slug (gamma) or token_id (midpoint)
        self.responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, params: dict[str, str]) -> Any:
        self.calls.append((url, dict(params)))
        key = params.get("slug") or params.get("token_id") or ""
        if key not in self.responses:
            raise RuntimeError(f"no canned response for {key}")
        return self.responses[key]


MARKET = TrackedMarket(
    slug="hormuz-closure-2026",
    question="Will the Strait of Hormuz be closed to shipping in 2026?",
    yes_token_id="tok-yes-123",
    theme="energy_chokepoint",
)


def _gamma_market(prob: float) -> dict[str, Any]:
    """A Gamma market dict with outcomePrices as a JSON-string array."""
    return {
        "question": MARKET.question,
        "slug": MARKET.slug,
        "outcomePrices": json.dumps([str(prob), str(round(1 - prob, 4))]),
        "clobTokenIds": json.dumps([MARKET.yes_token_id, "tok-no-456"]),
    }


def _insert_obs(
    eng: Engine, slug: str, prob: float, observed_at: datetime,
    theme: str = "energy_chokepoint", question: str = MARKET.question,
    notified: str | None = None,
) -> None:
    with eng.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO poly_market_probs "
                "(slug, question, theme, yes_prob, observed_at, source, "
                " notified_shift) "
                "VALUES (:slug, :q, :theme, :prob, :ts, 'gamma', :n)"
            ),
            {
                "slug": slug, "q": question, "theme": theme,
                "prob": prob, "ts": observed_at.isoformat(), "n": notified,
            },
        )


# ---------------------------------------------------------------------- #
# Gamma parse
# ---------------------------------------------------------------------- #


class TestParseOutcomeYes:
    def test_json_string_array(self) -> None:
        assert _parse_outcome_yes(json.dumps(["0.72", "0.28"])) == pytest.approx(0.72)

    def test_native_list(self) -> None:
        assert _parse_outcome_yes([0.4, 0.6]) == pytest.approx(0.4)

    def test_none(self) -> None:
        assert _parse_outcome_yes(None) is None

    def test_empty(self) -> None:
        assert _parse_outcome_yes(json.dumps([])) is None

    def test_unparseable(self) -> None:
        assert _parse_outcome_yes("not json") is None
        assert _parse_outcome_yes(["x", "y"]) is None


# ---------------------------------------------------------------------- #
# fetch_current_prob
# ---------------------------------------------------------------------- #


class TestFetchCurrentProb:
    def test_gamma_primary(self, engine: Engine) -> None:
        http = FakeHttp({MARKET.slug: _gamma_market(0.61)})
        sig = PolymarketSignal(engine, http_get_json=http)
        assert sig.fetch_current_prob(MARKET) == pytest.approx(0.61)
        # Gamma is tried first — no midpoint call needed.
        assert all("gamma" in u for u, _ in http.calls)

    def test_gamma_list_response(self, engine: Engine) -> None:
        http = FakeHttp({MARKET.slug: [_gamma_market(0.33)]})
        sig = PolymarketSignal(engine, http_get_json=http)
        assert sig.fetch_current_prob(MARKET) == pytest.approx(0.33)

    def test_midpoint_fallback(self, engine: Engine) -> None:
        # Gamma returns nothing usable → fall through to CLOB midpoint.
        http = FakeHttp({
            MARKET.slug: {"outcomePrices": None},
            MARKET.yes_token_id: {"mid": "0.47"},
        })
        sig = PolymarketSignal(engine, http_get_json=http)
        assert sig.fetch_current_prob(MARKET) == pytest.approx(0.47)

    def test_both_fail_returns_none(self, engine: Engine) -> None:
        http = FakeHttp({})  # every fetch raises
        sig = PolymarketSignal(engine, http_get_json=http)
        assert sig.fetch_current_prob(MARKET) is None

    def test_clamped(self, engine: Engine) -> None:
        http = FakeHttp({MARKET.slug: _gamma_market(1.5)})
        sig = PolymarketSignal(engine, http_get_json=http)
        assert sig.fetch_current_prob(MARKET) == pytest.approx(1.0)


# ---------------------------------------------------------------------- #
# poll + persist
# ---------------------------------------------------------------------- #


class TestPollProbabilities:
    def test_persists_observation(self, engine: Engine) -> None:
        http = FakeHttp({MARKET.slug: _gamma_market(0.55)})
        sig = PolymarketSignal(engine, http_get_json=http)
        observed = sig.poll_probabilities([MARKET])
        assert len(observed) == 1
        with engine.connect() as conn:
            rows = list(conn.execute(text(
                "SELECT slug, theme, yes_prob, source FROM poly_market_probs"
            )))
        assert len(rows) == 1
        assert rows[0].slug == MARKET.slug
        assert rows[0].theme == "energy_chokepoint"
        assert rows[0].yes_prob == pytest.approx(0.55)
        assert rows[0].source == "gamma"

    def test_per_market_failure_tolerant(self, engine: Engine) -> None:
        good = TrackedMarket("good", "Good?", "tg", "war_escalation")
        bad = TrackedMarket("bad", "Bad?", "tb", "war_escalation")
        http = FakeHttp({"good": _gamma_market(0.4)})  # 'bad' raises
        sig = PolymarketSignal(engine, http_get_json=http)
        observed = sig.poll_probabilities([bad, good])
        # Only the good market persisted; the bad one didn't kill the poll.
        assert {m.slug for m, _ in observed} == {"good"}
        with engine.connect() as conn:
            n = conn.execute(text(
                "SELECT COUNT(*) FROM poly_market_probs"
            )).scalar()
        assert n == 1


# ---------------------------------------------------------------------- #
# shift detection
# ---------------------------------------------------------------------- #


class TestDetectShifts:
    def test_rising_shift(self, engine: Engine) -> None:
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        _insert_obs(engine, MARKET.slug, 0.20, now - timedelta(hours=20))
        _insert_obs(engine, MARKET.slug, 0.34, now - timedelta(minutes=5))
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        shifts = sig.detect_shifts(window_hours=24, threshold=0.10, now=now)
        assert len(shifts) == 1
        s = shifts[0]
        assert s.slug == MARKET.slug
        assert s.rising
        assert s.delta == pytest.approx(0.14)
        assert s.delta_points == 14

    def test_falling_shift(self, engine: Engine) -> None:
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        _insert_obs(engine, MARKET.slug, 0.60, now - timedelta(hours=10))
        _insert_obs(engine, MARKET.slug, 0.42, now - timedelta(minutes=1))
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        shifts = sig.detect_shifts(window_hours=24, threshold=0.10, now=now)
        assert len(shifts) == 1
        assert not shifts[0].rising
        assert shifts[0].delta_points == -18

    def test_below_threshold_no_shift(self, engine: Engine) -> None:
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        _insert_obs(engine, MARKET.slug, 0.50, now - timedelta(hours=5))
        _insert_obs(engine, MARKET.slug, 0.56, now)  # +6 pts < 10
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        assert sig.detect_shifts(threshold=0.10, now=now) == []

    def test_window_boundary_excludes_old_obs(self, engine: Engine) -> None:
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        # The big move happened OUTSIDE the 24h window; only recent obs
        # count, and they're flat → no shift.
        _insert_obs(engine, MARKET.slug, 0.10, now - timedelta(hours=30))
        _insert_obs(engine, MARKET.slug, 0.50, now - timedelta(hours=2))
        _insert_obs(engine, MARKET.slug, 0.51, now - timedelta(minutes=1))
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        shifts = sig.detect_shifts(window_hours=24, threshold=0.10, now=now)
        # earliest in-window is 0.50, latest 0.51 → +1 pt, no shift.
        assert shifts == []

    def test_insufficient_history(self, engine: Engine) -> None:
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        _insert_obs(engine, MARKET.slug, 0.20, now)  # single obs
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        assert sig.detect_shifts(now=now) == []

    def test_dedup_not_renotified(self, engine: Engine) -> None:
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        _insert_obs(engine, MARKET.slug, 0.20, now - timedelta(hours=5))
        _insert_obs(engine, MARKET.slug, 0.36, now - timedelta(minutes=1))
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        shifts = sig.detect_shifts(now=now)
        assert len(shifts) == 1
        # Mark it notified (simulate notify_shift's dedup stamp).
        sig._mark_notified(shifts[0])
        # Same window, same latest row → no longer fires.
        assert sig.detect_shifts(now=now) == []

    def test_new_larger_shift_fires_after_dedup(self, engine: Engine) -> None:
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        _insert_obs(engine, MARKET.slug, 0.20, now - timedelta(hours=5))
        _insert_obs(engine, MARKET.slug, 0.36, now - timedelta(hours=1))
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        first = sig.detect_shifts(now=now)
        sig._mark_notified(first[0])
        # A LATER observation extends the move — new latest row, not
        # stamped → fires again.
        _insert_obs(engine, MARKET.slug, 0.50, now - timedelta(minutes=1))
        again = sig.detect_shifts(now=now)
        assert len(again) == 1
        assert again[0].delta_points == 30


# ---------------------------------------------------------------------- #
# notify + alert HTML
# ---------------------------------------------------------------------- #


class TestShiftAlert:
    def _shift(self, prob: float, delta: float, question: str) -> ProbShift:
        return ProbShift(
            slug="hormuz-closure-2026", question=question,
            theme="energy_chokepoint", latest_prob=prob,
            earliest_prob=prob - delta, delta=delta, window_hours=24,
            anchor_id=1,
        )

    def test_rising_alert_html(self) -> None:
        title, msg = build_shift_alert(
            self._shift(0.72, 0.14, "Will the Strait of Hormuz be closed?"),
        )
        assert title == "Prediction market shift"
        assert "<b>Prediction market shift</b>" in msg
        assert "<b>72% ↑ +14 (24h)</b>" in msg
        assert "<i>energy_chokepoint</i>" in msg
        assert "polymarket.com/event/hormuz-closure-2026" in msg
        assert "ESCALATION" in msg

    def test_falling_alert_html(self) -> None:
        _, msg = build_shift_alert(
            self._shift(0.30, -0.14, "Will the Strait of Hormuz be closed?"),
        )
        assert "<b>30% ↓ -14 (24h)</b>" in msg

    def test_question_escaped(self) -> None:
        _, msg = build_shift_alert(
            self._shift(0.50, 0.12, "War & <peace> in 2026?"),
        )
        assert "War &amp; &lt;peace&gt; in 2026?" in msg
        assert "<peace>" not in msg

    def test_notify_marks_and_sends(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.events import polymarket_signal as mod
        from src.research.notifications import DispatchResult

        sent: list[tuple[str, str]] = []

        def fake_notify(title: str, message: str, priority: int = 0, *, html: bool = False):
            sent.append((title, message))
            return DispatchResult(telegram_attempted=True, telegram_succeeded=True)

        monkeypatch.setattr(mod, "notify_operator", fake_notify)

        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        _insert_obs(engine, MARKET.slug, 0.20, now - timedelta(hours=3))
        _insert_obs(engine, MARKET.slug, 0.40, now - timedelta(minutes=1))
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        shifts = sig.detect_shifts(now=now)
        assert len(shifts) == 1
        sig.notify_shift(shifts[0])
        assert len(sent) == 1
        # Marked → not re-notified.
        assert sig.detect_shifts(now=now) == []


class TestImpliedRead:
    def test_rising_escalation_market(self) -> None:
        s = ProbShift("s", "Will Taiwan be blockaded?", "taiwan_semiconductor",
                      0.4, 0.25, 0.15, 24)
        assert "ESCALATION" in _implied_read(s)

    def test_rising_ceasefire_market_is_deescalation(self) -> None:
        s = ProbShift("s", "Will there be a ceasefire in Ukraine?",
                      "russia_ukraine", 0.5, 0.3, 0.2, 24)
        assert "DE-ESCALATION" in _implied_read(s)

    def test_ambiguous_falls_back(self) -> None:
        s = ProbShift("s", "Some vague question about a thing", "other",
                      0.5, 0.3, 0.2, 24)
        read = _implied_read(s)
        assert "rising YES" in read


# ---------------------------------------------------------------------- #
# latest_prob_for_theme
# ---------------------------------------------------------------------- #


class TestLatestProbForTheme:
    def test_returns_latest_per_slug(self, engine: Engine) -> None:
        now = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        _insert_obs(engine, "hormuz-closure-2026", 0.10, now - timedelta(hours=2),
                    theme="energy_chokepoint")
        _insert_obs(engine, "hormuz-closure-2026", 0.18, now - timedelta(minutes=1),
                    theme="energy_chokepoint")
        _insert_obs(engine, "malacca-blocked-2026", 0.05, now,
                    theme="energy_chokepoint")
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        out = sig.latest_prob_for_theme("energy_chokepoint")
        assert set(out) == {"hormuz-closure-2026", "malacca-blocked-2026"}
        assert out["hormuz-closure-2026"]["yes_prob"] == pytest.approx(0.18)
        # rising: latest 0.18 >= prior 0.10 → True
        assert out["hormuz-closure-2026"]["rising"] is True

    def test_other_theme_empty(self, engine: Engine) -> None:
        sig = PolymarketSignal(engine, http_get_json=FakeHttp({}))
        assert sig.latest_prob_for_theme("nonexistent") == {}


# ---------------------------------------------------------------------- #
# config load
# ---------------------------------------------------------------------- #


class TestLoadTrackedMarkets:
    def test_parses_entries(self, tmp_path: Path) -> None:
        cfg = tmp_path / "geo.yaml"
        cfg.write_text(yaml.safe_dump({"markets": [
            {"slug": "a", "question": "Q A?", "yes_token_id": "ta",
             "theme": "war_escalation"},
            {"slug": "b", "question": "Q B?", "yes_token_id": "tb",
             "theme": "taiwan_semiconductor"},
        ]}))
        markets = load_tracked_markets(cfg)
        assert [m.slug for m in markets] == ["a", "b"]
        assert markets[0].theme == "war_escalation"
        assert markets[0].yes_token_id == "ta"

    def test_missing_file_empty(self, tmp_path: Path) -> None:
        assert load_tracked_markets(tmp_path / "nope.yaml") == []

    def test_skips_incomplete(self, tmp_path: Path) -> None:
        cfg = tmp_path / "geo.yaml"
        cfg.write_text(yaml.safe_dump({"markets": [
            {"slug": "ok", "yes_token_id": "t", "theme": "x"},
            {"slug": "no-token", "theme": "x"},
            {"yes_token_id": "t2", "theme": "x"},
        ]}))
        markets = load_tracked_markets(cfg)
        assert [m.slug for m in markets] == ["ok"]

    def test_legacy_token_id_field(self, tmp_path: Path) -> None:
        cfg = tmp_path / "geo.yaml"
        cfg.write_text(yaml.safe_dump({"markets": [
            {"slug": "a", "token_id": "legacy", "theme": "x"},
        ]}))
        markets = load_tracked_markets(cfg)
        assert markets[0].yes_token_id == "legacy"
