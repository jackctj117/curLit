"""Account-currency risk and P&L for event legs (CL-vfw7).

Independent oracles: every expected number below is HAND-COMPUTED from
the instrument's quote convention (1 unit of BASE_QUOTE costs ``price``
QUOTE; a QUOTE amount is worth ``amount / USDQUOTE`` USD), not taken from
the implementation. The defect: ``size = equity * risk / stop_distance``
divided USD by JPY-per-unit — a 2026-08-03 USD_JPY short was ~150x too
small and its "+44.29" P&L was ¥44.29 (~$0.28).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.risk.currency import (
    AccountCurrencyConverter,
    ConversionUnavailable,
    quote_currency,
    resolve_account_currency,
    risk_sized_units,
)
from src.strategies.event_book import EventBook, EventPosition
from src.strategies.event_driven import EventDrivenConfig, EventDrivenStrategy

EQUITY = 100_000.0  # account currency (USD)
# Config defaults (unchanged by CL-vfw7): 0.5% risk, 1% stop.
RISK_BUDGET = EQUITY * 0.005  # $500


def fresh(price: float, *, age: timedelta = timedelta(0)) -> dict[str, Any]:
    ts = (datetime.now(UTC) - age).isoformat()
    return {"bid": price, "ask": price, "ts": ts}


def make_strat(tmp_path: Path, **cfg: Any) -> EventDrivenStrategy:
    """Uses ONLY the pre-CL-vfw7 constructor surface, so the sizing vectors
    below also run (and must FAIL) against the pre-fix code."""
    cfg.setdefault("event_book_state_path", str(tmp_path / "book.json"))
    return EventDrivenStrategy(EventDrivenConfig(**cfg))


def enter(
    strat: EventDrivenStrategy,
    instrument: str,
    prices: dict[str, Any],
    direction: str = "short",
) -> tuple[list[Any], list[Any], list[Any]]:
    assessment = {
        "urgency": 8,
        "confidence": 0.9,
        "affected": [
            {"instrument": instrument, "kind": "fx", "direction": direction, "reason": "t"},
        ],
    }
    return strat._enter_confirmed(
        {"id": 1, "headline": "vector"},
        assessment,
        prices,
        EQUITY,
        datetime.now(UTC),
    )


# =============================================================================
# Sizing vectors — account-currency risk at the stop == equity * risk_pct
# =============================================================================


class TestSizingVectors:
    def test_usd_jpy_short_sized_in_usd(self, tmp_path: Path) -> None:
        # USDJPY = 150. 1% stop → 1.5 JPY/unit = 1.5/150 = $0.01/unit.
        # $500 / $0.01 = 50,000 units. Pre-fix: 500 / 1.5 = 333 units.
        strat = make_strat(tmp_path)
        intents, entered, skipped = enter(strat, "USD_JPY", {"USDJPY": fresh(150.0)})
        assert skipped == []
        assert len(intents) == 1
        assert intents[0].target_position == pytest.approx(-50_000.0)

    def test_eur_jpy_cross_sized_in_usd(self, tmp_path: Path) -> None:
        # EURJPY = 160, USDJPY = 150. Stop 1.6 JPY/unit = 1.6/150 USD.
        # 500 / (1.6/150) = 46,875 units.
        strat = make_strat(tmp_path)
        prices = {"EURJPY": fresh(160.0), "USDJPY": fresh(150.0), "EURUSD": fresh(1.0667)}
        intents, _, skipped = enter(strat, "EUR_JPY", prices, direction="long")
        assert skipped == []
        assert intents[0].target_position == pytest.approx(46_875.0)

    def test_usd_mxn_sized_in_usd(self, tmp_path: Path) -> None:
        # USDMXN = 18. Stop 0.18 MXN/unit = $0.01 → 50,000 units.
        strat = make_strat(tmp_path)
        intents, _, skipped = enter(strat, "USD_MXN", {"USDMXN": fresh(18.0)})
        assert skipped == []
        assert intents[0].target_position == pytest.approx(-50_000.0)

    def test_eur_usd_unchanged(self, tmp_path: Path) -> None:
        # USD-quoted: rate 1. Stop 0.011 USD/unit → 500/0.011 = 45,454.54.
        strat = make_strat(tmp_path)
        intents, _, skipped = enter(strat, "EUR_USD", {"EURUSD": fresh(1.10)}, "long")
        assert skipped == []
        assert intents[0].target_position == pytest.approx(500.0 / 0.011)

    def test_entry_records_conversion_provenance(self, tmp_path: Path) -> None:
        strat = make_strat(tmp_path)
        enter(strat, "USD_JPY", {"USDJPY": fresh(150.0)})
        pos = strat.book.pending_entries["USD_JPY"].position
        assert pos.quote_ccy == "JPY"
        assert pos.entry_conversion is not None
        assert pos.entry_conversion.rate == pytest.approx(1 / 150.0)
        assert pos.entry_conversion.source_pair == "USD_JPY"
        state = json.loads((tmp_path / "book.json").read_text())
        persisted = state["pending_entries"]["USD_JPY"]["position"]
        assert persisted["quote_ccy"] == "JPY"
        assert persisted["entry_conversion"]["rate"] == pytest.approx(1 / 150.0)


class TestFailClosed:
    def test_stale_tick_skips_entry_no_order(self, tmp_path: Path) -> None:
        strat = make_strat(tmp_path)
        prices = {"USDJPY": fresh(150.0, age=timedelta(minutes=20))}
        intents, entered, skipped = enter(strat, "USD_JPY", prices)
        assert intents == [] and entered == []
        assert skipped == [("USD_JPY", "conversion_unavailable")]
        assert strat.book.pending_entries == {}

    def test_missing_conversion_pair_skips(self, tmp_path: Path) -> None:
        # EUR_JPY priced, but no USD_JPY (or JPY_USD) tick → JPY→USD unknown.
        strat = make_strat(tmp_path)
        intents, _, skipped = enter(strat, "EUR_JPY", {"EURJPY": fresh(160.0)}, "long")
        assert intents == []
        assert skipped == [("EUR_JPY", "conversion_unavailable")]

    def test_timestampless_tick_is_not_fresh(self, tmp_path: Path) -> None:
        strat = make_strat(tmp_path)
        intents, _, skipped = enter(strat, "USD_JPY", {"USDJPY": {"bid": 150.0, "ask": 150.0}})
        assert intents == []
        assert skipped == [("USD_JPY", "conversion_unavailable")]

    def test_broker_account_currency_mismatch_blocks_sizing(self) -> None:
        broker = SimpleNamespace(
            get_account=lambda: SimpleNamespace(equity=EQUITY, currency="EUR"),
        )
        assert EventDrivenStrategy._get_equity(broker, "USD") is None
        assert EventDrivenStrategy._get_equity(broker, "EUR") == EQUITY


# =============================================================================
# Converter
# =============================================================================


def _conv(prices: dict[str, Any], account: str = "USD") -> AccountCurrencyConverter:
    return AccountCurrencyConverter(price_source=lambda: prices, account_currency=account)


class TestConverter:
    def test_identity(self) -> None:
        now = datetime.now(UTC)
        c = _conv({}).rate("USD", now)
        assert c.rate == 1.0 and c.source_pair == "identity"

    def test_inverse_direct(self) -> None:
        c = _conv({"USD_JPY": fresh(150.0)}).rate("JPY", datetime.now(UTC))
        assert c.rate == pytest.approx(1 / 150.0)

    def test_forward_direct_uses_mid(self) -> None:
        tick = {"bid": 1.2690, "ask": 1.2710, "ts": datetime.now(UTC).isoformat()}
        c = _conv({"GBPUSD": tick}).rate("GBP", datetime.now(UTC))
        assert c.rate == pytest.approx(1.27)
        assert c.source_pair == "GBP_USD"

    def test_cross_via_usd_for_eur_account(self) -> None:
        # JPY→EUR with no EUR_JPY tick: (1/150 USD per JPY) × (1/1.25 EUR
        # per USD) = 1/187.5.
        prices = {"USD_JPY": fresh(150.0), "EUR_USD": fresh(1.25)}
        c = _conv(prices, "EUR").rate("JPY", datetime.now(UTC))
        assert c.rate == pytest.approx(1 / 187.5)
        assert c.source_pair == "USD_JPY*EUR_USD"

    def test_oanda_rfc3339_timestamp(self) -> None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.123456789Z")
        prices = {"USDJPY": {"bid": 150.0, "ask": 150.0, "time": stamp}}
        assert _conv(prices).rate("JPY", datetime.now(UTC)).rate == pytest.approx(1 / 150.0)

    @pytest.mark.parametrize(
        "tick",
        [
            {"bid": 150.0, "ask": 150.0},  # no timestamp
            {"bid": 0.0, "ask": 150.0, "ts": "now"},  # non-positive
            {"bid": 151.0, "ask": 150.0, "ts": "now"},  # crossed
            {"bid": float("nan"), "ask": 150.0, "ts": "now"},
            {"bid": 150.0, "ask": 150.0, "ts": "stale"},
            {"bid": 150.0, "ask": 150.0, "ts": "future"},
        ],
    )
    def test_invalid_ticks_unavailable(self, tick: dict[str, Any]) -> None:
        now = datetime.now(UTC)
        stamps = {
            "now": now,
            "stale": now - timedelta(minutes=16),
            "future": now + timedelta(minutes=5),
        }
        if "ts" in tick:
            tick = {**tick, "ts": stamps[tick["ts"]].isoformat()}
        with pytest.raises(ConversionUnavailable):
            _conv({"USD_JPY": tick}).rate("JPY", now)

    def test_quote_currency_parsing(self) -> None:
        assert quote_currency("USD_JPY") == "JPY"
        assert quote_currency("USDJPY") == "JPY"
        assert quote_currency("XAU_USD") == "USD"
        assert quote_currency("SPX500_USD") == "USD"
        assert quote_currency("SPX500") is None

    def test_account_currency_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CURLIT_ACCOUNT_CURRENCY", raising=False)
        assert resolve_account_currency() == "USD"
        monkeypatch.setenv("CURLIT_ACCOUNT_CURRENCY", "eur")
        assert resolve_account_currency() == "EUR"
        assert resolve_account_currency(SimpleNamespace(currency="GBP")) == "GBP"
        monkeypatch.setenv("CURLIT_ACCOUNT_CURRENCY", "dollars")
        with pytest.raises(ValueError):
            resolve_account_currency()


@settings(max_examples=300, deadline=None)
@given(
    equity=st.floats(min_value=1.0, max_value=1e9),
    risk_pct=st.floats(min_value=1e-5, max_value=0.1),
    stop=st.floats(min_value=1e-6, max_value=1e4),
    rate=st.floats(min_value=1e-4, max_value=1e4),
)
def test_account_risk_invariant(equity: float, risk_pct: float, stop: float, rate: float) -> None:
    """For ANY positive inputs, the account-currency loss at the stop of
    the computed size equals equity × risk_pct (old code: only if rate=1)."""
    units = risk_sized_units(equity * risk_pct, stop, rate)
    assert units > 0
    assert math.isclose(units * stop * rate, equity * risk_pct, rel_tol=1e-9)


@settings(max_examples=100, deadline=None)
@given(rate_a=st.floats(min_value=1e-3, max_value=1e3), k=st.floats(1.01, 100.0))
def test_size_monotone_decreasing_in_rate(rate_a: float, k: float) -> None:
    assert risk_sized_units(500.0, 1.5, rate_a * k) < risk_sized_units(500.0, 1.5, rate_a)


# =============================================================================
# P&L accounting
# =============================================================================


def _book(tmp_path: Path, prices: dict[str, Any], **kw: Any) -> EventBook:
    conv = _conv(prices)
    return EventBook(
        state_path=str(tmp_path / "book.json"),
        max_loss_pct=0.02,
        per_instrument_max_pct=0.55,
        haven_max_pct=0.60,
        max_holding_hours=4.0,
        reconcile_grace_sec=120,
        account_currency="USD",
        convert=conv.rate,
        **kw,
    )


#: Broker snapshot at trigger: the account holds our 222-unit short (a
#: capture of 0 would finalize the leg as a never-filled PHANTOM).
_HELD = SimpleNamespace(symbol="USD_JPY", quantity=-222.0)


def _short_jpy(hours_ago: float = 5.0) -> EventPosition:
    return EventPosition(
        symbol="USD_JPY",
        event_id=1,
        entry_ts=datetime.now(UTC) - timedelta(hours=hours_ago),
        entry_price=156.758,
        quantity=-222.0,
        direction=-1,
        stop_price=158.0,
        headline="2026-08-03 vector",
    )


class TestPnlAccounting:
    def test_usd_jpy_short_pnl_converted(self, tmp_path: Path) -> None:
        # (156.5585 - 156.758) × -222 = ¥44.289; at USDJPY 156.76 → $0.28253.
        prices = {"USDJPY": fresh(156.76)}
        book = _book(tmp_path, prices)
        book.open_positions["USD_JPY"] = _short_jpy()
        now = datetime.now(UTC)
        book.check_exits(lambda _s: 156.5585, now, broker_positions=[_HELD])
        records = book.confirm_exits([], now)
        assert len(records) == 1
        assert records[0].pnl_quote == pytest.approx(44.289)
        assert records[0].pnl_account == pytest.approx(44.289 / 156.76)
        assert book.realized_pnl == pytest.approx(0.28253, abs=1e-5)
        row = book.recent_closed[-1]
        assert row["quote_ccy"] == "JPY" and row["account_ccy"] == "USD"
        assert row["exit_price_basis"] == "trigger_price_estimate"
        assert row["exit_conversion"]["source_pair"] == "USD_JPY"

    def test_unavailable_exit_rate_never_booked_and_blocks_entries(
        self,
        tmp_path: Path,
    ) -> None:
        prices: dict[str, Any] = {}  # no USD_JPY tick at trigger or confirm
        book = _book(tmp_path, prices)
        pos = _short_jpy()
        pos.entry_price = 156.0  # short from 156.0, exit 156.5585 → a LOSS
        book.open_positions["USD_JPY"] = pos
        now = datetime.now(UTC)
        book.check_exits(lambda _s: 156.5585, now, broker_positions=[_HELD])
        records = book.confirm_exits([], now)
        assert records[0].pnl_account is None
        assert records[0].book_realized_pnl is None  # unknown, not "unchanged"
        assert book.realized_pnl == 0.0  # unknown is NOT zero-booked
        assert book.closed_trades == 1
        assert book.loss_cap_unknown_reason() == "unconverted_realized_loss"
        # A LATER fresh rate does not back-fill it (it is not the close-time
        # rate): still blocked after a restart with a live USDJPY tick.
        prices["USD_JPY"] = fresh(156.0)
        reloaded = _book(tmp_path, prices)
        assert reloaded.loss_cap_unknown_reason() == "unconverted_realized_loss"
        assert reloaded.realized_pnl == 0.0
        # Operator reconciliation from the broker statement books it.
        state = json.loads((tmp_path / "book.json").read_text())
        state["unconverted_closes"][0]["reconciled_pnl_account"] = -0.79
        (tmp_path / "book.json").write_text(json.dumps(state))
        fixed = _book(tmp_path, prices)
        assert fixed.loss_cap_unknown_reason() is None
        assert fixed.realized_pnl == pytest.approx(-0.79)
        assert fixed.recent_closed[-1]["pnl_account_source"] == "operator_reconciled"

    def test_stale_trigger_conversion_not_used_at_confirmation(self, tmp_path: Path) -> None:
        # Trigger captured a rate; the exit stayed pending two days (e.g.
        # rejected re-emits / restart). At confirmation the stored rate is
        # stale and no fresh tick exists → unconverted, not booked.
        prices: dict[str, Any] = {"USD_JPY": fresh(150.0)}
        book = _book(tmp_path, prices)
        book.open_positions["USD_JPY"] = _short_jpy()
        trigger = datetime.now(UTC)
        book.check_exits(lambda _s: 156.5585, trigger, broker_positions=[_HELD])
        assert book.pending_exits["USD_JPY"].exit_conversion is not None
        prices.clear()
        records = book.confirm_exits([], trigger + timedelta(days=2))
        assert records[0].pnl_account is None
        assert book.realized_pnl == 0.0
        assert len(book.unconverted_closes) == 1

    def test_delayed_confirmation_not_backfilled_with_todays_rate(
        self,
        tmp_path: Path,
    ) -> None:
        # Trigger 1 h ago (rate then 150); confirmation now with a FRESH
        # 156.76 tick. The fill time lies somewhere in between, so neither
        # rate is known to be the fill-time rate → unconverted.
        prices: dict[str, Any] = {"USD_JPY": fresh(150.0, age=timedelta(hours=1))}
        book = _book(tmp_path, prices)
        book.open_positions["USD_JPY"] = _short_jpy(hours_ago=6.0)
        trigger = datetime.now(UTC) - timedelta(hours=1)
        book.check_exits(lambda _s: 156.5585, trigger, broker_positions=[_HELD])
        assert book.pending_exits["USD_JPY"].exit_conversion is not None
        prices["USD_JPY"] = fresh(156.76)
        records = book.confirm_exits([], datetime.now(UTC))
        assert records[0].pnl_account is None
        assert book.realized_pnl == 0.0
        assert len(book.unconverted_closes) == 1

    def test_delayed_confirmation_of_usd_quoted_leg_books_identity(
        self,
        tmp_path: Path,
    ) -> None:
        # EUR_USD long 10,000 from 1.1000, trigger 1.0990 → -$10. USD→USD is
        # 1 at any time, so a late confirmation is still fully known.
        book = _book(tmp_path, {})
        book.open_positions["EUR_USD"] = EventPosition(
            "EUR_USD",
            1,
            datetime.now(UTC) - timedelta(hours=6),
            1.1000,
            10_000.0,
            1,
            1.0890,
        )
        trigger = datetime.now(UTC) - timedelta(hours=1)
        held = SimpleNamespace(symbol="EUR_USD", quantity=10_000.0)
        book.check_exits(lambda _s: 1.0990, trigger, broker_positions=[held])
        records = book.confirm_exits([], datetime.now(UTC))
        assert records[0].pnl_account == pytest.approx(-10.0)
        assert book.realized_pnl == pytest.approx(-10.0)
        assert book.loss_cap_unknown_reason() is None

    def test_confirmation_within_window_uses_trigger_rate(self, tmp_path: Path) -> None:
        prices: dict[str, Any] = {"USD_JPY": fresh(156.76)}
        book = _book(tmp_path, prices)
        book.open_positions["USD_JPY"] = _short_jpy()
        trigger = datetime.now(UTC)
        book.check_exits(lambda _s: 156.5585, trigger, broker_positions=[_HELD])
        prices["USD_JPY"] = fresh(150.0)  # moved; trigger-time rate stands
        records = book.confirm_exits([], trigger + timedelta(minutes=5))
        assert records[0].pnl_account == pytest.approx(44.289 / 156.76)

    def test_projected_total_unknown_while_earlier_close_unconverted(
        self,
        tmp_path: Path,
    ) -> None:
        book = _book(tmp_path, {"USD_JPY": fresh(156.76)})
        book.unconverted_closes.append(
            {
                "symbol": "EUR_JPY",
                "quote_ccy": "JPY",
                "pnl_quote": -160_000.0,
                "closed_at": datetime.now(UTC).isoformat(),
            }
        )
        book.open_positions["USD_JPY"] = _short_jpy()
        now = datetime.now(UTC)
        book.check_exits(lambda _s: 156.5585, now, broker_positions=[_HELD])
        records = book.confirm_exits([], now)
        assert records[0].pnl_account == pytest.approx(44.289 / 156.76)
        assert records[0].book_realized_pnl is None

    def test_unconverted_loss_blocks_strategy_entries(self, tmp_path: Path) -> None:
        strat = make_strat(tmp_path)
        strat.book.unconverted_closes.append(
            {
                "symbol": "USD_JPY",
                "quote_ccy": "JPY",
                "pnl_quote": -1000.0,
                "closed_at": datetime.now(UTC).isoformat(),
            }
        )
        intents, _, skipped = enter(strat, "EUR_USD", {"EURUSD": fresh(1.1)}, "long")
        assert intents == []
        assert skipped == [("EUR_USD", "unconverted_realized_loss")]

    def test_concentration_uses_account_notional(self, tmp_path: Path) -> None:
        # Open 50,000 USD_JPY @150 = ¥7.5M = $50,000 notional (50% of 100k).
        # Cap 55% → $5,000 headroom → a new 50k-unit leg trims to 5,000
        # units. Pre-fix compared ¥7.5M to $55k → skipped entirely.
        book = _book(tmp_path, {"USD_JPY": fresh(150.0)})
        book.open_positions["USD_JPY"] = EventPosition(
            "USD_JPY", 1, datetime.now(UTC), 150.0, 50_000.0, 1, 148.5
        )
        capped = book.concentration_capped_size("USD_JPY", 50_000.0, 150.0, EQUITY)
        assert capped == pytest.approx(5_000.0)

    def test_concentration_unavailable_raises(self, tmp_path: Path) -> None:
        book = _book(tmp_path, {})
        with pytest.raises(ConversionUnavailable):
            book.concentration_capped_size("USD_JPY", 50_000.0, 150.0, EQUITY)


# =============================================================================
# Legacy state migration (v1 → v2)
# =============================================================================


def _write(tmp_path: Path, payload: dict[str, Any]) -> None:
    (tmp_path / "book.json").write_text(json.dumps(payload))


class TestLegacyState:
    def test_v1_aggregate_labelled_legacy_not_account(self, tmp_path: Path) -> None:
        # The live file's +48.38 is a mixed ¥/$ sum.
        _write(tmp_path, {"version": 1, "realized_pnl": 48.38, "closed_trades": 7})
        book = _book(tmp_path, {})
        assert book.realized_pnl == 0.0
        assert book.legacy_mixed_currency_pnl == pytest.approx(48.38)
        assert book.closed_trades == 7
        assert book.currency_migrated_at is not None
        book.save()
        saved = json.loads((tmp_path / "book.json").read_text())
        assert saved["version"] == 2
        assert saved["account_currency"] == "USD"
        assert saved["realized_pnl"] == 0.0
        assert saved["legacy_mixed_currency_pnl"] == pytest.approx(48.38)
        # Reload v2: stays labelled, never merged.
        again = _book(tmp_path, {})
        assert again.realized_pnl == 0.0
        assert again.legacy_mixed_currency_pnl == pytest.approx(48.38)

    def test_versionless_file_is_v1(self, tmp_path: Path) -> None:
        _write(tmp_path, {"realized_pnl": -12.5})
        book = _book(tmp_path, {})
        assert book.legacy_mixed_currency_pnl == pytest.approx(-12.5)
        assert book.realized_pnl == 0.0

    def test_legacy_gain_not_credited_to_loss_budget(self, tmp_path: Path) -> None:
        # A +1,000 mixed sum can hide -$5,000 and +¥6,000 (≈ -$4,960): its
        # USD value is unknown either way, so it neither grants nor consumes
        # a known budget — the loss cap is UNKNOWN and entries are blocked.
        _write(tmp_path, {"version": 1, "realized_pnl": 1000.0, "closed_trades": 2})
        book = _book(tmp_path, {})
        assert book.loss_cap_unknown_reason() == "legacy_pnl_unreconciled"

    def test_v1_history_blocks_strategy_entries(self, tmp_path: Path) -> None:
        _write(tmp_path, {"version": 1, "realized_pnl": 48.38, "closed_trades": 7})
        strat = make_strat(tmp_path)
        intents, _, skipped = enter(strat, "EUR_USD", {"EURUSD": fresh(1.1)}, "long")
        assert intents == []
        assert skipped == [("EUR_USD", "legacy_pnl_unreconciled")]

    def test_flat_v1_book_reconciles_via_documented_procedure(self, tmp_path: Path) -> None:
        # Flat book (no legs) → nothing else would ever save; the migration
        # must persist at load so the operator can edit the v2 field.
        _write(tmp_path, {"version": 1, "realized_pnl": 48.38, "closed_trades": 7})
        assert _book(tmp_path, {}).legacy_unreconciled()
        state = json.loads((tmp_path / "book.json").read_text())
        assert state["version"] == 2
        assert state["legacy_mixed_currency_pnl"] == pytest.approx(48.38)
        assert state["legacy_closed_trades"] == 7
        state["legacy_reconciled_account_pnl"] = -12.0  # from broker records
        (tmp_path / "book.json").write_text(json.dumps(state))
        book = _book(tmp_path, {})
        assert book.loss_cap_unknown_reason() is None
        assert book.loss_cap_consumed() == pytest.approx(12.0)

    def test_empty_v1_history_does_not_block(self, tmp_path: Path) -> None:
        _write(tmp_path, {"version": 1, "realized_pnl": 0.0, "closed_trades": 0})
        assert _book(tmp_path, {}).loss_cap_unknown_reason() is None

    def test_operator_reconciled_legacy_counts_in_account_currency(
        self,
        tmp_path: Path,
    ) -> None:
        # Operator supplies the broker-statement value of the legacy period:
        # -1,500 USD. With -600 since migration: 2,100 consumed >= 2,000 cap
        # (the migration does not reset the budget).
        _write(
            tmp_path,
            {
                "version": 2,
                "account_currency": "USD",
                "realized_pnl": -600.0,
                "legacy_mixed_currency_pnl": 48.38,
                "legacy_closed_trades": 7,
                "legacy_reconciled_account_pnl": -1500.0,
            },
        )
        book = _book(tmp_path, {})
        assert book.loss_cap_unknown_reason() is None
        assert book.loss_cap_consumed() == pytest.approx(2100.0)
        assert book.breached(EQUITY)
        book.save()
        saved = json.loads((tmp_path / "book.json").read_text())
        assert saved["legacy_reconciled_account_pnl"] == -1500.0
        assert saved["legacy_mixed_currency_pnl"] == pytest.approx(48.38)

    def test_v2_legacy_without_count_stays_unreconciled(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {"version": 2, "account_currency": "USD", "legacy_mixed_currency_pnl": 0.0},
        )
        assert _book(tmp_path, {}).legacy_unreconciled()

    @pytest.mark.parametrize(
        "payload",
        [
            {"version": 3, "account_currency": "USD"},
            {"version": "2", "account_currency": "USD"},
            {"version": 2, "account_currency": "EUR"},
            {"version": 2},
            {"version": 2, "account_currency": "USD", "legacy_mixed_currency_pnl": "abc"},
            {"version": 2, "account_currency": "USD", "legacy_mixed_currency_pnl": "inf"},
            {"version": 2, "account_currency": "USD", "legacy_reconciled_account_pnl": "x"},
            {"version": 2, "account_currency": "USD", "legacy_reconciled_account_pnl": True},
            {
                "version": 2,
                "account_currency": "USD",
                "legacy_mixed_currency_pnl": 1.0,
                "legacy_closed_trades": -1,
            },
            {
                "version": 2,
                "account_currency": "USD",
                "unconverted_closes": [
                    {
                        "symbol": "USD_JPY",
                        "quote_ccy": "JPY",
                        "pnl_quote": -5.0,
                        "closed_at": "2026-10-06T00:00:00+00:00",
                        "reconciled_pnl_account": "nan",
                    }
                ],
            },
            {"version": 2, "account_currency": "USD", "unconverted_closes": {}},
            {
                "version": 2,
                "account_currency": "USD",
                "unconverted_closes": [{"symbol": "X", "quote_ccy": "JPY"}],
            },
            {"version": 1, "realized_pnl": "nan"},
        ],
    )
    def test_nonsense_state_fails_closed(self, tmp_path: Path, payload: dict[str, Any]) -> None:
        _write(tmp_path, payload)
        with pytest.raises(ValueError):
            _book(tmp_path, {})

    def test_legacy_position_without_currency_metadata_loads(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "version": 1,
                "realized_pnl": 0.0,
                "open_positions": {
                    "USD_JPY": {
                        "event_id": 3,
                        "entry_ts": datetime.now(UTC).isoformat(),
                        "entry_price": 150.0,
                        "quantity": -318.0,
                        "direction": -1,
                        "stop_price": 151.5,
                    }
                },
            },
        )
        book = _book(tmp_path, {"USDJPY": fresh(150.0)})
        pos = book.open_positions["USD_JPY"]
        assert pos.quote_ccy is None and pos.entry_conversion is None
        assert book.leg_quote_ccy(pos) == "JPY"


# =============================================================================
# Wiring
# =============================================================================


class TestWiring:
    def test_run_engine_wires_live_tick_dict(self, tmp_path: Path) -> None:
        from src.runtime.run_engine import wire_account_currency  # noqa: PLC0415

        strat = make_strat(tmp_path)
        live: dict[str, Any] = {}
        assert wire_account_currency([strat, object()], live) == 1
        intents, _, skipped = enter(strat, "USD_JPY", {"USDJPY": fresh(150.0)})
        # The wired converter reads the ENGINE dict, which is still empty.
        assert skipped == [("USD_JPY", "conversion_unavailable")]
        live["USDJPY"] = fresh(150.0)
        strat.book.pending_entries.clear()
        intents, _, skipped = enter(strat, "USD_JPY", {"USDJPY": fresh(150.0)})
        assert intents[0].target_position == pytest.approx(-50_000.0)

    def test_converter_account_currency_mismatch_rejected(self, tmp_path: Path) -> None:
        strat = make_strat(tmp_path)
        with pytest.raises(ValueError):
            strat.set_currency_converter(_conv({}, account="EUR"))
